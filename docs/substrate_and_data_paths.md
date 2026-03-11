# Oxidata substrate and data paths (fixed schema vs dynamic keys)

## 1. What the core substrate is doing

Oxidata is a shared-memory + handle + runtime-lifetime substrate for building high-throughput multiprocessing pipelines with minimal copying.

The core idea is:

- Allocate and/or attach shared memory segments.
- Represent data as lightweight handles into shared memory.
- Enforce lifetimes and borrow rules at runtime (to prevent use-after-free and illegal aliasing).
- Use a native ring buffer to pass handle metadata between processes without pickling whole payloads.

### 1.1 Shared memory primitives

- `oxidata.shm_arena.SharedMemoryArena`
  - Creates or attaches a POSIX shared memory segment.
  - On attach (`create=False`), the arena infers size from the OS mapping (`size=None`), so callers don’t need placeholder sizes.

- `oxidata.shm_arena.Handle`
  - A compact reference to a contiguous byte region inside a shared memory segment.
  - Conceptually: `(shm_name, offset, nbytes, kind)`.
  - A `Handle` is what you pass around cheaply between processes.

### 1.2 Lifetime + borrow enforcement

Oxidata includes runtime lifetime and borrowing constructs (in `oxidata.lifetimes` and `oxidata.scopes`) to make off-heap/shared-memory usage safer:

- `Scope` / `Arena`
- `Owned[T]` and borrow context managers
- `Frame` and `GlobalSegment`
  - `Frame` is an ergonomic “stack-frame-like” lifetime.
  - `GlobalSegment` is a “named shared segment” intended for attach/reuse patterns.

Also:

- Publish/freeze semantics (`publish` / `freeze`) enable “immutable-after-publish” patterns.

### 1.3 Multiprocessing transport substrate: slots + ring

The multiprocessing “dataloader” pieces in `oxidata.dataloader` are built on:

- A **slot arena** (a fixed-size shared memory segment divided into fixed-size slots).
- A **native ring buffer** (`oxidata.native.ShmRingBuffer`) that only transports compact handle tuples.

The flow is:

1. Producer writes payload bytes into a slot.
2. Producer pushes `(offset, nbytes, kind_tag)` into the ring.
3. Worker pops, opens data (bytes/blob/array), runs a function, returns results via `mp.Queue`.

Key APIs:

- `Producer.publish(payload: bytes)`
- `Producer.publish_blob(obj, codec=...)` (codec encodes to bytes, bytes stored in slot)
- `Producer.publish_array(np_array)` (fixed-shape ndarray path)

Workers:

- `WorkerPool(fn_bytes)`
- `BlobWorkerPool(codec, fn_obj)`
- `ArrayWorkerPool(dtype, shape, fn_arr)`

A small helper exists for index-based request/response:

- `IndexRequestor`
  - Buffers out-of-order responses.
  - Enforces `max_in_flight` backpressure.
  - Expects worker results that contain an index, e.g. `(i, value)`.

## 2. Data path families for ML samples

When integrating with ML frameworks (PyTorch / NumPy / Arrow), the main guideline is to avoid serializing large array/tensor payloads through a blob codec.

Instead:

- Keep bulk tensor/array buffers in shared memory.
- Only serialize small metadata (shapes, dtypes, offsets, field names).

There are two common ways to model a sample or a batch:

- Fixed schema (SoA / stable field set)
- Dynamic keys (dict-of-arrays / nested structures)

## 3. Fixed schema path

### 3.1 When to use

Use a fixed schema when:

- Your model input signature is stable (e.g. `input_ids`, `attention_mask`, `labels`).
- Keys and shapes are predictable.
- You want maximal throughput and minimal per-sample overhead.

### 3.2 Representation (SoA)

A fixed schema is naturally represented as a struct-of-arrays (SoA):

- One named column per feature.
- Each column is a contiguous typed array.

In oxidata you already have a schema/batch abstraction in `oxidata.soa`:

- `SoASchema`
- `SoABatch`

A conceptual producer-side shape:

```python
from oxidata.soa import SoASchema, SoABatch
from oxidata.shm_arena import SharedMemoryArena

schema = SoASchema.from_mapping({
    "x": "float32",
    "y": "int64",
})

arena = SharedMemoryArena(size=64 * 1024 * 1024, create=True)

batch = SoABatch.alloc(arena, schema=schema, length=1024)
cols, shms = SoABatch.open(arena, batch)
try:
    cols["x"][...] = ...
    cols["y"][...] = ...
finally:
    SoABatch.close_opened(shms)

# Transport: send a small descriptor (handles/offsets), not the bytes.
```

### 3.3 Transport: what crosses the process boundary

For fixed schema, you generally only need to pass:

- A batch descriptor (handles + lengths)
- Optionally a sample index (if batching is aligned by index)

Codec use:

- You can use `msgspec` or JSON for the descriptor, but the descriptor should be small.
- Avoid encoding the array bytes.

### 3.4 Worker-side opening

Workers attach to shared memory and open columns as NumPy views (or construct torch tensors from them).

In this model, “marshalling” is minimal:

- Pass a small descriptor.
- Interpret typed views using dtype/shape.

## 4. Dynamic keys path

### 4.1 When to use

Use dynamic keys when:

- Samples have optional or varying fields.
- You need nested structures.
- You’re prototyping and don’t want to lock in a schema.

### 4.2 Two sub-approaches

#### A) All-blob (simplest)

- Encode the entire sample dict as bytes using a codec.
- Put it into a slot.
- Worker decodes it.

This is fine for:

- small metadata objects
- small samples

This is not ideal when the dict contains large arrays.

#### B) Metadata blob + shm handles for arrays

- For each array/tensor-like field, allocate bytes in shared memory and produce a `Handle`.
- Build a metadata object that maps keys to handle info plus dtype/shape.
- Encode only the metadata (small) via `msgspec` (preferred) or JSON.

Example metadata shape:

```python
sample_desc = {
  "tensors": {
    "image": {"dtype": "uint8", "shape": [3, 224, 224], "offset": 123, "nbytes": 150528},
    "label": {"dtype": "int64", "shape": [], "offset": 456, "nbytes": 8},
  },
  "scalars": {"id": 17},
}
```

This supports dynamic keys while keeping large buffers in shared memory.

Codec recommendation:

- Prefer `msgspec` (if installed) for speed.
- JSON is acceptable for correctness/prototyping.

## 5. PyTorch integration shape: Map-style dataset adapter

For a PyTorch-like API, a map-style dataset adapter can sit on top of the substrate.

### 5.1 Recommended constraint

If your dataset adapter internally spawns oxidata workers and owns a ring/shm segment, run torch’s DataLoader with:

- `num_workers=0`

This prevents “double multiprocessing” (torch workers each spawning their own oxidata pools).

### 5.2 IO in workers (index-based)

This is the recommended first path for real workloads:

- Main process sends only an index request.
- Worker reconstructs the underlying dataset once per process (`make_dataset()`), then does `__getitem__`.
- Worker returns `(index, value)`.
- Main uses `IndexRequestor` to block on the correct response and buffer out-of-order results.

See `examples/torch_oxidata_index_dataset.py` for a working example.

### 5.3 IO in main (transform in workers)

The same request/response machinery works if IO is cheap and you only want to offload transforms:

- Main loads sample `s = base[i]`.
- Main writes sample into shm (bytes/blob/array/handles).
- Worker transforms and returns result.

The difference between “IO in main” vs “IO in workers” is primarily **what the request payload contains**, not the transport substrate.

## 6. Summary

- Prefer fixed schema / SoA when you want maximal throughput and predictable training inputs.
- Use dynamic keys when you need flexibility, but keep tensor/array bytes in shared memory.
- Use codecs only for small metadata, not for the bulk numeric buffers.
- Use `IndexRequestor` for map-style request/response patterns.
- For PyTorch integration, prefer `DataLoader(..., num_workers=0)` when oxidata manages worker processes.

## 7. Addendum: how these paths map onto the current `oxidata.dataloader` APIs

This section ties the two “data path” families to the concrete dataloader substrate you have today.

Recommended default for deep learning workloads:

- `TensorSampleProducer`
- `TensorSampleWorkerPool`
- `msgspec_json` descriptors for metadata only
- explicit torch staging (`tensor_tree_to_torch`, `pin_memory_tree`, `stage_tree_to_device`) after worker reconstruction
- samples may span multiple payload slots
- hard current limit: each individual tensor leaf must fit within one payload slot

Use blob transport only as fallback when a sample cannot be described cleanly as metadata plus array leaves.

This limit matters:

- multi-slot support currently distributes different leaves of the same sample across slots
- a single oversized leaf is rejected rather than fragmented
- if one tensor leaf exceeds `payload_slot_size`, the next required extension is segmented-leaf descriptors

This section ties the two “data path” families to the concrete dataloader substrate you have today:

- `Producer` + `ShmRingBuffer` for transport
- one of the worker pools (`ArrayWorkerPool`, `BlobWorkerPool`, `WorkerPool`)
- `IndexRequestor` for map-style request/response

### 7.1 Fixed schema via `ArrayWorkerPool` (single fixed-shape tensor per sample)

This is the most direct “fixed schema” mapping when each sample is a single fixed-shape array (or you can pack multiple fields into one fixed tensor).

Notes:

- No per-sample codec cost for the array bytes.
- Only the handle tuple crosses the ring.
- Worker reconstructs an `ndarray` view and runs `fn_arr`.

Sketch:

```python
import numpy as np

from oxidata.dataloader import Producer, ArrayWorkerPool, IndexRequestor

producer = Producer(ring_name="ring", shm_size=256 * 1024 * 1024, slot_size=4096)

def fn_arr(a: np.ndarray):
    # a is a view over shared-memory bytes
    # return (i, value) if you want IndexRequestor semantics
    return float(a.sum())

pool = ArrayWorkerPool(
    shm_name=producer.shm_name,
    ring_name=producer.ring_name,
    dtype="float32",
    shape=(1024,),
    fn_arr=fn_arr,
    num_workers=4,
)
pool.start()

q = pool.results()

# If you want map-style indexing, ensure worker returns (i, value)
# and wire IndexRequestor to publish an index request.
```

Notes:

- The built-in `ArrayWorkerPool` currently assumes a single `(dtype, shape)` configured at pool creation.
- For a multi-field fixed schema, you can either:
  - pack fields into a single fixed tensor, or
  - extend the substrate to support “multi-array per sample” (descriptor + multiple handles), which looks like the dynamic-keys descriptor below but with a fixed field set.

### 7.2 Fixed schema via SoA (`oxidata.soa`) (multi-field, columnar)

If you want a true fixed schema with multiple named arrays, SoA is the natural model.

How it fits the dataloader substrate:

- The “payload” you send over the ring should be a **small descriptor** (handles/offsets) referencing the SoA columns.
- The worker should not serialize the column bytes; it should only open views.

You can transport the descriptor via:

- `BlobWorkerPool` using `msgspec`/JSON for descriptor encoding, or
- `WorkerPool` using a compact binary header (future optimization).

### 7.3 Dynamic keys via “multi-handle descriptor” (dict-of-arrays)

When keys are dynamic, the recommended high-performance pattern is:

- Write each array buffer into shared memory and capture a handle.
- Send **metadata only** (keys, dtype, shape, offsets) as a blob.

A recommended descriptor shape:

```python
sample_desc = {
  "i": 123,
  "tensors": {
    "image": {"dtype": "uint8", "shape": [3, 224, 224], "offset": 1000, "nbytes": 150528},
    "label": {"dtype": "int64", "shape": [], "offset": 2000, "nbytes": 8},
  },
  "scalars": {"id": 123, "source": "train"},
}
```

Transport mapping:

- Use `Producer.publish_blob(sample_desc, codec="msgspec_json")` (preferred) or `codec="json"`.
- Use `BlobWorkerPool(codec=..., fn_obj=...)` if you want workers to *consume* these descriptors.

Where the array bytes live:

- The offsets/nbytes in the descriptor refer to regions of a shared memory segment.
- In the simplest version, those regions are “slots” inside the `Producer`’s slot arena.
- For larger/variable arrays, you typically want either larger slots or a different allocation strategy (arena allocations) and then only pass the resulting handles.

### 7.4 Recommendation: which mapping to start with

- If your sample is naturally a single fixed-shape tensor: start with `ArrayWorkerPool`.
- If your sample is multiple arrays with a stable signature: use SoA and pass a small descriptor.
- If your sample is a dict-of-arrays with dynamic keys: use a “multi-handle descriptor” and encode metadata with `msgspec`.

In all cases, keep the invariant: codecs are for metadata only; bulk numeric buffers should remain in shared memory and be referenced by handles.

## 8. Benchmark notes (local development machine)

Recent benchmark runs on:

- MacBook Pro `MacBookPro18,1`
- Apple M1 Pro
- 10 CPU cores
- 16 GB unified memory

Transport-path highlights:

| Payload | `mp.Queue` msg/s | `slot+ring+copy` msg/s | `slot+ring+view` msg/s |
|---|---:|---:|---:|
| 64 B | 116931 | 39013 | 98217 |
| 256 B | 88669 | 43522 | 72061 |
| 1 KiB | 56120 | 33434 | 57091 |
| 64 KiB | 11849 | 9137 | 11905 |
| 1 MiB | 686 | 496 | 637 |
| 4 MiB | 139 | 145 | 182 |
| 100 MiB | 7 | 8 | 11 |

Metadata-codec highlights from the same benchmark:

| Descriptor context | `json` ops/s | `msgspec_json` ops/s |
|---|---:|---:|
| 64 B payload descriptor | 52446 | 51940 |
| 256 B payload descriptor | 47309 | 58281 |
| 1 KiB payload descriptor | 43569 | 56615 |
| 64 KiB payload descriptor | 3857 | 16347 |
| 1 MiB payload descriptor | 271 | 1610 |
| 100 MiB payload descriptor | 2 | 13 |

Interpretation:

- the meaningful shared-memory comparison is `slot+ring+view`, not `slot+ring+copy`
- the copy variant still pays a worker-side materialization cost back into Python-owned memory
- the view variant is the closest measurement of the intended tensor-first path
- the codec table is a metadata-only measurement; it should not be read as a bulk tensor transport benchmark
- GPU transfer is still expected after this stage; these numbers are about CPU-side transport and staging only

## 9. Spawn DataLoader comparison (against torch-sharing baseline)

`benchmarks/bench_dataloader_spawn_memory.py` compares dataset storage strategies under `torch.utils.data.DataLoader(..., multiprocessing_context="spawn")`.

Compared methods:

- `python_list`: plain Python list of nested NumPy samples
- `torch_tensors`: dataset stores nested torch tensors directly
- `torch_serialize`: dataset stores a serialized list in torch `uint8` tensor storage
- `oxidata_descriptors`: dataset stores small Python descriptors while bulk NumPy leaves live in shared memory

Run configuration:

- `32` samples
- `4 MiB` per sample
- `4` workers
- `8` warmup samples
- `16` measured samples

Results on the same M1 Pro MacBook Pro:

| Method | Samples/s | Parent footprint | Workers footprint | Total footprint |
|---|---:|---:|---:|---:|
| `python_list` | 450.06 | 0.96 GiB | 1.11 GiB | 2.07 GiB |
| `torch_tensors` | 5175.13 | 1.10 GiB | 0.57 GiB | 1.67 GiB |
| `torch_serialize` | 360.30 | 1.10 GiB | 0.68 GiB | 1.78 GiB |
| `oxidata_descriptors` | 557.90 | 1.00 GiB | 0.65 GiB | 1.65 GiB |

How to read this:

- `torch_tensors` is the strongest reference baseline on this machine for spawn-based loading.
- `torch_serialize` is a relevant baseline because it shares storage better than plain Python objects, but it still decodes at item access.
- `oxidata_descriptors` keeps most of the memory benefit while preserving an explicit descriptor-plus-shared-buffer model.
- `python_list` is the representation Oxidata is explicitly trying to avoid for large nested tensor samples.

Metric caveat:

- macOS does not expose Linux-style `PSS` in the same way as the original article.
- This benchmark uses per-process `vmmap` physical footprint.
- Summed footprint should be treated as an approximation when comparing shared-storage approaches.

Scaling sweep on the same machine (`32` samples, `4` workers, `8` measured samples):

| Sample size | `python_list` total footprint / samples/s | `torch_tensors` total footprint / samples/s | `torch_serialize` total footprint / samples/s | `oxidata_descriptors` total footprint / samples/s |
|---|---:|---:|---:|---:|
| 1 MiB | 1.17 GiB / 1088.02 | 1.03 GiB / 3265.69 | 0.98 GiB / 1082.63 | 0.91 GiB / 1019.88 |
| 4 MiB | 2.12 GiB / 368.42 | 1.67 GiB / 3601.51 | 1.67 GiB / 353.16 | 1.53 GiB / 348.09 |
| 8 MiB | 3.65 GiB / 156.71 | 1.87 GiB / 5135.89 | 2.38 GiB / 139.84 | 2.40 GiB / 185.43 |
| 16 MiB | 6.00 GiB / 102.24 | 3.28 GiB / 4878.67 | 4.19 GiB / 89.55 | 4.12 GiB / 88.24 |

Pressure takeaway:

- `python_list` is the representation that blows up fastest as sample size rises.
- `torch_tensors` is the strongest direct PyTorch baseline when you can preconvert the dataset into torch storage.
- `oxidata_descriptors` tracks much closer to the shared-storage baselines than the plain Python representation.
- `torch_serialize` is memory-aware but decode-heavy, especially as samples get larger.

`fork` contrast on the same `32` samples x `4 MiB`, `4` workers, `16` measured configuration:

| Context | `python_list` total footprint | `torch_tensors` total footprint | `torch_serialize` total footprint | `oxidata_descriptors` total footprint |
|---|---:|---:|---:|---:|
| `spawn` | 2.12 GiB | 1.67 GiB | 1.78 GiB | 1.53 GiB |
| `fork` | 0.97 GiB | 0.99 GiB | 0.99 GiB | 0.96 GiB |

Interpretation:

- under `fork`, copy-on-write already suppresses most of the memory blowup
- under `spawn`, representation choice matters much more
- the main motivation for this substrate is the non-`fork` setting, not trying to beat `fork` on its natural memory-sharing path

## 10. File-backed / object-store-like source benchmark

`benchmarks/bench_dataloader_source_pipeline.py` adds a more realistic source stage:

- samples are written as files on disk
- workers either stream from files or consume a cached post-ingest representation
- `--sleep-ms` can simulate object-store/network latency per sample

Compared methods:

- `streaming_torch_files`
- `python_list_cache`
- `torch_tensors_cache`
- `torch_serialize_cache`
- `oxidata_descriptors_cache`

Representative run:

- `32` samples
- `4 MiB` per sample
- `4` workers
- `8` warmup
- `16` measured
- `spawn`
- `10 ms` synthetic latency per sample

| Method | Build time | Samples/s | Parent footprint | Workers footprint | Total footprint |
|---|---:|---:|---:|---:|---:|
| `streaming_torch_files` | 0.00 s | 313.18 | 0.92 GiB | 0.63 GiB | 1.55 GiB |
| `python_list_cache` | 1.01 s | 338.16 | 0.91 GiB | 1.11 GiB | 2.02 GiB |
| `torch_tensors_cache` | 0.55 s | 5286.20 | 0.82 GiB | 0.57 GiB | 1.40 GiB |
| `torch_serialize_cache` | 0.63 s | 232.42 | 0.91 GiB | 0.69 GiB | 1.60 GiB |
| `oxidata_descriptors_cache` | 0.64 s | 513.73 | 0.92 GiB | 0.65 GiB | 1.58 GiB |

How to read this:

- `torch_tensors_cache` is still the strongest direct baseline if full upfront tensor materialization is acceptable.
- `oxidata_descriptors_cache` keeps worker footprint much closer to the shared-storage baselines than `python_list_cache`.
- `streaming_torch_files` shows the cost of staying on the source path without a prebuilt cached representation.
- `torch_serialize_cache` still pays visible decode overhead.

## 11. End-to-end streaming benchmark

`benchmarks/bench_end_to_end_streaming.py` measures the full source-to-main-process path rather than a cached post-ingest representation:

- workers read sample files directly
- `--sleep-ms` can simulate object-store/network latency
- `streaming_torch_files` returns torch-native samples from workers
- `oxidata_streaming` writes bulk leaves into bounded shared-memory payload slots and returns only descriptors

Representative run:

- `32` samples
- `4 MiB` per sample
- `4` workers
- `8` warmup
- `16` measured
- `spawn`
- `10 ms` synthetic latency per sample
- `payload_slot_mib=8`
- `inflight_slots=16`
- `oxidata_stage_torch=True`

| Method | Samples/s | Parent footprint | Workers footprint | Total footprint |
|---|---:|---:|---:|---:|
| `streaming_torch_files` | 111.94 | 0.16 GiB | 0.64 GiB | 0.80 GiB |
| `oxidata_streaming` | 347.11 | 0.16 GiB | 0.69 GiB | 0.85 GiB |

Interpretation:

- this is the benchmark closest to the intended workload: source IO in workers, explicit cross-process transport, then reconstruction in the consumer
- `streaming_torch_files` pays the native torch multiprocessing path on every sample after file read and tensor construction
- `oxidata_streaming` keeps bulk numeric payloads in explicit shared memory and sends only lightweight descriptors across the process boundary
- the benchmark keeps only a bounded amount of work in flight so the result reflects steady-state transport instead of a prefilled result queue
- the footprint stays in the same range, but the transport model is much closer to the substrate this project is trying to provide
