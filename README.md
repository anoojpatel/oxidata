# Oxidata

Oxidata is a Python toolkit for building fast zero-copy / low-copy data structures and multiprocessing pipelines on top of POSIX shared memory, with runtime-enforced lifetime and borrowing discipline.

## Where it fits

Oxidata is intended for workloads that need multi-process concurrency but do not want the default Python behavior of:

- copying large payloads between processes
- pickling/unpickling (and the associated CPU + allocation overhead)
- losing control over memory lifetime and mutability when data becomes “shared”

Practically, it’s a substrate for:

- ML input pipelines that need to move batches/tensors efficiently across process boundaries
- producer/consumer systems that pass lightweight references (handles) instead of bytes
- shared off-heap caches that use immutable-after-publish semantics

## What it improves over basic Python

Compared to `multiprocessing.Queue` / pipes / manager objects (which primarily move Python objects via pickle):

- Performance
  - avoids repeated pickling/unpickling of large arrays/tensors
  - shifts transfer to shared memory plus small handle messages
  - optional native ring buffer and GIL-released shared-memory copies

- Ergonomics and safety
  - scoped lifetimes (`Frame`) and attachable global segments (`GlobalSegment`)
  - borrow/freeze patterns to reduce accidental mutation and use-after-free

## How it compares to common alternatives

- vs PyTorch `DataLoader` multiprocessing
  - Oxidata focuses on the transport substrate (shared-memory handles, ring, and slots) and can back a map-style dataset adapter.
  - It is a good fit when the bottleneck is serialization and copying of large samples/batches.

- vs using `multiprocessing.shared_memory` directly
  - Oxidata adds reusable building blocks: arenas, handles, scoped lifetimes, publish/freeze discipline, worker-pool substrate.
  - It reduces ad-hoc offset bookkeeping.

- vs Arrow-style columnar formats / Plasma-like object stores
  - Arrow is well-suited to standardized columnar interchange.
  - Oxidata targets tight inner-loop pipelines where you want explicit control over allocation, mutability, and lifetimes, and you want to pass handles through a ring quickly.

It’s designed for workloads like:

- ML input pipelines (PyTorch/NumPy/Arrow-style data) where pickling large tensors is a bottleneck
- Multiprocessing producer/consumer pipelines that pass *handles* instead of payloads
- Off-heap caches and shared segments with “immutable-after-publish” semantics

## What it provides

- Shared memory arena and handles
  - `SharedMemoryArena` allocates/attaches shared memory segments.
  - `Handle` references `(shm_name, offset, nbytes, kind)` so processes can exchange small metadata rather than big payloads.
  - Attach ergonomics: attaching infers size from the OS mapping (`size=None` when `create=False`).

- Runtime lifetimes and borrowing (safety for off-heap memory)
  - `Owned[T]` values can be borrowed immutably or mutably via context managers.
  - `Frame` and `GlobalSegment` provide ergonomic “scoped” and “named global” shared-memory lifetimes.
  - Publish/freeze patterns support “immutable-after-publish”.

- Multiprocessing dataloader substrate
  - A native shared-memory ring buffer transports *handle tuples*.
  - A slot arena stores payload bytes/arrays in shared memory.
  - `TensorSampleProducer` is the recommended path for nested tensor-like samples with NumPy leaves.
  - `Producer` publishes bytes/blobs/fixed-shape arrays for lower-level or fallback usage.
  - `WorkerPool`, `BlobWorkerPool`, `ArrayWorkerPool` attach in workers and execute your function.
  - `TensorSampleWorkerPool` reconstructs nested shared-memory-backed array views in workers.
  - `IndexRequestor` supports map-style request/response (buffering out-of-order results).

- Struct-of-arrays (SoA) utilities
  - `SoASchema` / `SoABatch` help represent fixed-schema data as columnar arrays.

## Documentation

- `docs/substrate_and_data_paths.md` explains:
  - the core substrate (shared memory, handles, lifetimes)
  - fixed-schema vs dynamic-key representations
  - how they map onto the current `oxidata.dataloader` APIs



## Install with uv

```bash
uv venv --python 3.12
uv sync
```

## Build native extension

```bash
uv run maturin develop --manifest-path oxidata_native/Cargo.toml
```

The native extension enables:

- `oxidata.native.ShmRingBuffer` (shared-memory ring buffer)
- GIL-released shared-memory copy helpers

If the native extension isn’t built, some multiprocessing pipeline features will be unavailable or skipped.

## Run tests

```bash
uv run python -m unittest discover -s tests -v
```

## Quick start

See `examples/`, `usecases/`, and `benchmarks/`.

### Recommended Path: nested tensor-like sample transport

```python
from oxidata.dataloader import TensorSampleProducer, TensorSampleWorkerPool

def fn_sample(sample):
    return int(sample["tokens"].sum())

producer = TensorSampleProducer(ring_name="oxidata-tree-ring")
pool = TensorSampleWorkerPool(
    metadata_shm_name=producer.metadata_shm_name,
    ring_name=producer.ring_name,
    fn_sample=fn_sample,
    ack_queue=producer.ack_queue,
)
```

This is the intended deep-learning path:

- NumPy leaves stay in shared memory.
- `msgspec_json`/JSON carries only small descriptors.
- Workers operate on array views, not Python blobs.
- One sample may span multiple payload slots.
- Important limit: each individual tensor leaf must currently fit within one payload slot.
- Torch conversion and CPU-to-GPU staging happen explicitly after worker reconstruction.

### Near-term roadmap

- Add an `OxidataDataLoader` that hides queues, acknowledgements, and slot recycling behind a DataLoader-like API.
- Add `return_format="numpy" | "torch"` and make tensor-oriented worker outputs the default path.
- Add built-in nested tensor-tree batching/collation instead of leaving batching to ad hoc user code.
- Add staging policies for `cpu`, `torch`, `torch_pinned`, and `device`.
- Add map-style dataset/source adapters for file-backed and object-store-backed data.
- Add benchmark support for repeated runs / medians so transport claims are based on stable measurements.

Blob transport remains available as a compatibility fallback when a sample cannot be described cleanly as metadata plus array leaves.

Warning:

- If a single tensor leaf is larger than `payload_slot_size`, `TensorSampleProducer` raises `MemoryError`.
- Multi-slot support currently applies across leaves within a sample, not within one leaf.
- If your workload has single leaves larger than a practical slot size, the next required feature is segmented-leaf descriptors.

### Torch staging

If workers reconstruct NumPy-backed tensor samples in shared memory, you can stage them explicitly:

```python
from oxidata import tensor_tree_to_torch, pin_memory_tree, stage_tree_to_device

cpu_tree = tensor_tree_to_torch(sample, pin_memory=True)
gpu_tree = stage_tree_to_device(cpu_tree, "cuda", non_blocking=True)
```

Notes:

- `tensor_tree_to_torch(..., pin_memory=True)` is the intended pinned-host-memory path when PyTorch is available.
- GPU transfer is still expected for training workloads; the goal is to avoid extra CPU copies before that transfer.

## Benchmarks

Measured on this machine:

- MacBook Pro `MacBookPro18,1`
- Apple M1 Pro, `10` CPU cores
- `16 GB` unified memory

Transport-path results from `benchmarks/bench_payload_matrix.py`:

| Payload | `mp.Queue` msg/s | `slot+ring+copy` msg/s | `slot+ring+view` msg/s |
|---|---:|---:|---:|
| 64 B | 116931 | 39013 | 98217 |
| 256 B | 88669 | 43522 | 72061 |
| 1 KiB | 56120 | 33434 | 57091 |
| 4 KiB | 55268 | 31119 | 59859 |
| 16 KiB | 15420 | 10214 | 10332 |
| 64 KiB | 11849 | 9137 | 11905 |
| 256 KiB | 3046 | 2440 | 2757 |
| 1 MiB | 686 | 496 | 637 |
| 4 MiB | 139 | 145 | 182 |
| 100 MiB | 7 | 8 | 11 |

Metadata-codec results from the same benchmark:

| Descriptor context | `json` ops/s | `msgspec_json` ops/s |
|---|---:|---:|
| 64 B payload descriptor | 52446 | 51940 |
| 256 B payload descriptor | 47309 | 58281 |
| 1 KiB payload descriptor | 43569 | 56615 |
| 4 KiB payload descriptor | 29953 | 48861 |
| 16 KiB payload descriptor | 12661 | 35996 |
| 64 KiB payload descriptor | 3857 | 16347 |
| 256 KiB payload descriptor | 1061 | 5862 |
| 1 MiB payload descriptor | 271 | 1610 |
| 4 MiB payload descriptor | 63 | 401 |
| 100 MiB payload descriptor | 2 | 13 |

Interpretation:

- `slot+ring+copy` is not the target deep-learning path; it copies payloads back into Python-owned bytes in the worker.
- `slot+ring+view` is the meaningful shared-memory result. It is competitive or better once the worker stays attached and operates on shared-memory views.
- `json` and `msgspec_json` here measure descriptor metadata roundtrip only, not bulk tensor transport.
- `msgspec_json` is the intended metadata codec because it is faster on structured descriptors while tensor bytes stay in shared memory.

Other benchmark results on the same machine:

- `benchmarks/bench_native_ringbuffer.py`: `322946 msg/s`
- `benchmarks/bench_blob_vs_pickle.py`: `pickle 0.0044s`, `shm+json 0.0208s`, `shm+msgspec 0.0081s` for `200` iterations
- `benchmarks/bench_dataloader_source_pipeline.py`: file-backed/object-store-like benchmark with optional per-sample sleep, comparing streaming file reads against cached spawn-safe representations

### Spawn DataLoader memory/speed benchmark

`benchmarks/bench_dataloader_spawn_memory.py` compares dataset representations under `torch.utils.data.DataLoader(..., multiprocessing_context="spawn")`:

- `python_list`: plain Python list of nested NumPy samples
- `torch_tensors`: nested torch tensors stored directly in the dataset
- `torch_serialize_full`: serialized full-sample list stored in torch `uint8` tensor storage
- `oxidata_descriptors`: Oxidata-style small Python descriptors pointing at shared-memory NumPy leaves

Measured on the same M1 Pro MacBook Pro with:

- `32` samples
- `4 MiB` per sample
- `4` workers
- `8` warmup samples
- `16` measured samples

| Method | Samples/s | Parent footprint | Workers footprint | Total footprint |
|---|---:|---:|---:|---:|
| `python_list` | 450.06 | 0.96 GiB | 1.11 GiB | 2.07 GiB |
| `torch_tensors` | 5175.13 | 1.10 GiB | 0.57 GiB | 1.67 GiB |
| `torch_serialize_full` | 360.30 | 1.10 GiB | 0.68 GiB | 1.78 GiB |
| `oxidata_descriptors` | 557.90 | 1.00 GiB | 0.65 GiB | 1.65 GiB |

Interpretation:

- `python_list` shows the worst combined memory footprint in this spawn setting.
- `torch_tensors` is the strongest baseline here on both speed and footprint.
- `torch_serialize_full` reduces worker footprint relative to `python_list`, but it still pays decode cost.
- `oxidata_descriptors` is close to the torch-sharing memory story while keeping the transport shape explicit and tensor-oriented.

Metric caveat:

- This benchmark uses per-process macOS `vmmap` physical footprint.
- Summed parent + worker footprint is still an approximation, not Linux-style `PSS`.

### File-backed / object-store-like source benchmark

`benchmarks/bench_dataloader_source_pipeline.py` adds file IO and optional synthetic source latency. It compares:

- `streaming_torch_files`: read sample files on demand and convert to torch in workers
- `streaming_torch_serialized_index`: blog-faithful torch-serialized metadata/index path, with real payload still read in workers
- `python_list_cache`: ingest all files into Python objects up front
- `torch_tensors_cache`: ingest all files into torch-backed storage up front
- `torch_serialize_full_cache`: ingest all files into a torch-backed serialized full-sample cache up front
- `oxidata_descriptors_cache`: ingest all files into Oxidata shared-memory descriptors up front

Representative run on this machine:

- `32` samples
- `4 MiB` per sample
- `4` workers
- `8` warmup
- `16` measured
- `spawn`
- `10 ms` synthetic source latency per sample

| Method | Build time | Samples/s | Parent footprint | Workers footprint | Total footprint |
|---|---:|---:|---:|---:|---:|
| `streaming_torch_files` | 0.00 s | 273.86 | 0.92 GiB | 0.64 GiB | 1.55 GiB |
| `streaming_torch_serialized_index` | 0.00 s | 349.58 | 0.74 GiB | 0.63 GiB | 1.38 GiB |
| `python_list_cache` | 1.06 s | 313.30 | 0.91 GiB | 1.11 GiB | 2.03 GiB |
| `torch_tensors_cache` | 0.56 s | 4647.17 | 0.91 GiB | 0.58 GiB | 1.49 GiB |
| `torch_serialize_full_cache` | 0.62 s | 239.10 | 0.94 GiB | 0.68 GiB | 1.63 GiB |
| `oxidata_descriptors_cache` | 0.63 s | 311.29 | 0.85 GiB | 0.65 GiB | 1.50 GiB |

Interpretation:

- `torch_tensors_cache` remains the strongest baseline when you can afford full upfront conversion into torch storage.
- `streaming_torch_serialized_index` is the closest match to the blog post's idea: share only the dataset index/metadata through a torch-backed serialized list, then read the real payload in workers.
- `torch_serialize_full_cache` is intentionally harsher than the blog pattern because it serializes the full sample payload, not just metadata.
- `oxidata_descriptors_cache` is much closer to the shared-storage memory story than `python_list_cache`.
- `streaming_torch_files` avoids the upfront ingest/build cost, but throughput is then bounded by source latency and per-sample reconstruction.

### End-to-end streaming benchmark

`benchmarks/bench_end_to_end_streaming.py` measures the full source-to-main-process path instead of a cached representation:

- sample files are read in worker processes
- `--sleep-ms` can simulate object-store/network latency
- `streaming_torch_files` returns torch-native samples from workers
- `streaming_torch_serialized_index` uses the blog-faithful torch-serialized metadata/index trick, then reads payloads in workers
- `oxidata_streaming` writes leaves into bounded shared-memory payload slots and sends only descriptors back

Representative run on this machine:

- `128` samples
- `4 MiB` per sample
- `4` workers
- `16` warmup
- `64` measured
- `spawn`
- `10 ms` synthetic source latency per sample
- `payload_slot_mib=8`
- `inflight_slots=16`
- `oxidata_stage_torch=True`

| Method | Samples/s | Parent footprint | Workers footprint | Total footprint |
|---|---:|---:|---:|---:|
| `streaming_torch_files` | 111.68 | 0.16 GiB | 0.64 GiB | 0.81 GiB |
| `streaming_torch_serialized_index` | 110.02 | 0.16 GiB | 0.64 GiB | 0.81 GiB |
| `oxidata_streaming` | 201.22 | 0.16 GiB | 0.72 GiB | 0.87 GiB |

Interpretation:

- this is the benchmark closest to the actual reason the library exists: source IO in workers, cross-process transport, and reconstruction in the training process
- `streaming_torch_files` must move the fully materialized sample through PyTorch's multiprocessing path on every item
- `streaming_torch_serialized_index` shows that the blog-faithful torch-serialized metadata/index trick helps the dataset representation story, but it does not remove the cost of transporting full materialized samples once workers have read them
- `oxidata_streaming` keeps bulk data in explicit shared memory and sends only descriptors across the process boundary
- the benchmark keeps only a bounded amount of work in flight so the measurement reflects steady-state transport rather than precomputed backlog
- the memory footprints stay close, but the transport story is materially different once the benchmark includes the real source stage

Scaling sweep on the same machine (`32` samples, `4` workers, `8` measured samples):

| Sample size | `python_list` total footprint / samples/s | `torch_tensors` total footprint / samples/s | `torch_serialize_full` total footprint / samples/s | `oxidata_descriptors` total footprint / samples/s |
|---|---:|---:|---:|---:|
| 1 MiB | 1.17 GiB / 1088.02 | 1.03 GiB / 3265.69 | 0.98 GiB / 1082.63 | 0.91 GiB / 1019.88 |
| 4 MiB | 2.12 GiB / 368.42 | 1.67 GiB / 3601.51 | 1.67 GiB / 353.16 | 1.53 GiB / 348.09 |
| 8 MiB | 3.65 GiB / 156.71 | 1.87 GiB / 5135.89 | 2.38 GiB / 139.84 | 2.40 GiB / 185.43 |
| 16 MiB | 6.00 GiB / 102.24 | 3.28 GiB / 4878.67 | 4.19 GiB / 89.55 | 4.12 GiB / 88.24 |

Pressure takeaway:

- `python_list` scales worst in combined footprint as sample size rises.
- `torch_tensors` remains the strongest baseline when the dataset can be pre-materialized as torch storage.
- `oxidata_descriptors` stays much closer to the shared-storage baselines than the plain Python representation.
- `torch_serialize_full` reduces memory pressure relative to `python_list`, but decode cost becomes more visible as samples grow.

`fork` contrast on the same `32` samples x `4 MiB`, `4` workers, `16` measured configuration:

| Context | `python_list` total footprint | `torch_tensors` total footprint | `torch_serialize_full` total footprint | `oxidata_descriptors` total footprint |
|---|---:|---:|---:|---:|
| `spawn` | 2.12 GiB | 1.67 GiB | 1.78 GiB | 1.53 GiB |
| `fork` | 0.97 GiB | 0.99 GiB | 0.99 GiB | 0.96 GiB |

Why this matters:

- under `fork`, copy-on-write already removes most of the memory blowup, so all four methods converge much more closely
- under `spawn`, explicit shared-storage approaches matter much more
- this project is primarily targeting the non-`fork` case, where plain Python objects scale poorly

### Example: scoped off-heap bytes with `Frame`

```python
from oxidata.scopes import Frame
from oxidata.offheap import borrow_region, borrow_region_mut

with Frame(size=8 * 1024 * 1024) as f:
    h = f.alloc_bytes(b"hello")

    with borrow_region(h) as r:
        print(bytes(r.view))

    with borrow_region_mut(h) as r:
        r.view[:5] = b"HELLO"

    pub = f.publish(h)  # immutable-after-publish
```

### Example: multiprocessing pipeline (producer/workers)

```python
from oxidata.dataloader import Producer, WorkerPool
from oxidata.native import available

if not available():
    raise SystemExit("native extension not built")

producer = Producer(ring_name="oxidata-demo-ring", shm_size=8 * 1024 * 1024, slot_size=256)

def fn_bytes(b: bytes) -> int:
    return len(b)

pool = WorkerPool(shm_name=producer.shm_name, ring_name=producer.ring_name, fn_bytes=fn_bytes, num_workers=2)
pool.start()

try:
    producer.publish(b"abc")
    pool.stop()
    pool.join(timeout=10)
    print(pool.results().get())
finally:
    producer.cleanup()
```

Notes:

- Worker callbacks run in spawned subprocesses. Top-level functions are safest.
- Closure-free Python functions also work; closures are not supported.

### Example: PyTorch map-style dataset adapter

For a high-level PyTorch-facing adapter that does IO in oxidata workers, see:

- `examples/torch_oxidata_index_dataset.py`

When your dataset adapter internally spawns oxidata workers, prefer:

- `torch.utils.data.DataLoader(..., num_workers=0)`
