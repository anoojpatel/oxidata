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
