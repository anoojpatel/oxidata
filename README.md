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
  - `Producer` publishes bytes/blobs/fixed-shape arrays.
  - `WorkerPool`, `BlobWorkerPool`, `ArrayWorkerPool` attach in workers and execute your function.
  - `IndexRequestor` supports map-style request/response (buffering out-of-order results).

- Struct-of-arrays (SoA) utilities
  - `SoASchema` / `SoABatch` help represent fixed-schema data as columnar arrays.

## Documentation

- `docs/substrate_and_data_paths.md` explains:
  - the core substrate (shared memory, handles, lifetimes)
  - fixed-schema vs dynamic-key representations
  - how they map onto the current `oxidata.dataloader` APIs



## Install (pure Python)

```bash
uv pip install -e .
```

## Build native extension

```bash
uv pip install maturin
maturin develop -m pyocaml_native/pyproject.toml
```

The native extension enables:

- `oxidata.native.ShmRingBuffer` (shared-memory ring buffer)
- GIL-released shared-memory copy helpers

If the native extension isn’t built, some multiprocessing pipeline features will be unavailable or skipped.

## Run tests

```bash
python3 -m unittest
```

## Quick start

See `examples/`, `usecases/`, and `benchmarks/`.

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

### Example: PyTorch map-style dataset adapter

For a high-level PyTorch-facing adapter that does IO in oxidata workers, see:

- `examples/torch_oxidata_index_dataset.py`

When your dataset adapter internally spawns oxidata workers, prefer:

- `torch.utils.data.DataLoader(..., num_workers=0)`
