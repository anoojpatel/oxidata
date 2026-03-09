from __future__ import annotations

import importlib
import marshal
import multiprocessing as mp
import pickle
import queue
import threading
import types
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from .native import ShmRingBuffer, available as native_available
from .shm_arena import Handle, SharedMemoryArena

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class HandleMsg:
    shm_name: str
    offset: int
    nbytes: int
    kind_tag: int

    def to_handle(self) -> Handle:
        return Handle(self.shm_name, self.offset, self.nbytes, "bytes")


@dataclass(frozen=True)
class SerializedCallable:
    module: str
    name: str
    code: bytes
    defaults: Any
    kwdefaults: Any


@dataclass(frozen=True)
class TensorTreeDescriptor:
    payload_shm_name: str
    slot_indices: tuple[int, ...]
    tree: Any


def _serialize_callable(fn: Callable[..., Any]) -> Callable[..., Any] | SerializedCallable:
    try:
        pickle.dumps(fn)
        return fn
    except Exception:
        if not isinstance(fn, types.FunctionType):
            raise TypeError("worker callback must be picklable or a Python function")
        if fn.__closure__:
            raise TypeError("worker callback closures are not supported with spawn")

        return SerializedCallable(
            module=str(fn.__module__),
            name=str(fn.__name__),
            code=marshal.dumps(fn.__code__),
            defaults=fn.__defaults__,
            kwdefaults=fn.__kwdefaults__,
        )


def _resolve_callable(fn: Callable[..., Any] | SerializedCallable) -> Callable[..., Any]:
    if callable(fn):
        return fn

    module = importlib.import_module(fn.module)
    code = marshal.loads(fn.code)
    resolved = types.FunctionType(code, vars(module), fn.name, fn.defaults)
    resolved.__kwdefaults__ = fn.kwdefaults
    return resolved


class SlotArena:
    def __init__(self, *, shm_size: int, slot_size: int, name: Optional[str] = None):
        if slot_size <= 0:
            raise ValueError("slot_size must be > 0")
        if shm_size < slot_size:
            raise ValueError("shm_size must be >= slot_size")

        self.arena = SharedMemoryArena(size=shm_size, name=name, create=True)
        self.slot_size = int(slot_size)
        self.capacity = int(shm_size // slot_size)
        self._i = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.arena.name

    def close(self) -> None:
        self.arena.close()

    def unlink(self) -> None:
        self.arena.unlink()

    def alloc_slot(self, payload: bytes) -> HandleMsg:
        if len(payload) > self.slot_size:
            payload = payload[: self.slot_size]

        with self._lock:
            idx = self._i % self.capacity
            self._i += 1

        offset = idx * self.slot_size
        self.arena.write_at(offset, payload)
        return HandleMsg(self.arena.name, offset, len(payload), ShmRingBuffer.KIND_BYTES)

    def alloc_slot_nbytes(self, nbytes: int, *, kind_tag: int) -> HandleMsg:
        n = int(nbytes)
        if n < 0:
            n = 0
        if n > self.slot_size:
            n = self.slot_size

        with self._lock:
            idx = self._i % self.capacity
            self._i += 1

        offset = idx * self.slot_size
        return HandleMsg(self.arena.name, offset, n, int(kind_tag))

    def slot_view(self, msg: HandleMsg) -> memoryview:
        return self.arena.view_at(msg.offset, msg.nbytes)


class _PayloadSlotWriter:
    def __init__(self, arena: SharedMemoryArena, *, slot_offset: int, slot_size: int, slot_index: int):
        self._arena = arena
        self._slot_offset = int(slot_offset)
        self._slot_size = int(slot_size)
        self._slot_index = int(slot_index)
        self._bump = 0

    @staticmethod
    def _align(n: int, alignment: int = 64) -> int:
        return (n + (alignment - 1)) & ~(alignment - 1)

    def alloc_ndarray(self, array: Any) -> dict[str, Any]:
        import numpy as np  # type: ignore

        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        nbytes = int(array.nbytes)
        start = self._align(self._bump)
        end = start + nbytes
        if end > self._slot_size:
            raise MemoryError("tensor tree payload slot out of space")

        absolute_offset = self._slot_offset + start
        buf = self._arena.view_at(absolute_offset, nbytes)
        try:
            dst = np.ndarray(shape=array.shape, dtype=array.dtype, buffer=buf)
            dst[...] = array
        finally:
            buf.release()

        self._bump = end
        return {
            "__oxidata__": "ndarray",
            "offset": int(absolute_offset),
            "nbytes": int(nbytes),
            "dtype": str(array.dtype),
            "shape": [int(x) for x in array.shape],
            "slot_index": self._slot_index,
        }


class _PayloadSlotAllocator:
    def __init__(self, arena: SharedMemoryArena, *, slot_size: int, acquire_slot: Callable[[], int]):
        self._arena = arena
        self._slot_size = int(slot_size)
        self._acquire_slot = acquire_slot
        self._writers: dict[int, _PayloadSlotWriter] = {}
        self._current_slot_index: Optional[int] = None

    def used_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._writers))

    def _writer_for_slot(self, slot_index: int) -> _PayloadSlotWriter:
        writer = self._writers.get(slot_index)
        if writer is None:
            writer = _PayloadSlotWriter(
                self._arena,
                slot_offset=int(slot_index) * self._slot_size,
                slot_size=self._slot_size,
                slot_index=slot_index,
            )
            self._writers[slot_index] = writer
        return writer

    def alloc_ndarray(self, array: Any) -> dict[str, Any]:
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("tensor tree transport requires numpy") from e

        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        if int(array.nbytes) > self._slot_size:
            raise MemoryError("tensor tree leaf exceeds payload slot size")

        if self._current_slot_index is None:
            self._current_slot_index = self._acquire_slot()

        writer = self._writer_for_slot(self._current_slot_index)
        try:
            return writer.alloc_ndarray(array)
        except MemoryError:
            self._current_slot_index = self._acquire_slot()
            writer = self._writer_for_slot(self._current_slot_index)
            return writer.alloc_ndarray(array)


def _encode_tree_leaf(alloc: _PayloadSlotAllocator, obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("tensor tree transport requires numpy") from e

    if isinstance(obj, np.ndarray):
        return alloc.alloc_ndarray(obj)

    if isinstance(obj, dict):
        return {str(k): _encode_tree_leaf(alloc, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_tree_leaf(alloc, v) for v in obj]
    if isinstance(obj, tuple):
        return {"__oxidata__": "tuple", "items": [_encode_tree_leaf(alloc, v) for v in obj]}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    raise TypeError(f"unsupported tensor tree leaf type: {type(obj)!r}")


def _open_tree_leaf(payload_shm: Any, obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("tensor tree transport requires numpy") from e

    if isinstance(obj, dict):
        tag = obj.get("__oxidata__")
        if tag == "ndarray":
            return np.ndarray(
                shape=tuple(int(x) for x in obj["shape"]),
                dtype=np.dtype(str(obj["dtype"])),
                buffer=payload_shm.buf,
                offset=int(obj["offset"]),
            )
        if tag == "tuple":
            return tuple(_open_tree_leaf(payload_shm, v) for v in obj["items"])
        return {k: _open_tree_leaf(payload_shm, v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_open_tree_leaf(payload_shm, v) for v in obj]
    return obj


class Producer:
    def __init__(
        self,
        *,
        ring_name: str,
        shm_size: int,
        slot_size: int,
        ring_capacity: int = 4096,
        ring_slot_size: int = 64,
    ):
        if not native_available():
            raise RuntimeError("Producer requires native extension for ShmRingBuffer")

        self.slots = SlotArena(shm_size=shm_size, slot_size=slot_size)
        self.ring = ShmRingBuffer.create(ring_name, capacity=ring_capacity, slot_size=ring_slot_size)

    @property
    def shm_name(self) -> str:
        return self.slots.name

    @property
    def ring_name(self) -> str:
        return self.ring.name()

    def publish(self, payload: bytes) -> None:
        msg = self.slots.alloc_slot(payload)
        while not self.ring.push_handle(offset=msg.offset, nbytes=msg.nbytes, kind_tag=msg.kind_tag):
            pass

    def publish_blob(self, obj: Any, *, codec: str = "json") -> None:
        from .blob_codec import codec_by_name

        c = codec_by_name(codec)
        data = c.encode(obj)
        msg = self.slots.alloc_slot(data)
        while not self.ring.push_handle(offset=msg.offset, nbytes=msg.nbytes, kind_tag=ShmRingBuffer.KIND_BLOB):
            pass

    def publish_array(self, array: Any) -> None:
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("publish_array requires numpy") from e

        if not isinstance(array, np.ndarray):
            raise TypeError("publish_array expects a numpy.ndarray")
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        nbytes = int(array.nbytes)
        msg = self.slots.alloc_slot_nbytes(nbytes, kind_tag=ShmRingBuffer.KIND_NDARRAY)
        buf = self.slots.slot_view(msg)

        dst = np.ndarray(shape=(msg.nbytes,), dtype=np.uint8, buffer=buf)
        src = np.frombuffer(array, dtype=np.uint8, count=msg.nbytes)
        dst[:] = src

        while not self.ring.push_handle(offset=msg.offset, nbytes=msg.nbytes, kind_tag=ShmRingBuffer.KIND_NDARRAY):
            pass

    def stop(self) -> None:
        while not self.ring.push_handle(offset=0, nbytes=0, kind_tag=0):
            pass

    def close(self) -> None:
        self.ring.close()
        self.slots.close()

    def unlink(self) -> None:
        self.ring.unlink()
        self.slots.unlink()

    def cleanup(self) -> None:
        try:
            self.close()
        finally:
            self.unlink()

    def __enter__(self) -> "Producer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TensorTreeProducer:
    def __init__(
        self,
        *,
        ring_name: str,
        metadata_shm_size: int = 8 * 1024 * 1024,
        metadata_slot_size: int = 64 * 1024,
        payload_shm_size: int = 512 * 1024 * 1024,
        payload_slot_size: int = 64 * 1024 * 1024,
        ring_capacity: int = 4096,
        ring_slot_size: int = 64,
        payload_name: Optional[str] = None,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if not native_available():
            raise RuntimeError("TensorTreeProducer requires native extension for ShmRingBuffer")
        if payload_slot_size <= 0:
            raise ValueError("payload_slot_size must be > 0")
        if payload_shm_size < payload_slot_size:
            raise ValueError("payload_shm_size must be >= payload_slot_size")

        self._ctx = ctx or mp.get_context("spawn")
        self.metadata = SlotArena(shm_size=metadata_shm_size, slot_size=metadata_slot_size)
        self.payloads = SharedMemoryArena(size=payload_shm_size, name=payload_name, create=True)
        self.payload_slot_size = int(payload_slot_size)
        self.payload_capacity = int(payload_shm_size // payload_slot_size)
        self._free_slots = deque(range(self.payload_capacity))
        self._ack_q: mp.Queue = self._ctx.Queue()
        self.ring = ShmRingBuffer.create(ring_name, capacity=ring_capacity, slot_size=ring_slot_size)

    @property
    def metadata_shm_name(self) -> str:
        return self.metadata.name

    @property
    def payload_shm_name(self) -> str:
        return self.payloads.name

    @property
    def ring_name(self) -> str:
        return self.ring.name()

    @property
    def ack_queue(self) -> "mp.Queue":
        return self._ack_q

    def _drain_acks(self) -> None:
        while True:
            try:
                item = self._ack_q.get_nowait()
            except queue.Empty:
                return
            slot_indices = item if isinstance(item, (list, tuple)) else [item]
            for slot_index in slot_indices:
                self._free_slots.append(int(slot_index))

    def _acquire_slot(self) -> int:
        self._drain_acks()
        if self._free_slots:
            return int(self._free_slots.popleft())

        slot_index = int(self._ack_q.get())
        self._free_slots.append(slot_index)
        return int(self._free_slots.popleft())

    def publish_tree(self, obj: Any, *, codec: str = "msgspec_json") -> None:
        from .blob_codec import codec_by_name

        alloc = _PayloadSlotAllocator(
            self.payloads,
            slot_size=self.payload_slot_size,
            acquire_slot=self._acquire_slot,
        )
        tree = _encode_tree_leaf(alloc, obj)
        desc = TensorTreeDescriptor(
            payload_shm_name=self.payloads.name,
            slot_indices=alloc.used_slots(),
            tree=tree,
        )
        payload = codec_by_name(codec).encode(
            {"payload_shm_name": desc.payload_shm_name, "slot_indices": list(desc.slot_indices), "tree": desc.tree}
        )
        msg = self.metadata.alloc_slot(payload)
        while not self.ring.push_handle(offset=msg.offset, nbytes=msg.nbytes, kind_tag=ShmRingBuffer.KIND_BLOB):
            pass

    def stop(self) -> None:
        while not self.ring.push_handle(offset=0, nbytes=0, kind_tag=0):
            pass

    def close(self) -> None:
        self.ring.close()
        self.metadata.close()
        self.payloads.close()

    def unlink(self) -> None:
        self.ring.unlink()
        self.metadata.unlink()
        self.payloads.unlink()

    def cleanup(self) -> None:
        try:
            self.close()
        finally:
            self.unlink()


def _worker_loop_bytes(shm_name: str, ring_name: str, fn_bytes: Callable[[bytes], R], q_out: "mp.Queue"):
    fn = _resolve_callable(fn_bytes)
    rb = ShmRingBuffer.attach(ring_name)
    try:
        from .mp import read_handle_bytes

        while True:
            t = rb.pop_handle()
            if t is None:
                continue
            offset, nbytes, kind_tag = t
            if int(offset) == 0 and int(nbytes) == 0 and int(kind_tag) == 0:
                break
            h = Handle(shm_name, int(offset), int(nbytes), "bytes")
            payload = read_handle_bytes(h)
            q_out.put(fn(payload))
    finally:
        rb.close()


def _worker_loop_array(
    shm_name: str,
    ring_name: str,
    dtype_s: str,
    shape: tuple[int, ...],
    fn_arr: Callable[[Any], R],
    q_out: "mp.Queue",
):
    fn = _resolve_callable(fn_arr)
    rb = ShmRingBuffer.attach(ring_name)
    try:
        import numpy as np  # type: ignore
        from multiprocessing import shared_memory

        dt = np.dtype(dtype_s)
        nbytes_expected = int(dt.itemsize)
        for d in shape:
            nbytes_expected *= int(d)

        while True:
            t = rb.pop_handle()
            if t is None:
                continue
            offset, nbytes, kind_tag = t
            if int(offset) == 0 and int(nbytes) == 0 and int(kind_tag) == 0:
                break
            if int(nbytes) < nbytes_expected:
                continue

            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            try:
                buf = shm.buf[int(offset) : int(offset) + nbytes_expected]
                arr = np.ndarray(shape=shape, dtype=dt, buffer=buf)
                q_out.put(fn(arr))
            finally:
                shm.close()
    finally:
        rb.close()


def _worker_loop_blob(shm_name: str, ring_name: str, codec_name: str, fn_obj: Callable[[Any], R], q_out: "mp.Queue"):
    fn = _resolve_callable(fn_obj)
    rb = ShmRingBuffer.attach(ring_name)
    try:
        from .mp import read_handle_bytes
        from .blob_codec import codec_by_name

        codec = codec_by_name(codec_name)

        while True:
            t = rb.pop_handle()
            if t is None:
                continue
            offset, nbytes, kind_tag = t
            if int(offset) == 0 and int(nbytes) == 0 and int(kind_tag) == 0:
                break
            h = Handle(shm_name, int(offset), int(nbytes), "bytes")
            data = read_handle_bytes(h)
            obj = codec.decode(data)
            q_out.put(fn(obj))
    finally:
        rb.close()


def _worker_loop_tensor_tree(
    metadata_shm_name: str,
    ring_name: str,
    codec_name: str,
    fn_obj: Callable[[Any], R],
    q_out: "mp.Queue",
    q_ack: "mp.Queue",
):
    from multiprocessing import shared_memory

    from .blob_codec import codec_by_name

    fn = _resolve_callable(fn_obj)
    rb = ShmRingBuffer.attach(ring_name)
    metadata_shm = shared_memory.SharedMemory(name=metadata_shm_name, create=False)
    payload_cache: dict[str, shared_memory.SharedMemory] = {}
    codec = codec_by_name(codec_name)
    try:
        while True:
            t = rb.pop_handle()
            if t is None:
                continue
            offset, nbytes, kind_tag = t
            if int(offset) == 0 and int(nbytes) == 0 and int(kind_tag) == 0:
                break

            data = bytes(metadata_shm.buf[int(offset) : int(offset) + int(nbytes)])
            desc = codec.decode(data)
            payload_name = str(desc["payload_shm_name"])
            slot_indices = tuple(int(x) for x in desc["slot_indices"])
            payload_shm = payload_cache.get(payload_name)
            if payload_shm is None:
                payload_shm = shared_memory.SharedMemory(name=payload_name, create=False)
                payload_cache[payload_name] = payload_shm

            sample = _open_tree_leaf(payload_shm, desc["tree"])
            try:
                q_out.put(fn(sample))
            finally:
                q_ack.put(slot_indices)
    finally:
        for payload_shm in payload_cache.values():
            payload_shm.close()
        metadata_shm.close()
        rb.close()


class WorkerPool(Generic[R]):
    def __init__(
        self,
        *,
        shm_name: str,
        ring_name: str,
        fn_bytes: Callable[[bytes], R],
        num_workers: int = 1,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._shm_name = str(shm_name)
        self._ring_name = str(ring_name)
        self._num_workers = int(num_workers)
        self._q_out: mp.Queue = ctx.Queue()
        self._procs = [
            ctx.Process(
                target=_worker_loop_bytes,
                args=(self._shm_name, self._ring_name, _serialize_callable(fn_bytes), self._q_out),
            )
            for _ in range(self._num_workers)
        ]

    def start(self) -> None:
        for p in self._procs:
            p.start()

    def results(self) -> "mp.Queue":
        return self._q_out

    def stop(self) -> None:
        if not self._procs:
            return
        rb = ShmRingBuffer.attach(self._ring_name)
        try:
            for _ in range(self._num_workers):
                while not rb.push_handle(offset=0, nbytes=0, kind_tag=0):
                    pass
        finally:
            rb.close()

    def join(self, timeout: Optional[float] = None) -> None:
        for p in self._procs:
            p.join(timeout=timeout)

    def terminate(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


class IndexRequestor(Generic[R]):
    def __init__(
        self,
        *,
        publish_index: Callable[[int], None],
        q_out: "mp.Queue",
        unpack: Optional[Callable[[Any], tuple[int, R]]] = None,
        max_in_flight: int = 256,
    ):
        self._publish_index = publish_index
        self._q_out = q_out
        self._max_in_flight = int(max_in_flight)
        self._buffer: dict[int, R] = {}
        self._in_flight: set[int] = set()

        if unpack is None:
            self._unpack = self._default_unpack
        else:
            self._unpack = unpack

    @staticmethod
    def _default_unpack(item: Any) -> tuple[int, Any]:
        if isinstance(item, tuple) and len(item) == 2:
            return int(item[0]), item[1]
        if hasattr(item, "i") and hasattr(item, "value"):
            return int(getattr(item, "i")), getattr(item, "value")
        raise TypeError("IndexRequestor expected (i, value) tuple or object with .i/.value")

    def get(self, i: int) -> R:
        idx = int(i)
        if idx in self._buffer:
            return self._buffer.pop(idx)

        if idx not in self._in_flight:
            while len(self._in_flight) >= self._max_in_flight:
                raw = self._q_out.get()
                got_i, got_v = self._unpack(raw)
                self._in_flight.discard(int(got_i))
                self._buffer[int(got_i)] = got_v

            self._publish_index(idx)
            self._in_flight.add(idx)

        while True:
            raw2 = self._q_out.get()
            got_i2, got_v2 = self._unpack(raw2)
            self._in_flight.discard(int(got_i2))
            if int(got_i2) == idx:
                return got_v2
            self._buffer[int(got_i2)] = got_v2


class ArrayWorkerPool(Generic[R]):
    def __init__(
        self,
        *,
        shm_name: str,
        ring_name: str,
        dtype: str,
        shape: tuple[int, ...],
        fn_arr: Callable[[Any], R],
        num_workers: int = 1,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._shm_name = str(shm_name)
        self._ring_name = str(ring_name)
        self._dtype = str(dtype)
        self._shape = tuple(int(x) for x in shape)
        self._num_workers = int(num_workers)
        self._q_out: mp.Queue = ctx.Queue()
        self._procs = [
            ctx.Process(
                target=_worker_loop_array,
                args=(
                    self._shm_name,
                    self._ring_name,
                    self._dtype,
                    self._shape,
                    _serialize_callable(fn_arr),
                    self._q_out,
                ),
            )
            for _ in range(self._num_workers)
        ]

    def start(self) -> None:
        for p in self._procs:
            p.start()

    def results(self) -> "mp.Queue":
        return self._q_out

    def stop(self) -> None:
        if not self._procs:
            return
        rb = ShmRingBuffer.attach(self._ring_name)
        try:
            for _ in range(self._num_workers):
                while not rb.push_handle(offset=0, nbytes=0, kind_tag=0):
                    pass
        finally:
            rb.close()

    def join(self, timeout: Optional[float] = None) -> None:
        for p in self._procs:
            p.join(timeout=timeout)

    def terminate(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


class BlobWorkerPool(Generic[R]):
    def __init__(
        self,
        *,
        shm_name: str,
        ring_name: str,
        codec: str,
        fn_obj: Callable[[Any], R],
        num_workers: int = 1,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._shm_name = str(shm_name)
        self._ring_name = str(ring_name)
        self._codec = str(codec)
        self._num_workers = int(num_workers)
        self._q_out: mp.Queue = ctx.Queue()
        self._procs = [
            ctx.Process(
                target=_worker_loop_blob,
                args=(self._shm_name, self._ring_name, self._codec, _serialize_callable(fn_obj), self._q_out),
            )
            for _ in range(self._num_workers)
        ]

    def start(self) -> None:
        for p in self._procs:
            p.start()

    def results(self) -> "mp.Queue":
        return self._q_out

    def stop(self) -> None:
        if not self._procs:
            return
        rb = ShmRingBuffer.attach(self._ring_name)
        try:
            for _ in range(self._num_workers):
                while not rb.push_handle(offset=0, nbytes=0, kind_tag=0):
                    pass
        finally:
            rb.close()

    def join(self, timeout: Optional[float] = None) -> None:
        for p in self._procs:
            p.join(timeout=timeout)

    def terminate(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


class TensorTreeWorkerPool(Generic[R]):
    def __init__(
        self,
        *,
        metadata_shm_name: str,
        ring_name: str,
        fn_obj: Callable[[Any], R],
        codec: str = "msgspec_json",
        ack_queue: Optional["mp.Queue"] = None,
        num_workers: int = 1,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._metadata_shm_name = str(metadata_shm_name)
        self._ring_name = str(ring_name)
        self._codec = str(codec)
        self._ack_q: mp.Queue = ack_queue if ack_queue is not None else ctx.Queue()
        self._num_workers = int(num_workers)
        self._q_out: mp.Queue = ctx.Queue()
        self._procs = [
            ctx.Process(
                target=_worker_loop_tensor_tree,
                args=(
                    self._metadata_shm_name,
                    self._ring_name,
                    self._codec,
                    _serialize_callable(fn_obj),
                    self._q_out,
                    self._ack_q,
                ),
            )
            for _ in range(self._num_workers)
        ]

    def start(self) -> None:
        for p in self._procs:
            p.start()

    def results(self) -> "mp.Queue":
        return self._q_out

    def stop(self) -> None:
        if not self._procs:
            return
        rb = ShmRingBuffer.attach(self._ring_name)
        try:
            for _ in range(self._num_workers):
                while not rb.push_handle(offset=0, nbytes=0, kind_tag=0):
                    pass
        finally:
            rb.close()

    def join(self, timeout: Optional[float] = None) -> None:
        for p in self._procs:
            p.join(timeout=timeout)

    def terminate(self) -> None:
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


class TensorSampleProducer(TensorTreeProducer):
    """Opinionated producer for nested tensor-like samples.

    The recommended path is:
    - NumPy leaves in shared memory
    - msgspec/json for small metadata only
    - bounded in-flight payload slots
    """

    def publish_sample(self, sample: Any, *, codec: str = "msgspec_json") -> None:
        self.publish_tree(sample, codec=codec)


class TensorSampleWorkerPool(TensorTreeWorkerPool[R]):
    """Opinionated worker pool for nested tensor-like samples."""

    def __init__(
        self,
        *,
        metadata_shm_name: str,
        ring_name: str,
        fn_sample: Callable[[Any], R],
        codec: str = "msgspec_json",
        ack_queue: Optional["mp.Queue"] = None,
        num_workers: int = 1,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        super().__init__(
            metadata_shm_name=metadata_shm_name,
            ring_name=ring_name,
            fn_obj=fn_sample,
            codec=codec,
            ack_queue=ack_queue,
            num_workers=num_workers,
            ctx=ctx,
        )
