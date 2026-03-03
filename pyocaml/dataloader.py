from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from .native import ShmRingBuffer, native_available
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
        # kind is informational here; callers can branch on kind_tag.
        return Handle(self.shm_name, self.offset, self.nbytes, "bytes")


class SlotArena:
    """Fixed-size slot allocator over a SharedMemoryArena.

    This is the recommended initial reclamation strategy for dataloading:
    - payloads are written into slots
    - producer publishes (offset,nbytes) metadata
    - slots are reused in a ring (overwrite)

    Safety model: immutable-after-publish; consumers must treat payload as read-only.
    """

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
        """Reserve a slot for `nbytes` (clamped to slot_size) without writing."""
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


class Producer:
    """Produces byte payloads into a SlotArena and publishes metadata onto a queue."""

    def __init__(self, *, ring_name: str, shm_size: int, slot_size: int, ring_capacity: int = 4096, ring_slot_size: int = 64):
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
        # Backpressure: spin until enqueued
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
        """Publish a fixed-shape, fixed-dtype ndarray payload.

        This is intended to be used with ArrayWorkerPool which knows dtype/shape.
        """
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

        # Copy directly into shared memory (no intermediate Python bytes object).
        # We only publish `nbytes` bytes; slot may be larger.
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
        # Only call once you're sure no other process still needs the segment.
        self.ring.unlink()
        self.slots.unlink()

    def cleanup(self) -> None:
        """Close and unlink all OS resources.

        Call this only after workers have exited / detached.
        """
        try:
            self.close()
        finally:
            self.unlink()

    def __enter__(self) -> "Producer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Do not unlink automatically: workers may still be attached.
        self.close()


def _worker_loop_bytes(shm_name: str, ring_name: str, fn_bytes: Callable[[bytes], R], q_out: "mp.Queue"):
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
            q_out.put(fn_bytes(payload))
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
                q_out.put(fn_arr(arr))
            finally:
                shm.close()
    finally:
        rb.close()


def _worker_loop_blob(shm_name: str, ring_name: str, codec_name: str, fn_obj: Callable[[Any], R], q_out: "mp.Queue"):
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
            q_out.put(fn_obj(obj))
    finally:
        rb.close()


class WorkerPool(Generic[R]):
    """Spawn workers consuming handle messages from a ShmRingBuffer."""

    def __init__(self, *, shm_name: str, ring_name: str, fn_bytes: Callable[[bytes], R], num_workers: int = 1, ctx: Optional[mp.context.BaseContext] = None):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._shm_name = str(shm_name)
        self._ring_name = str(ring_name)
        self._num_workers = int(num_workers)
        self._q_out: mp.Queue = ctx.Queue()
        self._procs = [
            ctx.Process(target=_worker_loop_bytes, args=(self._shm_name, self._ring_name, fn_bytes, self._q_out))
            for _ in range(self._num_workers)
        ]

    def start(self) -> None:
        for p in self._procs:
            p.start()

    def results(self) -> "mp.Queue":
        return self._q_out

    def stop(self) -> None:
        """Send one stop token per worker."""
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


class ArrayWorkerPool(Generic[R]):
    """Spawn workers that interpret messages as fixed-shape ndarrays."""

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
            ctx.Process(target=_worker_loop_array, args=(self._shm_name, self._ring_name, self._dtype, self._shape, fn_arr, self._q_out))
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
    """Spawn workers that decode blob messages (json/msgspec_json) from slots."""

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
            ctx.Process(target=_worker_loop_blob, args=(self._shm_name, self._ring_name, self._codec, fn_obj, self._q_out))
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
