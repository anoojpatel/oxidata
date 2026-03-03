from __future__ import annotations

import os
import struct
import threading
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Optional, Tuple


class ArenaClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class Handle:
    shm_name: str
    offset: int
    nbytes: int
    kind: str
    meta: Tuple[Any, ...] = ()

    def __reduce__(self):
        return (Handle, (self.shm_name, self.offset, self.nbytes, self.kind, self.meta))


class SharedMemoryArena:
    def __init__(self, size: Optional[int], name: Optional[str] = None, create: bool = True):
        self._lock = threading.RLock()
        self._closed = False

        if name is None:
            name = f"oxidata-{os.getpid()}-{uuid.uuid4().hex}"

        if create:
            if size is None:
                raise ValueError("size is required when create=True")
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=int(size))
            self._size = int(size)
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._size = int(len(self._shm.buf))

        self._bump = 0

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def size(self) -> int:
        return self._size

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._shm.close()

    def unlink(self) -> None:
        with self._lock:
            self._shm.unlink()

    def __enter__(self) -> "SharedMemoryArena":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> None:
        with self._lock:
            self._ensure_open()
            self._bump = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArenaClosed("SharedMemoryArena is closed")

    @staticmethod
    def _align(n: int, alignment: int = 8) -> int:
        return (n + (alignment - 1)) & ~(alignment - 1)

    def alloc_bytes(self, data: bytes, *, kind: str = "bytes") -> Handle:
        with self._lock:
            self._ensure_open()
            n = len(data)
            start = self._align(self._bump)
            end = start + n
            if end > self._size:
                raise MemoryError("SharedMemoryArena out of space")
            self._shm.buf[start:end] = data
            self._bump = end
            return Handle(self._shm.name, start, n, kind)

    def write_at(self, offset: int, data: bytes) -> int:
        with self._lock:
            self._ensure_open()
            if offset < 0 or offset >= self._size:
                return 0
            n = min(len(data), self._size - int(offset))
            self._shm.buf[int(offset) : int(offset) + n] = data[:n]
            return int(n)

    def read_at(self, offset: int, nbytes: int) -> bytes:
        with self._lock:
            self._ensure_open()
            if offset < 0 or offset >= self._size:
                return b""
            n = min(int(nbytes), self._size - int(offset))
            return bytes(self._shm.buf[int(offset) : int(offset) + n])

    def view_at(self, offset: int, nbytes: int) -> memoryview:
        with self._lock:
            self._ensure_open()
            if offset < 0 or offset >= self._size:
                return memoryview(b"")
            n = min(int(nbytes), self._size - int(offset))
            return self._shm.buf[int(offset) : int(offset) + n]

    def open_view(self, h: Handle) -> Tuple[memoryview, shared_memory.SharedMemory]:
        shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
        return shm.buf[h.offset : h.offset + h.nbytes], shm

    def read_bytes(self, h: Handle) -> bytes:
        shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
        try:
            return bytes(shm.buf[h.offset : h.offset + h.nbytes])
        finally:
            shm.close()

    def alloc_int64(self, value: int) -> Handle:
        return self.alloc_bytes(struct.pack("<q", int(value)), kind="i64")

    def read_int64(self, h: Handle) -> int:
        b = self.read_bytes(h)
        (v,) = struct.unpack("<q", b)
        return int(v)

    def alloc_float64(self, value: float) -> Handle:
        return self.alloc_bytes(struct.pack("<d", float(value)), kind="f64")

    def read_float64(self, h: Handle) -> float:
        b = self.read_bytes(h)
        (v,) = struct.unpack("<d", b)
        return float(v)

    def alloc_utf8(self, s: str) -> Handle:
        data = s.encode("utf-8")
        return self.alloc_bytes(data, kind="utf8")

    def read_utf8(self, h: Handle) -> str:
        return self.read_bytes(h).decode("utf-8")

    def alloc_numpy(self, array: Any) -> Handle:
        import numpy as np  # type: ignore

        if not isinstance(array, np.ndarray):
            raise TypeError("alloc_numpy expects a numpy.ndarray")
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        h = self.alloc_bytes(array.tobytes(order="C"), kind="ndarray")
        meta = (str(array.dtype), tuple(array.shape))
        return Handle(h.shm_name, h.offset, h.nbytes, h.kind, meta=meta)

    def alloc_ndarray(self, *, shape: Tuple[int, ...], dtype: Any) -> Handle:
        import numpy as np  # type: ignore

        dt = np.dtype(dtype)
        nbytes = int(dt.itemsize)
        for d in shape:
            nbytes *= int(d)

        with self._lock:
            self._ensure_open()
            start = self._align(self._bump)
            end = start + nbytes
            if end > self._size:
                raise MemoryError("SharedMemoryArena out of space")
            self._bump = end

        meta = (str(dt), tuple(shape))
        return Handle(self._shm.name, start, nbytes, "ndarray", meta=meta)

    def open_numpy(self, h: Handle) -> Any:
        if h.kind != "ndarray":
            raise TypeError(f"handle kind is {h.kind!r}, expected 'ndarray'")

        import numpy as np  # type: ignore

        dtype_s, shape = h.meta
        shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
        buf = shm.buf[h.offset : h.offset + h.nbytes]
        arr = np.ndarray(shape=shape, dtype=np.dtype(dtype_s), buffer=buf)
        return arr, shm
