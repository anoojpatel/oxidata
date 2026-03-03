from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, Iterator, Optional, Tuple, TypeVar

from multiprocessing import shared_memory

from .lifetimes import Scope, Arena, Owned
from .shm_arena import SharedMemoryArena, Handle

T = TypeVar("T")


@dataclass(frozen=True)
class OpenedRegion:
    view: memoryview
    shm: Any

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass


class OffHeapScope:
    def __init__(self, *, size: int = 64 * 1024 * 1024, name: Optional[str] = None):
        self._scope = Scope()
        self._arena = SharedMemoryArena(size=size, name=name, create=True)
        self._owned_arena = Arena(self._scope)

    @property
    def scope(self) -> Scope:
        return self._scope

    @property
    def arena(self) -> SharedMemoryArena:
        return self._arena

    def alloc_bytes(self, data: bytes, *, kind: str = "bytes") -> Owned[Handle]:
        h = self._arena.alloc_bytes(data, kind=kind)
        return self._owned_arena.alloc(h)

    def alloc_utf8(self, s: str) -> Owned[Handle]:
        h = self._arena.alloc_utf8(s)
        return self._owned_arena.alloc(h)

    def alloc_numpy(self, array: Any) -> Owned[Handle]:
        h = self._arena.alloc_numpy(array)
        return self._owned_arena.alloc(h)

    def alloc_ndarray(self, *, shape: Tuple[int, ...], dtype: Any) -> Owned[Handle]:
        h = self._arena.alloc_ndarray(shape=shape, dtype=dtype)
        return self._owned_arena.alloc(h)

    def __enter__(self) -> "OffHeapScope":
        self._scope.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._scope.__exit__(exc_type, exc, tb)
        finally:
            try:
                self._arena.close()
            finally:
                try:
                    self._arena.unlink()
                except Exception:
                    pass


@contextmanager
def borrow_region(owned_handle: Owned[Handle]) -> Iterator[OpenedRegion]:
    with owned_handle.borrow() as b:
        h = b.get()
        shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
        mv = shm.buf[h.offset : h.offset + h.nbytes]
        try:
            yield OpenedRegion(view=mv, shm=shm)
        finally:
            try:
                shm.close()
            except Exception:
                pass


@contextmanager
def borrow_region_mut(owned_handle: Owned[Handle]) -> Iterator[OpenedRegion]:
    with owned_handle.borrow_mut() as b:
        h = b.get()
        shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
        mv = shm.buf[h.offset : h.offset + h.nbytes]
        try:
            yield OpenedRegion(view=mv, shm=shm)
        finally:
            try:
                shm.close()
            except Exception:
                pass


@dataclass(frozen=True)
class OffHeap(Generic[T]):
    handle: Owned[Handle]

    def freeze(self) -> None:
        self.handle.freeze()

    def frozen(self) -> bool:
        return self.handle.frozen()


@dataclass(frozen=True)
class OffHeapBytes(OffHeap[bytes]):
    def read(self) -> bytes:
        with borrow_region(self.handle) as r:
            return bytes(r.view)


@dataclass(frozen=True)
class OffHeapBlob(OffHeap[T]):
    codec: Any

    def decode(self) -> T:
        with self.handle.borrow() as b:
            h = b.get()
            shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
            try:
                data = bytes(shm.buf[h.offset : h.offset + h.nbytes])
            finally:
                shm.close()
        return self.codec.decode(data)


@dataclass(frozen=True)
class OffHeapArray(OffHeap[Any]):
    def open_numpy(self):
        with self.handle.borrow() as b:
            h = b.get()
        from .mp import open_handle_numpy

        return open_handle_numpy(h)
