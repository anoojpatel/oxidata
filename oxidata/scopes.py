from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, Tuple, TypeVar

from .lifetimes import Arena, Owned, Scope
from .shm_arena import Handle, SharedMemoryArena

T = TypeVar("T")


@dataclass(frozen=True)
class Published(Generic[T]):
    owned: Owned[T]

    def handle(self) -> Owned[T]:
        return self.owned


class Frame:
    def __init__(self, *, size: int = 64 * 1024 * 1024, name: Optional[str] = None):
        self._scope = Scope()
        self._arena = SharedMemoryArena(size=size, name=name, create=True)
        self._owned = Arena(self._scope)
        self._closed = False

    @property
    def scope(self) -> Scope:
        return self._scope

    @property
    def arena(self) -> SharedMemoryArena:
        return self._arena

    @property
    def shm_name(self) -> str:
        return self._arena.name

    def var(self, value: T) -> Owned[T]:
        return self._owned.alloc(value)

    def alloc_bytes(self, data: bytes, *, kind: str = "bytes") -> Owned[Handle]:
        h = self._arena.alloc_bytes(data, kind=kind)
        return self._owned.alloc(h)

    def alloc_utf8(self, s: str) -> Owned[Handle]:
        h = self._arena.alloc_utf8(s)
        return self._owned.alloc(h)

    def alloc_numpy(self, array: Any) -> Owned[Handle]:
        h = self._arena.alloc_numpy(array)
        return self._owned.alloc(h)

    def alloc_ndarray(self, *, shape: Tuple[int, ...], dtype: Any) -> Owned[Handle]:
        h = self._arena.alloc_ndarray(shape=shape, dtype=dtype)
        return self._owned.alloc(h)

    def publish(self, owned: Owned[T]) -> Published[T]:
        owned.freeze()
        return Published(owned)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._scope.__exit__(None, None, None)
        finally:
            try:
                self._arena.close()
            finally:
                try:
                    self._arena.unlink()
                except Exception:
                    pass

    def __enter__(self) -> "Frame":
        self._scope.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class GlobalSegment:
    def __init__(self, *, name: str, size: Optional[int], create: bool):
        self._name = str(name)
        self._arena = SharedMemoryArena(size=size, name=self._name, create=create)
        self._scope = Scope()
        self._owned = Arena(self._scope)
        self._closed = False
        self._owner = bool(create)

    @classmethod
    def create(cls, name: str, *, size: int) -> "GlobalSegment":
        return cls(name=name, size=size, create=True)

    @classmethod
    def attach(cls, name: str) -> "GlobalSegment":
        return cls(name=name, size=None, create=False)

    @property
    def name(self) -> str:
        return self._arena.name

    def alloc_bytes(self, data: bytes, *, kind: str = "bytes") -> Owned[Handle]:
        h = self._arena.alloc_bytes(data, kind=kind)
        return self._owned.alloc(h)

    def alloc_ndarray(self, *, shape: Tuple[int, ...], dtype: Any) -> Owned[Handle]:
        h = self._arena.alloc_ndarray(shape=shape, dtype=dtype)
        return self._owned.alloc(h)

    def publish(self, owned: Owned[T]) -> Published[T]:
        owned.freeze()
        return Published(owned)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._scope.__exit__(None, None, None)
        finally:
            self._arena.close()

    def unlink(self) -> None:
        if not self._owner:
            return
        self._arena.unlink()

    def __enter__(self) -> "GlobalSegment":
        self._scope.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
