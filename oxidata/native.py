from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class NativeNotAvailable(RuntimeError):
    pass


def _try_import_native():
    for module_name in ("pyocaml_native", "oxidata_native.oxidata_native"):
        try:
            module = __import__(module_name, fromlist=["*"])
            return module
        except Exception:
            continue
    return None


_native = _try_import_native()


def available() -> bool:
    return _native is not None


def require_native():
    if _native is None:
        raise NativeNotAvailable(
            "native extension is not available. Build it with `uv run maturin develop --manifest-path oxidata_native/Cargo.toml`."
        )
    return _native


def shm_readinto(
    shm_name: str,
    offset: int,
    out: bytearray,
    out_offset: int = 0,
    nbytes: int | None = None,
) -> int:
    n = require_native()
    return int(n.shm_readinto(str(shm_name), int(offset), out, int(out_offset), nbytes))


def shm_write(shm_name: str, offset: int, data: bytes) -> int:
    n = require_native()
    return int(n.shm_write(str(shm_name), int(offset), data))


def handle_readinto(h: Any, out: bytearray, out_offset: int = 0, nbytes: int | None = None) -> int:
    want = int(h.nbytes) if nbytes is None else int(nbytes)
    return shm_readinto(str(h.shm_name), int(h.offset), out, out_offset=out_offset, nbytes=want)


def handle_write(h: Any, data: bytes) -> int:
    if len(data) > int(h.nbytes):
        data = data[: int(h.nbytes)]
    return shm_write(str(h.shm_name), int(h.offset), data)


@dataclass
class AtomicI64:
    _inner: object

    def __init__(self, value: int = 0):
        n = require_native()
        self._inner = n.AtomicI64(int(value))

    def load(self) -> int:
        return int(self._inner.load())

    def store(self, value: int) -> None:
        self._inner.store(int(value))

    def fetch_add(self, delta: int) -> int:
        return int(self._inner.fetch_add(int(delta)))

    def load_i64(self) -> int:
        return self.load()

    def store_i64(self, value: int) -> None:
        self.store(value)


@dataclass
class RwBytes:
    _inner: object

    def __init__(self, size: int):
        n = require_native()
        self._inner = n.RwBytes(int(size))

    def size(self) -> int:
        return int(self._inner.size())

    def readinto(self, out: bytearray, offset: int = 0) -> int:
        return int(self._inner.readinto(out, int(offset)))

    def write(self, data: bytes, offset: int = 0) -> int:
        return int(self._inner.write(data, int(offset)))


@dataclass
class ShmRingBuffer:
    _inner: object

    def __init__(self, name: str, capacity: int, slot_size: int, create: bool):
        n = require_native()
        self._inner = n.ShmRingBuffer(str(name), int(capacity), int(slot_size), bool(create))

    @classmethod
    def create(cls, name: str, *, capacity: int, slot_size: int) -> "ShmRingBuffer":
        return cls(name, int(capacity), int(slot_size), True)

    @classmethod
    def attach(cls, name: str) -> "ShmRingBuffer":
        return cls(name, 0, 0, False)

    def name(self) -> str:
        return str(self._inner.name())

    def capacity(self) -> int:
        return int(self._inner.capacity())

    def slot_size(self) -> int:
        return int(self._inner.slot_size())

    def push(self, data: bytes) -> bool:
        return bool(self._inner.push(data))

    KIND_BYTES = 1
    KIND_NDARRAY = 2
    KIND_BLOB = 3

    def pop_into(self, out: bytearray, out_offset: int = 0) -> int:
        return int(self._inner.pop_into(out, int(out_offset)))

    def pop(self) -> bytes | None:
        buf = bytearray(self.slot_size())
        n = self.pop_into(buf, 0)
        if n <= 0:
            return None
        return bytes(buf[:n])

    def push_handle(self, *, offset: int, nbytes: int, kind_tag: int = KIND_BYTES) -> bool:
        return bool(self._inner.push_handle(int(offset), int(nbytes), int(kind_tag)))

    def pop_handle(self):
        return self._inner.pop_handle()

    def close(self) -> None:
        self._inner.close()

    def unlink(self) -> None:
        self._inner.unlink()
