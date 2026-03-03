from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .shm_arena import Handle, SharedMemoryArena


class Codec(Protocol):
    name: str

    def encode(self, obj: Any) -> bytes: ...

    def decode(self, data: bytes) -> Any: ...


@dataclass(frozen=True)
class JsonCodec:
    name: str = "json"

    def encode(self, obj: Any) -> bytes:
        # Compact encoding; assumes obj is JSON-serializable.
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


@dataclass(frozen=True)
class MsgspecJsonCodec:
    """Uses msgspec if installed; falls back to JSON elsewhere.

    Note: We avoid importing msgspec at module import time.
    """

    name: str = "msgspec_json"

    def encode(self, obj: Any) -> bytes:
        import msgspec  # type: ignore

        enc = msgspec.json.Encoder()
        return enc.encode(obj)

    def decode(self, data: bytes) -> Any:
        import msgspec  # type: ignore

        dec = msgspec.json.Decoder()
        return dec.decode(data)


def default_codec() -> Codec:
    try:
        import msgspec  # noqa: F401  # type: ignore

        return MsgspecJsonCodec()
    except Exception:
        return JsonCodec()


def codec_by_name(name: str) -> Codec:
    if name == "json":
        return JsonCodec()
    if name == "msgspec_json":
        return MsgspecJsonCodec()
    raise ValueError(f"unknown codec: {name}")


def alloc_object(arena: SharedMemoryArena, obj: Any, *, codec: Codec | None = None) -> Handle:
    """Encode object to bytes and store in shared memory as a blob."""
    if codec is None:
        codec = default_codec()
    data = codec.encode(obj)
    return arena.alloc_bytes(data, kind=f"blob:{codec.name}")


def open_object(arena: SharedMemoryArena, h: Handle, *, codec: Codec | None = None) -> Any:
    """Read bytes from shared memory and decode into an object.

    Note: decoding creates Python objects (not zero-copy). The *transfer* remains
    zero-copy; decoding cost is comparable to msgspec/json decode.
    """
    if codec is None:
        codec = default_codec()

    data = arena.read_bytes(h)
    return codec.decode(data)
