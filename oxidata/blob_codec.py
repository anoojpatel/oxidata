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
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


@dataclass(frozen=True)
class MsgspecJsonCodec:
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
    if codec is None:
        codec = default_codec()
    data = codec.encode(obj)
    return arena.alloc_bytes(data, kind=f"blob:{codec.name}")


def open_object(arena: SharedMemoryArena, h: Handle, *, codec: Codec | None = None) -> Any:
    if codec is None:
        codec = default_codec()

    data = arena.read_bytes(h)
    return codec.decode(data)
