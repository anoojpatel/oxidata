from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .shm_arena import Handle, SharedMemoryArena


@dataclass(frozen=True)
class SoASchema:
    fields: Tuple[Tuple[str, str], ...]

    @staticmethod
    def from_mapping(fields: Mapping[str, Any]) -> "SoASchema":
        items = tuple((str(k), str(v)) for k, v in fields.items())
        return SoASchema(fields=items)

    def names(self) -> Tuple[str, ...]:
        return tuple(n for n, _ in self.fields)

    def dtype_of(self, name: str) -> str:
        for n, dt in self.fields:
            if n == name:
                return dt
        raise KeyError(name)


@dataclass(frozen=True)
class SoABatchHandle:
    schema: SoASchema
    length: int
    columns: Tuple[Tuple[str, Handle], ...]

    def __reduce__(self):
        return (SoABatchHandle, (self.schema, self.length, self.columns))


class SoABatch:
    @staticmethod
    def alloc(arena: SharedMemoryArena, *, schema: SoASchema, length: int) -> SoABatchHandle:
        import numpy as np  # type: ignore

        cols = []
        for name, dtype_s in schema.fields:
            h = arena.alloc_ndarray(shape=(length,), dtype=np.dtype(dtype_s))
            cols.append((name, h))
        return SoABatchHandle(schema=schema, length=int(length), columns=tuple(cols))

    @staticmethod
    def open(arena: SharedMemoryArena, batch: SoABatchHandle) -> Tuple[Dict[str, Any], Any]:
        shms: Dict[str, Any] = {}
        cols: Dict[str, Any] = {}
        for name, h in batch.columns:
            arr, shm = arena.open_numpy(h)
            cols[name] = arr
            shms[name] = shm

        return cols, shms

    @staticmethod
    def close_opened(shm_keeper: Any) -> None:
        if isinstance(shm_keeper, dict):
            for shm in shm_keeper.values():
                try:
                    shm.close()
                except Exception:
                    pass
