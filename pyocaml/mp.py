from __future__ import annotations

"""Multiprocessing helpers.

The core design is that shared-memory payloads are addressed by `Handle`, which is
cheap to pickle and pass across processes. No object-graph pickling.

This module provides a small, explicit API for common open/close patterns.
"""

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Tuple

from .shm_arena import Handle


@dataclass(frozen=True)
class OpenedView:
    view: memoryview
    shm: shared_memory.SharedMemory

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass


def open_handle_view(h: Handle) -> OpenedView:
    shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
    v = shm.buf[h.offset : h.offset + h.nbytes]
    return OpenedView(view=v, shm=shm)


def read_handle_bytes(h: Handle) -> bytes:
    shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
    try:
        return bytes(shm.buf[h.offset : h.offset + h.nbytes])
    finally:
        shm.close()


def open_handle_numpy(h: Handle) -> Tuple[Any, shared_memory.SharedMemory]:
    """Open a NumPy ndarray view for a handle produced by arena (kind == 'ndarray')."""

    if h.kind != "ndarray":
        raise TypeError(f"handle kind is {h.kind!r}, expected 'ndarray'")

    import numpy as np  # type: ignore

    dtype_s, shape = h.meta
    shm = shared_memory.SharedMemory(name=h.shm_name, create=False)
    buf = shm.buf[h.offset : h.offset + h.nbytes]
    arr = np.ndarray(shape=shape, dtype=np.dtype(dtype_s), buffer=buf)
    return arr, shm
