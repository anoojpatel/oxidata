from __future__ import annotations

from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:
        raise RuntimeError("torch staging utilities require PyTorch") from e


def _map_tree(obj: Any, fn) -> Any:
    if isinstance(obj, dict):
        return {k: _map_tree(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_tree(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map_tree(v, fn) for v in obj)
    return fn(obj)


def tensor_tree_to_torch(tree: Any, *, pin_memory: bool = False) -> Any:
    torch = require_torch()

    def convert(obj: Any) -> Any:
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if np is not None and isinstance(obj, np.ndarray):
            t = torch.from_numpy(obj)
            return t.pin_memory() if pin_memory else t
        if isinstance(obj, torch.Tensor):
            return obj.pin_memory() if pin_memory and not obj.is_pinned() else obj
        return obj

    return _map_tree(tree, convert)


def pin_memory_tree(tree: Any) -> Any:
    torch = require_torch()

    def convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj if obj.is_pinned() else obj.pin_memory()
        return obj

    return _map_tree(tree, convert)


def stage_tree_to_device(tree: Any, device: str | Any, *, non_blocking: bool = True) -> Any:
    torch = require_torch()

    def convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device=device, non_blocking=non_blocking)
        return obj

    return _map_tree(tree, convert)
