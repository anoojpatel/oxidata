from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generator, Generic, Optional, TypeVar, Union

from .lifetimes import Borrowed, Owned

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class Borrow(Generic[T]):
    owned: Owned[T]
    mutable: bool = False


def borrow(owned: Owned[T]) -> Borrow[T]:
    return Borrow(owned=owned, mutable=False)


def borrow_mut(owned: Owned[T]) -> Borrow[T]:
    return Borrow(owned=owned, mutable=True)


def run(gen: Generator[Any, Any, R]) -> R:
    stack = []
    sent: Any = None
    try:
        while True:
            try:
                req = gen.send(sent)
            except StopIteration as e:
                return e.value  # type: ignore[return-value]

            if isinstance(req, Borrow):
                if req.mutable:
                    cm = req.owned.borrow_mut()
                else:
                    cm = req.owned.borrow()

                borrowed = cm.__enter__()
                stack.append(cm)
                sent = borrowed
            else:
                raise TypeError(f"unhandled effect request: {type(req)!r}")
    finally:
        while stack:
            cm = stack.pop()
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass


def run_fn(fn: Callable[..., Generator[Any, Any, R]], *args: Any, **kwargs: Any) -> R:
    return run(fn(*args, **kwargs))
