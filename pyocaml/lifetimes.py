from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, Iterator, Optional, TypeVar
import threading

T = TypeVar("T")


class LifetimeError(RuntimeError):
    pass


class BorrowError(RuntimeError):
    pass


class Scope:
    """A scope represents a lifetime domain.

    When a scope exits, all `Owned` values created within it become invalid.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._alive = True
        self._owned = []  # type: list[Owned[object]]

    def _register(self, owned: "Owned[object]") -> None:
        with self._lock:
            if not self._alive:
                raise LifetimeError("cannot allocate into a dead Scope")
            self._owned.append(owned)

    def _invalidate_all(self) -> None:
        with self._lock:
            self._alive = False
            for o in self._owned:
                o._invalidate_from_scope()
            self._owned.clear()

    def alive(self) -> bool:
        with self._lock:
            return self._alive

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._invalidate_all()


class Arena:
    """An arena is a convenience allocator bound to a `Scope`."""

    def __init__(self, scope: Scope):
        self._scope = scope

    def alloc(self, value: T) -> "Owned[T]":
        owned: Owned[T] = Owned(value=value, scope=self._scope)
        self._scope._register(owned)  # type: ignore[arg-type]
        return owned


@dataclass
class _BorrowState:
    readers: int = 0
    writer: bool = False


class Owned(Generic[T]):
    """A lifetime-managed object.

    Borrow rules:
    - Multiple immutable borrows allowed concurrently.
    - Exactly one mutable borrow allowed, and only if no readers.

    This is *runtime enforced* (Python can’t do static checking), but the API
    forces you to be explicit, similar to Oxidized OCaml capabilities.
    """

    __slots__ = ("_value", "_scope_ref", "_alive", "_lock", "_state", "_frozen")

    def __init__(self, value: T, scope: Scope):
        self._value: T = value
        self._scope_ref: Scope = scope
        self._alive: bool = True
        self._lock = threading.RLock()
        self._state = _BorrowState()
        self._frozen: bool = False

    def _invalidate_from_scope(self) -> None:
        with self._lock:
            self._alive = False

    def alive(self) -> bool:
        with self._lock:
            return self._alive and self._scope_ref.alive()

    def _ensure_alive(self) -> None:
        if not self.alive():
            raise LifetimeError("use-after-free: Owned value outlived its Scope")

    @contextmanager
    def borrow(self) -> Iterator["Borrowed[T]"]:
        """Immutable borrow."""
        with self._lock:
            self._ensure_alive()
            if self._state.writer:
                raise BorrowError("cannot immutably borrow while mutably borrowed")
            self._state.readers += 1

        try:
            yield Borrowed(self, mutable=False)
        finally:
            with self._lock:
                self._state.readers -= 1

    @contextmanager
    def borrow_mut(self) -> Iterator["Borrowed[T]"]:
        """Mutable borrow."""
        with self._lock:
            self._ensure_alive()
            if self._frozen:
                raise BorrowError("cannot mutably borrow a frozen value")
            if self._state.writer or self._state.readers > 0:
                raise BorrowError("cannot mutably borrow while already borrowed")
            self._state.writer = True

        try:
            yield Borrowed(self, mutable=True)
        finally:
            with self._lock:
                self._state.writer = False

    def into_inner(self) -> T:
        """Move the value out.

        After `into_inner`, this `Owned` becomes invalid.
        """
        with self._lock:
            self._ensure_alive()
            if self._state.writer or self._state.readers > 0:
                raise BorrowError("cannot move out while borrowed")
            self._alive = False
            return self._value

    def freeze(self) -> None:
        """Freeze this value: future mutable borrows are rejected."""
        with self._lock:
            self._ensure_alive()
            if self._state.writer or self._state.readers > 0:
                raise BorrowError("cannot freeze while borrowed")
            self._frozen = True

    def frozen(self) -> bool:
        with self._lock:
            return bool(self._frozen)


class Borrowed(Generic[T]):
    """A borrowed view into an `Owned[T]`.

    This object is only valid inside the context manager that produced it.
    """

    __slots__ = ("_owned", "_mutable")

    def __init__(self, owned: Owned[T], mutable: bool):
        self._owned = owned
        self._mutable = mutable

    def get(self) -> T:
        self._owned._ensure_alive()
        return self._owned._value

    def set(self, value: T) -> None:
        if not self._mutable:
            raise BorrowError("cannot set through immutable borrow")
        self._owned._ensure_alive()
        self._owned._value = value

    def mutate(self) -> T:
        """Return the underlying value for in-place mutation.

        Note: this is only safe if `T` is a mutable structure and you hold a
        mutable borrow.
        """
        if not self._mutable:
            raise BorrowError("cannot mutate through immutable borrow")
        self._owned._ensure_alive()
        return self._owned._value
