"""
Shared mutable objects with minimal copy-on-write overhead.
"""

import threading
import weakref
from typing import Any, Optional, Dict, TypeVar, Generic
from contextlib import contextmanager
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class VersionedValue:
    value: Any
    version: int
    readers: int = 0


class SharedObject(Generic[T]):
    def __init__(self, initial_value: T):
        self._value = VersionedValue(initial_value, 0)
        self._lock = threading.RWLock() if hasattr(threading, "RWLock") else threading.RLock()
        self._read_count = 0
        self._write_count = 0
        self._observers = weakref.WeakSet()

    @contextmanager
    def read(self):
        with self._lock:
            self._value.readers += 1
            current_value = self._value.value

        try:
            yield current_value
        finally:
            with self._lock:
                self._value.readers -= 1
                self._read_count += 1

    @contextmanager
    def write(self):
        with self._lock:
            while self._value.readers > 0:
                self._lock.release()
                threading.Event().wait(0.001)
                self._lock.acquire()

            old_value = self._value.value

        try:
            yield old_value
        finally:
            with self._lock:
                if self._value.value != old_value:
                    self._value.version += 1
                self._write_count += 1
                self._notify_observers()

    def get(self) -> T:
        with self.read() as value:
            return value

    def set(self, value: T) -> None:
        with self.write():
            self._value.value = value

    def version(self) -> int:
        with self._lock:
            return self._value.version

    def add_observer(self, callback) -> None:
        self._observers.add(callback)

    def _notify_observers(self) -> None:
        for observer in self._observers:
            try:
                observer(self._value.value, self._value.version)
            except Exception:
                pass


class MemoryManager:
    def __init__(self, heap_size: int = 8192):
        self.heap_size = heap_size
        self.shared_objects: Dict[int, SharedObject] = {}
        self.object_counter = 0
        self._lock = threading.RLock()

    def create_shared(self, initial_value: T) -> SharedObject[T]:
        with self._lock:
            obj_id = self._next_id()
            shared_obj = SharedObject(initial_value)
            self.shared_objects[obj_id] = shared_obj
            return shared_obj

    def _next_id(self) -> int:
        self.object_counter += 1
        return self.object_counter

    def get_shared(self, obj_id: int) -> Optional[SharedObject]:
        with self._lock:
            return self.shared_objects.get(obj_id)


if not hasattr(threading, "RWLock"):

    class RWLock:
        def __init__(self):
            self._read_ready = threading.Condition(threading.RLock())
            self._readers = 0

        def __enter__(self):
            self.acquire_read()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release_read()

        def acquire_read(self):
            self._read_ready.acquire()
            try:
                self._readers += 1
            finally:
                self._read_ready.release()

        def release_read(self):
            self._read_ready.acquire()
            try:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notifyAll()
            finally:
                self._read_ready.release()

        def acquire_write(self):
            self._read_ready.acquire()
            while self._readers > 0:
                self._read_ready.wait()

        def release_write(self):
            self._read_ready.release()

        def acquire(self):
            self.acquire_write()

        def release(self):
            self.release_write()

    threading.RWLock = RWLock
