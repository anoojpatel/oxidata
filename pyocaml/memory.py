"""
Shared mutable objects with minimal copy-on-write overhead.
"""

import threading
import weakref
from typing import Any, Optional, Dict, List, TypeVar, Generic, Union
from contextlib import contextmanager
from dataclasses import dataclass
from .core import Value, Block

T = TypeVar('T')

@dataclass
class VersionedValue:
    """A value with version tracking for CoW optimization."""
    value: Any
    version: int
    readers: int = 0
    
class SharedObject(Generic[T]):
    """
    A shared mutable object that minimizes CoW overhead through
    version tracking and reader-writer semantics.
    """
    
    def __init__(self, initial_value: T):
        self._value = VersionedValue(initial_value, 0)
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.RLock()
        self._read_count = 0
        self._write_count = 0
        self._observers = weakref.WeakSet()
        
    @contextmanager
    def read(self):
        """Read access to the shared object."""
        with self._lock:
            self._value.readers += 1
            current_version = self._value.version
            current_value = self._value.value
            
        try:
            yield current_value
        finally:
            with self._lock:
                self._value.readers -= 1
                self._read_count += 1
                
    @contextmanager
    def write(self):
        """Write access to the shared object with version bump."""
        with self._lock:
            # Wait for all readers to finish
            while self._value.readers > 0:
                self._lock.release()
                threading.Event().wait(0.001)  # Small delay
                self._lock.acquire()
                
            old_value = self._value.value
            old_version = self._value.version
            
        try:
            yield old_value
        finally:
            with self._lock:
                # Only increment version if value actually changed
                if self._value.value != old_value:
                    self._value.version += 1
                self._write_count += 1
                self._notify_observers()
                
    def get(self) -> T:
        """Get current value (convenient read access)."""
        with self.read() as value:
            return value
            
    def set(self, value: T) -> None:
        """Set new value (convenient write access)."""
        with self.write() as old_value:
            self._value.value = value
            
    def version(self) -> int:
        """Get current version of the value."""
        with self._lock:
            return self._value.version
            
    def add_observer(self, callback) -> None:
        """Add an observer that gets notified on changes."""
        self._observers.add(callback)
        
    def _notify_observers(self) -> None:
        """Notify all observers of changes."""
        for observer in self._observers:
            try:
                observer(self._value.value, self._value.version)
            except Exception:
                pass  # Ignore observer errors

class MemoryManager:
    """
    Manages shared objects with optimized memory layout and minimal CoW.
    """
    
    def __init__(self, heap_size: int = 8192):
        self.heap_size = heap_size
        self.shared_objects: Dict[int, SharedObject] = {}
        self.object_counter = 0
        self._lock = threading.RLock()
        self._memory_pool = []
        self._pool_lock = threading.RLock()
        
    def create_shared(self, initial_value: T) -> SharedObject[T]:
        """Create a new shared object."""
        with self._lock:
            obj_id = self._next_id()
            shared_obj = SharedObject(initial_value)
            self.shared_objects[obj_id] = shared_obj
            return shared_obj
            
    def create_shared_block(self, tag: int, size: int) -> SharedObject[Block]:
        """Create a shared block object."""
        block = Block(tag, size)
        return self.create_shared(block)
        
    def _next_id(self) -> int:
        """Get next object ID."""
        self.object_counter += 1
        return self.object_counter
        
    def get_shared(self, obj_id: int) -> Optional[SharedObject]:
        """Get shared object by ID."""
        with self._lock:
            return self.shared_objects.get(obj_id)
            
    def allocate_from_pool(self, size: int) -> bytearray:
        """Allocate memory from pool to reduce fragmentation."""
        with self._pool_lock:
            # Find a suitable block in the pool
            for i, block in enumerate(self._memory_pool):
                if len(block) >= size:
                    allocated = block[:size]
                    self._memory_pool[i] = block[size:]
                    if not self._memory_pool[i]:
                        del self._memory_pool[i]
                    return allocated
                    
            # If no suitable block found, allocate new memory
            return bytearray(size)
            
    def return_to_pool(self, memory: bytearray) -> None:
        """Return memory to the pool for reuse."""
        with self._pool_lock:
            if len(memory) > 0:
                self._memory_pool.append(memory)
                
    def compact_memory(self) -> None:
        """Compact memory pool by merging adjacent blocks."""
        with self._pool_lock:
            if len(self._memory_pool) < 2:
                return
                
            # Sort blocks by address (simplified)
            self._memory_pool.sort(key=len, reverse=True)
            
            # Merge small blocks
            merged = []
            current = None
            for block in self._memory_pool:
                if current is None:
                    current = block
                elif len(current) + len(block) <= self.heap_size // 10:
                    current.extend(block)
                else:
                    merged.append(current)
                    current = block
                    
            if current:
                merged.append(current)
                
            self._memory_pool = merged

class AtomicReference(Generic[T]):
    """
    Atomic reference for lock-free access to shared objects.
    Uses threading primitives for atomic operations.
    """
    
    def __init__(self, initial_value: T):
        self._value = initial_value
        self._lock = threading.Lock()
        
    def get(self) -> T:
        """Get current value atomically."""
        with self._lock:
            return self._value
            
    def set(self, value: T) -> None:
        """Set new value atomically."""
        with self._lock:
            self._value = value
            
    def compare_and_set(self, expected: T, new_value: T) -> bool:
        """Compare and set atomically."""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False
            
    def update(self, updater_func) -> T:
        """Update value using function atomically."""
        with self._lock:
            old_value = self._value
            self._value = updater_func(old_value)
            return self._value

# Add RWLock implementation if not available
if not hasattr(threading, 'RWLock'):
    class RWLock:
        """Simple read-write lock implementation."""
        
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
