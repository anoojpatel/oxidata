"""
Core memory management abstractions mimicking OCaml's heap model.
"""

import weakref
import threading
from typing import Any, Optional, Dict, List, Union, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod

T = TypeVar('T')

class Value(ABC):
    """Base class for all values in the OCaml-like heap."""
    
    def __init__(self):
        self._mark = False
        self._forwarding = None
        
    @abstractmethod
    def size(self) -> int:
        """Return the size of this value in words."""
        pass
        
    @abstractmethod
    def children(self) -> List['Value']:
        """Return all child values for GC traversal."""
        pass
        
    def mark(self) -> None:
        """Mark this value for garbage collection."""
        self._mark = True
        
    def unmark(self) -> None:
        """Unmark this value for garbage collection."""
        self._mark = False
        
    def is_marked(self) -> bool:
        """Check if this value is marked."""
        return self._mark

class Block(Value):
    """Represents a heap block like OCaml's blocks with header and fields."""
    
    def __init__(self, tag: int, size: int):
        super().__init__()
        self.tag = tag
        self._fields = [None] * size
        
    def size(self) -> int:
        return len(self._fields) + 1  # +1 for header
        
    def children(self) -> List[Value]:
        return [field for field in self._fields if isinstance(field, Value)]
        
    def __getitem__(self, index: int) -> Any:
        return self._fields[index]
        
    def __setitem__(self, index: int, value: Any) -> None:
        self._fields[index] = value

class Immediate(Value):
    """Represents immediate values (integers, characters, etc.) that don't need allocation."""
    
    def __init__(self, value: Union[int, bool]):
        super().__init__()
        self.value = value
        
    def size(self) -> int:
        return 0  # Immediate values don't occupy heap space
        
    def children(self) -> List[Value]:
        return []

class Heap:
    """OCaml-style heap with generational garbage collection."""
    
    def __init__(self, young_size: int = 1024, old_size: int = 4096):
        self.young_generation = []
        self.old_generation = []
        self.young_size = young_size
        self.old_size = old_size
        self.allocation_pointer = 0
        self._lock = threading.RLock()
        
    def allocate(self, block: Block) -> Block:
        """Allocate a block in the young generation."""
        with self._lock:
            if self.allocation_pointer + block.size() > self.young_size:
                self._minor_gc()
                
            if self.allocation_pointer + block.size() > self.young_size:
                # Still not enough space, do major GC
                self._major_gc()
                
            self.young_generation.append(block)
            self.allocation_pointer += block.size()
            return block
            
    def _minor_gc(self) -> None:
        """Perform minor garbage collection on young generation."""
        # Simple mark-and-sweep for young generation
        for obj in self.young_generation:
            obj.unmark()
            
        # Mark roots (simplified - in real implementation would scan stack)
        for obj in self.young_generation:
            if obj.is_marked():
                self._mark_children(obj)
                
        # Sweep unmarked objects
        self.young_generation = [obj for obj in self.young_generation if obj.is_marked()]
        self.allocation_pointer = sum(obj.size() for obj in self.young_generation)
        
    def _major_gc(self) -> None:
        """Perform major garbage collection, moving objects to old generation."""
        # Promote surviving objects to old generation
        surviving = [obj for obj in self.young_generation if obj.is_marked()]
        self.old_generation.extend(surviving)
        
        # Clear young generation
        self.young_generation.clear()
        self.allocation_pointer = 0
        
        # Compact old generation if needed
        if len(self.old_generation) > self.old_size:
            self._compact_old_generation()
            
    def _mark_children(self, obj: Value) -> None:
        """Mark all children of an object."""
        for child in obj.children():
            if not child.is_marked():
                child.mark()
                self._mark_children(child)
                
    def _compact_old_generation(self) -> None:
        """Compact the old generation to reduce fragmentation."""
        # Simple compaction - remove dead objects
        self.old_generation = [obj for obj in self.old_generation if obj.is_marked()]

class Closure(Value):
    """Represents a function closure like in OCaml."""
    
    def __init__(self, function, env: List[Value]):
        super().__init__()
        self.function = function
        self.env = env
        
    def size(self) -> int:
        return len(self.env) + 1
        
    def children(self) -> List[Value]:
        return self.env
        
    def apply(self, *args) -> Any:
        """Apply the closure with its environment."""
        return self.function(self.env, *args)
