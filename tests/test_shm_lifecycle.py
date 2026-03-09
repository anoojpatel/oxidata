import unittest
import uuid

from oxidata.offheap import OffHeapScope, borrow_region
from oxidata.shm_arena import SharedMemoryArena


class TestSharedMemoryLifecycle(unittest.TestCase):
    def test_open_view_close_releases_memoryview(self):
        arena = SharedMemoryArena(size=4096)
        try:
            handle = arena.alloc_bytes(b"abcdef")
            view, opened = arena.open_view(handle)
            self.assertEqual(bytes(view), b"abcdef")
            opened.close()
        finally:
            arena.close()
            arena.unlink()

    def test_attach_cycle_can_reopen_same_segment(self):
        arena = SharedMemoryArena(size=4096, name=f"oxd-atc-{uuid.uuid4().hex[:8]}", create=True)
        try:
            handle = arena.alloc_bytes(b"hello")

            attached = SharedMemoryArena(size=None, name=arena.name, create=False)
            try:
                self.assertEqual(attached.read_bytes(handle), b"hello")
            finally:
                attached.close()

            attached2 = SharedMemoryArena(size=None, name=arena.name, create=False)
            try:
                self.assertEqual(attached2.read_bytes(handle), b"hello")
            finally:
                attached2.close()
        finally:
            arena.close()
            arena.unlink()

    def test_offheap_borrow_releases_view_before_scope_exit(self):
        with OffHeapScope(size=4096) as scope:
            handle = scope.alloc_bytes(b"xyz")
            with borrow_region(handle) as region:
                self.assertEqual(bytes(region.view[:3]), b"xyz")


if __name__ == "__main__":
    unittest.main()
