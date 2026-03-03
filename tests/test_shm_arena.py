import unittest

from oxidata.shm_arena import SharedMemoryArena


class TestSharedMemoryArena(unittest.TestCase):
    def test_alloc_and_read_bytes(self):
        arena = SharedMemoryArena(size=4096)
        try:
            h = arena.alloc_bytes(b"hello")
            self.assertEqual(arena.read_bytes(h), b"hello")
        finally:
            arena.close()
            arena.unlink()

    def test_open_view_zero_copy_roundtrip(self):
        arena = SharedMemoryArena(size=4096)
        try:
            payload = b"abcdef"
            h = arena.alloc_bytes(payload)

            mv, shm = arena.open_view(h)
            try:
                self.assertEqual(bytes(mv), payload)
            finally:
                shm.close()
        finally:
            arena.close()
            arena.unlink()

    def test_alloc_int64(self):
        arena = SharedMemoryArena(size=4096)
        try:
            h = arena.alloc_int64(123456789)
            self.assertEqual(arena.read_int64(h), 123456789)
        finally:
            arena.close()
            arena.unlink()

    def test_alloc_utf8(self):
        arena = SharedMemoryArena(size=4096)
        try:
            h = arena.alloc_utf8("héllo")
            self.assertEqual(arena.read_utf8(h), "héllo")
        finally:
            arena.close()
            arena.unlink()


if __name__ == "__main__":
    unittest.main()
