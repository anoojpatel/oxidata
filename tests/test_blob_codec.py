import unittest

from oxidata.shm_arena import SharedMemoryArena
from oxidata.blob_codec import JsonCodec, alloc_object, open_object


class TestBlobCodec(unittest.TestCase):
    def test_json_blob_roundtrip(self):
        arena = SharedMemoryArena(size=4096)
        try:
            obj = {"a": 1, "b": [1, 2, 3], "s": "hi"}
            h = alloc_object(arena, obj, codec=JsonCodec())
            out = open_object(arena, h, codec=JsonCodec())
            self.assertEqual(out, obj)
        finally:
            arena.close()
            arena.unlink()


if __name__ == "__main__":
    unittest.main()
