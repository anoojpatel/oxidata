import unittest


class TestNativeShm(unittest.TestCase):
    def test_native_shm_write_and_readinto(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.shm_arena import SharedMemoryArena
        from oxidata.native import handle_readinto, handle_write

        arena = SharedMemoryArena(size=4096)
        try:
            h = arena.alloc_bytes(b"" * 16, kind="bytes")
            # overwrite using native write
            n = handle_write(h, b"hello-native")
            self.assertGreaterEqual(n, 5)

            out = bytearray(16)
            n2 = handle_readinto(h, out, 0, 16)
            self.assertGreaterEqual(n2, 5)
            self.assertEqual(bytes(out[:11]), b"hello-native")
        finally:
            arena.close()
            arena.unlink()


if __name__ == "__main__":
    unittest.main()
