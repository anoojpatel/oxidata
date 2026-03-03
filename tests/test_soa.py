import unittest


class TestSoA(unittest.TestCase):
    def test_soa_alloc_open_roundtrip(self):
        try:
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("numpy not installed")
            return

        from oxidata.shm_arena import SharedMemoryArena
        from oxidata.soa import SoASchema, SoABatch

        arena = SharedMemoryArena(size=1024 * 1024)
        try:
            schema = SoASchema.from_mapping({"x": "int32", "y": "float32"})
            batch = SoABatch.alloc(arena, schema=schema, length=4)
            cols, shms = SoABatch.open(arena, batch)
            try:
                cols["x"][...] = np.array([1, 2, 3, 4], dtype=np.int32)
                cols["y"][...] = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

                # Re-open to ensure data is actually in shared memory
                cols2, shms2 = SoABatch.open(arena, batch)
                try:
                    self.assertTrue(np.array_equal(cols2["x"], np.array([1, 2, 3, 4], dtype=np.int32)))
                    self.assertTrue(np.allclose(cols2["y"], np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)))
                finally:
                    SoABatch.close_opened(shms2)
            finally:
                SoABatch.close_opened(shms)
        finally:
            arena.close()
            arena.unlink()


if __name__ == "__main__":
    unittest.main()
