import multiprocessing as mp
import unittest


def _worker_read_bytes(q_in: "mp.Queue", q_out: "mp.Queue"):
    from oxidata.mp import read_handle_bytes

    h = q_in.get()
    q_out.put(read_handle_bytes(h))


def _worker_read_soa(q_in: "mp.Queue", q_out: "mp.Queue"):
    try:
        import numpy as np  # type: ignore
    except Exception:
        q_out.put(("SKIP", "numpy not installed"))
        return

    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.soa import SoABatch

    arena_name, batch = q_in.get()

    # Attach to existing shared memory segment.
    arena = SharedMemoryArena(size=None, name=arena_name, create=False)
    try:
        cols, shms = SoABatch.open(arena, batch)
        try:
            x = cols["x"].copy()  # child-local copy for assertion transport
            y = cols["y"].copy()
            q_out.put((x.tolist(), y.tolist()))
        finally:
            SoABatch.close_opened(shms)
    finally:
        arena.close()


class TestMultiprocessingHandles(unittest.TestCase):
    def test_spawn_queue_handle_bytes(self):
        ctx = mp.get_context("spawn")
        q_in = ctx.Queue()
        q_out = ctx.Queue()

        from oxidata.shm_arena import SharedMemoryArena

        arena = SharedMemoryArena(size=4096)
        try:
            h = arena.alloc_bytes(b"hello-mp")

            p = ctx.Process(target=_worker_read_bytes, args=(q_in, q_out))
            p.start()
            q_in.put(h)
            out = q_out.get(timeout=5)
            p.join(timeout=5)
            self.assertEqual(out, b"hello-mp")
        finally:
            arena.close()
            arena.unlink()

    def test_spawn_queue_soa_batch(self):
        try:
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("numpy not installed")
            return

        ctx = mp.get_context("spawn")
        q_in = ctx.Queue()
        q_out = ctx.Queue()

        from oxidata.shm_arena import SharedMemoryArena
        from oxidata.soa import SoASchema, SoABatch

        arena = SharedMemoryArena(size=1024 * 1024)
        try:
            schema = SoASchema.from_mapping({"x": "int32", "y": "float32"})
            batch = SoABatch.alloc(arena, schema=schema, length=3)

            cols, shms = SoABatch.open(arena, batch)
            try:
                cols["x"][...] = np.array([1, 2, 3], dtype=np.int32)
                cols["y"][...] = np.array([0.25, 0.5, 0.75], dtype=np.float32)
            finally:
                SoABatch.close_opened(shms)

            p = ctx.Process(target=_worker_read_soa, args=(q_in, q_out))
            p.start()
            q_in.put((arena.name, batch))
            out = q_out.get(timeout=10)
            p.join(timeout=10)

            if isinstance(out, tuple) and len(out) == 2 and out[0] == "SKIP":
                self.skipTest(out[1])
                return

            x_list, y_list = out
            self.assertEqual(x_list, [1, 2, 3])
            self.assertEqual([round(v, 2) for v in y_list], [0.25, 0.5, 0.75])
        finally:
            arena.close()
            arena.unlink()


if __name__ == "__main__":
    unittest.main()
