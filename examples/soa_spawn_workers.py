import multiprocessing as mp


def _worker(arena_name: str, batch, q_out: "mp.Queue"):
    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.soa import SoABatch

    arena = SharedMemoryArena(size=None, name=arena_name, create=False)
    try:
        cols, shms = SoABatch.open(arena, batch)
        try:
            # Example: compute a small reduction
            x_sum = int(cols["x"].sum())
            q_out.put(x_sum)
        finally:
            SoABatch.close_opened(shms)
    finally:
        arena.close()


def main():
    try:
        import numpy as np  # type: ignore
    except Exception:
        raise SystemExit("numpy not installed")

    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.soa import SoASchema, SoABatch

    ctx = mp.get_context("spawn")

    arena = SharedMemoryArena(size=1024 * 1024)
    try:
        schema = SoASchema.from_mapping({"x": "int32", "y": "float32"})
        batch = SoABatch.alloc(arena, schema=schema, length=1000)
        cols, shms = SoABatch.open(arena, batch)
        try:
            cols["x"][...] = np.arange(1000, dtype=np.int32)
            cols["y"][...] = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        finally:
            SoABatch.close_opened(shms)

        q_out = ctx.Queue()
        p = ctx.Process(target=_worker, args=(arena.name, batch, q_out))
        p.start()
        out = q_out.get(timeout=10)
        p.join(timeout=10)

        print("sum(x)=", out)
    finally:
        arena.close()
        arena.unlink()


if __name__ == "__main__":
    main()
