import multiprocessing as mp
import time


def _worker_sum_x(arena_name: str, batch, q_out: "mp.Queue"):
    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.soa import SoABatch

    arena = SharedMemoryArena(size=None, name=arena_name, create=False)
    try:
        cols, shms = SoABatch.open(arena, batch)
        try:
            q_out.put(int(cols["x"].sum()))
        finally:
            SoABatch.close_opened(shms)
    finally:
        arena.close()


def _worker_sum_x_pickle(x_list, q_out: "mp.Queue"):
    q_out.put(sum(x_list))


def bench_spawn_soa(length: int = 100_000, iters: int = 50) -> float:
    import numpy as np  # type: ignore

    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.soa import SoASchema, SoABatch

    ctx = mp.get_context("spawn")

    arena = SharedMemoryArena(size=32 * 1024 * 1024)
    try:
        schema = SoASchema.from_mapping({"x": "int32"})
        batch = SoABatch.alloc(arena, schema=schema, length=length)

        cols, shms = SoABatch.open(arena, batch)
        try:
            cols["x"][...] = np.arange(length, dtype=np.int32)
        finally:
            SoABatch.close_opened(shms)

        t0 = time.perf_counter()
        for _ in range(iters):
            q_out = ctx.Queue()
            p = ctx.Process(target=_worker_sum_x, args=(arena.name, batch, q_out))
            p.start()
            _ = q_out.get(timeout=20)
            p.join(timeout=20)
        return time.perf_counter() - t0
    finally:
        arena.close()
        arena.unlink()


def bench_spawn_pickle(length: int = 100_000, iters: int = 50) -> float:
    # Baseline: transfer a python list (forces pickling) and compute sum in worker.
    ctx = mp.get_context("spawn")

    x = list(range(length))
    t0 = time.perf_counter()
    for _ in range(iters):
        q_out = ctx.Queue()
        p = ctx.Process(target=_worker_sum_x_pickle, args=(x, q_out))
        p.start()
        _ = q_out.get(timeout=20)
        p.join(timeout=20)
    return time.perf_counter() - t0


def main():
    try:
        import numpy as np  # noqa: F401
    except Exception:
        raise SystemExit("numpy not installed")

    length = 100_000
    iters = 20
    tp = bench_spawn_pickle(length=length, iters=iters)
    ts = bench_spawn_soa(length=length, iters=iters)

    print(f"spawn+pickle(list[int]) : {tp:.4f}s for {iters} iters (len={length})")
    print(f"spawn+SoA(handle)      : {ts:.4f}s for {iters} iters (len={length})")


if __name__ == "__main__":
    main()
