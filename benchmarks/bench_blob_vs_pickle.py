import pickle
import time


def bench_pickle(obj, iters: int = 1000) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        _ = pickle.loads(b)
    return time.perf_counter() - t0


def bench_shm_json(obj, iters: int = 1000) -> float:
    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.blob_codec import JsonCodec, alloc_object, open_object

    arena = SharedMemoryArena(size=10 * 1024 * 1024)
    try:
        t0 = time.perf_counter()
        for _ in range(iters):
            arena.reset()
            h = alloc_object(arena, obj, codec=JsonCodec())
            _ = open_object(arena, h, codec=JsonCodec())
        return time.perf_counter() - t0
    finally:
        arena.close()
        arena.unlink()


def bench_shm_msgspec(obj, iters: int = 1000) -> float:
    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.blob_codec import MsgspecJsonCodec, alloc_object, open_object

    arena = SharedMemoryArena(size=10 * 1024 * 1024)
    try:
        t0 = time.perf_counter()
        for _ in range(iters):
            arena.reset()
            h = alloc_object(arena, obj, codec=MsgspecJsonCodec())
            _ = open_object(arena, h, codec=MsgspecJsonCodec())
        return time.perf_counter() - t0
    finally:
        arena.close()
        arena.unlink()


def main():
    obj = {
        "id": 123,
        "name": "example",
        "values": list(range(1000)),
        "nested": {"a": 1, "b": 2, "c": [3, 4, 5]},
    }

    iters = 200
    tp = bench_pickle(obj, iters=iters)
    ts = bench_shm_json(obj, iters=iters)
    print(f"pickle: {tp:.4f}s for {iters}")
    print(f"shm+json: {ts:.4f}s for {iters}")

    try:
        import msgspec  # noqa: F401

        tm = bench_shm_msgspec(obj, iters=iters)
        print(f"shm+msgspec: {tm:.4f}s for {iters}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
