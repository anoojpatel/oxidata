import time

from oxidata.shm_arena import SharedMemoryArena


def main():
    arena = SharedMemoryArena(size=64 * 1024 * 1024, create=True)
    try:
        t0 = time.perf_counter()
        for _ in range(2000):
            _ = arena.alloc_bytes(b"x" * 1024)
            if _ % 100 == 0:
                arena.reset()
        dt = time.perf_counter() - t0
        print("done in", dt)
    finally:
        try:
            arena.close()
        finally:
            arena.unlink()


if __name__ == "__main__":
    main()
