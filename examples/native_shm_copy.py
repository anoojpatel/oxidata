from oxidata.native import available as native_available, handle_readinto, handle_write
from oxidata.shm_arena import SharedMemoryArena


def main():
    if not native_available():
        raise SystemExit("pyocaml_native not built")

    arena = SharedMemoryArena(size=4096)
    try:
        h = arena.alloc_bytes(b"x" * 64)
        handle_write(h, b"hello")
        out = bytearray(16)
        handle_readinto(h, out, 0, 16)
        print(bytes(out[:5]))
    finally:
        arena.close()
        arena.unlink()


if __name__ == "__main__":
    main()
