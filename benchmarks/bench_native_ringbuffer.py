import time


def main():
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("pyocaml_native not built")

    from oxidata.native import ShmRingBuffer

    ring_name = "oxidata-bench-ring"
    rb = ShmRingBuffer.create(ring_name, capacity=4096, slot_size=256)
    try:
        n = 100_000
        payload = b"x" * 64

        t0 = time.perf_counter()
        pushed = 0
        while pushed < n:
            if rb.push(payload):
                pushed += 1

        popped = 0
        while popped < n:
            msg = rb.pop()
            if msg is None:
                continue
            popped += 1

        dt = time.perf_counter() - t0
        print(f"pushed+popped {n} msgs in {dt:.4f}s => {n/dt:.0f} msg/s")
    finally:
        rb.close()
        rb.unlink()


if __name__ == "__main__":
    main()
