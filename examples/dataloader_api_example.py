import time


def main():
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("pyocaml_native not built")

    from oxidata.dataloader import Producer, WorkerPool

    ring_name = "oxidata-dl-api-ring"

    def fn_bytes(b: bytes) -> int:
        # Example worker fn: compute a trivial statistic.
        return len(b)

    producer = Producer(ring_name=ring_name, shm_size=8 * 1024 * 1024, slot_size=256)
    pool = WorkerPool(shm_name=producer.shm_name, ring_name=producer.ring_name, fn_bytes=fn_bytes, num_workers=2)

    try:
        pool.start()

        for i in range(1000):
            producer.publish(f"sample-{i}".encode("utf-8"))

        # stop workers and collect results
        pool.stop()
        pool.join(timeout=20)

        q = pool.results()
        out = []
        while not q.empty():
            out.append(q.get())

        print("results collected:", len(out))
        print("first 5:", out[:5])
    finally:
        # Ensure OS resources cleaned up after workers are done.
        producer.cleanup()


if __name__ == "__main__":
    main()
