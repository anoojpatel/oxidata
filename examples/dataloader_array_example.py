def main():
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("native extension not built")

    try:
        import numpy as np  # type: ignore
    except Exception:
        raise SystemExit("numpy not installed")

    from oxidata.dataloader import Producer, ArrayWorkerPool

    ring_name = "oxidata-dl-array-ring"

    def fn_arr(a):
        # Example: sum the array
        return float(a.sum())

    producer = Producer(ring_name=ring_name, shm_size=64 * 1024 * 1024, slot_size=4096)
    pool = ArrayWorkerPool(
        shm_name=producer.shm_name,
        ring_name=producer.ring_name,
        dtype="float32",
        shape=(1024,),
        fn_arr=fn_arr,
        num_workers=2,
    )

    try:
        pool.start()

        for _ in range(1000):
            x = np.ones((1024,), dtype=np.float32)
            producer.publish_array(x)

        pool.stop()
        pool.join(timeout=20)

        q = pool.results()
        out = []
        while not q.empty():
            out.append(q.get())

        print("results:", len(out))
        print("first 5:", out[:5])
    finally:
        producer.cleanup()


if __name__ == "__main__":
    main()
