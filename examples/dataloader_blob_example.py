def main():
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("native extension not built")

    from oxidata.dataloader import Producer, BlobWorkerPool

    ring_name = "oxidata-dl-blob-ring"

    def fn_obj(o) -> int:
        # Example: compute size of a nested list
        return len(o["values"])  # type: ignore[index]

    producer = Producer(ring_name=ring_name, shm_size=8 * 1024 * 1024, slot_size=4096)
    pool = BlobWorkerPool(
        shm_name=producer.shm_name,
        ring_name=producer.ring_name,
        codec="json",
        fn_obj=fn_obj,
        num_workers=2,
    )

    try:
        pool.start()

        for i in range(100):
            producer.publish_blob({"id": i, "values": list(range(1000))}, codec="json")

        pool.stop()
        pool.join(timeout=20)

        q = pool.results()
        out = []
        while not q.empty():
            out.append(q.get())

        print("results collected:", len(out))
        print("first 5:", out[:5])
    finally:
        producer.cleanup()


if __name__ == "__main__":
    main()
