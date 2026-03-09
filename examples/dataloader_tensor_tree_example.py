def main():
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("native extension not built")

    try:
        import numpy as np  # type: ignore
    except Exception:
        raise SystemExit("numpy not installed")

    from oxidata.dataloader import TensorSampleProducer, TensorSampleWorkerPool

    def fn_sample(sample):
        return {
            "tokens_sum": int(sample["tokens"].sum()),
            "signal_mean": float(sample["nested"]["signal"].mean()),
            "sample_id": int(sample["meta"]["sample_id"]),
        }

    producer = TensorSampleProducer(
        ring_name="oxidata-dl-tree-ring",
        metadata_shm_size=1024 * 1024,
        metadata_slot_size=32 * 1024,
        payload_shm_size=64 * 1024 * 1024,
    )
    pool = TensorSampleWorkerPool(
        metadata_shm_name=producer.metadata_shm_name,
        ring_name=producer.ring_name,
        fn_sample=fn_sample,
        ack_queue=producer.ack_queue,
        num_workers=1,
    )

    sample = {
        "tokens": np.arange(16, dtype=np.int32),
        "nested": {"signal": np.linspace(0.0, 1.0, 8, dtype=np.float32)},
        "meta": {"sample_id": 42},
    }

    try:
        pool.start()
        producer.publish_sample(sample)
        pool.stop()
        print(pool.results().get(timeout=10))
        pool.join(timeout=10)
    finally:
        producer.cleanup()


if __name__ == "__main__":
    main()
