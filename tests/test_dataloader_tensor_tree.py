import unittest


class TestDataloaderTensorTree(unittest.TestCase):
    def test_tensor_tree_workerpool_end_to_end(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        try:
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("numpy not installed")
            return

        from oxidata.dataloader import TensorSampleProducer, TensorSampleWorkerPool

        def fn_sample(sample):
            return {
                "tokens_sum": int(sample["tokens"].sum()),
                "signal_mean": float(sample["nested"]["signal"].mean()),
                "pair_total": int(sample["nested"]["pair"][0].sum() + sample["nested"]["pair"][1].sum()),
                "sample_id": int(sample["meta"]["sample_id"]),
            }

        producer = TensorSampleProducer(
            ring_name="oxd-tt-1",
            metadata_shm_size=1024 * 1024,
            metadata_slot_size=32 * 1024,
            payload_shm_size=16 * 1024 * 1024,
            payload_slot_size=16 * 1024 * 1024,
        )
        pool = TensorSampleWorkerPool(
            metadata_shm_name=producer.metadata_shm_name,
            ring_name=producer.ring_name,
            fn_sample=fn_sample,
            ack_queue=producer.ack_queue,
            num_workers=1,
        )

        sample = {
            "tokens": np.arange(8, dtype=np.int32),
            "nested": {
                "signal": np.linspace(0.0, 1.0, 4, dtype=np.float32),
                "pair": (
                    np.array([1, 2], dtype=np.int16),
                    np.array([3, 4], dtype=np.int16),
                ),
            },
            "meta": {"sample_id": 17},
        }

        try:
            pool.start()
            producer.publish_sample(sample)
            pool.stop()
            result = pool.results().get(timeout=10)
            pool.join(timeout=10)
        finally:
            producer.cleanup()

        self.assertEqual(result["tokens_sum"], 28)
        self.assertAlmostEqual(result["signal_mean"], 0.5, places=6)
        self.assertEqual(result["pair_total"], 10)
        self.assertEqual(result["sample_id"], 17)

    def test_tensor_tree_recycles_bounded_payload_slots(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        try:
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("numpy not installed")
            return

        from oxidata.dataloader import TensorSampleProducer, TensorSampleWorkerPool

        def fn_sample(sample):
            return int(sample["tokens"].sum())

        producer = TensorSampleProducer(
            ring_name="oxd-tt-2",
            metadata_shm_size=1024 * 1024,
            metadata_slot_size=32 * 1024,
            payload_shm_size=4096,
            payload_slot_size=4096,
        )
        pool = TensorSampleWorkerPool(
            metadata_shm_name=producer.metadata_shm_name,
            ring_name=producer.ring_name,
            fn_sample=fn_sample,
            ack_queue=producer.ack_queue,
            num_workers=1,
        )

        try:
            pool.start()
            producer.publish_sample({"tokens": np.arange(16, dtype=np.int32)})
            self.assertEqual(pool.results().get(timeout=10), sum(range(16)))
            producer.publish_sample({"tokens": np.arange(8, dtype=np.int32)})
            self.assertEqual(pool.results().get(timeout=10), sum(range(8)))
            pool.stop()
            pool.join(timeout=10)
        finally:
            producer.cleanup()

    def test_tensor_tree_sample_can_span_multiple_slots(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        try:
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("numpy not installed")
            return

        from oxidata.dataloader import TensorSampleProducer, TensorSampleWorkerPool

        def fn_sample(sample):
            return int(sample["a"].sum() + sample["b"].sum())

        producer = TensorSampleProducer(
            ring_name="oxd-tt-4",
            metadata_shm_size=1024 * 1024,
            metadata_slot_size=32 * 1024,
            payload_shm_size=4096,
            payload_slot_size=2048,
        )
        pool = TensorSampleWorkerPool(
            metadata_shm_name=producer.metadata_shm_name,
            ring_name=producer.ring_name,
            fn_sample=fn_sample,
            ack_queue=producer.ack_queue,
            num_workers=1,
        )

        sample = {
            "a": np.arange(256, dtype=np.int32),
            "b": np.arange(256, dtype=np.int32),
        }

        try:
            pool.start()
            producer.publish_sample(sample)
            self.assertEqual(pool.results().get(timeout=10), 2 * sum(range(256)))
            producer.publish_sample(sample)
            self.assertEqual(pool.results().get(timeout=10), 2 * sum(range(256)))
            pool.stop()
            pool.join(timeout=10)
        finally:
            producer.cleanup()

    def test_tensor_tree_rejects_sample_larger_than_slot(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        try:
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("numpy not installed")
            return

        from oxidata.dataloader import TensorSampleProducer

        producer = TensorSampleProducer(
            ring_name="oxd-tt-3",
            metadata_shm_size=1024 * 1024,
            metadata_slot_size=32 * 1024,
            payload_shm_size=4096,
            payload_slot_size=512,
        )
        try:
            with self.assertRaises(MemoryError):
                producer.publish_sample({"tokens": np.arange(1024, dtype=np.int32)})
        finally:
            producer.cleanup()


if __name__ == "__main__":
    unittest.main()
