import unittest


def _top_level_len(b: bytes) -> int:
    return len(b)


class TestDataloaderSpawnCallbacks(unittest.TestCase):
    def test_top_level_callback_supported(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.dataloader import Producer, WorkerPool

        producer = Producer(ring_name="oxidata-test-spawn-top-level", shm_size=2 * 1024 * 1024, slot_size=128)
        pool = WorkerPool(
            shm_name=producer.shm_name,
            ring_name=producer.ring_name,
            fn_bytes=_top_level_len,
            num_workers=1,
        )

        try:
            pool.start()
            producer.publish(b"abcdef")
            pool.stop()
            self.assertEqual(pool.results().get(timeout=10), 6)
            pool.join(timeout=10)
        finally:
            producer.cleanup()

    def test_local_callback_without_closure_supported(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.dataloader import Producer, WorkerPool

        def local_len(b: bytes) -> int:
            return len(b)

        producer = Producer(ring_name="oxidata-test-spawn-local", shm_size=2 * 1024 * 1024, slot_size=128)
        pool = WorkerPool(
            shm_name=producer.shm_name,
            ring_name=producer.ring_name,
            fn_bytes=local_len,
            num_workers=1,
        )

        try:
            pool.start()
            producer.publish(b"abcd")
            pool.stop()
            self.assertEqual(pool.results().get(timeout=10), 4)
            pool.join(timeout=10)
        finally:
            producer.cleanup()

    def test_closure_callback_rejected(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.dataloader import Producer, WorkerPool

        extra = 1

        def closure_len(b: bytes) -> int:
            return len(b) + extra

        producer = Producer(ring_name="oxidata-test-spawn-closure", shm_size=2 * 1024 * 1024, slot_size=128)
        try:
            with self.assertRaises(TypeError):
                WorkerPool(
                    shm_name=producer.shm_name,
                    ring_name=producer.ring_name,
                    fn_bytes=closure_len,
                    num_workers=1,
                )
        finally:
            producer.cleanup()


if __name__ == "__main__":
    unittest.main()
