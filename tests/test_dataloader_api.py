import unittest


class TestDataloaderAPI(unittest.TestCase):
    def test_producer_workerpool_end_to_end(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.dataloader import Producer, WorkerPool

        ring_name = "oxidata-test-dl-ring"

        def fn_bytes(b: bytes) -> int:
            return len(b)

        producer = Producer(ring_name=ring_name, shm_size=2 * 1024 * 1024, slot_size=128)
        pool = WorkerPool(shm_name=producer.shm_name, ring_name=producer.ring_name, fn_bytes=fn_bytes, num_workers=1)

        try:
            pool.start()
            producer.publish(b"abc")
            producer.publish(b"hello")

            pool.stop()
            pool.join(timeout=10)

            q = pool.results()
            got = []
            # Queue may have 2 items; pull what's available.
            while not q.empty():
                got.append(q.get())

            self.assertTrue(any(x in got for x in (3, 5)))
        finally:
            producer.cleanup()


if __name__ == "__main__":
    unittest.main()
