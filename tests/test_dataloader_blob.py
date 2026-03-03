import unittest


class TestDataloaderBlob(unittest.TestCase):
    def test_blob_workerpool_end_to_end(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.dataloader import Producer, BlobWorkerPool

        ring_name = "oxidata-test-dl-blob-ring"

        def fn_obj(o) -> int:
            return int(o["id"])  # type: ignore[index]

        producer = Producer(ring_name=ring_name, shm_size=4 * 1024 * 1024, slot_size=4096)
        pool = BlobWorkerPool(
            shm_name=producer.shm_name,
            ring_name=producer.ring_name,
            codec="json",
            fn_obj=fn_obj,
            num_workers=1,
        )

        try:
            pool.start()
            producer.publish_blob({"id": 7, "values": [1, 2, 3]}, codec="json")
            pool.stop()
            pool.join(timeout=10)

            q = pool.results()
            got = []
            while not q.empty():
                got.append(q.get())

            self.assertIn(7, got)
        finally:
            producer.cleanup()


if __name__ == "__main__":
    unittest.main()
