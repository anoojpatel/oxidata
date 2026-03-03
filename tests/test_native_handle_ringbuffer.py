import unittest


class TestNativeHandleRingBuffer(unittest.TestCase):
    def test_push_pop_handle(self):
        from oxidata.native import available, ShmRingBuffer

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        name = "oxidata-test-handle-ring"
        rb = ShmRingBuffer.create(name, capacity=8, slot_size=64)
        try:
            self.assertTrue(rb.push_handle(offset=123, nbytes=456, kind_tag=ShmRingBuffer.KIND_BYTES))
            t = rb.pop_handle()
            self.assertIsNotNone(t)
            off, nb, kt = t
            self.assertEqual(int(off), 123)
            self.assertEqual(int(nb), 456)
            self.assertEqual(int(kt), ShmRingBuffer.KIND_BYTES)
        finally:
            rb.close()
            rb.unlink()


if __name__ == "__main__":
    unittest.main()
