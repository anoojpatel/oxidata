import unittest


class TestNativeRingBuffer(unittest.TestCase):
    def test_push_pop(self):
        from oxidata.native import available, ShmRingBuffer

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        name = "oxidata-test-ring"
        rb = ShmRingBuffer.create(name, capacity=8, slot_size=64)
        try:
            self.assertTrue(rb.push(b"hello"))
            out = rb.pop()
            self.assertEqual(out, b"hello")
            self.assertIsNone(rb.pop())
        finally:
            rb.close()
            rb.unlink()


if __name__ == "__main__":
    unittest.main()
