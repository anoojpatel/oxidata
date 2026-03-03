import unittest


class TestNative(unittest.TestCase):
    def test_atomic_i64(self):
        from oxidata.native import available, AtomicI64

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        a = AtomicI64(1)
        self.assertEqual(a.load(), 1)
        self.assertEqual(a.fetch_add(2), 1)
        self.assertEqual(a.load(), 3)
        a.store(10)
        self.assertEqual(a.load(), 10)

    def test_rwbytes(self):
        from oxidata.native import available

        if not available():
            self.skipTest("pyocaml_native extension not built")
            return

        from oxidata.native import RwBytes

        b = RwBytes(16)
        self.assertEqual(b.size(), 16)
        n = b.write(b"abc", 0)
        self.assertEqual(n, 3)
        out = bytearray(4)
        n2 = b.readinto(out, 0)
        self.assertEqual(n2, 4)
        self.assertEqual(bytes(out)[:3], b"abc")


if __name__ == "__main__":
    unittest.main()
