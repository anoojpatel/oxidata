import unittest

from oxidata.offheap import OffHeapScope, borrow_region, borrow_region_mut
from oxidata.lifetimes import BorrowError, LifetimeError


class TestOffHeapBorrowing(unittest.TestCase):
    def test_offheap_scope_invalidates_handle(self):
        owned = None
        with OffHeapScope(size=4096) as s:
            owned = s.alloc_bytes(b"abc")
            with borrow_region(owned) as r:
                self.assertEqual(bytes(r.view[:3]), b"abc")

        self.assertIsNotNone(owned)
        with self.assertRaises(LifetimeError):
            with borrow_region(owned):
                pass

    def test_borrow_rules_apply_to_offheap_handle(self):
        with OffHeapScope(size=4096) as s:
            h = s.alloc_bytes(b"hello")
            with borrow_region(h):
                with self.assertRaises(BorrowError):
                    with borrow_region_mut(h):
                        pass

    def test_mut_borrow_can_write(self):
        with OffHeapScope(size=4096) as s:
            h = s.alloc_bytes(b"xxxxx")
            with borrow_region_mut(h) as r:
                r.view[:5] = b"hello"
            with borrow_region(h) as r2:
                self.assertEqual(bytes(r2.view[:5]), b"hello")

    def test_freeze_blocks_mutation(self):
        with OffHeapScope(size=4096) as s:
            h = s.alloc_bytes(b"abc")
            h.freeze()
            with borrow_region(h) as r:
                self.assertEqual(bytes(r.view[:3]), b"abc")
            with self.assertRaises(BorrowError):
                with borrow_region_mut(h):
                    pass


if __name__ == "__main__":
    unittest.main()
