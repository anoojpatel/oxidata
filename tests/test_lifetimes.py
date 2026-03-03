import unittest

from oxidata.lifetimes import Scope, Arena, LifetimeError, BorrowError


class TestLifetimes(unittest.TestCase):
    def test_scope_invalidation_use_after_free(self):
        arena_owned = None
        with Scope() as scope:
            arena = Arena(scope)
            arena_owned = arena.alloc(123)
            self.assertTrue(arena_owned.alive())

        self.assertIsNotNone(arena_owned)
        self.assertFalse(arena_owned.alive())

        with self.assertRaises(LifetimeError):
            with arena_owned.borrow() as b:
                _ = b.get()

    def test_multiple_readers_ok(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc({"k": "v"})
            with x.borrow() as r1:
                with x.borrow() as r2:
                    self.assertEqual(r1.get(), r2.get())

    def test_mut_borrow_excludes_readers(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc([1, 2, 3])
            with x.borrow() as r:
                with self.assertRaises(BorrowError):
                    with x.borrow_mut():
                        pass

    def test_mut_borrow_excludes_other_mut(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc([1, 2, 3])
            with x.borrow_mut() as w:
                w.mutate().append(4)
                with self.assertRaises(BorrowError):
                    with x.borrow_mut():
                        pass

    def test_into_inner_moves_value(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc(7)
            v = x.into_inner()
            self.assertEqual(v, 7)
            self.assertFalse(x.alive())
            with self.assertRaises(LifetimeError):
                with x.borrow():
                    pass


if __name__ == "__main__":
    unittest.main()
