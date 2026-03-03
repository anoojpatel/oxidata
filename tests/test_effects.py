import unittest

from oxidata.lifetimes import Scope, Arena, BorrowError
from oxidata.effects import borrow, borrow_mut, run


class TestEffects(unittest.TestCase):
    def test_effects_borrow_read(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc(10)

            def prog():
                b = yield borrow(x)
                return b.get() + 1

            self.assertEqual(run(prog()), 11)

    def test_effects_borrow_mut(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc([1])

            def prog():
                b = yield borrow_mut(x)
                b.mutate().append(2)
                b2 = yield borrow(x)
                return list(b2.get())

            self.assertEqual(run(prog()), [1, 2])

    def test_effects_enforces_borrow_rules(self):
        with Scope() as scope:
            a = Arena(scope)
            x = a.alloc([1])

            def prog():
                r = yield borrow(x)
                # Attempt to re-enter a mutable borrow while read is held
                _ = r.get()
                w = yield borrow_mut(x)
                w.mutate().append(2)
                return 0

            with self.assertRaises(BorrowError):
                run(prog())


if __name__ == "__main__":
    unittest.main()
