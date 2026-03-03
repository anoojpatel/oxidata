from oxidata.scopes import Frame
from oxidata.offheap import borrow_region, borrow_region_mut


def main():
    with Frame(size=8 * 1024 * 1024) as f:
        h = f.alloc_bytes(b"hello world")
        with borrow_region(h) as r:
            print(bytes(r.view))

        with borrow_region_mut(h) as r:
            r.view[:5] = b"HELLO"

        with borrow_region(h) as r:
            print(bytes(r.view))

        pub = f.publish(h)
        try:
            with borrow_region_mut(pub.owned) as r:
                r.view[:1] = b"x"
        except Exception as e:
            print("expected frozen mutation failure:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
