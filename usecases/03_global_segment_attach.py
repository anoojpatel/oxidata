from oxidata.scopes import GlobalSegment
from oxidata.mp import read_handle_bytes


def main():
    seg = GlobalSegment.create("oxidata-global-demo", size=8 * 1024 * 1024)
    try:
        h = seg.alloc_bytes(b"shared")
        pub = seg.publish(h)

        seg2 = GlobalSegment.attach("oxidata-global-demo")
        try:
            with pub.owned.borrow() as b:
                h2 = b.get()
            print("attached read:", read_handle_bytes(h2))
        finally:
            seg2.close()
    finally:
        seg.close()
        seg.unlink()


if __name__ == "__main__":
    main()
