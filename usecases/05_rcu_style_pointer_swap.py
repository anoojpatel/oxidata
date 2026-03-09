import time

from oxidata.scopes import GlobalSegment
from oxidata.blob_codec import codec_by_name
from oxidata.native import AtomicI64, available as native_available
from oxidata.mp import read_handle_bytes


def main():
    if not native_available():
        raise SystemExit("native extension not built")

    codec = codec_by_name("json")
    seg = GlobalSegment.create("oxidata-rcu-demo", size=32 * 1024 * 1024)
    try:
        table: list[object] = []
        current = AtomicI64(0)

        def publish_obj(obj):
            data = codec.encode(obj)
            h = seg.alloc_bytes(data)
            seg.publish(h)
            table.append(h)
            current.store_i64(len(table) - 1)

        publish_obj({"version": 0})

        for v in range(1, 5):
            publish_obj({"version": v})
            idx = int(current.load_i64())
            h = table[idx]
            with h.borrow() as b:
                hh = b.get()
            obj = codec.decode(read_handle_bytes(hh))
            print("reader sees:", obj)
            time.sleep(0.1)
    finally:
        seg.close()
        seg.unlink()


if __name__ == "__main__":
    main()
