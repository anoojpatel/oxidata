from oxidata.scopes import GlobalSegment
from oxidata.blob_codec import codec_by_name
from oxidata.mp import read_handle_bytes


def main():
    codec = codec_by_name("json")
    seg = GlobalSegment.create("oxidata-cache-demo", size=16 * 1024 * 1024)
    try:
        index: dict[str, object] = {}

        def put(key: str, obj):
            data = codec.encode(obj)
            h = seg.alloc_bytes(data)
            seg.publish(h)
            index[key] = h

        def get(key: str):
            h = index[key]
            with h.borrow() as b:
                hh = b.get()
            raw = read_handle_bytes(hh)
            return codec.decode(raw)

        put("a", {"x": 1, "y": [1, 2, 3]})
        print(get("a"))
    finally:
        seg.close()
        seg.unlink()


if __name__ == "__main__":
    main()
