import multiprocessing as mp


def _consumer(rb_name: str, q_out: "mp.Queue"):
    from oxidata.native import ShmRingBuffer

    rb = ShmRingBuffer.attach(rb_name)
    try:
        # Pop until sentinel
        items = []
        while True:
            msg = rb.pop()
            if msg is None:
                continue
            if msg == b"__STOP__":
                break
            items.append(msg)
        q_out.put(items)
    finally:
        rb.close()


def main():
    ctx = mp.get_context("spawn")

    from oxidata.native import available as native_available, ShmRingBuffer

    if not native_available():
        raise SystemExit("native extension not built")

    rb_name = "oxidata-example-ring"
    rb = ShmRingBuffer.create(rb_name, capacity=1024, slot_size=256)

    q_out = ctx.Queue()
    p = ctx.Process(target=_consumer, args=(rb_name, q_out))
    p.start()

    for i in range(10):
        payload = f"item-{i}".encode("utf-8")
        while not rb.push(payload):
            pass

    while not rb.push(b"__STOP__"):
        pass

    items = q_out.get(timeout=10)
    p.join(timeout=10)

    rb.close()
    rb.unlink()

    print("consumer got", items)


if __name__ == "__main__":
    main()
