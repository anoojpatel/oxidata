import multiprocessing as mp


def _consumer(arena_name: str, rb_name: str, q_out: "mp.Queue"):
    from oxidata.native import ShmRingBuffer
    from oxidata.mp import read_handle_bytes
    from oxidata.shm_arena import Handle

    rb = ShmRingBuffer.attach(rb_name)
    try:
        out = []
        while True:
            t = rb.pop_handle()
            if t is None:
                continue
            offset, nbytes, kind_tag = t
            # sentinel convention: offset==0,nbytes==0,kind_tag==0
            if int(offset) == 0 and int(nbytes) == 0 and int(kind_tag) == 0:
                break

            h = Handle(arena_name, int(offset), int(nbytes), "bytes")
            out.append(read_handle_bytes(h))

        q_out.put(out)
    finally:
        rb.close()


def main():
    from oxidata.native import available as native_available, ShmRingBuffer
    from oxidata.shm_arena import SharedMemoryArena

    if not native_available():
        raise SystemExit("native extension not built")

    ctx = mp.get_context("spawn")

    arena = SharedMemoryArena(size=4 * 1024 * 1024)
    rb_name = "oxidata-dl-ring"
    rb = ShmRingBuffer.create(rb_name, capacity=1024, slot_size=64)

    q_out = ctx.Queue()
    p = ctx.Process(target=_consumer, args=(arena.name, rb_name, q_out))
    p.start()

    try:
        for i in range(100):
            payload = f"sample-{i}".encode("utf-8")
            h = arena.alloc_bytes(payload)
            while not rb.push_handle(offset=h.offset, nbytes=h.nbytes, kind_tag=ShmRingBuffer.KIND_BYTES):
                pass

        while not rb.push_handle(offset=0, nbytes=0, kind_tag=0):
            pass

        out = q_out.get(timeout=20)
        p.join(timeout=20)
        print("received", len(out), "items")
        print(out[:3])
    finally:
        rb.close()
        rb.unlink()
        arena.close()
        arena.unlink()


if __name__ == "__main__":
    main()
