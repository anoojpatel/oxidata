import multiprocessing as mp
import time


def _worker_mpqueue(q_in: "mp.Queue", q_out: "mp.Queue"):
    # Receives raw bytes (pickled by mp) and returns total length.
    total = 0
    while True:
        b = q_in.get()
        if b is None:
            break
        total += len(b)
    q_out.put(total)


def _worker_slotring(shm_name: str, ring_name: str, n_msgs: int, q_out: "mp.Queue"):
    from oxidata.native import ShmRingBuffer
    from oxidata.mp import read_handle_bytes
    from oxidata.shm_arena import Handle

    rb = ShmRingBuffer.attach(ring_name)
    try:
        total = 0
        seen = 0
        while seen < n_msgs:
            t = rb.pop_handle()
            if t is None:
                continue
            offset, nbytes, kind_tag = t
            h = Handle(shm_name, int(offset), int(nbytes), "bytes")
            b = read_handle_bytes(h)
            total += len(b)
            seen += 1
        q_out.put(total)
    finally:
        rb.close()


def bench_mpqueue(payload_size: int = 1024, n_msgs: int = 50_000) -> float:
    ctx = mp.get_context("spawn")
    q_in = ctx.Queue(maxsize=1024)
    q_out = ctx.Queue()

    payload = b"x" * int(payload_size)

    p = ctx.Process(target=_worker_mpqueue, args=(q_in, q_out))
    p.start()

    t0 = time.perf_counter()
    for _ in range(int(n_msgs)):
        q_in.put(payload)
    q_in.put(None)
    _ = q_out.get(timeout=30)
    p.join(timeout=30)
    return time.perf_counter() - t0


def bench_slotring(payload_size: int = 1024, n_msgs: int = 50_000) -> float:
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("native extension not built")

    from oxidata.dataloader import Producer

    ring_name = "oxidata-bench-slotring"
    ctx = mp.get_context("spawn")
    q_out = ctx.Queue()

    producer = Producer(ring_name=ring_name, shm_size=64 * 1024 * 1024, slot_size=max(256, payload_size))
    try:
        p = ctx.Process(target=_worker_slotring, args=(producer.shm_name, producer.ring_name, int(n_msgs), q_out))
        p.start()

        payload = b"x" * int(payload_size)
        t0 = time.perf_counter()
        for _ in range(int(n_msgs)):
            producer.publish(payload)
        _ = q_out.get(timeout=30)
        p.join(timeout=30)
        return time.perf_counter() - t0
    finally:
        producer.cleanup()


def main():
    payload_size = 1024
    n_msgs = 20_000

    t_mp = bench_mpqueue(payload_size=payload_size, n_msgs=n_msgs)
    print(f"mp.Queue (pickled bytes): {n_msgs/t_mp:.0f} msg/s")

    try:
        t_sr = bench_slotring(payload_size=payload_size, n_msgs=n_msgs)
        print(f"slot+ring (handles only): {n_msgs/t_sr:.0f} msg/s")
    except SystemExit as e:
        print(e)


if __name__ == "__main__":
    main()
