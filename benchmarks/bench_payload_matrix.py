import multiprocessing as mp
import time
import uuid


PAYLOAD_SIZES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 104857600]


def _n_msgs_for_payload(payload_size: int) -> int:
    if payload_size <= 256:
        return 20_000
    if payload_size <= 4096:
        return 10_000
    if payload_size <= 65536:
        return 2_000
    if payload_size <= 262144:
        return 500
    if payload_size <= 1048576:
        return 100
    if payload_size <= 4194304:
        return 25
    return 2


def _worker_mpqueue(q_in: "mp.Queue", q_out: "mp.Queue"):
    total = 0
    while True:
        payload = q_in.get()
        if payload is None:
            break
        total += len(payload)
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
            item = rb.pop_handle()
            if item is None:
                continue
            offset, nbytes, _kind_tag = item
            handle = Handle(shm_name, int(offset), int(nbytes), "bytes")
            total += len(read_handle_bytes(handle))
            seen += 1
        q_out.put(total)
    finally:
        rb.close()


def _worker_slotring_view(shm_name: str, ring_name: str, n_msgs: int, q_out: "mp.Queue"):
    from multiprocessing import shared_memory

    from oxidata.native import ShmRingBuffer

    rb = ShmRingBuffer.attach(ring_name)
    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    try:
        total = 0
        seen = 0
        while seen < n_msgs:
            item = rb.pop_handle()
            if item is None:
                continue
            offset, nbytes, _kind_tag = item
            start = int(offset)
            end = start + int(nbytes)
            view = shm.buf[start:end]
            total += len(view)
            view.release()
            seen += 1
        q_out.put(total)
    finally:
        shm.close()
        rb.close()


def bench_mpqueue(payload_size: int, n_msgs: int) -> float:
    ctx = mp.get_context("spawn")
    q_in = ctx.Queue(maxsize=1024)
    q_out = ctx.Queue()
    payload = b"x" * int(payload_size)

    p = ctx.Process(target=_worker_mpqueue, args=(q_in, q_out))
    p.start()
    try:
        t0 = time.perf_counter()
        for _ in range(int(n_msgs)):
            q_in.put(payload)
        q_in.put(None)
        _ = q_out.get(timeout=30)
        p.join(timeout=30)
        return time.perf_counter() - t0
    finally:
        if p.is_alive():
            p.terminate()


def bench_slotring(payload_size: int, n_msgs: int) -> float:
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("native extension not built")

    from oxidata.dataloader import Producer

    ctx = mp.get_context("spawn")
    q_out = ctx.Queue()
    ring_name = f"oxd-bsr-{uuid.uuid4().hex[:8]}"
    producer = Producer(
        ring_name=ring_name,
        shm_size=max(64 * 1024 * 1024, int(payload_size) * int(n_msgs) * 2),
        slot_size=max(256, int(payload_size)),
    )

    p = ctx.Process(target=_worker_slotring, args=(producer.shm_name, producer.ring_name, int(n_msgs), q_out))
    p.start()
    try:
        payload = b"x" * int(payload_size)
        t0 = time.perf_counter()
        for _ in range(int(n_msgs)):
            producer.publish(payload)
        _ = q_out.get(timeout=30)
        p.join(timeout=30)
        return time.perf_counter() - t0
    finally:
        if p.is_alive():
            p.terminate()
        producer.cleanup()


def bench_slotring_view(payload_size: int, n_msgs: int) -> float:
    from oxidata.native import available as native_available

    if not native_available():
        raise SystemExit("native extension not built")

    from oxidata.dataloader import Producer

    ctx = mp.get_context("spawn")
    q_out = ctx.Queue()
    ring_name = f"oxd-bsv-{uuid.uuid4().hex[:8]}"
    producer = Producer(
        ring_name=ring_name,
        shm_size=max(64 * 1024 * 1024, int(payload_size) * int(n_msgs) * 2),
        slot_size=max(256, int(payload_size)),
    )

    p = ctx.Process(target=_worker_slotring_view, args=(producer.shm_name, producer.ring_name, int(n_msgs), q_out))
    p.start()
    try:
        payload = b"x" * int(payload_size)
        t0 = time.perf_counter()
        for _ in range(int(n_msgs)):
            producer.publish(payload)
        _ = q_out.get(timeout=30)
        p.join(timeout=30)
        return time.perf_counter() - t0
    finally:
        if p.is_alive():
            p.terminate()
        producer.cleanup()


def bench_blob_roundtrip(payload_size: int, n_msgs: int, codec_name: str) -> float:
    from oxidata.blob_codec import codec_by_name
    from oxidata.shm_arena import SharedMemoryArena

    codec = codec_by_name(codec_name)
    payload = {
        "payload": "x" * int(payload_size),
        "meta": {"payload_size": int(payload_size)},
    }

    arena = SharedMemoryArena(size=max(32 * 1024 * 1024, int(payload_size) * 4))
    try:
        t0 = time.perf_counter()
        for _ in range(int(n_msgs)):
            arena.reset()
            handle = arena.alloc_bytes(codec.encode(payload), kind=f"blob:{codec.name}")
            _ = codec.decode(arena.read_bytes(handle))
        return time.perf_counter() - t0
    finally:
        arena.close()
        arena.unlink()


def _format_rate(n_msgs: int, elapsed: float) -> str:
    return f"{(n_msgs / elapsed):>10.0f}"


def main():
    rows = []
    codecs = ["json"]
    try:
        import msgspec  # noqa: F401

        codecs.append("msgspec_json")
    except Exception:
        pass

    for payload_size in PAYLOAD_SIZES:
        n_msgs = _n_msgs_for_payload(payload_size)
        mp_dt = bench_mpqueue(payload_size, n_msgs)
        row = {
            "payload": payload_size,
            "n_msgs": n_msgs,
            "mp_queue": _format_rate(n_msgs, mp_dt),
        }

        try:
            sr_dt = bench_slotring(payload_size, n_msgs)
            row["slot_ring_copy"] = _format_rate(n_msgs, sr_dt)
        except SystemExit as exc:
            row["slot_ring_copy"] = str(exc)

        try:
            srv_dt = bench_slotring_view(payload_size, n_msgs)
            row["slot_ring_view"] = _format_rate(n_msgs, srv_dt)
        except SystemExit as exc:
            row["slot_ring_view"] = str(exc)

        for codec_name in codecs:
            blob_dt = bench_blob_roundtrip(payload_size, max(50, min(n_msgs, 500)), codec_name)
            row[codec_name] = _format_rate(max(50, min(n_msgs, 500)), blob_dt)

        rows.append(row)

    headers = ["payload", "n_msgs", "mp_queue", "slot_ring_copy", "slot_ring_view"] + codecs
    print("transport path throughput (bulk payload movement)")
    print("payload bytes | messages | mp.Queue msg/s | slot+ring+copy msg/s | slot+ring+view msg/s")
    for row in rows:
        print(
            f"{row['payload']:>13} | "
            f"{row['n_msgs']:>8} | "
            f"{row['mp_queue']:>14} | "
            f"{row['slot_ring_copy']:>21} | "
            f"{row['slot_ring_view']:>21}"
        )

    print()
    print("metadata codec throughput (descriptor-only roundtrip, not bulk tensor transport)")
    print("payload bytes | descriptor ops | " + " | ".join(codecs))
    for row in rows:
        blob_ops = max(50, min(int(row["n_msgs"]), 500))
        print(
            f"{row['payload']:>13} | "
            f"{blob_ops:>14} | "
            + " | ".join(f"{row[h]:>16}" for h in codecs)
        )

    print()
    print("Interpretation: `slot+ring+copy` includes a per-message copy back to Python bytes.")
    print("Interpretation: `slot+ring+view` keeps the worker attached and only touches shared-memory views.")
    print("Interpretation: codec numbers measure descriptor metadata only; tensor bytes should stay out of the codec path.")


if __name__ == "__main__":
    main()
