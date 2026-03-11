import argparse
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _first_only(batch):
    return batch[0]


def _touch_tree(obj: Any) -> int:
    if isinstance(obj, dict):
        return sum(_touch_tree(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_touch_tree(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return int(obj.reshape(-1)[0])
    if isinstance(obj, torch.Tensor):
        return int(obj.reshape(-1)[0].item())
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return 0


def _physical_footprint_bytes(pid: int) -> int:
    out = subprocess.check_output(["vmmap", "-summary", str(pid)], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"Physical footprint:\s+([0-9.]+)([KMG])", out)
    if match is None:
        raise RuntimeError(f"could not parse vmmap output for pid {pid}")
    value = float(match.group(1))
    unit = match.group(2)
    scale = {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}[unit]
    return int(value * scale)


def _format_gib(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 3):.2f} GiB"


def _make_numpy_sample(sample_bytes: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_f32 = max(1, sample_bytes // 4)
    split = max(1, n_f32 // 2)
    signal = rng.standard_normal(split, dtype=np.float32)
    aux = rng.standard_normal(n_f32 - split, dtype=np.float32)
    return {
        "genomics": {
            "signal": signal.reshape(-1, 1),
            "aux": aux.reshape(-1, 1),
        },
        "meta": {"sample_id": int(seed)},
    }


def _write_sample_files(root: Path, *, samples: int, sample_bytes: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(int(samples)):
        sample = _make_numpy_sample(sample_bytes, i)
        path = root / f"sample_{i:05d}.npz"
        np.savez(path, signal=sample["genomics"]["signal"], aux=sample["genomics"]["aux"], sample_id=i)
        out.append(path)
    return out


def _read_sample_file(path: Path, *, sleep_ms: int = 0) -> dict[str, Any]:
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    with np.load(path, allow_pickle=False) as data:
        signal = np.array(data["signal"], copy=True)
        aux = np.array(data["aux"], copy=True)
        sample_id = int(data["sample_id"])
    return {
        "genomics": {"signal": signal, "aux": aux},
        "meta": {"sample_id": sample_id},
    }


def _tree_numpy_to_torch(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _tree_numpy_to_torch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tree_numpy_to_torch(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tree_numpy_to_torch(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj.copy())
    return obj


class StreamingTorchFileDataset(Dataset):
    def __init__(self, paths: list[Path], *, sleep_ms: int):
        self.paths = paths
        self.sleep_ms = int(sleep_ms)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        return _tree_numpy_to_torch(_read_sample_file(self.paths[int(idx)], sleep_ms=self.sleep_ms))


@dataclass
class BenchResult:
    method: str
    samples_per_s: float
    parent_fp_bytes: int
    workers_fp_bytes: int


def _run_torch_streaming(
    paths: list[Path],
    *,
    num_workers: int,
    warmup: int,
    measure: int,
    sleep_ms: int,
    multiprocessing_context: str,
) -> BenchResult:
    ds = StreamingTorchFileDataset(paths, sleep_ms=sleep_ms)
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
        collate_fn=_first_only,
        persistent_workers=True,
    )
    it = iter(loader)
    for _ in range(warmup):
        _touch_tree(next(it))
    worker_pids = [int(w.pid) for w in getattr(it, "_workers", [])]
    parent_fp = _physical_footprint_bytes(os.getpid())
    workers_fp = sum(_physical_footprint_bytes(pid) for pid in worker_pids)
    t0 = time.perf_counter()
    checksum = 0
    for _ in range(measure):
        checksum += _touch_tree(next(it))
    dt = time.perf_counter() - t0
    if checksum == -1:
        print("impossible")
    del it
    del loader
    return BenchResult("streaming_torch_files", measure / dt, parent_fp, workers_fp)


def _oxidata_worker(
    worker_id: int,
    req_q: "mp.Queue",
    res_q: "mp.Queue",
    free_slots_q: "mp.Queue",
    paths: list[str],
    payload_shm_name: str,
    payload_slot_size: int,
    sleep_ms: int,
):
    from oxidata.dataloader import _PayloadSlotAllocator, _encode_tree_leaf
    from oxidata.shm_arena import SharedMemoryArena

    arena = SharedMemoryArena(size=None, name=payload_shm_name, create=False)
    try:
        while True:
            idx = req_q.get()
            if idx is None:
                break
            sample = _read_sample_file(Path(paths[int(idx)]), sleep_ms=sleep_ms)
            acquired: list[int] = []

            def acquire_slot() -> int:
                slot = int(free_slots_q.get())
                acquired.append(slot)
                return slot

            alloc = _PayloadSlotAllocator(arena, slot_size=payload_slot_size, acquire_slot=acquire_slot)
            tree = _encode_tree_leaf(alloc, sample)
            res_q.put(
                {
                    "i": int(idx),
                    "payload_shm_name": payload_shm_name,
                    "slot_indices": list(alloc.used_slots()),
                    "tree": tree,
                    "worker_id": int(worker_id),
                }
            )
    finally:
        arena.close()


def _run_oxidata_streaming(
    paths: list[Path],
    *,
    num_workers: int,
    warmup: int,
    measure: int,
    sleep_ms: int,
    payload_slot_mib: int,
    inflight_slots: int,
    stage_torch: bool,
) -> BenchResult:
    from oxidata.dataloader import _open_tree_leaf
    from oxidata.shm_arena import SharedMemoryArena
    from oxidata.torch_stage import tensor_tree_to_torch
    from multiprocessing import shared_memory

    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    res_q = ctx.Queue()
    free_slots_q = ctx.Queue()
    payload_slot_size = int(payload_slot_mib) * 1024 * 1024
    payload_arena = SharedMemoryArena(size=payload_slot_size * int(inflight_slots), create=True)
    for i in range(int(inflight_slots)):
        free_slots_q.put(i)

    workers = [
        ctx.Process(
            target=_oxidata_worker,
            args=(
                i,
                req_q,
                res_q,
                free_slots_q,
                [str(p) for p in paths],
                payload_arena.name,
                payload_slot_size,
                int(sleep_ms),
            ),
        )
        for i in range(int(num_workers))
    ]
    for p in workers:
        p.start()

    shm = shared_memory.SharedMemory(name=payload_arena.name, create=False)
    try:
        total = warmup + measure
        max_inflight = max(1, min(int(inflight_slots), max(1, int(num_workers) * 2), total))
        sent = 0
        for _ in range(max_inflight):
            req_q.put(sent % len(paths))
            sent += 1

        def consume_one() -> int:
            nonlocal sent
            item = res_q.get()
            sample = _open_tree_leaf(shm, item["tree"])
            if stage_torch:
                sample = tensor_tree_to_torch(sample, pin_memory=False)
            checksum = _touch_tree(sample)
            for slot in item["slot_indices"]:
                free_slots_q.put(int(slot))
            if sent < total:
                req_q.put(sent % len(paths))
                sent += 1
            return checksum

        for _ in range(warmup):
            consume_one()

        parent_fp = _physical_footprint_bytes(os.getpid())
        workers_fp = sum(_physical_footprint_bytes(int(p.pid)) for p in workers if p.pid is not None)

        t0 = time.perf_counter()
        checksum = 0
        for _ in range(measure):
            checksum += consume_one()
        dt = time.perf_counter() - t0
        if checksum == -1:
            print("impossible")
        return BenchResult("oxidata_streaming", measure / dt, parent_fp, workers_fp)
    finally:
        try:
            shm.close()
        except Exception:
            pass
        for _ in workers:
            req_q.put(None)
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        payload_arena.close()
        payload_arena.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--sample-mib", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--measure", type=int, default=16)
    ap.add_argument("--context", type=str, default="spawn")
    ap.add_argument("--sleep-ms", type=int, default=10)
    ap.add_argument("--payload-slot-mib", type=int, default=8)
    ap.add_argument("--inflight-slots", type=int, default=16)
    ap.add_argument("--oxidata-stage-torch", action="store_true")
    ap.add_argument(
        "--methods",
        nargs="+",
        choices=("streaming_torch_files", "oxidata_streaming"),
        default=("streaming_torch_files", "oxidata_streaming"),
    )
    args = ap.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="oxidata-e2e-bench-"))
    try:
        paths = _write_sample_files(
            tmpdir,
            samples=int(args.samples),
            sample_bytes=int(args.sample_mib) * 1024 * 1024,
        )
        print(
            f"end-to-end file-backed benchmark: {args.samples} samples x {args.sample_mib} MiB, "
            f"workers={args.workers}, warmup={args.warmup}, measure={args.measure}, "
            f"context={args.context}, sleep_ms={args.sleep_ms}, "
            f"payload_slot_mib={args.payload_slot_mib}, inflight_slots={args.inflight_slots}, "
            f"oxidata_stage_torch={bool(args.oxidata_stage_torch)}"
        )
        print("method | samples/s | parent footprint | workers footprint | total footprint")

        for method in args.methods:
            if method == "streaming_torch_files":
                res = _run_torch_streaming(
                    paths,
                    num_workers=int(args.workers),
                    warmup=int(args.warmup),
                    measure=int(args.measure),
                    sleep_ms=int(args.sleep_ms),
                    multiprocessing_context=str(args.context),
                )
            else:
                res = _run_oxidata_streaming(
                    paths,
                    num_workers=int(args.workers),
                    warmup=int(args.warmup),
                    measure=int(args.measure),
                    sleep_ms=int(args.sleep_ms),
                    payload_slot_mib=int(args.payload_slot_mib),
                    inflight_slots=int(args.inflight_slots),
                    stage_torch=bool(args.oxidata_stage_torch),
                )
            total = res.parent_fp_bytes + res.workers_fp_bytes
            print(
                f"{res.method:>22} | "
                f"{res.samples_per_s:>9.2f} | "
                f"{_format_gib(res.parent_fp_bytes):>16} | "
                f"{_format_gib(res.workers_fp_bytes):>17} | "
                f"{_format_gib(total):>15}"
            )

        print()
        print("Metric note: macOS `vmmap` physical footprint is used per process. Summed worker+parent footprint is still an approximation.")
        print("Benchmark note: this is the full source->worker->transport->main-process path, not a cached-representation benchmark.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
