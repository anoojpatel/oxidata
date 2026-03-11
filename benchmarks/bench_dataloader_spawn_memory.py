import argparse
import os
import pickle
import re
import subprocess
import time
from dataclasses import dataclass
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


class PythonListDataset(Dataset):
    def __init__(self, data: list[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[int(idx)]


class TorchTensorDataset(Dataset):
    def __init__(self, data: list[Any]):
        self.data = [_tree_numpy_to_torch(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[int(idx)]


class TorchSerializedFullSampleDataset(Dataset):
    def __init__(self, data: list[Any]):
        payloads = [pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL) for x in data]
        sizes = [len(x) for x in payloads]
        offsets = np.asarray([0] + sizes, dtype=np.int64).cumsum()
        blob = np.concatenate([np.frombuffer(x, dtype=np.uint8) for x in payloads])
        self._offsets = torch.from_numpy(offsets)
        self._blob = torch.from_numpy(blob)

    def __len__(self) -> int:
        return int(self._offsets.numel() - 1)

    def __getitem__(self, idx: int) -> Any:
        start = int(self._offsets[int(idx)].item())
        end = int(self._offsets[int(idx) + 1].item())
        mv = memoryview(self._blob.numpy())[start:end]
        return pickle.loads(mv)


class OxidataDescriptorDataset(Dataset):
    def __init__(self, data: list[Any]):
        from oxidata.dataloader import _PayloadSlotAllocator, _encode_tree_leaf, _open_tree_leaf
        from oxidata.shm_arena import SharedMemoryArena

        total_bytes = sum(
            sum(v.nbytes for v in [x["genomics"]["signal"], x["genomics"]["aux"]]) for x in data
        )
        self._arena = SharedMemoryArena(size=max(64 * 1024 * 1024, total_bytes * 2), create=True)
        self._encode_tree_leaf = _encode_tree_leaf
        self._open_tree_leaf = _open_tree_leaf
        self._arena_name = self._arena.name
        self._descriptors = []
        allocator = _PayloadSlotAllocator(
            self._arena,
            slot_size=max(8 * 1024 * 1024, total_bytes),
            acquire_slot=lambda: 0,
        )
        for sample in data:
            self._descriptors.append(self._encode_tree_leaf(allocator, sample))
        self._worker_shm = None

    def __len__(self) -> int:
        return len(self._descriptors)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_worker_shm"] = None
        state["_arena"] = None
        return state

    def __getitem__(self, idx: int) -> Any:
        from multiprocessing import shared_memory

        if self._worker_shm is None:
            self._worker_shm = shared_memory.SharedMemory(name=self._arena_name, create=False)
        return self._open_tree_leaf(self._worker_shm, self._descriptors[int(idx)])

    def close(self) -> None:
        if self._worker_shm is not None:
            try:
                self._worker_shm.close()
            except Exception:
                pass
            self._worker_shm = None
        if self._arena is not None:
            try:
                self._arena.close()
            finally:
                try:
                    self._arena.unlink()
                except Exception:
                    pass
            self._arena = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass
class BenchResult:
    method: str
    samples_per_s: float
    parent_fp_bytes: int
    workers_fp_bytes: int


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _run_loader_bench(
    method: str,
    dataset: Dataset,
    *,
    num_workers: int,
    warmup: int,
    measure: int,
    multiprocessing_context: str,
) -> BenchResult:
    loader = DataLoader(
        dataset,
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
    parent_pid = os.getpid()
    parent_fp = _physical_footprint_bytes(parent_pid)
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
    return BenchResult(method=method, samples_per_s=measure / dt, parent_fp_bytes=parent_fp, workers_fp_bytes=workers_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--sample-mib", type=int, default=8)
    ap.add_argument("--sample-mib-list", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--measure", type=int, default=32)
    ap.add_argument("--context", type=str, default="spawn")
    args = ap.parse_args()

    sample_mibs = _parse_int_list(args.sample_mib_list) if args.sample_mib_list else [int(args.sample_mib)]

    for sample_mib in sample_mibs:
        sample_bytes = int(sample_mib) * 1024 * 1024
        data = [_make_numpy_sample(sample_bytes, i) for i in range(int(args.samples))]

        methods = [
            ("python_list", PythonListDataset(data)),
            ("torch_tensors", TorchTensorDataset(data)),
            ("torch_serialize_full", TorchSerializedFullSampleDataset(data)),
            ("oxidata_descriptors", OxidataDescriptorDataset(data)),
        ]

        print(
            f"dataset: {args.samples} samples x {sample_mib} MiB, "
            f"workers={args.workers}, warmup={args.warmup}, measure={args.measure}, context={args.context}"
        )
        print("method | samples/s | parent footprint | workers footprint | total footprint")
        for name, ds in methods:
            try:
                res = _run_loader_bench(
                    name,
                    ds,
                    num_workers=int(args.workers),
                    warmup=int(args.warmup),
                    measure=int(args.measure),
                    multiprocessing_context=str(args.context),
                )
                total_fp = res.parent_fp_bytes + res.workers_fp_bytes
                print(
                    f"{res.method:>18} | "
                    f"{res.samples_per_s:>9.2f} | "
                    f"{_format_gib(res.parent_fp_bytes):>16} | "
                    f"{_format_gib(res.workers_fp_bytes):>17} | "
                    f"{_format_gib(total_fp):>15}"
                )
            except Exception as e:
                print(f"{name:>18} | {'ERROR':>9} | {type(e).__name__}: {e}")
            finally:
                if hasattr(ds, "close"):
                    ds.close()
        print()

    print("Metric note: macOS `vmmap` physical footprint is used per process. Summed worker+parent footprint is still an approximation.")


if __name__ == "__main__":
    main()
