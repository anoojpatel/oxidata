import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, TypeVar


def _require_torch():
    try:
        import torch  # type: ignore
        from torch.utils.data import Dataset  # type: ignore

        return torch, Dataset
    except Exception as e:
        raise SystemExit("torch not installed") from e


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class _Item(Generic[R]):
    i: int
    value: R


class _OrderedEmitter(Generic[R]):
    def __init__(self):
        self._next = 0
        self._buffer: dict[int, R] = {}

    def push(self, i: int, value: R) -> Iterable[R]:
        if i == self._next:
            out = [value]
            self._next += 1
            while self._next in self._buffer:
                out.append(self._buffer.pop(self._next))
                self._next += 1
            return out

        self._buffer[int(i)] = value
        return ()


class _IndexWorker(Generic[R]):
    def __init__(self, *, make_dataset: Callable[[], Any], worker_map: Callable[[Any], R]):
        self._make_dataset = make_dataset
        self._worker_map = worker_map
        self._ds: Any = None

    def __call__(self, msg: dict[str, Any]) -> _Item[R]:
        if self._ds is None:
            self._ds = self._make_dataset()
        i = int(msg["i"])
        sample = self._ds[i]
        return _Item(i=i, value=self._worker_map(sample))


class OxidataIndexDataset(Generic[R]):
    """PyTorch-facing adapter using oxidata multiprocessing + shared-memory.

    Pattern (recommended): IO+decode in oxidata workers.

    - Main process publishes indices.
    - Worker reconstructs underlying dataset and performs __getitem__.
    - Worker returns (index, transformed_sample).

    Use with torch DataLoader(num_workers=0) since oxidata already manages worker procs.
    """

    def __init__(
        self,
        *,
        length: int,
        make_dataset: Callable[[], Any],
        worker_map: Callable[[Any], R],
        ring_name: str = "oxidata-torch-index-ring",
        shm_size: int = 64 * 1024 * 1024,
        slot_size: int = 4096,
        num_workers: int = 2,
        ctx: Optional[mp.context.BaseContext] = None,
        max_in_flight: int = 256,
    ):
        self._length = int(length)
        self._make_dataset = make_dataset
        self._worker_map = worker_map
        self._ring_name = str(ring_name)
        self._shm_size = int(shm_size)
        self._slot_size = int(slot_size)
        self._num_workers = int(num_workers)
        self._ctx = ctx or mp.get_context("spawn")
        self._max_in_flight = int(max_in_flight)

        self._producer = None
        self._pool = None
        self._q = None
        self._req = None

    def __len__(self) -> int:
        return self._length

    def _ensure_started(self) -> None:
        from oxidata.dataloader import Producer, BlobWorkerPool
        from oxidata.native import available as native_available

        if not native_available():
            raise RuntimeError("oxidata native extension not available")

        if self._producer is not None:
            return

        producer = Producer(ring_name=self._ring_name, shm_size=self._shm_size, slot_size=self._slot_size)
        worker = _IndexWorker(make_dataset=self._make_dataset, worker_map=self._worker_map)
        pool = BlobWorkerPool(
            shm_name=producer.shm_name,
            ring_name=producer.ring_name,
            codec="json",
            fn_obj=worker,
            num_workers=self._num_workers,
            ctx=self._ctx,
        )

        pool.start()

        self._producer = producer
        self._pool = pool
        self._q = pool.results()

    def close(self) -> None:
        if self._pool is None or self._producer is None:
            return

        try:
            self._pool.stop()
            self._pool.join(timeout=30)
        finally:
            self._producer.cleanup()
            self._pool = None
            self._producer = None
            self._q = None
            self._req = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, i: int) -> R:
        i = int(i)
        if i < 0 or i >= self._length:
            raise IndexError(i)

        self._ensure_started()
        assert self._producer is not None
        assert self._q is not None

        if self._req is None:
            from oxidata.dataloader import IndexRequestor

            self._req = IndexRequestor(
                publish_index=lambda idx: self._producer.publish_blob({"i": int(idx)}, codec="json"),
                q_out=self._q,
                max_in_flight=self._max_in_flight,
            )

        return self._req.get(i)


class _DemoBaseDataset:
    def __len__(self):
        return 1000

    def __getitem__(self, i: int):
        return {"i": i, "x": float(i) * 0.5}


def _make_demo_dataset() -> Any:
    return _DemoBaseDataset()


def _demo_worker_map(sample: dict[str, Any]):
    import torch  # type: ignore

    return torch.tensor([sample["i"], int(sample["x"] * 1000)], dtype=torch.int64)


def main():
    torch, Dataset = _require_torch()

    length = 1000

    class TorchAdapter(Dataset):
        def __init__(self):
            super().__init__()
            self._inner = OxidataIndexDataset(
                length=length,
                make_dataset=_make_demo_dataset,
                worker_map=_demo_worker_map,
                num_workers=4,
                max_in_flight=256,
                ring_name="oxidata-example-torch-index-ring",
            )

        def __len__(self) -> int:
            return len(self._inner)

        def __getitem__(self, idx: int):
            return self._inner[idx]

    ds = TorchAdapter()
    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0, shuffle=True)

    for batch in dl:
        print(batch.shape, batch[0])
        break


if __name__ == "__main__":
    main()
