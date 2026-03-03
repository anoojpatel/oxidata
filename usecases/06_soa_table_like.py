from oxidata.scopes import Frame
from oxidata.soa import SoASchema


def main():
    try:
        import numpy as np  # type: ignore
    except Exception:
        raise SystemExit("numpy not installed")

    with Frame(size=64 * 1024 * 1024) as f:
        schema = SoASchema.from_mapping({"x": "float32", "y": "int64"})
        h = schema.alloc(f.arena, n=1024)
        batch = schema.open(h)

        batch.x[:] = np.arange(1024, dtype=np.float32)
        batch.y[:] = np.arange(1024, dtype=np.int64)

        print("x[0],y[0]", batch.x[0], batch.y[0])
        print("x.sum", float(batch.x.sum()))


if __name__ == "__main__":
    main()
