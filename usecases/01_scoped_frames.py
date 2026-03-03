from oxidata.scopes import Frame


def main():
    with Frame(size=8 * 1024 * 1024) as f:
        counter = f.var(0)
        with counter.borrow_mut() as b:
            b.set(b.get() + 1)

        with counter.borrow() as b:
            v = b.get()
        print("counter in frame:", v)

    try:
        with counter.borrow() as b:  # type: ignore[name-defined]
            _ = b.get()
    except Exception as e:
        print("expected use-after-free:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
