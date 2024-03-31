#! /usr/bin/env python


def bezout_10(n):
    """For n coprime with 10, returns x such that 10 x ≡ 1 (mod n)."""
    assert n > 0
    q, r = divmod(n + 5, 10)
    if r == 2:
        return -3 * q + 1
    if r == 4:
        return q
    if r == 6:
        return -q
    if r == 8:
        return 3 * q + 1
    return 0


def divisibility_by(n):
    if n <= 0:
        raise ValueError(f"expected a positive integer, but got {n}")
    x = bezout_10(n)
    if x:
        assert (10 * x) % n == 1
        assert pow(10, -1, n) == x % n
        return f"10a+b is divisible by {n} iff a{x:+}b is divisible by {n}"
    else:
        return f"{n} is not coprime with 10"


def main():
    import sys

    for line in sys.stdin:
        try:
            n = int(line, 0)
            print(divisibility_by(n))
        except ValueError as err:
            print(err, file=sys.stderr)


if __name__ == "__main__":
    main()
