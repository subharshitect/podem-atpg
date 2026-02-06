# Example Circuits

## toy1.bench
Simple AND feeding a NOT (NAND behavior).

Example faults:
- `n1/SA0` should be **DETECTED** with `a=1, b=1` (drives D through the NOT).
- `n1/SA1` should be **DETECTED** with `a=0, b=X` or `a=X, b=0` (drives DBAR).

## toy2.bench
Redundant logic: `out = OR(n1, a)` with `n1 = AND(a, b)`.

Example fault:
- `n1/SA0` should be **UNTESTABLE** because `a=1` masks observation at the output.

## toy3.bench
Two-level XOR parity.

Example faults:
- `n1/SA0` should be **DETECTED** with `a=1, b=0, c=0`.
- `n1/SA1` should be **DETECTED** with `a=0, b=0, c=0`.
