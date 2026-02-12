# vf_curve.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def vf_amp(fref: float, *, vf: int, fbase: float, abase: float, amax: float, vboost: float, fboost: float) -> float:
    if vf == 0:
        return abase
    A = abase * (fref / fbase)
    if vboost > 0:
        w = clamp(1.0 - (fref / max(fboost, 1e-9)), 0.0, 1.0)
        A += vboost * w
    return clamp(A, 0.0, amax)


def main():
    ap = argparse.ArgumentParser(description="Plot V/f amplitude curve A(f).")
    ap.add_argument("--freqs", type=str, default="2 5 10 15 25 50")
    ap.add_argument("--vf", type=int, default=1)
    ap.add_argument("--fbase", type=float, default=50.0)
    ap.add_argument("--abase", type=float, default=0.9)
    ap.add_argument("--amax", type=float, default=0.98)
    ap.add_argument("--vboost", type=float, default=0.0)
    ap.add_argument("--fboost", type=float, default=5.0)
    ap.add_argument("--out", type=str, default="out/report/vf_curve.png")
    args = ap.parse_args()

    freqs = [float(x) for x in args.freqs.split()]
    Avals = [
        vf_amp(
            f,
            vf=args.vf,
            fbase=args.fbase,
            abase=args.abase,
            amax=args.amax,
            vboost=args.vboost,
            fboost=args.fboost,
        )
        for f in freqs
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, Avals, marker="o")
    ax.set_xlabel("Command frequency fref (Hz)")
    ax.set_ylabel("Modulation amplitude A")
    ax.set_title(f"V/f scheduling curve (vf={args.vf})  fbase={args.fbase}Hz abase={args.abase} amax={args.amax}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
