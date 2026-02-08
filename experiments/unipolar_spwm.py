# unipolar_spwm.py
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def triangle_wave(t: np.ndarray, f: float, amp: float = 1.0) -> np.ndarray:
    p = (t * f) % 1.0
    tri = 4.0 * np.abs(p - 0.5) - 1.0
    return amp * tri


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main():
    ap = argparse.ArgumentParser(description="Minimal unipolar SPWM with optional V/f control.")
    ap.add_argument("--crr", type=float, default=1.0, help="carrier amplitude (triangle peak)")
    ap.add_argument("--A", type=float, default=0.9, help="reference sine amplitude (used if --vf=0)")
    ap.add_argument("--fref", type=float, default=50.0, help="reference frequency (Hz)")
    ap.add_argument("--fcar", type=float, default=2000.0, help="carrier frequency (Hz)")
    ap.add_argument("--fs", type=float, default=200000.0, help="sample rate (Hz)")
    ap.add_argument("--phase", type=float, default=0.0, help="sine phase (rad)")
    ap.add_argument("--vdc", type=float, default=1.0, help="DC bus voltage (V)")
    ap.add_argument("--out", type=str, default="out/unipolar_spwm.png")

    # V/f control knobs
    ap.add_argument("--vf", type=int, default=0, help="1 enables V/f control, 0 uses --A directly")
    ap.add_argument("--fbase", type=float, default=50.0, help="base frequency for V/f (Hz)")
    ap.add_argument("--abase", type=float, default=0.9, help="A at base frequency for V/f")
    ap.add_argument("--amax", type=float, default=0.98, help="max allowed A (avoid overmodulation)")
    ap.add_argument("--vboost", type=float, default=0.0, help="low-f voltage boost added to A (0..)")
    ap.add_argument("--fboost", type=float, default=5.0, help="boost applies below this freq (Hz)")

    args = ap.parse_args()

    if args.fs <= 0 or args.fref <= 0 or args.fcar <= 0:
        raise ValueError("fs, fref, fcar must be > 0")
    if args.fbase <= 0:
        raise ValueError("fbase must be > 0")
    if args.crr <= 0:
        raise ValueError("crr must be > 0")

    # V/f amplitude selection
    if args.vf != 0:
        A_vf = args.abase * (args.fref / args.fbase)

        # Optional low-frequency boost (common in motor V/f drives)
        # Add boost only at low freq; taper to 0 by fboost.
        if args.vboost > 0:
            w = clamp(1.0 - (args.fref / max(args.fboost, 1e-9)), 0.0, 1.0)
            A_vf += args.vboost * w

        A_use = clamp(A_vf, 0.0, args.amax)
    else:
        A_use = args.A

    # One fundamental cycle: 0..2π
    Tref = 1.0 / args.fref
    n = int(math.ceil(args.fs * Tref))
    t = np.arange(n, dtype=np.float64) / args.fs
    theta = 2.0 * np.pi * args.fref * t

    # Reference and carrier
    vm = A_use * np.sin(theta + args.phase)
    vcr = triangle_wave(t, args.fcar, amp=args.crr)

    # Unipolar modulation: two legs
    vg1 = (vm >= vcr).astype(np.float64)
    vg3 = ((-vm) >= vcr).astype(np.float64)

    vAN = args.vdc * vg1
    vBN = args.vdc * vg3
    vAB = vAN - vBN

    # Carrier-window average for envelope intuition
    win = max(1, int(round(args.fs / args.fcar)))
    vAB_avg = moving_average(vAB, win)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(theta, vm, linewidth=2, label=r"$v_m$")
    axes[0].plot(theta, vcr, linestyle="--", label=r"$v_{cr}$")
    axes[0].set_ylabel("amp")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"Unipolar SPWM (one cycle) | fref={args.fref} Hz, fcar={args.fcar} Hz, "
        f"A={A_use:.3f} (vf={args.vf})"
    )

    axes[1].step(theta, vg1, where="post", label=r"$v_{g1}$")
    axes[1].step(theta, vg3, where="post", label=r"$v_{g3}$")
    axes[1].set_ylim(-0.2, 1.2)
    axes[1].set_ylabel("gate")
    axes[1].legend(loc="upper right")

    axes[2].step(theta, vAN, where="post", label=r"$v_{AN}$")
    axes[2].step(theta, vBN, where="post", label=r"$v_{BN}$")
    axes[2].set_ylabel("V")
    axes[2].legend(loc="upper right")

    axes[3].step(theta, vAB, where="post", label=r"$v_{AB}$")
    axes[3].plot(theta, vAB_avg, linewidth=2, label=r"$\langle v_{AB}\rangle$")
    axes[3].set_ylabel("V")
    axes[3].set_xlabel(r"$\theta = 2\pi f_{ref} t$ (rad)")
    axes[3].legend(loc="upper right")

    axes[3].set_xlim(0.0, 2.0 * np.pi)
    axes[3].set_xticks([0.0, np.pi, 2.0 * np.pi])
    axes[3].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
