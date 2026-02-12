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


def quantize_unit(x: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantize x in [-1,1] to signed fixed-point with 'bits' total bits.
    bits=0 disables quantization.
    """
    if bits <= 0:
        return x
    if bits < 2:
        raise ValueError("qbits must be 0 or >=2")
    q = (1 << (bits - 1)) - 1
    y = np.round(x * q) / q
    return np.clip(y, -1.0, 1.0)


def rl_current(v: np.ndarray, dt: float, r: float, l: float) -> np.ndarray:
    """
    Simple R-L load: di/dt = (v - r*i)/l
    Forward Euler, good enough for report-level intuition.
    """
    if l <= 0:
        raise ValueError("L must be > 0")
    i = np.zeros_like(v, dtype=np.float64)
    for k in range(1, len(v)):
        di = (v[k - 1] - r * i[k - 1]) / l
        i[k] = i[k - 1] + dt * di
    return i


def main():
    ap = argparse.ArgumentParser(description="Unipolar SPWM with V/f control, data dump, FFT-ready.")
    ap.add_argument("--crr", type=float, default=1.0, help="carrier amplitude (triangle peak)")
    ap.add_argument("--A", type=float, default=0.9, help="reference sine amplitude (used if --vf=0)")
    ap.add_argument("--fref", type=float, default=50.0, help="reference frequency (Hz)")
    ap.add_argument("--fcar", type=float, default=2000.0, help="carrier frequency (Hz)")
    ap.add_argument("--fs", type=float, default=200000.0, help="sample rate (Hz)")
    ap.add_argument("--phase", type=float, default=0.0, help="sine phase (rad)")
    ap.add_argument("--vdc", type=float, default=1.0, help="DC bus voltage (V)")
    ap.add_argument("--out", type=str, default="out/unipolar_spwm.png")
    ap.add_argument("--dump", type=str, default="", help="optional .npz dump path (stores all waveforms)")
    ap.add_argument("--tag", type=str, default="", help="optional title tag")

    # V/f control knobs
    ap.add_argument("--vf", type=int, default=0, help="1 enables V/f control, 0 uses --A directly")
    ap.add_argument("--fbase", type=float, default=50.0, help="base frequency for V/f (Hz)")
    ap.add_argument("--abase", type=float, default=0.9, help="A at base frequency for V/f")
    ap.add_argument("--amax", type=float, default=0.98, help="max allowed A (avoid overmodulation)")
    ap.add_argument("--vboost", type=float, default=0.0, help="low-f voltage boost added to A (0..)")
    ap.add_argument("--fboost", type=float, default=5.0, help="boost applies below this freq (Hz)")

    # Quantization
    ap.add_argument("--qbits", type=int, default=0, help="0 disables, else quantize vm and carrier to qbits")

    # Simple load model
    ap.add_argument("--load", type=int, default=0, help="1 enables simple R-L load current simulation")
    ap.add_argument("--r", type=float, default=1.0, help="R for load model (ohm, normalized)")
    ap.add_argument("--l", type=float, default=0.01, help="L for load model (H, normalized)")

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
        if args.vboost > 0:
            w = clamp(1.0 - (args.fref / max(args.fboost, 1e-9)), 0.0, 1.0)
            A_vf += args.vboost * w
        A_use = clamp(A_vf, 0.0, args.amax)
    else:
        A_use = args.A

    vf_ratio = A_use / max(args.fref, 1e-9)
    print(f"[info] vf={args.vf} fref={args.fref}Hz fcar={args.fcar}Hz A_use={A_use:.6f} V_over_f={vf_ratio:.9f} qbits={args.qbits}")

    # One fundamental cycle: 0..2π
    Tref = 1.0 / args.fref
    n = int(math.ceil(args.fs * Tref))
    t = np.arange(n, dtype=np.float64) / args.fs
    theta = 2.0 * np.pi * args.fref * t

    # Reference and carrier (pre-quant)
    vm = A_use * np.sin(theta + args.phase)
    vcr = triangle_wave(t, args.fcar, amp=args.crr)

    # Quantize in normalized domain [-1,1] then scale back
    if args.qbits > 0:
        vmn = quantize_unit(vm / max(A_use, 1e-12), args.qbits) * A_use
        vcrn = quantize_unit(vcr / max(args.crr, 1e-12), args.qbits) * args.crr
        vm, vcr = vmn, vcrn

    # Unipolar modulation: two legs
    vg1 = (vm >= vcr).astype(np.float64)
    vg3 = ((-vm) >= vcr).astype(np.float64)

    # Complementary gates (useful later for dead-time RTL discussion)
    vg2 = 1.0 - vg1
    vg4 = 1.0 - vg3

    vAN = args.vdc * vg1
    vBN = args.vdc * vg3
    vAB = vAN - vBN

    # Carrier-window average for envelope intuition
    win = max(1, int(round(args.fs / args.fcar)))
    vAB_avg = moving_average(vAB, win)

    # Load current (optional)
    i_load = None
    if args.load != 0:
        dt = 1.0 / args.fs
        i_load = rl_current(vAB, dt, r=args.r, l=args.l)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 5 if i_load is not None else 4
    fig, axes = plt.subplots(rows, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(theta, vm, linewidth=2, label=r"$v_m$")
    axes[0].plot(theta, vcr, linestyle="--", label=r"$v_{cr}$")
    axes[0].set_ylabel("amp")
    axes[0].legend(loc="upper right")

    title_tag = f" | {args.tag}" if args.tag else ""
    axes[0].set_title(
        f"Unipolar SPWM (one cycle){title_tag}\n"
        f"fref={args.fref} Hz, fcar={args.fcar} Hz, A={A_use:.3f} (vf={args.vf}), qbits={args.qbits}"
    )

    axes[1].step(theta, vg1, where="post", label=r"$v_{g1}$ (A-top)")
    axes[1].step(theta, vg3, where="post", label=r"$v_{g3}$ (B-top)")
    axes[1].step(theta, vg2, where="post", label=r"$v_{g2}$ (A-bot)", alpha=0.6)
    axes[1].step(theta, vg4, where="post", label=r"$v_{g4}$ (B-bot)", alpha=0.6)
    axes[1].set_ylim(-0.2, 1.2)
    axes[1].set_ylabel("gate")
    axes[1].legend(loc="upper right", ncols=2)

    axes[2].step(theta, vAN, where="post", label=r"$v_{AN}$")
    axes[2].step(theta, vBN, where="post", label=r"$v_{BN}$")
    axes[2].set_ylabel("V")
    axes[2].legend(loc="upper right")

    axes[3].step(theta, vAB, where="post", label=r"$v_{AB}$")
    axes[3].plot(theta, vAB_avg, linewidth=2, label=r"$\langle v_{AB}\rangle$")
    axes[3].set_ylabel("V")
    axes[3].legend(loc="upper right")

    if i_load is not None:
        axes[4].plot(theta, i_load, linewidth=2, label=r"$i_{load}$ (R-L)")
        axes[4].set_ylabel("A (norm)")
        axes[4].set_xlabel(r"$\theta = 2\pi f_{ref} t$ (rad)")
        axes[4].legend(loc="upper right")
    else:
        axes[3].set_xlabel(r"$\theta = 2\pi f_{ref} t$ (rad)")

    axes[-1].set_xlim(0.0, 2.0 * np.pi)
    axes[-1].set_xticks([0.0, np.pi, 2.0 * np.pi])
    axes[-1].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    # Optional dump for report scripts
    if args.dump:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            fs=args.fs,
            fref=args.fref,
            fcar=args.fcar,
            A_use=A_use,
            vf=args.vf,
            fbase=args.fbase,
            abase=args.abase,
            amax=args.amax,
            vboost=args.vboost,
            fboost=args.fboost,
            qbits=args.qbits,
            vdc=args.vdc,
            crr=args.crr,
            t=t,
            theta=theta,
            vm=vm,
            vcr=vcr,
            vg1=vg1,
            vg2=vg2,
            vg3=vg3,
            vg4=vg4,
            vAN=vAN,
            vBN=vBN,
            vAB=vAB,
            vAB_avg=vAB_avg,
            i_load=i_load if i_load is not None else np.array([]),
            r=args.r,
            l=args.l,
        )
        print(f"[info] dump={dump_path}")


if __name__ == "__main__":
    main()
