# fourier_series_plot.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def fourier_coeffs(x: np.ndarray, K: int):
    """
    Compute Fourier series coefficients for a 2π-periodic signal x(theta),
    sampled uniformly over theta in [0, 2π).
    Returns: a0, a[1..K], b[1..K]
    """
    n = len(x)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dtheta = 2.0 * np.pi / n

    a0 = (1.0 / np.pi) * np.sum(x) * dtheta

    a = np.zeros(K + 1, dtype=np.float64)
    b = np.zeros(K + 1, dtype=np.float64)

    for k in range(1, K + 1):
        a[k] = (1.0 / np.pi) * np.sum(x * np.cos(k * theta)) * dtheta
        b[k] = (1.0 / np.pi) * np.sum(x * np.sin(k * theta)) * dtheta

    return a0, a, b, theta


def recon_from_coeffs(a0: float, a: np.ndarray, b: np.ndarray, theta: np.ndarray, K: int):
    y = 0.5 * a0 * np.ones_like(theta)
    for k in range(1, K + 1):
        y += a[k] * np.cos(k * theta) + b[k] * np.sin(k * theta)
    return y


def main():
    ap = argparse.ArgumentParser(description="Fourier series reconstruction plots from run_dump.npz")
    ap.add_argument("--inp", type=str, default="out/report/run_dump.npz")
    ap.add_argument("--K", type=int, default=25, help="number of harmonics in reconstruction")
    ap.add_argument("--outdir", type=str, default="out/report/analysis")
    ap.add_argument("--zoom", type=float, default=2.0 * np.pi, help="theta window to show (rad)")
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    vAB = d["vAB"].astype(np.float64)
    fref = float(d["fref"])
    A_use = float(d["A_use"])
    qbits = int(d.get("qbits", 0))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a0, a, b, theta = fourier_coeffs(vAB, args.K)
    yK = recon_from_coeffs(a0, a, b, theta, args.K)

    # fundamental component only
    y1 = a[1] * np.cos(theta) + b[1] * np.sin(theta)

    # Reconstruction overlay plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(theta, vAB, linewidth=1.0, label=r"$v_{AB}(\theta)$ (original)")
    ax.plot(theta, yK, linewidth=2.0, label=rf"Fourier recon (K={args.K})")
    ax.plot(theta, y1, linewidth=2.0, label=r"fundamental (k=1)")

    ax.set_xlim(0.0, float(args.zoom))
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("V")
    ax.set_title(f"Fourier series reconstruction of vAB | fref={fref:g} Hz  A={A_use:.3f}  qbits={qbits}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()

    p1 = outdir / f"fourier_recon_f{fref:g}_K{args.K}_q{qbits}.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)

    # Harmonic magnitude bar plot (use amplitude sqrt(a^2 + b^2))
    mags = np.sqrt(a[1:] ** 2 + b[1:] ** 2)
    ks = np.arange(1, args.K + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(ks, mags)
    ax.set_xlabel("Harmonic index k")
    ax.set_ylabel(r"$\sqrt{a_k^2 + b_k^2}$")
    ax.set_title(f"Harmonic magnitudes from Fourier series | up to K={args.K} | fref={fref:g} Hz")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    p2 = outdir / f"fourier_harmonics_f{fref:g}_K{args.K}_q{qbits}.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)

    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
