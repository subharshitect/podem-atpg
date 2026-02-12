# report_analysis.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def fft_spectrum(x: np.ndarray, fs: float):
    """
    Return freq axis and magnitude spectrum (single-sided) for x.
    Uses Hann window.
    """
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    mag = np.abs(X) / (np.sum(w) / 2.0)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return f, mag


def fundamental_metrics(vab: np.ndarray, fs: float, f1: float, max_harm: int = 40):
    """
    THD proxy: sqrt(sum_{k=2..K} Vk^2) / V1
    by sampling the FFT magnitude at harmonic bins nearest to k*f1.
    """
    f, mag = fft_spectrum(vab, fs)
    def pick(freq):
        idx = int(np.argmin(np.abs(f - freq)))
        return mag[idx]

    v1 = pick(f1)
    harm = []
    for k in range(2, max_harm + 1):
        harm.append(pick(k * f1))
    harm = np.array(harm)
    thd = float(np.sqrt(np.sum(harm ** 2)) / max(v1, 1e-12))
    return float(v1), thd, f, mag


def edge_count(sig01: np.ndarray):
    """
    Switching activity proxy: count 0->1 and 1->0 transitions.
    """
    s = (sig01 > 0.5).astype(np.int8)
    return int(np.sum(np.abs(np.diff(s))))


def main():
    ap = argparse.ArgumentParser(description="Generate research plots from SPWM .npz dumps.")
    ap.add_argument("--inp", type=str, required=True, help="input .npz dump file")
    ap.add_argument("--outdir", type=str, default="out/report/analysis", help="output directory")
    ap.add_argument("--max_harm", type=int, default=40)
    ap.add_argument("--spec_max_hz", type=float, default=0.0, help="0 means auto, else limit spectrum x-axis")
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    fs = float(d["fs"])
    fref = float(d["fref"])
    vab = d["vAB"].astype(np.float64)
    vg1 = d["vg1"].astype(np.float64)
    vg3 = d["vg3"].astype(np.float64)
    A_use = float(d["A_use"])
    qbits = int(d["qbits"])

    v1, thd, f, mag = fundamental_metrics(vab, fs, fref, max_harm=args.max_harm)
    sw = edge_count(vg1) + edge_count(vg3)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Spectrum plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f, mag)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|VAB(f)| (arb)")
    ttl = f"Spectrum of vAB | fref={fref}Hz A={A_use:.3f} qbits={qbits}  V1={v1:.3g} THD~{thd:.3g}  edges={sw}"
    ax.set_title(ttl)
    if args.spec_max_hz > 0:
        ax.set_xlim(0, args.spec_max_hz)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    spec_path = outdir / f"spectrum_f{fref:g}_q{qbits}.png"
    fig.savefig(spec_path, dpi=180)
    plt.close(fig)

    # Save metrics text
    met_path = outdir / f"metrics_f{fref:g}_q{qbits}.txt"
    met_path.write_text(
        f"fref={fref}\nA_use={A_use}\nV1_proxy={v1}\nTHD_proxy={thd}\nswitch_edges_proxy={sw}\n"
    )

    print(f"Wrote: {spec_path}")
    print(f"Wrote: {met_path}")


if __name__ == "__main__":
    main()
