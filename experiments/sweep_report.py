# sweep_report.py
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from report_analysis import fundamental_metrics, edge_count


def run_one(out_npz: Path, out_png: Path, args, fref: float, fcar: float):
    cmd = [
        args.py, "unipolar_spwm.py",
        "--crr", str(args.crr),
        "--fref", str(fref),
        "--fcar", str(fcar),
        "--fs", str(args.fs),
        "--phase", str(args.phase),
        "--vdc", str(args.vdc),
        "--vf", str(args.vf),
        "--fbase", str(args.fbase),
        "--abase", str(args.abase),
        "--amax", str(args.amax),
        "--vboost", str(args.vboost),
        "--fboost", str(args.fboost),
        "--A", str(args.A),
        "--qbits", str(args.qbits),
        "--load", str(args.load),
        "--r", str(args.r),
        "--l", str(args.l),
        "--out", str(out_png),
        "--dump", str(out_npz),
        "--tag", f"sweep fref={fref:g}Hz",
    ]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Sweep fref and generate research plots (V1, THD, switching).")
    ap.add_argument("--py", type=str, default="python3")
    ap.add_argument("--outdir", type=str, default="out/report/sweep2")
    ap.add_argument("--freqs", type=str, default="2 5 10 15 25 50")
    ap.add_argument("--mr", type=float, default=20.0, help="carrier ratio: fcar = mr * fref")

    # forward knobs to unipolar_spwm
    ap.add_argument("--fs", type=float, default=200000.0)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--crr", type=float, default=1.0)
    ap.add_argument("--vdc", type=float, default=1.0)

    ap.add_argument("--vf", type=int, default=1)
    ap.add_argument("--fbase", type=float, default=50.0)
    ap.add_argument("--abase", type=float, default=0.9)
    ap.add_argument("--amax", type=float, default=0.98)
    ap.add_argument("--vboost", type=float, default=0.0)
    ap.add_argument("--fboost", type=float, default=5.0)
    ap.add_argument("--A", type=float, default=0.9)

    ap.add_argument("--qbits", type=int, default=0)
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--l", type=float, default=0.01)

    ap.add_argument("--max_harm", type=int, default=40)
    args = ap.parse_args()

    freqs = [float(x) for x in args.freqs.split()]
    outdir = Path(args.outdir)
    dumps = outdir / "dumps"
    waves = outdir / "waves"
    outdir.mkdir(parents=True, exist_ok=True)
    dumps.mkdir(parents=True, exist_ok=True)
    waves.mkdir(parents=True, exist_ok=True)

    rows = []
    for fref in freqs:
        fcar = args.mr * fref
        npz = dumps / f"run_f{fref:g}_q{args.qbits}.npz"
        png = waves / f"wave_f{fref:g}_q{args.qbits}.png"
        run_one(npz, png, args, fref=fref, fcar=fcar)

        d = np.load(npz, allow_pickle=True)
        fs = float(d["fs"])
        vab = d["vAB"].astype(np.float64)
        vg1 = d["vg1"].astype(np.float64)
        vg3 = d["vg3"].astype(np.float64)
        A_use = float(d["A_use"])

        v1, thd, _, _ = fundamental_metrics(vab, fs, fref, max_harm=args.max_harm)
        sw = edge_count(vg1) + edge_count(vg3)
        rows.append((fref, fcar, A_use, v1, thd, sw))

    rows = np.array(rows, dtype=np.float64)
    fref = rows[:, 0]
    A_use = rows[:, 2]
    v1 = rows[:, 3]
    thd = rows[:, 4]
    sw = rows[:, 5]

    # Plot 1: A_use vs fref (actual V/f schedule)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fref, A_use, marker="o")
    ax.set_xlabel("fref (Hz)")
    ax.set_ylabel("A_use")
    ax.set_title(f"V/f schedule across sweep (vf={args.vf}, qbits={args.qbits})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    p = outdir / "sweep_vf_curve.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)

    # Plot 2: fundamental amplitude proxy vs fref
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fref, v1, marker="o")
    ax.set_xlabel("fref (Hz)")
    ax.set_ylabel("V1 proxy (arb)")
    ax.set_title("Fundamental amplitude proxy vs frequency")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    p = outdir / "sweep_fundamental.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)

    # Plot 3: THD proxy vs fref
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fref, thd, marker="o")
    ax.set_xlabel("fref (Hz)")
    ax.set_ylabel("THD proxy")
    ax.set_title(f"THD proxy vs frequency (max_harm={args.max_harm})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    p = outdir / "sweep_thd.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)

    # Plot 4: switching activity proxy vs fref
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fref, sw, marker="o")
    ax.set_xlabel("fref (Hz)")
    ax.set_ylabel("edge count proxy")
    ax.set_title("Switching activity proxy vs frequency (vg1+vg3 edges)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    p = outdir / "sweep_switching.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)

    # Save CSV
    csv_path = outdir / "sweep_metrics.csv"
    header = "fref,fcar,A_use,V1_proxy,THD_proxy,switch_edges_proxy\n"
    lines = [header]
    for r in rows:
        lines.append(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{int(r[5])}\n")
    csv_path.write_text("".join(lines))

    print(f"Wrote sweep plots + CSV to: {outdir}")


if __name__ == "__main__":
    main()
