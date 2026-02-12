# deadtime_plot.py
import argparse
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def insert_deadtime(g_top: np.ndarray, dt_samples: int):
    """
    Given top gate g_top (0/1), create bottom gate as NOT(top) then apply dead-time:
    when top turns off, delay bottom turn-on by dt_samples
    when bottom turns off, delay top turn-on by dt_samples
    Returns: g_top_dt, g_bot_dt
    """
    g = (g_top > 0.5).astype(np.int8)
    gbar = (1 - g).astype(np.int8)

    top_dt = g.copy()
    bot_dt = gbar.copy()

    if dt_samples <= 0:
        return top_dt, bot_dt

    # enforce non-overlap: if either tries to turn on too early, hold it low
    # simple rule: detect rising edges and delay them by dt_samples
    def delay_rise(x):
        y = x.copy()
        rises = np.where((x[1:] == 1) & (x[:-1] == 0))[0] + 1
        for r in rises:
            y[r : min(len(y), r + dt_samples)] = 0
        return y

    top_dt = delay_rise(top_dt)
    bot_dt = delay_rise(bot_dt)

    # final safety: never allow both high
    both = (top_dt == 1) & (bot_dt == 1)
    bot_dt[both] = 0
    return top_dt, bot_dt


def main():
    ap = argparse.ArgumentParser(description="Dead-time timing plot from run_dump.npz")
    ap.add_argument("--inp", type=str, default="out/report/run_dump.npz")
    ap.add_argument("--dt", type=int, default=20, help="dead-time in samples")
    ap.add_argument("--out", type=str, default="out/report/analysis/deadtime_timing.png")
    ap.add_argument("--start", type=int, default=0, help="start sample for zoom")
    ap.add_argument("--span", type=int, default=2000, help="number of samples to show")
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    fs = float(d["fs"])
    fref = float(d["fref"])
    vg1 = d["vg1"].astype(np.float64)  # leg A top
    vg3 = d["vg3"].astype(np.float64)  # leg B top

    s = max(0, int(args.start))
    e = min(len(vg1), s + int(args.span))

    a_top = (vg1[s:e] > 0.5).astype(np.int8)
    b_top = (vg3[s:e] > 0.5).astype(np.int8)

    a_top_dt, a_bot_dt = insert_deadtime(a_top, args.dt)
    b_top_dt, b_bot_dt = insert_deadtime(b_top, args.dt)

    t = np.arange(e - s) / fs

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    # stack signals with offsets
    ax.step(t, a_top + 6, where="post", label="Leg A top (raw)")
    ax.step(t, (1 - a_top) + 5, where="post", label="Leg A bot (raw)")
    ax.step(t, a_top_dt + 4, where="post", label="Leg A top (dead-time)")
    ax.step(t, a_bot_dt + 3, where="post", label="Leg A bot (dead-time)")

    ax.step(t, b_top + 2, where="post", label="Leg B top (raw)")
    ax.step(t, (1 - b_top) + 1, where="post", label="Leg B bot (raw)")
    ax.step(t, b_top_dt + 0, where="post", label="Leg B top (dead-time)")
    ax.step(t, b_bot_dt - 1, where="post", label="Leg B bot (dead-time)")

    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Dead-time insertion timing (dt={args.dt} samples) | fs={fs:g} Hz | fref={fref:g} Hz")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", ncols=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)

if __name__ == "__main__":
    main()