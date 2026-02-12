# rtl_spwm_amaranth.py
# Minimal RTL-ish unipolar SPWM in Amaranth + Python simulation that saves a single PNG.

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from amaranth import Elaboratable, Module, Signal, Mux
from amaranth.sim import Simulator


def gen_sine_lut(n: int, bits: int) -> np.ndarray:
    """
    Signed sine LUT in Q( bits-1 ) format, values in [-2^(bits-1), +2^(bits-1)-1]
    """
    assert n > 0 and bits >= 2
    peak = (1 << (bits - 1)) - 1
    xs = np.sin(2.0 * np.pi * np.arange(n) / n)
    q = np.round(xs * peak).astype(np.int64)
    q = np.clip(q, -peak - 1, peak)
    return q


class UnipolarSPWMCore(Elaboratable):
    """
    Unipolar SPWM core:
    - Phase accumulator + sine LUT (vm)
    - Up-down counter triangle carrier (vcr)
    - Comparators to produce vg1, vg3
    - Outputs vAN, vBN, vAB as signed integer levels
    """

    def __init__(self, *, lut, phase_bits: int, car_max: int, amp_max: int, vdc: int):
        self.lut = lut
        self.lut_len = int(len(lut))
        self.phase_bits = int(phase_bits)

        self.car_max = int(car_max)      # carrier counter peak (triangle spans [-car_max, +car_max])
        self.amp_max = int(amp_max)      # amplitude scaling range (0..amp_max)
        self.vdc = int(vdc)              # vdc output in integer units

        # Inputs
        self.phase_inc = Signal(self.phase_bits)         # frequency command as phase increment
        self.amp = Signal(range(self.amp_max + 1))       # V/f amplitude command (0..amp_max)

        # Outputs
        self.theta_idx = Signal(range(self.lut_len))     # LUT index for plotting
        self.vm = Signal(signed(16))                     # signed modulating value (scaled LUT)
        self.vcr = Signal(signed(16))                    # signed carrier
        self.vg1 = Signal()
        self.vg3 = Signal()
        self.vAN = Signal(range(self.vdc + 1))
        self.vBN = Signal(range(self.vdc + 1))
        self.vAB = Signal(signed(16))

    def elaborate(self, platform):
        m = Module()

        # Phase accumulator
        phase = Signal(self.phase_bits)
        m.d.sync += phase.eq(phase + self.phase_inc)

        # LUT index from MSBs
        idx_bits = int(math.ceil(math.log2(self.lut_len)))
        shift = self.phase_bits - idx_bits
        idx = Signal(idx_bits)
        m.d.comb += idx.eq(phase >> shift)

        # Map idx into actual LUT range if lut_len not power of 2
        # For simplicity: modulo via compare, but easiest is power-of-2 LUT length.
        # We will assume lut_len is power-of-2 for now.
        m.d.comb += self.theta_idx.eq(idx)

        # Carrier up-down counter
        car = Signal(signed(16), reset=0)
        dirn = Signal(reset=1)  # 1 = up, 0 = down

        with m.If(dirn):
            m.d.sync += car.eq(car + 1)
            with m.If(car >= self.car_max - 1):
                m.d.sync += dirn.eq(0)
        with m.Else():
            m.d.sync += car.eq(car - 1)
            with m.If(car <= -self.car_max + 1):
                m.d.sync += dirn.eq(1)

        # Sine LUT read (combinational via Python list baked into a switch)
        # Minimal and synthesis-friendly for small LUTs.
        lut_val = Signal(signed(16))
        with m.Switch(idx):
            for i, v in enumerate(self.lut):
                with m.Case(i):
                    m.d.comb += lut_val.eq(int(v))

        # Scale sine by amplitude command: vm = lut_val * amp / amp_max
        # Use integer math: (lut_val * amp) // amp_max
        vm_scaled = Signal(signed(16))
        prod = Signal(signed(32))
        m.d.comb += prod.eq(lut_val.as_signed() * self.amp)
        m.d.comb += vm_scaled.eq(prod // self.amp_max)

        # Comparators for unipolar SPWM legs
        vg1 = Signal()
        vg3 = Signal()
        m.d.comb += vg1.eq(vm_scaled >= car)
        m.d.comb += vg3.eq((-vm_scaled) >= car)

        # Pole voltages (0 or vdc)
        vAN = Signal(range(self.vdc + 1))
        vBN = Signal(range(self.vdc + 1))
        m.d.comb += vAN.eq(Mux(vg1, self.vdc, 0))
        m.d.comb += vBN.eq(Mux(vg3, self.vdc, 0))

        # vAB = vAN - vBN
        vAB = Signal(signed(16))
        m.d.comb += vAB.eq(vAN.as_signed() - vBN.as_signed())

        # Export
        m.d.comb += [
            self.vm.eq(vm_scaled),
            self.vcr.eq(car),
            self.vg1.eq(vg1),
            self.vg3.eq(vg3),
            self.vAN.eq(vAN),
            self.vBN.eq(vBN),
            self.vAB.eq(vAB),
        ]

        return m


def vf_amp_command(fref: float, *, vf: int, fbase: float, abase: float, amax: float, vboost: float, fboost: float) -> float:
    if vf == 0:
        return abase
    if fbase <= 0:
        raise ValueError("fbase must be > 0")

    A = abase * (fref / fbase)

    if vboost > 0:
        # linear taper to 0 by fboost
        w = max(0.0, min(1.0, 1.0 - (fref / max(fboost, 1e-9))))
        A += vboost * w

    A = max(0.0, min(amax, A))
    return A


def main():
    ap = argparse.ArgumentParser(description="RTL-ish unipolar SPWM using Amaranth, dumps a single PNG with subplots.")
    ap.add_argument("--fs", type=float, default=200000.0)
    ap.add_argument("--fref", type=float, default=50.0)
    ap.add_argument("--fcar", type=float, default=2000.0)

    # V/f knobs
    ap.add_argument("--vf", type=int, default=1)
    ap.add_argument("--fbase", type=float, default=50.0)
    ap.add_argument("--abase", type=float, default=0.9)
    ap.add_argument("--amax", type=float, default=0.98)
    ap.add_argument("--vboost", type=float, default=0.0)
    ap.add_argument("--fboost", type=float, default=5.0)

    # RTL params
    ap.add_argument("--lut", type=int, default=256, help="sine LUT size (power of 2 recommended)")
    ap.add_argument("--lut_bits", type=int, default=14, help="sine LUT signed bits")
    ap.add_argument("--phase_bits", type=int, default=24, help="phase accumulator bits")
    ap.add_argument("--vdc", type=int, default=1000, help="Vdc in integer units for plotting")
    ap.add_argument("--out", type=str, default="out/rtl_pwm.png")
    args = ap.parse_args()

    if args.fs <= 0 or args.fref <= 0 or args.fcar <= 0:
        raise ValueError("fs, fref, fcar must be > 0")

    # Build LUT
    lut = gen_sine_lut(args.lut, args.lut_bits)

    # Carrier counter peak so that triangle freq matches fcar:
    # Triangle period in cycles = 4*car_max (up and down across full range)
    # fcar = fs / (4*car_max)  => car_max = fs / (4*fcar)
    car_max = int(round(args.fs / (4.0 * args.fcar)))
    car_max = max(2, car_max)

    # V/f amplitude as a float in [0,1], then map to integer amp_cmd
    A = vf_amp_command(
        args.fref,
        vf=args.vf,
        fbase=args.fbase,
        abase=args.abase,
        amax=args.amax,
        vboost=args.vboost,
        fboost=args.fboost,
    )
    amp_max = 1024
    amp_cmd = int(round(A * amp_max))
    amp_cmd = max(0, min(amp_max, amp_cmd))

    # Phase increment for desired fref:
    # fref = (phase_inc / 2^phase_bits) * fs  => phase_inc = fref * 2^phase_bits / fs
    phase_inc = int(round(args.fref * (1 << args.phase_bits) / args.fs))
    phase_inc = max(1, min((1 << args.phase_bits) - 1, phase_inc))

    core = UnipolarSPWMCore(lut=lut, phase_bits=args.phase_bits, car_max=car_max, amp_max=amp_max, vdc=args.vdc)

    sim = Simulator(core)
    sim.add_clock(1.0 / args.fs)

    # Simulate one fundamental cycle
    Tref = 1.0 / args.fref
    n = int(math.ceil(args.fs * Tref))

    # Trace buffers
    theta = np.zeros(n, dtype=np.float64)
    vm = np.zeros(n, dtype=np.float64)
    vcr = np.zeros(n, dtype=np.float64)
    vg1 = np.zeros(n, dtype=np.float64)
    vg3 = np.zeros(n, dtype=np.float64)
    vAN = np.zeros(n, dtype=np.float64)
    vBN = np.zeros(n, dtype=np.float64)
    vAB = np.zeros(n, dtype=np.float64)

    def proc():
        # program inputs
        yield core.phase_inc.eq(phase_inc)
        yield core.amp.eq(amp_cmd)

        for i in range(n):
            # theta for x-axis (0..2π)
            theta[i] = 2.0 * np.pi * (i / n)

            vm[i] = (yield core.vm)
            vcr[i] = (yield core.vcr)
            vg1[i] = (yield core.vg1)
            vg3[i] = (yield core.vg3)
            vAN[i] = (yield core.vAN)
            vBN[i] = (yield core.vBN)
            vAB[i] = (yield core.vAB)

            yield  # next clock

    sim.add_sync_process(proc)
    sim.run()

    # Normalize for nicer comparison in subplot 1
    vm_n = vm / ((1 << (args.lut_bits - 1)) - 1 + 1e-9)
    vcr_n = vcr / (car_max + 1e-9)

    # Simple carrier-window average of vAB (purely for visualization)
    win = max(1, int(round(args.fs / args.fcar)))
    k = np.ones(win) / float(win)
    vAB_avg = np.convolve(vAB, k, mode="same")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(theta, vm_n, linewidth=2, label="vm (norm)")
    axes[0].plot(theta, vcr_n, linestyle="--", label="vcr (norm)")
    axes[0].set_ylabel("norm")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"RTL-ish Unipolar SPWM (Amaranth) | fref={args.fref}Hz fcar={args.fcar}Hz A={A:.3f} (vf={args.vf})")

    axes[1].step(theta, vg1, where="post", label="vg1")
    axes[1].step(theta, vg3, where="post", label="vg3")
    axes[1].set_ylim(-0.2, 1.2)
    axes[1].set_ylabel("gate")
    axes[1].legend(loc="upper right")

    axes[2].step(theta, vAN, where="post", label="vAN")
    axes[2].step(theta, vBN, where="post", label="vBN")
    axes[2].set_ylabel("V (int)")
    axes[2].legend(loc="upper right")

    axes[3].step(theta, vAB, where="post", label="vAB")
    axes[3].plot(theta, vAB_avg, linewidth=2, label="avg(vAB)")
    axes[3].set_ylabel("V (int)")
    axes[3].set_xlabel("theta (rad)")
    axes[3].legend(loc="upper right")

    axes[3].set_xlim(0.0, 2.0 * np.pi)
    axes[3].set_xticks([0.0, np.pi, 2.0 * np.pi])
    axes[3].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    # Amaranth needs signed() helper import, keep it here to avoid polluting top
    from amaranth import signed
    main()
