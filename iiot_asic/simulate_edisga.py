# FILE: simulate_edisga.py
# Event Driven Industrial Sensor Gateway ASIC (EDISGA)
# Single-file, deterministic, RTL-style simulation + artifact generation.
#
# Contract:
# - Makefile creates a unique outdir with subfolders: figures/, tables/, logs/, waveforms/
# - This script MUST NOT reuse prior outputs. It refuses to proceed if any required subdir is non-empty.
# - All artifacts are written only under outdir.
#
# Run:
#   python3 simulate_edisga.py --outdir build_EDISGA_... --seed 7 --duration_s 240
#
# Outputs (under outdir):
#   config.json
#   waveforms/waveforms.csv
#   waveforms/trends.csv
#   logs/historian_records.csv
#   logs/frames.csv
#   logs/packets.bin
#   figures/*.png
#   tables/*.csv  (generic tabular artifacts)
#   tables/*.json (structured summaries)
#   tables/*.txt  (human-readable notes)

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Utilities
# ----------------------------

def die(msg: str) -> None:
    raise RuntimeError(msg)


def ensure_dir_exists(path: str) -> None:
    if not os.path.isdir(path):
        die(f"Expected directory does not exist: {path}")


def ensure_clean_sandbox(outdir: str) -> Dict[str, str]:
    # Makefile creates outdir and subfolders. We accept existing outdir only if required subdirs are empty.
    ensure_dir_exists(outdir)
    required = ["figures", "tables", "logs", "waveforms"]
    paths: Dict[str, str] = {}
    for sub in required:
        p = os.path.join(outdir, sub)
        if os.path.exists(p):
            if not os.path.isdir(p):
                die(f"Expected directory but found file: {p}")
            if os.listdir(p):
                die(f"Refusing to write into non-empty directory: {p}")
        else:
            os.makedirs(p, exist_ok=False)
        paths[sub] = p
    return paths


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def q88_from_float(x: float) -> int:
    return int(round(x * 256.0))


def q88_to_float(x: int) -> float:
    return x / 256.0


def write_table_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_kv_txt(path: str, title: str, items: List[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(title.strip() + "\n")
        f.write("=" * len(title.strip()) + "\n\n")
        for k, v in items:
            f.write(f"{k}: {v}\n")


# ----------------------------
# CRC16-CCITT (poly 0x1021)
# ----------------------------

CRC16_POLY = 0x1021
CRC16_INIT = 0xFFFF


def crc16_ccitt(data: bytes, poly: int = CRC16_POLY, init: int = CRC16_INIT) -> int:
    # Non-reflected, xorout=0, width=16.
    crc = init & 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if (crc & 0x8000) != 0:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


# ----------------------------
# RTL-ish modules
# ----------------------------

@dataclass
class MovingAverageQ88:
    win: int
    buf: List[int]
    s: int
    idx: int

    @staticmethod
    def create(win: int) -> "MovingAverageQ88":
        return MovingAverageQ88(win=win, buf=[0] * win, s=0, idx=0)

    def push(self, x: int) -> int:
        # x is Q8.8
        old = self.buf[self.idx]
        self.s -= old
        self.buf[self.idx] = x
        self.s += x
        self.idx = (self.idx + 1) % self.win
        # Average with integer division (synth-friendly)
        return int(self.s // self.win)


@dataclass
class DebouncedLatch:
    # Debounce + hysteresis latch
    debounce_n: int
    set_th_q88: int
    clr_th_q88: int
    cnt: int = 0
    latched: int = 0

    def step(self, x_q88: int, mode_high: bool) -> Tuple[int, int]:
        # mode_high True: alarm when x >= set_th, clear when x <= clr_th
        # mode_high False: alarm when x <= set_th, clear when x >= clr_th
        prev = self.latched

        def cond_set() -> bool:
            return x_q88 >= self.set_th_q88 if mode_high else x_q88 <= self.set_th_q88

        def cond_clr() -> bool:
            return x_q88 <= self.clr_th_q88 if mode_high else x_q88 >= self.clr_th_q88

        edge = 0
        if self.latched == 0:
            if cond_set():
                self.cnt += 1
                if self.cnt >= self.debounce_n:
                    self.latched = 1
                    self.cnt = 0
            else:
                self.cnt = 0
        else:
            if cond_clr():
                self.cnt += 1
                if self.cnt >= self.debounce_n:
                    self.latched = 0
                    self.cnt = 0
            else:
                self.cnt = 0

        if prev == 0 and self.latched == 1:
            edge = 1
        return self.latched, edge


@dataclass
class FIFO:
    depth: int
    mem: List[bytes]
    rd: int = 0
    wr: int = 0
    occ: int = 0
    drops: int = 0

    @staticmethod
    def create(depth: int) -> "FIFO":
        return FIFO(depth=depth, mem=[b""] * depth)

    def push(self, item: bytes) -> int:
        # Returns 1 if pushed, 0 if dropped due to full
        if self.occ >= self.depth:
            self.drops += 1
            return 0
        self.mem[self.wr] = item
        self.wr = (self.wr + 1) % self.depth
        self.occ += 1
        return 1

    def pop(self) -> Tuple[int, bytes]:
        if self.occ <= 0:
            return 0, b""
        item = self.mem[self.rd]
        self.rd = (self.rd + 1) % self.depth
        self.occ -= 1
        return 1, item

    def peek(self) -> Tuple[int, bytes]:
        if self.occ <= 0:
            return 0, b""
        return 1, self.mem[self.rd]


# ----------------------------
# Record + packet formats
# ----------------------------

# Fixed-width record (little endian). Payload is 32 bytes.
# Fields:
#  u32 ts_ms
#  u8  fsm_state
#  u8  proto_id
#  u16 alarms_mask
#  i16 temp_q88
#  i16 moist_q88
#  i16 press_q88
#  i16 vib_q88
#  u8  heater_en
#  u8  fan_pwm
#  u8  valve
#  u8  event_type   (0=periodic, 1=alarm_edge)
#  u16 event_code
#  u16 fifo_occ
#  u32 rsv0
#  u32 rsv1
#
# Total: 32 bytes
RECORD_STRUCT = struct.Struct("<I B B H h h h h B B B B H H I I")


def pack_record(
    ts_ms: int,
    fsm_state: int,
    proto_id: int,
    alarms_mask: int,
    temp_q88: int,
    moist_q88: int,
    press_q88: int,
    vib_q88: int,
    heater_en: int,
    fan_pwm: int,
    valve: int,
    event_type: int,
    event_code: int,
    fifo_occ: int,
) -> bytes:
    return RECORD_STRUCT.pack(
        ts_ms & 0xFFFFFFFF,
        fsm_state & 0xFF,
        proto_id & 0xFF,
        alarms_mask & 0xFFFF,
        int(clamp(temp_q88, -32768, 32767)),
        int(clamp(moist_q88, -32768, 32767)),
        int(clamp(press_q88, -32768, 32767)),
        int(clamp(vib_q88, -32768, 32767)),
        heater_en & 0xFF,
        fan_pwm & 0xFF,
        valve & 0xFF,
        event_type & 0xFF,
        event_code & 0xFFFF,
        fifo_occ & 0xFFFF,
        0,
        0,
    )


SYNC = b"\xA5\x5A"


def header_modbus_like(proto_addr: int, func: int, payload_len: int) -> bytes:
    # [addr][func][len_lo][len_hi]
    return struct.pack("<B B H", proto_addr & 0xFF, func & 0xFF, payload_len & 0xFFFF)


def header_cip_like(service: int, cls: int, inst: int, payload_len: int) -> bytes:
    # [service][class][instance][len_lo][len_hi]
    return struct.pack("<B B B H", service & 0xFF, cls & 0xFF, inst & 0xFF, payload_len & 0xFFFF)


def make_frame(proto_id: int, payload: bytes) -> Tuple[bytes, bytes]:
    # Returns (header, full_frame_bytes_including_sync)
    if proto_id == 0:
        hdr = header_modbus_like(proto_addr=0x11, func=0x03, payload_len=len(payload))
    else:
        hdr = header_cip_like(service=0x4C, cls=0x20, inst=0x01, payload_len=len(payload))
    crc = crc16_ccitt(hdr + payload)
    frame = SYNC + hdr + payload + struct.pack("<H", crc)
    return hdr, frame


# ----------------------------
# Plant FSM (industrial dryer)
# ----------------------------

FSM_IDLE = 0
FSM_RAMP = 1
FSM_STEADY = 2
FSM_FAULT = 3
FSM_COOLDOWN = 4

FSM_NAMES = {
    FSM_IDLE: "IDLE",
    FSM_RAMP: "RAMP",
    FSM_STEADY: "STEADY",
    FSM_FAULT: "FAULT",
    FSM_COOLDOWN: "COOLDOWN",
}


@dataclass
class DryerFSM:
    state: int = FSM_IDLE
    dwell: int = 0
    fault_latched: int = 0

    # Hysteresis thresholds (Q8.8)
    target_temp_q88: int = q88_from_float(80.0)
    ramp_enter_temp_q88: int = q88_from_float(40.0)
    steady_temp_hi_q88: int = q88_from_float(85.0)
    steady_temp_lo_q88: int = q88_from_float(75.0)
    moist_done_q88: int = q88_from_float(15.0)

    # Fault thresholds
    high_temp_fault_q88: int = q88_from_float(98.0)
    vib_fault_q88: int = q88_from_float(8.0)

    # Dwell counts in ticks
    ramp_min_dwell: int = 20      # 2.0 s
    cooldown_dwell: int = 30      # 3.0 s

    # Actuators
    heater_en: int = 0
    fan_pwm: int = 0      # 0..255
    valve: int = 0

    def step(self, temp_q88: int, moist_q88: int, vib_q88: int) -> None:
        # Default actuators (combinational-like based on state)
        self.heater_en = 0
        self.fan_pwm = 0
        self.valve = 0

        if self.state == FSM_IDLE:
            self.fan_pwm = 30
            self.valve = 0
            self.heater_en = 0
            self.dwell = 0
            self.fault_latched = 0
            # Start condition: moisture high
            if moist_q88 > q88_from_float(30.0):
                self.state = FSM_RAMP

        elif self.state == FSM_RAMP:
            self.heater_en = 1
            self.fan_pwm = 160
            self.valve = 0
            self.dwell += 1

            if temp_q88 >= self.high_temp_fault_q88 or vib_q88 >= self.vib_fault_q88:
                self.state = FSM_FAULT
                self.fault_latched = 1
                self.dwell = 0
            else:
                if self.dwell >= self.ramp_min_dwell and temp_q88 >= self.ramp_enter_temp_q88:
                    self.state = FSM_STEADY
                    self.dwell = 0

        elif self.state == FSM_STEADY:
            if temp_q88 < self.steady_temp_lo_q88:
                self.heater_en = 1
            elif temp_q88 > self.steady_temp_hi_q88:
                self.heater_en = 0
            else:
                self.heater_en = 1 if (self.dwell % 4) < 2 else 0

            self.fan_pwm = 200
            self.valve = 1 if moist_q88 > q88_from_float(25.0) else 0
            self.dwell += 1

            if temp_q88 >= self.high_temp_fault_q88 or vib_q88 >= self.vib_fault_q88:
                self.state = FSM_FAULT
                self.fault_latched = 1
                self.dwell = 0
            else:
                if moist_q88 <= self.moist_done_q88:
                    self.state = FSM_COOLDOWN
                    self.dwell = 0

        elif self.state == FSM_FAULT:
            self.heater_en = 0
            self.fan_pwm = 255
            self.valve = 1
            self.dwell += 1
            if temp_q88 <= q88_from_float(70.0) and vib_q88 <= q88_from_float(5.0) and self.dwell >= 25:
                self.state = FSM_COOLDOWN
                self.dwell = 0

        elif self.state == FSM_COOLDOWN:
            self.heater_en = 0
            self.fan_pwm = 180
            self.valve = 1
            self.dwell += 1
            if self.dwell >= self.cooldown_dwell:
                self.state = FSM_IDLE
                self.dwell = 0

        else:
            self.state = FSM_IDLE
            self.dwell = 0


# ----------------------------
# Simulation driver
# ----------------------------

@dataclass
class SimConfig:
    title: str = "Event Driven Industrial Sensor Gateway ASIC (EDISGA)"
    tick_ms: int = 100
    oversample_m: int = 8
    ma_win: int = 8
    duration_s: int = 240
    fifo_depth: int = 256
    byte_budget_per_tick: int = 40
    proto_switch_period_ticks: int = 50
    ready_drop_prob: float = 0.08
    seed: int = 7

    # Alarm debounce and thresholds
    debounce_ticks: int = 3

    high_temp_set_c: float = 95.0
    high_temp_clr_c: float = 90.0

    low_moist_set_pct: float = 12.0
    low_moist_clr_pct: float = 16.0

    high_press_set_kpa: float = 140.0
    high_press_clr_kpa: float = 132.0

    high_vib_set: float = 7.5
    high_vib_clr: float = 6.0


def synth_sensor_subsample(
    tick: int,
    sub: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    fsm_state: int,
    heater_en: int,
    fan_pwm: int,
    valve: int,
) -> Tuple[float, float, float, float]:
    # Deterministic-ish but noisy sensor evolution.
    t = tick * cfg.tick_ms / 1000.0 + sub * (cfg.tick_ms / 1000.0) / cfg.oversample_m

    ambient_temp = 28.0 + 0.5 * math.sin(2 * math.pi * t / 90.0)
    ambient_moist = 55.0 + 1.0 * math.sin(2 * math.pi * t / 120.0)
    ambient_press = 101.0 + 0.2 * math.sin(2 * math.pi * t / 60.0)
    ambient_vib = 1.0 + 0.05 * math.sin(2 * math.pi * t / 15.0)

    temp = ambient_temp + (20.0 if heater_en else 0.0) * (0.35 + 0.65 * (fan_pwm / 255.0))
    moist = ambient_moist - (25.0 * (fan_pwm / 255.0)) - (10.0 if valve else 0.0) - (5.0 if heater_en else 0.0)
    press = ambient_press + (10.0 * (fan_pwm / 255.0)) + (4.0 if valve else 0.0)
    vib = ambient_vib + (1.0 + 2.0 * (fan_pwm / 255.0)) + (0.5 if valve else 0.0)

    if 60.0 < t < 70.0:
        temp += 8.0 * math.exp(-0.25 * (t - 60.0))
    if 150.0 < t < 165.0:
        vib += 6.0 * (1.0 - math.exp(-0.5 * (t - 150.0)))
    if 110.0 < t < 118.0:
        moist += 18.0 * math.exp(-0.45 * (t - 110.0))

    temp += rng.normal(0.0, 0.35)
    moist += rng.normal(0.0, 0.9)
    press += rng.normal(0.0, 0.25)
    vib += abs(rng.normal(0.0, 0.18))

    temp = max(-10.0, min(140.0, temp))
    moist = max(0.0, min(100.0, moist))
    press = max(60.0, min(220.0, press))
    vib = max(0.0, min(30.0, vib))
    return temp, moist, press, vib


def run_sim(outdir: str, seed: int, duration_s: int) -> None:
    paths = ensure_clean_sandbox(outdir)
    figdir = paths["figures"]
    tabdir = paths["tables"]
    logdir = paths["logs"]
    wavedir = paths["waveforms"]

    cfg = SimConfig(seed=seed, duration_s=duration_s)
    ticks = int(round((cfg.duration_s * 1000) / cfg.tick_ms))

    rng = np.random.default_rng(cfg.seed)

    ma_temp = MovingAverageQ88.create(cfg.ma_win)
    ma_moist = MovingAverageQ88.create(cfg.ma_win)
    ma_press = MovingAverageQ88.create(cfg.ma_win)
    ma_vib = MovingAverageQ88.create(cfg.ma_win)

    fsm = DryerFSM()

    al_high_temp = DebouncedLatch(cfg.debounce_ticks, q88_from_float(cfg.high_temp_set_c), q88_from_float(cfg.high_temp_clr_c))
    al_low_moist = DebouncedLatch(cfg.debounce_ticks, q88_from_float(cfg.low_moist_set_pct), q88_from_float(cfg.low_moist_clr_pct))
    al_high_press = DebouncedLatch(cfg.debounce_ticks, q88_from_float(cfg.high_press_set_kpa), q88_from_float(cfg.high_press_clr_kpa))
    al_high_vib = DebouncedLatch(cfg.debounce_ticks, q88_from_float(cfg.high_vib_set), q88_from_float(cfg.high_vib_clr))

    BIT_HIGH_TEMP = 0
    BIT_LOW_MOIST = 1
    BIT_HIGH_PRESS = 2
    BIT_HIGH_VIB = 3

    fifo = FIFO.create(cfg.fifo_depth)

    sending = 0
    send_buf = b""
    send_off = 0
    send_frame_len = 0
    send_hdr = b""
    send_crc = 0
    send_payload = b""
    send_proto = 0
    frame_seq = 0

    t_ms = np.zeros(ticks, dtype=np.int64)

    raw_temp = np.zeros(ticks, dtype=np.int32)
    raw_moist = np.zeros(ticks, dtype=np.int32)
    raw_press = np.zeros(ticks, dtype=np.int32)
    raw_vib = np.zeros(ticks, dtype=np.int32)

    fil_temp = np.zeros(ticks, dtype=np.int32)
    fil_moist = np.zeros(ticks, dtype=np.int32)
    fil_press = np.zeros(ticks, dtype=np.int32)
    fil_vib = np.zeros(ticks, dtype=np.int32)

    fsm_state = np.zeros(ticks, dtype=np.int32)
    heater = np.zeros(ticks, dtype=np.int32)
    fan = np.zeros(ticks, dtype=np.int32)
    valve = np.zeros(ticks, dtype=np.int32)

    alarms_mask = np.zeros(ticks, dtype=np.int32)
    alarm_edges = np.zeros(ticks, dtype=np.int32)
    event_code = np.zeros(ticks, dtype=np.int32)

    fifo_occ = np.zeros(ticks, dtype=np.int32)
    fifo_drops = np.zeros(ticks, dtype=np.int32)

    proto_id = np.zeros(ticks, dtype=np.int32)
    valid = np.zeros(ticks, dtype=np.int32)
    ready = np.zeros(ticks, dtype=np.int32)
    fire = np.zeros(ticks, dtype=np.int32)
    bytes_sent = np.zeros(ticks, dtype=np.int32)
    frame_inflight = np.zeros(ticks, dtype=np.int32)

    wf_csv = os.path.join(wavedir, "waveforms.csv")
    trends_csv = os.path.join(wavedir, "trends.csv")
    records_csv = os.path.join(logdir, "historian_records.csv")
    frames_csv = os.path.join(logdir, "frames.csv")
    packets_bin = os.path.join(logdir, "packets.bin")

    wf_f = open(wf_csv, "w", newline="", encoding="utf-8")
    wf_w = csv.writer(wf_f)
    wf_w.writerow([
        "tick", "ts_ms",
        "temp_raw_q88", "moist_raw_q88", "press_raw_q88", "vib_raw_q88",
        "temp_filt_q88", "moist_filt_q88", "press_filt_q88", "vib_filt_q88",
        "fsm_state", "heater_en", "fan_pwm", "valve",
        "alarms_mask", "alarm_edge", "event_code",
        "fifo_occ", "fifo_drops",
        "proto_id",
        "valid", "ready", "fire", "bytes_sent", "frame_inflight",
    ])

    rec_f = open(records_csv, "w", newline="", encoding="utf-8")
    rec_w = csv.writer(rec_f)
    rec_w.writerow([
        "ts_ms", "tick", "event_type", "event_code",
        "fsm_state", "proto_id", "alarms_mask",
        "temp_C", "moist_pct", "press_kPa", "vib",
        "heater_en", "fan_pwm", "valve",
        "fifo_occ_after_push",
    ])

    frm_f = open(frames_csv, "w", newline="", encoding="utf-8")
    frm_w = csv.writer(frm_f)
    frm_w.writerow([
        "frame_seq", "ts_ms", "tick", "proto_id",
        "payload_len", "header_hex", "crc16_hex", "frame_len",
    ])

    pkt_f = open(packets_bin, "wb")

    enqueue_ticks: List[int] = []
    lat_periodic: List[int] = []
    lat_event: List[int] = []

    for k in range(ticks):
        ts = k * cfg.tick_ms
        t_ms[k] = ts

        pid = 0 if (k // cfg.proto_switch_period_ticks) % 2 == 0 else 1
        proto_id[k] = pid

        sub_temp = 0.0
        sub_moist = 0.0
        sub_press = 0.0
        sub_vib = 0.0
        for s in range(cfg.oversample_m):
            a_temp, a_moist, a_press, a_vib = synth_sensor_subsample(
                tick=k,
                sub=s,
                cfg=cfg,
                rng=rng,
                fsm_state=fsm.state,
                heater_en=fsm.heater_en,
                fan_pwm=fsm.fan_pwm,
                valve=fsm.valve,
            )
            sub_temp += a_temp
            sub_moist += a_moist
            sub_press += a_press
            sub_vib += a_vib

        temp_f = sub_temp / cfg.oversample_m
        moist_f = sub_moist / cfg.oversample_m
        press_f = sub_press / cfg.oversample_m
        vib_f = sub_vib / cfg.oversample_m

        temp_q = q88_from_float(temp_f)
        moist_q = q88_from_float(moist_f)
        press_q = q88_from_float(press_f)
        vib_q = q88_from_float(vib_f)

        raw_temp[k] = temp_q
        raw_moist[k] = moist_q
        raw_press[k] = press_q
        raw_vib[k] = vib_q

        tf = ma_temp.push(temp_q)
        mf = ma_moist.push(moist_q)
        pf = ma_press.push(press_q)
        vf = ma_vib.push(vib_q)

        fil_temp[k] = tf
        fil_moist[k] = mf
        fil_press[k] = pf
        fil_vib[k] = vf

        fsm.step(tf, mf, vf)

        fsm_state[k] = fsm.state
        heater[k] = fsm.heater_en
        fan[k] = fsm.fan_pwm
        valve[k] = fsm.valve

        a0, e0 = al_high_temp.step(tf, mode_high=True)
        a1, e1 = al_low_moist.step(mf, mode_high=False)
        a2, e2 = al_high_press.step(pf, mode_high=True)
        a3, e3 = al_high_vib.step(vf, mode_high=True)

        mask = (a0 << BIT_HIGH_TEMP) | (a1 << BIT_LOW_MOIST) | (a2 << BIT_HIGH_PRESS) | (a3 << BIT_HIGH_VIB)
        alarms_mask[k] = mask

        edge_mask = (e0 << BIT_HIGH_TEMP) | (e1 << BIT_LOW_MOIST) | (e2 << BIT_HIGH_PRESS) | (e3 << BIT_HIGH_VIB)
        alarm_edges[k] = edge_mask

        ev_code = 0
        if edge_mask != 0:
            if (edge_mask >> BIT_HIGH_TEMP) & 1:
                ev_code = 0x10
            elif (edge_mask >> BIT_LOW_MOIST) & 1:
                ev_code = 0x11
            elif (edge_mask >> BIT_HIGH_PRESS) & 1:
                ev_code = 0x12
            elif (edge_mask >> BIT_HIGH_VIB) & 1:
                ev_code = 0x13
        event_code[k] = ev_code

        occ_before = fifo.occ

        rec_periodic = pack_record(
            ts, fsm.state, pid, mask, tf, mf, pf, vf, fsm.heater_en, fsm.fan_pwm, fsm.valve,
            event_type=0, event_code=0, fifo_occ=occ_before
        )
        pushed_p = fifo.push(rec_periodic)
        if pushed_p:
            enqueue_ticks.append(k)
        rec_w.writerow([
            ts, k, "PERIODIC", "0x00",
            FSM_NAMES[fsm.state], pid, f"0x{mask:04X}",
            f"{q88_to_float(tf):.2f}", f"{q88_to_float(mf):.2f}", f"{q88_to_float(pf):.2f}", f"{q88_to_float(vf):.2f}",
            fsm.heater_en, fsm.fan_pwm, fsm.valve,
            fifo.occ,
        ])

        if ev_code != 0:
            rec_event = pack_record(
                ts, fsm.state, pid, mask, tf, mf, pf, vf, fsm.heater_en, fsm.fan_pwm, fsm.valve,
                event_type=1, event_code=ev_code, fifo_occ=fifo.occ
            )
            pushed_e = fifo.push(rec_event)
            if pushed_e:
                enqueue_ticks.append(k)
            rec_w.writerow([
                ts, k, "ALARM_EDGE", f"0x{ev_code:02X}",
                FSM_NAMES[fsm.state], pid, f"0x{mask:04X}",
                f"{q88_to_float(tf):.2f}", f"{q88_to_float(mf):.2f}", f"{q88_to_float(pf):.2f}", f"{q88_to_float(vf):.2f}",
                fsm.heater_en, fsm.fan_pwm, fsm.valve,
                fifo.occ,
            ])

        fifo_occ[k] = fifo.occ
        fifo_drops[k] = fifo.drops

        rdy = 1 if rng.random() >= cfg.ready_drop_prob else 0
        ready[k] = rdy

        sent_this_tick = 0
        fired = 0

        if sending == 0:
            v = 1 if fifo.occ > 0 else 0
            valid[k] = v
            if v and rdy:
                ok, payload = fifo.peek()
                if ok:
                    hdr, frame = make_frame(pid, payload)
                    send_hdr = hdr
                    send_payload = payload
                    send_crc = crc16_ccitt(hdr + payload)
                    send_buf = frame
                    send_off = 0
                    send_frame_len = len(frame)
                    send_proto = pid
                    sending = 1
                    frame_seq += 1
                    fired = 1
                    frm_w.writerow([
                        frame_seq, ts, k, pid,
                        len(payload),
                        hdr.hex(),
                        f"{send_crc:04X}",
                        send_frame_len,
                    ])
                    pkt_f.write(frame)
        else:
            valid[k] = 1

        if sending and rdy:
            budget = cfg.byte_budget_per_tick
            rem = send_frame_len - send_off
            n = budget if rem > budget else rem
            send_off += n
            sent_this_tick = n
            if send_off >= send_frame_len:
                okp, _ = fifo.pop()
                if okp and enqueue_ticks:
                    enq_k = enqueue_ticks.pop(0)
                    ev_t = send_payload[19]  # event_type byte offset in record
                    lat = k - enq_k
                    if ev_t == 0:
                        lat_periodic.append(lat)
                    else:
                        lat_event.append(lat)
                sending = 0
                send_buf = b""
                send_off = 0
                send_frame_len = 0

        fire[k] = fired
        bytes_sent[k] = sent_this_tick
        frame_inflight[k] = 1 if sending else 0

        wf_w.writerow([
            k, ts,
            int(raw_temp[k]), int(raw_moist[k]), int(raw_press[k]), int(raw_vib[k]),
            int(fil_temp[k]), int(fil_moist[k]), int(fil_press[k]), int(fil_vib[k]),
            int(fsm_state[k]), int(heater[k]), int(fan[k]), int(valve[k]),
            int(alarms_mask[k]), int(alarm_edges[k]), int(event_code[k]),
            int(fifo_occ[k]), int(fifo_drops[k]),
            int(proto_id[k]),
            int(valid[k]), int(ready[k]), int(fire[k]), int(bytes_sent[k]), int(frame_inflight[k]),
        ])

    wf_f.close()
    rec_f.close()
    frm_f.close()
    pkt_f.close()

    with open(trends_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ts_s",
            "temp_C_raw", "temp_C_filt",
            "moist_pct_raw", "moist_pct_filt",
            "press_kPa_raw", "press_kPa_filt",
            "vib_raw", "vib_filt",
            "fsm_state", "heater_en", "fan_pwm", "valve",
            "alarms_mask", "fifo_occ",
        ])
        for k in range(ticks):
            w.writerow([
                t_ms[k] / 1000.0,
                q88_to_float(raw_temp[k]), q88_to_float(fil_temp[k]),
                q88_to_float(raw_moist[k]), q88_to_float(fil_moist[k]),
                q88_to_float(raw_press[k]), q88_to_float(fil_press[k]),
                q88_to_float(raw_vib[k]), q88_to_float(fil_vib[k]),
                int(fsm_state[k]), int(heater[k]), int(fan[k]), int(valve[k]),
                f"0x{int(alarms_mask[k]):04X}", int(fifo_occ[k]),
            ])

    # ----------------------------
    # Figures
    # ----------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def savefig(name: str) -> None:
        p = os.path.join(figdir, name)
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()

    t_s = t_ms / 1000.0

    def plot_sensor(name: str, raw: np.ndarray, flt: np.ndarray, ylabel: str) -> None:
        plt.figure(figsize=(9, 3.2))
        plt.plot(t_s, raw.astype(np.float64) / 256.0, label="raw")
        plt.plot(t_s, flt.astype(np.float64) / 256.0, label="filtered")
        plt.xlabel("time (s)")
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        savefig(name)

    plot_sensor("sensor_temp_raw_vs_filt.png", raw_temp, fil_temp, "temperature (C)")
    plot_sensor("sensor_moist_raw_vs_filt.png", raw_moist, fil_moist, "moisture (%)")
    plot_sensor("sensor_press_raw_vs_filt.png", raw_press, fil_press, "pressure (kPa)")
    plot_sensor("sensor_vib_raw_vs_filt.png", raw_vib, fil_vib, "vibration (a.u.)")

    plt.figure(figsize=(9, 5.4))
    plt.plot(t_s, raw_temp.astype(np.float64) / 256.0, label="temp raw")
    plt.plot(t_s, fil_temp.astype(np.float64) / 256.0, label="temp filt")
    plt.plot(t_s, raw_moist.astype(np.float64) / 256.0, label="moist raw")
    plt.plot(t_s, fil_moist.astype(np.float64) / 256.0, label="moist filt")
    plt.xlabel("time (s)")
    plt.ylabel("scaled units")
    plt.legend(loc="best", ncol=2)
    savefig("sensors_combined_raw_vs_filt.png")

    plt.figure(figsize=(9, 2.2))
    plt.step(t_s, fsm_state, where="post")
    plt.yticks([0, 1, 2, 3, 4], ["IDLE", "RAMP", "STEADY", "FAULT", "COOLDOWN"])
    plt.xlabel("time (s)")
    plt.ylabel("FSM state")
    savefig("fsm_timeline.png")

    plt.figure(figsize=(9, 3.2))
    plt.step(t_s, heater, where="post", label="HEATER_EN")
    plt.plot(t_s, fan, label="FAN_PWM")
    plt.step(t_s, valve, where="post", label="VALVE")
    plt.xlabel("time (s)")
    plt.ylabel("actuator level")
    plt.legend(loc="best")
    savefig("actuators.png")

    plt.figure(figsize=(9, 2.8))
    plt.step(t_s, alarms_mask, where="post", label="alarms_mask (bitfield)")
    plt.step(t_s, alarm_edges * 0x10000, where="post", label="alarm_edges (scaled)")
    plt.xlabel("time (s)")
    plt.ylabel("mask")
    plt.legend(loc="best")
    savefig("alarms_masks.png")

    plt.figure(figsize=(9, 3.2))
    plt.plot(t_s, fifo_occ, label="fifo_occ")
    plt.plot(t_s, fifo_drops, label="fifo_drops")
    plt.xlabel("time (s)")
    plt.ylabel("count")
    plt.legend(loc="best")
    savefig("fifo_occupancy_and_drops.png")

    plt.figure(figsize=(9, 3.2))
    plt.step(t_s, valid, where="post", label="valid")
    plt.step(t_s, ready, where="post", label="ready")
    plt.step(t_s, fire, where="post", label="start_fire")
    plt.step(t_s, frame_inflight, where="post", label="frame_inflight")
    plt.xlabel("time (s)")
    plt.ylabel("level")
    plt.legend(loc="best", ncol=4)
    savefig("handshake_valid_ready_fire.png")

    plt.figure(figsize=(9, 3.0))
    plt.plot(t_s, bytes_sent, label="bytes_sent_per_tick")
    plt.xlabel("time (s)")
    plt.ylabel("bytes/tick")
    plt.legend(loc="best")
    savefig("throughput_bytes.png")

    plt.figure(figsize=(9, 2.2))
    plt.step(t_s, proto_id, where="post")
    plt.yticks([0, 1], ["Modbus-like", "CIP-like"])
    plt.xlabel("time (s)")
    plt.ylabel("protocol")
    savefig("protocol_selection_over_time.png")

    def plot_hist(name: str, data: List[int], title: str) -> None:
        plt.figure(figsize=(6.4, 3.2))
        if len(data) == 0:
            plt.text(0.5, 0.5, "no samples", ha="center", va="center")
            plt.axis("off")
        else:
            plt.hist(data, bins=20)
            plt.xlabel("latency (ticks)")
            plt.ylabel("count")
        plt.title(title)
        savefig(name)

    plot_hist("latency_periodic_hist.png", lat_periodic, "Periodic record latency (ticks)")
    plot_hist("latency_event_hist.png", lat_event, "Event record latency (ticks)")

    def zoom_plot(name: str, t0: float, t1: float, series: List[Tuple[str, np.ndarray, str]]) -> None:
        plt.figure(figsize=(9, 3.2))
        m = (t_s >= t0) & (t_s <= t1)
        for label, arr, unit in series:
            plt.plot(t_s[m], arr[m].astype(np.float64) / 256.0, label=f"{label} ({unit})")
        plt.xlabel("time (s)")
        plt.ylabel("value")
        plt.legend(loc="best")
        savefig(name)

    zoom_plot("zoom_temp_drift_fault.png", 55.0, 75.0, [("temp_filt", fil_temp, "C"), ("vib_filt", fil_vib, "a.u.")])
    zoom_plot("zoom_moisture_drop.png", 105.0, 122.0, [("moist_filt", fil_moist, "%"), ("temp_filt", fil_temp, "C")])
    zoom_plot("zoom_disturbance_1.png", 145.0, 170.0, [("vib_filt", fil_vib, "a.u."), ("temp_filt", fil_temp, "C")])

    # ----------------------------
    # Generic tables and summaries (no LaTeX)
    # ----------------------------

    config_obj = dataclasses.asdict(cfg)
    config_obj["ticks"] = ticks
    config_obj["crc16"] = {
        "poly_hex": f"0x{CRC16_POLY:04X}",
        "init_hex": f"0x{CRC16_INIT:04X}",
        "refin": False,
        "refout": False,
        "xorout_hex": "0x0000",
    }
    config_obj["record_bytes"] = RECORD_STRUCT.size
    config_obj["frame_sync_hex"] = SYNC.hex()
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_obj, f, indent=2, sort_keys=True)

    # Record layout as CSV (machine friendly) and TXT (human friendly)
    record_layout_rows = [
        ["ts_ms", "u32", "timestamp in milliseconds"],
        ["fsm_state", "u8", "FSM state id"],
        ["proto_id", "u8", "protocol id"],
        ["alarms_mask", "u16", "alarm latch bitfield"],
        ["temp_q88", "i16", "temperature in Q8.8"],
        ["moist_q88", "i16", "moisture in Q8.8"],
        ["press_q88", "i16", "pressure in Q8.8"],
        ["vib_q88", "i16", "vibration in Q8.8"],
        ["heater_en", "u8", "heater enable"],
        ["fan_pwm", "u8", "fan PWM (0..255)"],
        ["valve", "u8", "valve state"],
        ["event_type", "u8", "0=periodic, 1=alarm_edge"],
        ["event_code", "u16", "event code"],
        ["fifo_occ", "u16", "fifo occupancy snapshot"],
        ["rsv0", "u32", "reserved"],
        ["rsv1", "u32", "reserved"],
    ]
    write_table_csv(
        os.path.join(tabdir, "record_layout.csv"),
        header=["field", "type", "meaning"],
        rows=record_layout_rows,
    )

    write_kv_txt(
        os.path.join(tabdir, "record_layout.txt"),
        "Record layout (fixed width payload)",
        [(r[0], f"{r[1]} | {r[2]}") for r in record_layout_rows],
    )

    # Config metadata CSV
    meta_rows = [
        ["Project", cfg.title],
        ["Seed", str(cfg.seed)],
        ["Tick period", f"{cfg.tick_ms} ms"],
        ["Duration", f"{cfg.duration_s} s"],
        ["Total ticks", str(ticks)],
        ["Oversampling M", str(cfg.oversample_m)],
        ["Moving average window W", str(cfg.ma_win)],
        ["FIFO depth (records)", str(cfg.fifo_depth)],
        ["Byte budget per tick", str(cfg.byte_budget_per_tick)],
        ["Protocol switch period (ticks)", str(cfg.proto_switch_period_ticks)],
        ["CRC16 polynomial", f"0x{CRC16_POLY:04X} (CCITT)"],
        ["CRC16 init", f"0x{CRC16_INIT:04X}"],
        ["Record bytes", str(RECORD_STRUCT.size)],
        ["Frame SYNC", SYNC.hex()],
    ]
    write_table_csv(
        os.path.join(tabdir, "config_metadata.csv"),
        header=["parameter", "value"],
        rows=meta_rows,
    )

    # Thresholds CSV
    thr_rows = [
        ["HIGH_TEMP set", f"{cfg.high_temp_set_c:.1f} C"],
        ["HIGH_TEMP clear", f"{cfg.high_temp_clr_c:.1f} C"],
        ["LOW_MOISTURE set", f"{cfg.low_moist_set_pct:.1f} %"],
        ["LOW_MOISTURE clear", f"{cfg.low_moist_clr_pct:.1f} %"],
        ["HIGH_PRESSURE set", f"{cfg.high_press_set_kpa:.1f} kPa"],
        ["HIGH_PRESSURE clear", f"{cfg.high_press_clr_kpa:.1f} kPa"],
        ["HIGH_VIBRATION set", f"{cfg.high_vib_set:.2f} a.u."],
        ["HIGH_VIBRATION clear", f"{cfg.high_vib_clr:.2f} a.u."],
    ]
    write_table_csv(
        os.path.join(tabdir, "thresholds.csv"),
        header=["signal", "threshold"],
        rows=thr_rows,
    )

    # CRC info CSV + JSON
    crc_rows = [
        ["width", "16"],
        ["poly_hex", f"0x{CRC16_POLY:04X}"],
        ["init_hex", f"0x{CRC16_INIT:04X}"],
        ["refin", "false"],
        ["refout", "false"],
        ["xorout_hex", "0x0000"],
        ["scope", "CRC computed over header + payload (SYNC excluded)"],
    ]
    write_table_csv(
        os.path.join(tabdir, "crc_config.csv"),
        header=["item", "value"],
        rows=crc_rows,
    )
    with open(os.path.join(tabdir, "crc_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "width": 16,
                "poly_hex": f"0x{CRC16_POLY:04X}",
                "init_hex": f"0x{CRC16_INIT:04X}",
                "refin": False,
                "refout": False,
                "xorout_hex": "0x0000",
                "scope": "CRC computed over header + payload (SYNC excluded)",
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # CRC validation vectors (CSV)
    vectors: List[List[str]] = []
    for i in range(6):
        kk = min(i * 7, ticks - 1)
        pidv = int(proto_id[kk])
        payload = pack_record(
            int(t_ms[kk]),
            int(fsm_state[kk]),
            pidv,
            int(alarms_mask[kk]),
            int(fil_temp[kk]),
            int(fil_moist[kk]),
            int(fil_press[kk]),
            int(fil_vib[kk]),
            int(heater[kk]),
            int(fan[kk]),
            int(valve[kk]),
            event_type=0,
            event_code=0,
            fifo_occ=int(fifo_occ[kk]),
        )
        hdr, _ = make_frame(pidv, payload)
        crc = crc16_ccitt(hdr + payload)
        vectors.append([str(pidv), hdr.hex(), (payload[:12].hex() + "..."), f"0x{crc:04X}"])

    write_table_csv(
        os.path.join(tabdir, "crc_vectors.csv"),
        header=["proto_id", "header_hex", "payload_prefix_hex", "crc16_hex"],
        rows=vectors,
    )

    # Latency summary CSV + JSON
    def stats(lst: List[int]) -> Tuple[float, float, float, float]:
        if len(lst) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        a = np.array(lst, dtype=np.float64)
        return (float(np.min(a)), float(np.mean(a)), float(np.percentile(a, 95)), float(np.max(a)))

    p_min, p_mean, p_p95, p_max = stats(lat_periodic)
    e_min, e_mean, e_p95, e_max = stats(lat_event)

    write_table_csv(
        os.path.join(tabdir, "latency_summary.csv"),
        header=["record_type", "min_ticks", "mean_ticks", "p95_ticks", "max_ticks", "num_samples"],
        rows=[
            ["Periodic", f"{p_min:.1f}", f"{p_mean:.2f}", f"{p_p95:.1f}", f"{p_max:.1f}", str(len(lat_periodic))],
            ["Event", f"{e_min:.1f}", f"{e_mean:.2f}", f"{e_p95:.1f}", f"{e_max:.1f}", str(len(lat_event))],
        ],
    )

    total_frames = frame_seq
    total_bytes = int(np.sum(bytes_sent))
    sim_time_s = cfg.duration_s
    avg_bps = total_bytes / sim_time_s if sim_time_s > 0 else 0.0
    num_events = int(np.sum(event_code != 0))
    total_records_attempted = ticks + num_events

    summary_obj = {
        "ticks": ticks,
        "records_attempted": total_records_attempted,
        "event_records_attempted": num_events,
        "fifo_drops_records": int(fifo.drops),
        "frames_emitted": int(total_frames),
        "total_bytes_transmitted": int(total_bytes),
        "average_throughput_Bps": float(avg_bps),
        "ticks_with_backpressure_ready0": int(np.sum(ready == 0)),
        "latency_ticks": {
            "periodic": {"min": p_min, "mean": p_mean, "p95": p_p95, "max": p_max, "n": len(lat_periodic)},
            "event": {"min": e_min, "mean": e_mean, "p95": e_p95, "max": e_max, "n": len(lat_event)},
        },
    }
    with open(os.path.join(tabdir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2, sort_keys=True)

    write_table_csv(
        os.path.join(tabdir, "run_summary.csv"),
        header=["metric", "value"],
        rows=[
            ["Total historian records attempted", str(total_records_attempted)],
            ["Event records attempted", str(num_events)],
            ["FIFO drops (records)", str(int(fifo.drops))],
            ["Frames emitted", str(total_frames)],
            ["Total bytes transmitted", str(total_bytes)],
            ["Average throughput (B/s)", f"{avg_bps:.1f}"],
            ["Ticks with backpressure (ready=0)", str(int(np.sum(ready == 0)))],
        ],
    )

    # Historian excerpt as CSV (first 20 data rows)
    excerpt_rows: List[List[str]] = []
    with open(records_csv, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        _hdr = next(rdr)
        for i, row in enumerate(rdr):
            if i >= 20:
                break
            excerpt_rows.append(row)

    write_table_csv(
        os.path.join(tabdir, "historian_records_excerpt.csv"),
        header=[
            "ts_ms", "tick", "event_type", "event_code",
            "fsm_state", "proto_id", "alarms_mask",
            "temp_C", "moist_pct", "press_kPa", "vib",
            "heater_en", "fan_pwm", "valve",
            "fifo_occ_after_push",
        ],
        rows=excerpt_rows,
    )

    # Notes as plain text
    notes = [
        ("notes", "Run demonstrates event-driven and periodic logging over a 100 ms tick."),
        ("notes", "Alarm edges trigger immediate event records with captured tick timestamps."),
        ("notes", "FIFO occupancy rises during backpressure (ready deassertions); overflow behavior is explicit via drops."),
    ]
    with open(os.path.join(tabdir, "analysis_notes.txt"), "w", encoding="utf-8") as f:
        for _, line in notes:
            f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration_s", type=int, default=240)
    args = ap.parse_args()

    run_sim(args.outdir, args.seed, args.duration_s)


if __name__ == "__main__":
    main()
