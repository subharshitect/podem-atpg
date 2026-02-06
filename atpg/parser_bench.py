"""Parser for simple .bench netlists."""

from __future__ import annotations

import re
from typing import List, Tuple

from .circuit import Circuit, CircuitError, Gate, SUPPORTED_GATES


class BenchParseError(Exception):
    pass


_INPUT_RE = re.compile(r"^INPUT\(([^)]+)\)\s*$", re.IGNORECASE)
_OUTPUT_RE = re.compile(r"^OUTPUT\(([^)]+)\)\s*$", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^([^=]+)=\s*([A-Z]+)\(([^)]*)\)\s*$", re.IGNORECASE)


def _clean_line(line: str) -> str:
    line = line.split("#", 1)[0]
    return line.strip()


def parse_bench(text: str) -> Circuit:
    primary_inputs: List[str] = []
    primary_outputs: List[str] = []
    gates: List[Gate] = []

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if match := _INPUT_RE.match(line):
            name = match.group(1).strip()
            primary_inputs.append(name)
            continue
        if match := _OUTPUT_RE.match(line):
            name = match.group(1).strip()
            primary_outputs.append(name)
            continue
        if match := _ASSIGN_RE.match(line):
            output = match.group(1).strip()
            gate_type = match.group(2).strip().upper()
            inputs = [item.strip() for item in match.group(3).split(",") if item.strip()]
            if gate_type not in SUPPORTED_GATES:
                raise BenchParseError(f"Unknown gate type: {gate_type}")
            if gate_type in {"NOT", "BUF"} and len(inputs) != 1:
                raise BenchParseError(f"Gate {gate_type} expects exactly one input")
            if gate_type not in {"NOT", "BUF"} and len(inputs) < 2:
                raise BenchParseError(f"Gate {gate_type} expects at least two inputs")
            gates.append(Gate(gate_type=gate_type, output=output, inputs=inputs))
            continue
        raise BenchParseError(f"Unrecognized line: {raw}")

    if not primary_outputs:
        raise BenchParseError("No OUTPUT declarations found")

    return Circuit(gates=gates, primary_inputs=primary_inputs, primary_outputs=primary_outputs)


def parse_bench_file(path: str) -> Circuit:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise BenchParseError(f"Failed to read netlist: {exc}") from exc
    return parse_bench(text)
