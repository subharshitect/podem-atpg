"""Five-valued logic utilities for ATPG."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple


class Logic5(Enum):
    ZERO = 0
    ONE = 1
    X = 2
    D = 3
    DBAR = 4

    def is_x(self) -> bool:
        return self is Logic5.X

    def is_d(self) -> bool:
        return self in (Logic5.D, Logic5.DBAR)

    def to_char(self) -> str:
        return {
            Logic5.ZERO: "0",
            Logic5.ONE: "1",
            Logic5.X: "X",
            Logic5.D: "D",
            Logic5.DBAR: "DBAR",
        }[self]

    def to_bool(self) -> Optional[int]:
        mapping = {
            Logic5.ZERO: 0,
            Logic5.ONE: 1,
            Logic5.D: 1,
            Logic5.DBAR: 0,
            Logic5.X: None,
        }
        return mapping[self]

    @staticmethod
    def from_pair(good: Optional[int], faulty: Optional[int]) -> "Logic5":
        if good is None or faulty is None:
            return Logic5.X
        if good == faulty == 0:
            return Logic5.ZERO
        if good == faulty == 1:
            return Logic5.ONE
        if good == 1 and faulty == 0:
            return Logic5.D
        if good == 0 and faulty == 1:
            return Logic5.DBAR
        return Logic5.X

    def as_pair(self) -> Tuple[Optional[int], Optional[int]]:
        if self is Logic5.ZERO:
            return 0, 0
        if self is Logic5.ONE:
            return 1, 1
        if self is Logic5.D:
            return 1, 0
        if self is Logic5.DBAR:
            return 0, 1
        return None, None


@dataclass(frozen=True)
class GateEvalResult:
    good: Optional[int]
    faulty: Optional[int]

    def to_logic5(self) -> Logic5:
        return Logic5.from_pair(self.good, self.faulty)


def _and(values: Iterable[Optional[int]]) -> Optional[int]:
    has_none = False
    for v in values:
        if v is None:
            has_none = True
        elif v == 0:
            return 0
    if has_none:
        return None
    return 1


def _or(values: Iterable[Optional[int]]) -> Optional[int]:
    has_none = False
    for v in values:
        if v is None:
            has_none = True
        elif v == 1:
            return 1
    if has_none:
        return None
    return 0


def _xor(values: Iterable[Optional[int]]) -> Optional[int]:
    total = 0
    for v in values:
        if v is None:
            return None
        total ^= v
    return total


def _invert(v: Optional[int]) -> Optional[int]:
    if v is None:
        return None
    return 1 - v


def evaluate_gate(gate_type: str, inputs: Iterable[Logic5]) -> GateEvalResult:
    """Evaluate a gate in five-valued logic.

    Returns good/faulty pairs before fault injection at the output.
    """
    good_inputs = []
    faulty_inputs = []
    for value in inputs:
        good, faulty = value.as_pair()
        good_inputs.append(good)
        faulty_inputs.append(faulty)

    gate_type = gate_type.upper()
    if gate_type == "AND":
        good = _and(good_inputs)
        faulty = _and(faulty_inputs)
    elif gate_type == "OR":
        good = _or(good_inputs)
        faulty = _or(faulty_inputs)
    elif gate_type == "NAND":
        good = _invert(_and(good_inputs))
        faulty = _invert(_and(faulty_inputs))
    elif gate_type == "NOR":
        good = _invert(_or(good_inputs))
        faulty = _invert(_or(faulty_inputs))
    elif gate_type == "NOT":
        (g0, f0) = Logic5.X.as_pair()
        for value in inputs:
            g0, f0 = value.as_pair()
            break
        good = _invert(g0)
        faulty = _invert(f0)
    elif gate_type == "BUF":
        (g0, f0) = Logic5.X.as_pair()
        for value in inputs:
            g0, f0 = value.as_pair()
            break
        good = g0
        faulty = f0
    elif gate_type == "XOR":
        good = _xor(good_inputs)
        faulty = _xor(faulty_inputs)
    elif gate_type == "XNOR":
        good = _invert(_xor(good_inputs))
        faulty = _invert(_xor(faulty_inputs))
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")

    return GateEvalResult(good=good, faulty=faulty)


def controlling_value(gate_type: str) -> Optional[Logic5]:
    gate_type = gate_type.upper()
    if gate_type in ("AND", "NAND"):
        return Logic5.ZERO
    if gate_type in ("OR", "NOR"):
        return Logic5.ONE
    return None


def non_controlling_value(gate_type: str) -> Optional[Logic5]:
    gate_type = gate_type.upper()
    if gate_type in ("AND", "NAND"):
        return Logic5.ONE
    if gate_type in ("OR", "NOR"):
        return Logic5.ZERO
    if gate_type == "XOR":
        return Logic5.ZERO
    if gate_type == "XNOR":
        return Logic5.ONE
    return None


def invert_value(value: Logic5) -> Logic5:
    if value is Logic5.ZERO:
        return Logic5.ONE
    if value is Logic5.ONE:
        return Logic5.ZERO
    if value is Logic5.D:
        return Logic5.DBAR
    if value is Logic5.DBAR:
        return Logic5.D
    return Logic5.X
