"""Circuit data structures and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .logic5 import Logic5, GateEvalResult, evaluate_gate


SUPPORTED_GATES = {"AND", "OR", "NAND", "NOR", "NOT", "BUF", "XOR", "XNOR"}


@dataclass(frozen=True)
class Gate:
    gate_type: str
    output: str
    inputs: List[str]


class CircuitError(Exception):
    pass


class Circuit:
    def __init__(self, gates: List[Gate], primary_inputs: List[str], primary_outputs: List[str]):
        self.gates = gates
        self.primary_inputs = primary_inputs
        self.primary_outputs = primary_outputs
        self.values: Dict[str, Logic5] = {}
        self.gate_map: Dict[str, Gate] = {gate.output: gate for gate in gates}
        self.topo_order: List[Gate] = self._topological_order()
        self._init_values()

    def _init_values(self) -> None:
        nets = set(self.primary_inputs + self.primary_outputs)
        for gate in self.gates:
            nets.add(gate.output)
            nets.update(gate.inputs)
        self.values = {net: Logic5.X for net in nets}

    def _topological_order(self) -> List[Gate]:
        incoming: Dict[str, int] = {}
        deps: Dict[str, List[Gate]] = {}
        outputs = {gate.output for gate in self.gates}
        for gate in self.gates:
            incoming[gate.output] = sum(1 for inp in gate.inputs if inp in outputs)
            for inp in gate.inputs:
                deps.setdefault(inp, []).append(gate)

        ready = [gate for gate in self.gates if incoming[gate.output] == 0]
        order: List[Gate] = []

        while ready:
            gate = ready.pop()
            order.append(gate)
            for nxt in deps.get(gate.output, []):
                incoming[nxt.output] -= 1
                if incoming[nxt.output] == 0:
                    ready.append(nxt)

        if len(order) != len(self.gates):
            raise CircuitError("Sequential or cyclic dependencies detected; only combinational circuits supported.")
        return order

    def reset(self) -> None:
        for net in self.values:
            self.values[net] = Logic5.X

    def assign(self, net: str, value: Logic5, stack: List[tuple[str, Logic5]]) -> bool:
        prev = self.values.get(net, Logic5.X)
        if prev is value:
            return True
        if prev is not Logic5.X and value is not Logic5.X and prev is not value:
            return False
        stack.append((net, prev))
        self.values[net] = value
        return True

    def evaluate_gate(self, gate: Gate, fault: Optional["Fault"]) -> Logic5:
        inputs = [self.values[inp] for inp in gate.inputs]
        result: GateEvalResult = evaluate_gate(gate.gate_type, inputs)
        if fault is not None and gate.output == fault.net:
            stuck = 0 if fault.stuck_at == 0 else 1
            if result.good is None:
                return Logic5.X
            return Logic5.from_pair(result.good, stuck)
        return result.to_logic5()

    def fanin_values(self, gate: Gate) -> List[Logic5]:
        return [self.values[name] for name in gate.inputs]

    def set_pi(self, name: str, value: Logic5, stack: List[tuple[str, Logic5]]) -> bool:
        if name not in self.primary_inputs:
            raise CircuitError(f"Unknown primary input: {name}")
        return self.assign(name, value, stack)

    def get_value(self, net: str) -> Logic5:
        return self.values.get(net, Logic5.X)


from .fault import Fault  # noqa: E402  # avoid circular import at runtime
