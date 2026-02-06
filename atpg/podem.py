"""PODEM algorithm implementation."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .circuit import Circuit
from .fault import Fault
from .logic5 import Logic5, controlling_value, invert_value, non_controlling_value
from .utils import PodemResult, vector_from_values


class PodemAbort(Exception):
    pass


def check_test(circuit: Circuit) -> bool:
    return any(circuit.get_value(po) in (Logic5.D, Logic5.DBAR) for po in circuit.primary_outputs)


def compute_d_frontier(circuit: Circuit) -> List:
    frontier = []
    for gate in circuit.topo_order:
        if circuit.get_value(gate.output) is not Logic5.X:
            continue
        if any(val in (Logic5.D, Logic5.DBAR) for val in circuit.fanin_values(gate)):
            frontier.append(gate)
    return frontier


def objective(circuit: Circuit, fault: Fault) -> Tuple[str, Logic5]:
    fault_value = circuit.get_value(fault.net)
    if fault_value in (Logic5.X, Logic5.ZERO, Logic5.ONE):
        desired = Logic5.ONE if fault.stuck_at == 0 else Logic5.ZERO
        return fault.net, desired

    frontier = compute_d_frontier(circuit)
    if not frontier:
        raise PodemAbort("D-frontier empty")
    gate = frontier[0]
    target_value = non_controlling_value(gate.gate_type)
    for inp in gate.inputs:
        if circuit.get_value(inp) is Logic5.X:
            return inp, target_value or Logic5.X
    raise PodemAbort("No X input in D-frontier gate")


def backtrace(circuit: Circuit, net_name: str, desired_value: Logic5) -> Tuple[str, Logic5]:
    if net_name in circuit.primary_inputs:
        return net_name, desired_value
    gate = circuit.gate_map.get(net_name)
    if gate is None:
        return net_name, desired_value

    gate_type = gate.gate_type.upper()
    inverted = gate_type in {"NAND", "NOR", "NOT", "XNOR"}
    base_type = gate_type
    if gate_type in {"NAND", "AND"}:
        base_type = "AND"
    elif gate_type in {"NOR", "OR"}:
        base_type = "OR"
    elif gate_type in {"XNOR", "XOR"}:
        base_type = "XOR"

    target = invert_value(desired_value) if inverted else desired_value

    if base_type == "AND":
        desired_in = Logic5.ONE if target is Logic5.ONE else Logic5.ZERO
    elif base_type == "OR":
        desired_in = Logic5.ZERO if target is Logic5.ZERO else Logic5.ONE
    elif base_type == "XOR":
        desired_in = target
    else:
        desired_in = target

    for inp in gate.inputs:
        if circuit.get_value(inp) is Logic5.X:
            return backtrace(circuit, inp, desired_in)

    return backtrace(circuit, gate.inputs[0], desired_in)


def imply(circuit: Circuit, pi_name: str, pi_value: Logic5, fault: Fault, stack: List[tuple[str, Logic5]]) -> bool:
    if not circuit.set_pi(pi_name, pi_value, stack):
        return False
    for gate in circuit.topo_order:
        new_value = circuit.evaluate_gate(gate, fault)
        if not circuit.assign(gate.output, new_value, stack):
            return False
    return True


def undo_to_level(stack: List[tuple[str, Logic5]], circuit: Circuit, level: int) -> None:
    while len(stack) > level:
        net, prev = stack.pop()
        circuit.values[net] = prev


def podem(
    circuit: Circuit,
    fault: Fault,
    timeout_s: Optional[float] = None,
    max_depth: Optional[int] = None,
    verbose: bool = False,
) -> PodemResult:
    start = time.monotonic()
    stack: List[tuple[str, Logic5]] = []
    circuit.reset()

    def timed_out() -> bool:
        return timeout_s is not None and (time.monotonic() - start) > timeout_s

    def recurse(depth: int) -> PodemResult:
        if timed_out():
            return PodemResult(status="ABORTED", test_vector={}, po_observations={}, depth=depth, reason="timeout")
        if max_depth is not None and depth > max_depth:
            return PodemResult(status="ABORTED", test_vector={}, po_observations={}, depth=depth, reason="max_depth")

        if check_test(circuit):
            test_vector = vector_from_values(circuit.values, circuit.primary_inputs)
            po_obs = {po: circuit.get_value(po) for po in circuit.primary_outputs}
            return PodemResult(status="DETECTED", test_vector=test_vector, po_observations=po_obs, depth=depth)

        fault_value = circuit.get_value(fault.net)
        if fault_value in (Logic5.D, Logic5.DBAR) and not compute_d_frontier(circuit):
            return PodemResult(status="UNTESTABLE", test_vector={}, po_observations={}, depth=depth)

        try:
            obj_net, obj_val = objective(circuit, fault)
        except PodemAbort:
            return PodemResult(status="UNTESTABLE", test_vector={}, po_observations={}, depth=depth)

        pi_name, pi_val = backtrace(circuit, obj_net, obj_val)
        for trial in (pi_val, invert_value(pi_val)):
            level = len(stack)
            if not imply(circuit, pi_name, trial, fault, stack):
                undo_to_level(stack, circuit, level)
                continue
            if len(stack) == level:
                continue
            if verbose:
                pass
            result = recurse(depth + 1)
            if result.status == "DETECTED":
                return result
            if result.status == "ABORTED":
                return result
            undo_to_level(stack, circuit, level)
        return PodemResult(status="UNTESTABLE", test_vector={}, po_observations={}, depth=depth)

    result = recurse(0)
    if result.status != "DETECTED":
        result.test_vector = vector_from_values(circuit.values, circuit.primary_inputs)
        result.po_observations = {po: circuit.get_value(po) for po in circuit.primary_outputs}
    return result
