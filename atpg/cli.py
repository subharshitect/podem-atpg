"""Command line interface for ATPG."""

from __future__ import annotations

import argparse
from typing import Any

from .fault import Fault, FaultError
from .parser_bench import BenchParseError, parse_bench_file
from .podem import podem


def _run_command(args: argparse.Namespace) -> int:
    try:
        circuit = parse_bench_file(args.netlist)
        fault = Fault.parse(args.fault)
    except (BenchParseError, FaultError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    if fault.net not in circuit.gate_map:
        print("Error: Fault must target a gate output net (not a primary input).")
        return 2

    result = podem(
        circuit,
        fault,
        timeout_s=args.timeout_s,
        max_depth=args.max_depth,
        verbose=args.verbose,
    )

    if args.verbose and result.trace and not args.json:
        for line in result.trace:
            print(line)
            print()

    if args.json:
        print(result.to_json())
        return 0

    print(f"Status: {result.status}")
    if result.reason:
        print(f"Reason: {result.reason}")
    print("Test vector:")
    for name, value in result.test_vector.items():
        print(f"  {name}: {value.to_char()}")
    if result.po_observations:
        print("PO observations:")
        for name, value in result.po_observations.items():
            print(f"  {name}: {value.to_char()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PODEM-based ATPG for combinational circuits")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run PODEM ATPG")
    run_parser.add_argument("--netlist", required=True, help="Path to .bench netlist")
    run_parser.add_argument("--fault", required=True, help="Fault specification <net>/SA0 or <net>/SA1")
    run_parser.add_argument("--timeout_s", type=float, default=None, help="Timeout in seconds")
    run_parser.add_argument("--max_depth", type=int, default=None, help="Maximum recursion depth")
    run_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    run_parser.add_argument("--json", action="store_true", help="Output JSON")
    run_parser.set_defaults(func=_run_command)

    return parser


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
