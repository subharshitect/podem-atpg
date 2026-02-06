"""Run ATPG for all gate-output stuck-at faults in given .bench netlists."""

from __future__ import annotations

import argparse
from pathlib import Path

from atpg.fault import Fault  # noqa: E402
from atpg.parser_bench import parse_bench_file  # noqa: E402
from atpg.podem import podem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlists", nargs="+", required=True, help="List of .bench files")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    ap.add_argument("--timeout_s", type=float, default=None)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for netlist_path in args.netlists:
        circuit = parse_bench_file(netlist_path)
        netlist_stem = Path(netlist_path).stem

        gate_outputs = sorted(circuit.gate_map.keys())
        for net in gate_outputs:
            for sa in (0, 1):
                fault = Fault(net=net, stuck_at=sa)
                result = podem(
                    circuit=parse_bench_file(netlist_path),
                    fault=fault,
                    timeout_s=args.timeout_s,
                    max_depth=args.max_depth,
                    verbose=args.verbose,
                )
                fname = f"{netlist_stem}_{net}_SA{sa}.json"
                (outdir / fname).write_text(result.to_json())


if __name__ == "__main__":
    main()
