"""Render LaTeX table of results for the report."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_path = Path("artifacts/results.json")
    if not results_path.exists():
        raise SystemExit("Run 'make examples' first to generate artifacts/results.json")

    rows = json.loads(results_path.read_text())
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Netlist & Net & Fault & Status & Depth \\\\",
        "\\midrule",
    ]
    for r in rows:
        netlist = r.get("netlist", "?")
        net = r.get("net", "?")
        fault = r.get("fault", "?")
        status = r.get("status", "?")
        depth = r.get("depth", "")
        lines.append(f"{netlist} & {net} & {fault} & {status} & {depth} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{ATPG results across example netlists and gate-output stuck-at faults.}",
            "\\label{tab:results}",
            "\\end{table}",
        ]
    )
    Path("report/artifacts/results.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
