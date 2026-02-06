#!/usr/bin/env python3
"""Render LaTeX table of results for the report."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_path = Path("artifacts/results.json")
    if not results_path.exists():
        raise SystemExit("artifacts/results.json missing. Run `make examples` first.")

    rows = json.loads(results_path.read_text())

    out_dir = Path("report/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Netlist & Fault & Status & Test Vector \\\\",
        "\\midrule",
    ]
    for row in rows:
        parts = row["file"].replace(".json", "").split("_")
        netlist = parts[0]
        fault = f"{parts[1]}/{parts[2]}"
        status = row.get("status", "")
        vector = row.get("test_vector", "")
        netlist = netlist.replace("_", "\\_")
        fault = fault.replace("_", "\\_")
        vector = vector.replace("_", "\\_")
        lines.append(f"{netlist} & {fault} & {status} & {vector} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Example ATPG results.}",
            "\\label{tab:results}",
            "\\end{table}",
        ]
    )
    (out_dir / "results.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
