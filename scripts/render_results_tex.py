"""Render LaTeX table of results for the report."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_path = Path("artifacts/results.json")
    if not results_path.exists():
        raise SystemExit("Run 'make examples' first to generate results.json")

    rows = json.loads(results_path.read_text())
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
        status = row["status"]
        vector = row["test_vector"]
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
    Path("report/artifacts/results.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
