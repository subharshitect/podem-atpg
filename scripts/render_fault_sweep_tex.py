from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _escape_tex(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def main() -> None:
    artifacts = Path("artifacts")
    out_dir = Path("report/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(artifacts.glob("*.json")):
        if p.name == "results.json":
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rows.append((p.name, data))

    if not rows:
        raise SystemExit("No artifacts/*.json found. Run make examples first.")

    total = len(rows)
    detected = sum(1 for _, d in rows if d.get("status") == "DETECTED")
    untestable = sum(1 for _, d in rows if d.get("status") == "UNTESTABLE")
    aborted = sum(1 for _, d in rows if d.get("status") == "ABORTED")

    runtimes = [_as_float(d.get("runtime_ms")) for _, d in rows if d.get("runtime_ms") is not None]
    depths = [_as_float(d.get("depth")) for _, d in rows]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary_lines = [
        "\\subsection{Fault sweep summary}",
        "\\begin{itemize}",
        f"\\item Total faults evaluated: {total}",
        f"\\item Detected: {detected} ({(100.0 * detected / total):.1f}\\%)",
        f"\\item Untestable: {untestable} ({(100.0 * untestable / total):.1f}\\%)",
        f"\\item Aborted: {aborted} ({(100.0 * aborted / total):.1f}\\%)",
        f"\\item Average search depth: {avg(depths):.2f}",
        f"\\item Average runtime (ms): {avg(runtimes):.2f}",
        "\\end{itemize}",
    ]
    (out_dir / "fault_sweep_summary.tex").write_text("\n".join(summary_lines))

    # Big table (trim to keep PDF sane, but still large)
    # We keep all rows, but you can cap later if needed.
    table = []
    table.append("\\subsection{Fault sweep detailed results}")
    table.append("\\begin{footnotesize}")
    table.append("\\begin{longtable}{llllrrr}")
    table.append("\\caption{Fault sweep results across all gate-output stuck-at faults in the provided netlists.}\\\\")
    table.append("\\toprule")
    table.append("Artifact & Status & Test vector & PO obs & Depth & Runtime(ms) & Backtracks\\\\")
    table.append("\\midrule")
    table.append("\\endfirsthead")
    table.append("\\toprule")
    table.append("Artifact & Status & Test vector & PO obs & Depth & Runtime(ms) & Backtracks\\\\")
    table.append("\\midrule")
    table.append("\\endhead")
    table.append("\\midrule")
    table.append("\\multicolumn{7}{r}{Continued on next page}\\\\")
    table.append("\\endfoot")
    table.append("\\bottomrule")
    table.append("\\endlastfoot")

    for fname, d in rows:
        status = str(d.get("status", ""))
        tv = d.get("test_vector", {})
        po = d.get("po_observations", {})
        depth = int(_as_float(d.get("depth")))
        runtime_ms = _as_float(d.get("runtime_ms"))
        backtracks = int(_as_float(d.get("backtracks")))

        tv_s = ", ".join(f"{k}={v}" for k, v in tv.items()) if isinstance(tv, dict) else str(tv)
        po_s = ", ".join(f"{k}={v}" for k, v in po.items()) if isinstance(po, dict) else str(po)

        row = (
            f"{_escape_tex(fname)} & "
            f"{_escape_tex(status)} & "
            f"{_escape_tex(tv_s)} & "
            f"{_escape_tex(po_s)} & "
            f"{depth} & "
            f"{runtime_ms:.2f} & "
            f"{backtracks}\\\\"
        )
        table.append(row)

    table.append("\\end{longtable}")
    table.append("\\end{footnotesize}")

    (out_dir / "fault_sweep_table.tex").write_text("\n".join(table))


if __name__ == "__main__":
    main()
