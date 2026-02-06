"""Summarize JSON outputs from example runs into JSON, CSV, and summary stats."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _parse_filename(name: str) -> tuple[str, str, str]:
    # expected: <netlist>_<net>_SA0.json or SA1.json
    stem = name.removesuffix(".json")
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "?", "?"
    netlist = parts[0]
    fault = f"{parts[-2]}/{parts[-1]}"
    net = "_".join(parts[1:-2])
    return netlist, net, fault


def main() -> None:
    outdir = Path("artifacts")
    rows = []

    for path in sorted(outdir.glob("*.json")):
        if path.name in {"results.json", "summary.json"}:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue

        netlist, net, fault = _parse_filename(path.name)
        status = data.get("status", "UNKNOWN")
        depth = data.get("depth", None)
        tv = data.get("test_vector", {})
        tv_str = ", ".join(f"{k}={v}" for k, v in tv.items())

        rows.append(
            {
                "file": path.name,
                "netlist": netlist,
                "net": net,
                "fault": fault,
                "status": status,
                "depth": depth,
                "test_vector": tv_str,
            }
        )

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    with (outdir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "netlist", "net", "fault", "status", "depth", "test_vector"],
        )
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    by_netlist: dict[str, dict[str, int]] = {}

    for r in rows:
        s = r["status"]
        counts[s] = counts.get(s, 0) + 1
        nl = r["netlist"]
        by_netlist.setdefault(nl, {})
        by_netlist[nl][s] = by_netlist[nl].get(s, 0) + 1

    summary = {
        "total_cases": len(rows),
        "status_counts": counts,
        "by_netlist": by_netlist,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
