"""Summarize JSON outputs from example runs."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    outputs = []
    for path in sorted(Path("artifacts").glob("*.json")):
        if path.name == "results.json":
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        outputs.append(
            {
                "file": path.name,
                "status": data["status"],
                "test_vector": ", ".join(f"{k}={v}" for k, v in data["test_vector"].items()),
            }
        )
    Path("artifacts/results.json").write_text(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
