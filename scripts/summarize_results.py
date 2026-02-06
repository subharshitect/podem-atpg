#!/usr/bin/env python3
"""Summarize JSON outputs from example runs."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    outputs = []
    for path in sorted(artifacts.glob("*.json")):
        data = json.loads(path.read_text())
        tv = data.get("test_vector", {})
        outputs.append(
            {
                "file": path.name,
                "status": data.get("status"),
                "test_vector": ", ".join(f"{k}={v}" for k, v in tv.items()),
            }
        )

    (artifacts / "results.json").write_text(json.dumps(outputs, indent=2) + "\n")


if __name__ == "__main__":
    main()
