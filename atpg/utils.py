from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from .logic5 import Logic5


@dataclass
class PodemResult:
    status: str
    test_vector: Dict[str, Logic5]
    po_observations: Dict[str, Logic5]
    depth: int
    reason: Optional[str] = None
    trace: Optional[list[str]] = None  # <-- ADD THIS

    # new fields
    runtime_ms: Optional[float] = None
    decisions: Optional[int] = None
    implications: Optional[int] = None
    backtracks: Optional[int] = None

    def to_json(self) -> str:
        payload = {
            "status": self.status,
            "test_vector": {k: v.to_char() for k, v in self.test_vector.items()},
            "po_observations": {k: v.to_char() for k, v in self.po_observations.items()},
            "depth": self.depth,
            "reason": self.reason,
            "runtime_ms": self.runtime_ms,
            "decisions": self.decisions,
            "implications": self.implications,
            "backtracks": self.backtracks,
            "trace": self.trace or [],  # <-- ADD THIS
        }
        return json.dumps(payload, indent=2, sort_keys=True)
