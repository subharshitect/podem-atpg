"""Utility helpers for ATPG."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .logic5 import Logic5


@dataclass
class PodemResult:
    status: str
    test_vector: Dict[str, Logic5]
    po_observations: Dict[str, Logic5]
    depth: int
    reason: Optional[str] = None

    def to_json(self) -> str:
        payload = {
            "status": self.status,
            "test_vector": {k: v.to_char() for k, v in self.test_vector.items()},
            "po_observations": {k: v.to_char() for k, v in self.po_observations.items()},
            "depth": self.depth,
            "reason": self.reason,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def vector_from_values(values: Dict[str, Logic5], primary_inputs: List[str]) -> Dict[str, Logic5]:
    return {pi: values.get(pi, Logic5.X) for pi in primary_inputs}
