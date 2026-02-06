"""Fault model definitions."""

from __future__ import annotations

from dataclasses import dataclass


class FaultError(Exception):
    pass


@dataclass(frozen=True)
class Fault:
    net: str
    stuck_at: int

    @classmethod
    def parse(cls, text: str) -> "Fault":
        try:
            net, sa = text.split("/")
        except ValueError as exc:
            raise FaultError("Fault format must be <net>/SA0 or <net>/SA1") from exc
        sa = sa.upper()
        if sa not in {"SA0", "SA1"}:
            raise FaultError("Fault format must be <net>/SA0 or <net>/SA1")
        return cls(net=net.strip(), stuck_at=0 if sa == "SA0" else 1)
