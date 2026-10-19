"""
src/countermeasure/d3fend_mapper.py
------------------------------------
Scenario-specific countermeasure mapping (notebook-aligned).

Each scenario (S1, S2) has an explicit list of candidate
countermeasures with operation + parameters. This matches the
manuscript Table IX/X presentation.

Loads from ``config/countermeasures.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..utils.io import load_config
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CountermeasureSpec:
    """Single countermeasure entry from the catalogue."""
    cid: str
    name: str
    d3fend: str                          # e.g. "D3-NTF"
    tier: str                            # high / medium / low
    op: str                              # SET_V / SCALE_E_TARGET / ...
    params: dict                         # operation-specific parameters
    intervention_type: str               # I / E / V (for legacy compat)
    M: int = 0                           # manpower
    T: int = 0                           # technology
    C: int = 0                           # cost
    rubric_note: str = ""

    @property
    def difficulty_components(self) -> dict[str, int]:
        """Return M/T/C as a dict for legacy difficulty.compute()."""
        return {"manpower": self.M, "technology": self.T, "cost": self.C}


class D3FENDMapper:
    """Resolve scenario IDs into countermeasure spec lists."""

    def __init__(self, countermeasures_yaml: str):
        cfg = load_config(countermeasures_yaml)

        # Notebook-aligned: scenario_countermeasures
        self._by_scenario: dict[str, list[CountermeasureSpec]] = {}
        for sid, entries in cfg.get("scenario_countermeasures", {}).items():
            specs = []
            for e in entries:
                specs.append(CountermeasureSpec(
                    cid=e["cid"],
                    name=e["name"],
                    d3fend=e.get("d3fend", ""),
                    tier=e.get("tier", "medium"),
                    op=e.get("op", "SCALE_E_TARGET"),
                    params=dict(e.get("params", {})),
                    intervention_type=e.get("intervention_type", "E"),
                    M=int(e.get("M", 0)),
                    T=int(e.get("T", 0)),
                    C=int(e.get("C", 0)),
                    rubric_note=e.get("rubric_note", ""),
                ))
            self._by_scenario[sid] = specs

        log.info(
            "D3FENDMapper loaded: %d scenarios, %d total countermeasures",
            len(self._by_scenario),
            sum(len(v) for v in self._by_scenario.values()),
        )

    # ------------------------------------------------------------------ API
    def for_scenario(self, scenario_id: str) -> list[CountermeasureSpec]:
        """Return the candidate list for a scenario id (S1, S2, ...)."""
        specs = self._by_scenario.get(scenario_id, [])
        if not specs:
            log.warning("No countermeasures defined for scenario %s", scenario_id)
        return specs

    def candidates(self, scenario_id_or_technique: str) -> list[CountermeasureSpec]:
        """Backward-compat alias used by older experiment code paths.

        Accepts a scenario id (preferred). If it does not match any
        scenario id, returns [] and logs a warning.
        """
        return self.for_scenario(scenario_id_or_technique)

    def all_scenarios(self) -> Iterable[str]:
        return self._by_scenario.keys()

    def get(self, cid: str) -> CountermeasureSpec | None:
        for specs in self._by_scenario.values():
            for s in specs:
                if s.cid == cid:
                    return s
        return None
