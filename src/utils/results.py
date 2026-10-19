"""
src/utils/results.py
---------------------
Standardised per-experiment output layout.

Every experiment writes into:

    results/<exp_name>/
        tables/   *.csv     (every dataframe save)
        figures/  *.png     (every plot)
        summary.json        (machine-readable scalar summary)
        meta.yaml           (run metadata: config snapshot, timestamp,
                             scenario list, dataset version)

This is the structure consumed by ``scripts/render_results.py`` and the
manuscript's table/figure auto-generation pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .logging import get_logger

log = get_logger(__name__)


class ExperimentOutput:
    """Per-experiment output handler.

    Usage
    -----
    >>> exp = ExperimentOutput("exp15_cvss_live_aggregation_sensitivity", config=config)
    >>> exp.save_table(df, "v_s_per_method.csv")
    >>> exp.save_figure(fig, "scatter_S1.png")
    >>> exp.save_summary({"best_method": "max", "n_scenarios": 5})
    >>> exp.close()                  # writes meta.yaml
    """

    TABLES_DIR = "tables"
    FIGURES_DIR = "figures"
    SUMMARY_FILE = "summary.json"
    META_FILE = "meta.yaml"

    def __init__(
        self,
        name: str,
        results_root: str | Path = "results",
        config: dict | None = None,
        scenarios: list[str] | None = None,
    ) -> None:
        self.name = name
        self.root = Path(results_root) / name
        self.tables = self.root / self.TABLES_DIR
        self.figures = self.root / self.FIGURES_DIR
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

        self._config_snapshot = config
        self._scenarios = scenarios or []
        self._summary: dict[str, Any] = {}
        self._tables_written: list[str] = []
        self._figures_written: list[str] = []
        self._t0 = time.time()

    # --------------------------------------------------------------- save API
    def save_table(
        self,
        df: pd.DataFrame,
        name: str,
        index: bool = False,
        also_xlsx: bool = False,
    ) -> Path:
        """Save a DataFrame into the experiment's ``tables/`` directory.

        Parameters
        ----------
        df : DataFrame
        name : str
            Filename ending in ``.csv``. ``.xlsx`` is added automatically
            if ``also_xlsx`` is True.
        index : bool
        also_xlsx : bool
            Also write a matching ``.xlsx`` for IEEE-style table import.
        """
        if not name.endswith(".csv"):
            name = f"{name}.csv"
        path = self.tables / name
        df.to_csv(path, index=index)
        self._tables_written.append(name)

        if also_xlsx:
            xlsx_path = self.tables / name.replace(".csv", ".xlsx")
            try:
                df.to_excel(xlsx_path, index=index)
                self._tables_written.append(xlsx_path.name)
            except ImportError:
                log.warning("xlsx requested but openpyxl not installed.")
        return path

    def save_figure(
        self,
        fig: Any,
        name: str,
        dpi: int = 200,
        also_pdf: bool = False,
    ) -> Path:
        """Save a matplotlib figure to ``figures/``.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        name : str
            Filename ending in ``.png``.
        dpi : int, default 200
        also_pdf : bool
            Save matching ``.pdf`` for vector publication output.
        """
        if not name.endswith(".png"):
            name = f"{name}.png"
        path = self.figures / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        self._figures_written.append(name)
        if also_pdf:
            pdf_path = self.figures / name.replace(".png", ".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            self._figures_written.append(pdf_path.name)
        return path

    def add_summary(self, key: str, value: Any) -> None:
        """Add a scalar/dict to the experiment's summary.json payload."""
        self._summary[key] = value

    def save_summary(self, payload: dict | None = None) -> Path:
        """Merge ``payload`` into the running summary and write it."""
        if payload:
            self._summary.update(payload)
        path = self.root / self.SUMMARY_FILE
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._summary, fh, indent=2, default=str)
        return path

    def close(self) -> Path:
        """Write meta.yaml capturing run provenance."""
        # Make sure summary.json exists even if user forgot to call.
        if self._summary:
            self.save_summary()
        meta = {
            "experiment": self.name,
            "wallclock_seconds": time.time() - self._t0,
            "scenarios": self._scenarios,
            "tables": self._tables_written,
            "figures": self._figures_written,
            "dataset_version": (
                self._config_snapshot.get("dataset", {}).get("version")
                if self._config_snapshot else None
            ),
            "config_snapshot": self._config_snapshot,
        }
        path = self.root / self.META_FILE
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)
        log.info("Experiment %s closed -> %s", self.name, self.root)
        return path
