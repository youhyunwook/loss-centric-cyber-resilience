"""
src/preprocessing/data_loader.py
---------------------------------
Load raw CIC-IDS-2018 CSVs and scenario definitions.

The CIC-IDS-2018 dataset distributes one CSV per capture day. We accept
either:
  * a single CSV path  (one file)
  * a directory path   (will glob ``*.csv``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.io import load_config
from ..utils.logging import get_logger

log = get_logger(__name__)


# CIC-IDS-2018 columns we use downstream.
# The dataset has many flow features; we only need IPs, ports, label.
REQUIRED_COLS = [
    "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Protocol", "Timestamp", "Label",
]


def _read_one(path: Path) -> pd.DataFrame:
    """Read one CIC-IDS-2018 CSV with column normalisation."""
    df = pd.read_csv(path, low_memory=False)
    # Column names in CIC-IDS-2018 sometimes have leading/trailing whitespace.
    df.columns = [c.strip() for c in df.columns]
    # Some releases use "Src IP" vs "Source IP" — normalise.
    rename_map = {
        "Source IP": "Src IP",
        "Source Port": "Src Port",
        "Destination IP": "Dst IP",
        "Destination Port": "Dst Port",
    }
    df = df.rename(columns=rename_map)
    return df


def load_cic_ids_2018(
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load CIC-IDS-2018 flow records.

    Parameters
    ----------
    path : str or Path
        Either a single CSV file or a directory containing day-level CSVs.
    columns : iterable of str, optional
        Subset of columns to keep. Defaults to ``REQUIRED_COLS``.

    Returns
    -------
    pandas.DataFrame
        Concatenated flow records.
    """
    path = Path(path)
    columns = list(columns) if columns is not None else list(REQUIRED_COLS)

    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs found in {path}")
        log.info("Loading %d CSV files from %s", len(csv_files), path)
        frames = [_read_one(p) for p in csv_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        log.info("Loading single CSV %s", path)
        df = _read_one(path)

    # Keep only requested columns that exist (CIC files vary slightly).
    keep = [c for c in columns if c in df.columns]
    missing = set(columns) - set(keep)
    if missing:
        log.warning("Missing columns dropped: %s", sorted(missing))
    df = df[keep].copy()

    log.info("Loaded %d flow records, %d cols", len(df), df.shape[1])
    return df


def load_scenarios(
    path: str | Path,
    only_active: bool = True,
) -> list[dict]:
    """Load attack scenario definitions from ``scenarios.yaml``.

    Parameters
    ----------
    path : str or Path
        YAML file produced by ``config/scenarios.yaml``.
    only_active : bool, default True
        If True, drop scenarios with ``active: false`` (kept in YAML for
        future-work extensibility, not run in current revision).

    Returns
    -------
    list[dict]
        One dict per scenario with keys: id, name, cic_labels,
        mitre_tactic, mitre_technique, capec, cwe, representative_cve, ...
    """
    cfg = load_config(path)
    scenarios = cfg.get("scenarios", [])
    n_total = len(scenarios)
    if only_active:
        scenarios = [s for s in scenarios if s.get("active", True)]
        log.info(
            "Loaded %d/%d active scenarios from %s",
            len(scenarios), n_total, path,
        )
    else:
        log.info("Loaded %d scenarios (incl. inactive) from %s", n_total, path)
    return scenarios
