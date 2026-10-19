"""
src/utils/io.py
---------------
YAML config loading and dataframe IO helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Parameters
    ----------
    path : str or Path
        Path to a ``.yaml`` file.

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge multiple config dicts; later dicts override earlier ones."""
    out: dict[str, Any] = {}
    for cfg in configs:
        if cfg:
            out.update(cfg)
    return out


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def load_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV into a DataFrame with sensible defaults."""
    return pd.read_csv(path, **kwargs)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
