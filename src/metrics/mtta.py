"""
src/metrics/mtta.py
-------------------
Mean Time To Awareness measurement.

Two parts:
  1. ``measure_mtta`` — wall-clock the pipeline call.
  2. ``mtta_improvement_table`` — produce a Multi-Baseline comparison
     table replacing the single 300 s baseline (Reviewers 2, 3).
"""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


def measure_mtta(
    pipeline_fn: Callable[[], object],
    n_runs: int = 5,
    warmup: int = 1,
) -> dict[str, float]:
    """Wall-clock a pipeline callable.

    Parameters
    ----------
    pipeline_fn : zero-arg callable
        Pipeline to time.
    n_runs : int, default 5
    warmup : int, default 1
        Discarded warm-up runs (e.g., to amortise import / cache cost).

    Returns
    -------
    dict
        ``{"mean_seconds", "std_seconds", "min_seconds", "max_seconds", "n_runs"}``.
    """
    for _ in range(warmup):
        pipeline_fn()

    times: list[float] = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        pipeline_fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        log.info("MTTA run %d/%d: %.4fs", i + 1, n_runs, elapsed)

    s = pd.Series(times)
    return {
        "mean_seconds": float(s.mean()),
        "std_seconds": float(s.std(ddof=0)),
        "min_seconds": float(s.min()),
        "max_seconds": float(s.max()),
        "n_runs": n_runs,
    }


def mtta_improvement_table(
    measured_seconds: float,
    baselines: list[dict],
) -> pd.DataFrame:
    """Build the Multi-Baseline MTTA comparison table.

    Replaces the single 300 s baseline with multiple, contextually
    distinct comparators (Reviewer 2 / 3).

    Parameters
    ----------
    measured_seconds : float
        Pipeline mean MTTA.
    baselines : list of dict
        Each dict has keys: name, seconds, source, relevance.
        See ``config['mtta']['baselines']``.

    Returns
    -------
    DataFrame
        Columns: baseline, baseline_seconds, source, relevance,
        framework_seconds, improvement_pct, speedup_x.
    """
    rows = []
    for b in baselines:
        b_s = float(b["seconds"])
        improvement_pct = (1.0 - measured_seconds / b_s) * 100.0
        speedup = b_s / measured_seconds if measured_seconds > 0 else float("inf")
        rows.append({
            "baseline": b["name"],
            "baseline_seconds": b_s,
            "source": b.get("source", ""),
            "relevance": b.get("relevance", ""),
            "framework_seconds": measured_seconds,
            "improvement_pct": improvement_pct,
            "speedup_x": speedup,
        })
    return pd.DataFrame(rows)
