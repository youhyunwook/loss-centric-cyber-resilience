"""
src/countermeasure/difficulty.py
---------------------------------
Implementation difficulty D(c) = C(c) + T(c) + M(c).

Each component is binary (0/1) per the rubric in
``config/countermeasures.yaml``. A floor of 1 prevents
divide-by-zero in Score = dLoss / D(c).

Manuscript Sec. IV.C (revised with rubric for Reviewer 1, 3).
"""

from __future__ import annotations

from .d3fend_mapper import CountermeasureSpec


def compute_difficulty(
    spec: CountermeasureSpec,
    components: list[str] | None = None,
    perturbation_pct: float = 0.0,
    min_value: int = 1,
) -> float:
    """Compute D(c) with optional perturbation for sensitivity analysis.

    Parameters
    ----------
    spec : CountermeasureSpec
    components : list of str, optional
        Component keys to sum. Defaults to all keys in
        ``spec.difficulty_components``.
    perturbation_pct : float
        Multiplicative perturbation in [-1.0, +1.0]
        (e.g., 0.25 -> +25%; -0.25 -> -25%).
        Used by the D(c) sensitivity experiment.
    min_value : int, default 1
        Floor — D(c) is clamped to >= ``min_value``.

    Returns
    -------
    float
        Perturbed difficulty value.
    """
    comps = components or list(spec.difficulty_components.keys())
    base = float(sum(spec.difficulty_components.get(c, 0) for c in comps))
    perturbed = base * (1.0 + perturbation_pct)
    return max(float(min_value), perturbed)
