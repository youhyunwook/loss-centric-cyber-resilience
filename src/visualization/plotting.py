"""
src/visualization/plotting.py
------------------------------
Shared, publication-quality plot utilities. Every figure produced by the
experiments goes through one of these helpers so the manuscript figures
have a consistent style.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def set_paper_style() -> None:
    """Apply the IEEE Access-friendly Matplotlib style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ----------------------------------------------------------------------------
# Specific plots
# ----------------------------------------------------------------------------
def plot_scatter_correlation(
    df: pd.DataFrame,
    x: str = "I",
    y: str = "ER",
    rho: float | None = None,
    pvalue: float | None = None,
    title: str = "",
    xlabel: str = "Asset importance I(n)",
    ylabel: str = "Attack exposure ER(n)",
):
    """Scatter plot for the I vs ER correlation analysis (R2 #5)."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(df[x], df[y], alpha=0.4, s=14, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rho is not None and pvalue is not None:
        ax.set_title(f"{title}\nSpearman ρ = {rho:.4f} (p = {pvalue:.3g})")
    elif title:
        ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig


def plot_grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    category_col: str,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
):
    """Grouped bar chart, e.g. V(S) per (scenario, aggregation method)."""
    set_paper_style()
    pivot = df.pivot(index=group_col, columns=category_col, values=value_col)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel or value_col)
    if title:
        ax.set_title(title)
    ax.legend(title=category_col, fontsize=8)
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_sensitivity_lines(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
):
    """Line plot with markers for sensitivity sweeps (tau, K, D(c) ...)."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    if hue:
        for key, grp in df.groupby(hue):
            ax.plot(grp[x], grp[y], marker="o", linewidth=1.5, label=str(key))
        ax.legend(title=hue, fontsize=8)
    else:
        ax.plot(df[x], df[y], marker="o", linewidth=1.6, color="#1f4e79")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_centrality_heatmap(
    matrix: pd.DataFrame,
    title: str = "Centrality ranking agreement (Spearman ρ)",
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
):
    """Heatmap of pairwise Spearman ρ across centrality methods (R3 #5)."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ρ")
    fig.tight_layout()
    return fig


def plot_mtta_comparison(
    comparison_df: pd.DataFrame,
    title: str = "MTTA: framework vs. manual baselines",
):
    """Horizontal bar of baseline_seconds with framework_seconds reference.

    Expects columns: ``baseline``, ``baseline_seconds``,
    ``framework_seconds`` (constant across rows is fine).
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(comparison_df))
    ax.barh(
        y, comparison_df["baseline_seconds"],
        color="#9ec5e8", edgecolor="black", linewidth=0.4,
        label="Manual baseline (s)",
    )
    framework_s = comparison_df["framework_seconds"].iloc[0]
    ax.axvline(
        framework_s, color="#c0392b", linestyle="--", linewidth=1.5,
        label=f"Framework: {framework_s:.2f} s",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(comparison_df["baseline"])
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds, log scale)")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
