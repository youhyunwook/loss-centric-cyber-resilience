"""
src/preprocessing/flow_filter.py
---------------------------------
Separate benign vs. attack flows and filter to a specific scenario.

The communication graph G_normal is built from BENIGN traffic only,
to avoid scan/flood traffic distorting structural centrality
(manuscript Sec. III.A and Reviewer 3 #5 / #6).

Internal-only filtering (RFC1918) is also provided to match the original
manuscript pipeline, which restricted G_normal to internal-internal
flows. This rejects egress to public CDNs, dropbox endpoints, etc., that
otherwise inflate the graph with structurally uninformative leaves.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


# RFC 1918 private ranges
_RFC1918_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]


def _networks(networks: Iterable[str] | None = None) -> list[ipaddress._BaseNetwork]:
    return [ipaddress.ip_network(n) for n in (networks or [])] or _RFC1918_NETS


def _is_internal_ip(ip: str) -> bool:
    """Return True iff ``ip`` parses as an RFC 1918 private address."""
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return False
    return any(addr in net for net in _RFC1918_NETS)


def is_internal_series(series: pd.Series) -> pd.Series:
    """Vectorised RFC 1918 check.

    Uses string prefix tests for speed (avoids per-row ipaddress parsing
    on millions of rows). Equivalent to the notebook's `is_internal_series`
    helper in cell [27].

    Parameters
    ----------
    series : pd.Series
        IPv4 dotted-quad strings.

    Returns
    -------
    pd.Series of bool
    """
    s = series.astype(str)
    is_10 = s.str.startswith("10.")
    is_192 = s.str.startswith("192.168.")
    # 172.16.0.0/12 -> second octet in [16, 31]
    is_172 = s.str.startswith("172.")
    if is_172.any():
        # Refine 172.* with second-octet test
        second_octet = s.str.split(".", n=2, expand=True)[1]
        second_octet_int = pd.to_numeric(second_octet, errors="coerce")
        is_172 = is_172 & second_octet_int.between(16, 31)
    return is_10 | is_192 | is_172


def is_in_networks_series(
    series: pd.Series,
    networks: Iterable[str] | None = None,
) -> pd.Series:
    """Vectorised membership check for configured IPv4 networks."""
    if not networks:
        return is_internal_series(series)
    nets = list(networks)
    s = series.astype(str)
    # Fast path for the original manuscript subnet.
    if nets == ["172.31.0.0/16"]:
        return s.str.startswith("172.31.")
    parsed = s.map(lambda x: _ip_in_networks(x, nets))
    return parsed.astype(bool)


def _ip_in_networks(ip: str, networks: Iterable[str]) -> bool:
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return False
    return any(addr in net for net in _networks(networks))


def filter_internal_only(
    df: pd.DataFrame,
    src_col: str = "Src IP",
    dst_col: str = "Dst IP",
    internal_networks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep only flows whose BOTH src and dst are RFC 1918 (internal).

    Used to build G_normal in the manuscript pipeline.
    """
    n_before = len(df)
    mask = is_in_networks_series(df[src_col], internal_networks) & is_in_networks_series(
        df[dst_col], internal_networks,
    )
    out = df[mask].copy()
    log.info(
        "Internal-only filter: %d -> %d flows (%.1f%% kept)",
        n_before, len(out), 100.0 * len(out) / max(n_before, 1),
    )
    return out


def filter_internal_destination_only(
    df: pd.DataFrame,
    src_col: str = "Src IP",
    dst_col: str = "Dst IP",
    internal_networks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep flows whose destination belongs to the configured internal network."""
    n_before = len(df)
    mask = is_in_networks_series(df[dst_col], internal_networks)
    out = df[mask].copy()
    log.info(
        "Internal-destination filter: %d -> %d flows (%.1f%% kept)",
        n_before, len(out), 100.0 * len(out) / max(n_before, 1),
    )
    return out


def separate_benign_attack(
    df: pd.DataFrame,
    benign_label: str = "Benign",
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split flows into (benign, attack) frames.

    Returns
    -------
    benign_df, attack_df : tuple of DataFrame
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found.")
    # Some CIC files have whitespace in labels.
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    benign_mask = df[label_col].str.casefold() == benign_label.casefold()
    benign_df = df[benign_mask].copy()
    attack_df = df[~benign_mask].copy()

    log.info(
        "Split: %d benign, %d attack (%d total)",
        len(benign_df), len(attack_df), len(df),
    )
    return benign_df, attack_df


def filter_flows(
    attack_df: pd.DataFrame,
    cic_labels: Iterable[str],
    label_col: str = "Label",
) -> pd.DataFrame:
    """Filter attack flows to a single scenario by its CIC-IDS-2018 labels.

    Parameters
    ----------
    attack_df : DataFrame
        Output of :func:`separate_benign_attack` (attack frame).
    cic_labels : iterable of str
        CIC-IDS-2018 labels for the scenario (case-insensitive).

    Returns
    -------
    DataFrame
        Attack flows belonging to the scenario.
    """
    cic_labels = [s.strip().casefold() for s in cic_labels]
    mask = attack_df[label_col].astype(str).str.strip().str.casefold().isin(cic_labels)
    out = attack_df[mask].copy()
    log.info(
        "Scenario filter %s -> %d flows",
        list(cic_labels), len(out),
    )
    return out


def drop_invalid_flows(
    df: pd.DataFrame,
    drop_na_subset: Iterable[str] = ("Src IP", "Dst IP", "Label"),
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Standard hygiene: drop NaN IPs and exact duplicates."""
    n_before = len(df)
    df = df.dropna(subset=list(drop_na_subset))
    if drop_duplicates:
        df = df.drop_duplicates()
    log.info("Hygiene: %d -> %d flows", n_before, len(df))
    return df


def exclude_attempted_flows(
    df: pd.DataFrame,
    suffix: str = "-Attempted",
    label_col: str = "Label",
) -> pd.DataFrame:
    """Drop "*-Attempted" rows specific to the Improved CSE-CIC-IDS-2018.

    The Liu et al. (2022) corrections introduce sub-labels for incomplete
    attacks (port closed, no payload, startup/teardown artefacts). Those
    flows do not represent a successfully executed attack and would
    distort scenario analysis if treated as attack data. Exclude them by
    default.

    Parameters
    ----------
    df : DataFrame
    suffix : str, default "-Attempted"
        Case-insensitive suffix marking attempted (failed) attacks.
    label_col : str, default "Label"

    Returns
    -------
    DataFrame
        Frame with "*-Attempted" rows removed.
    """
    if label_col not in df.columns:
        return df
    n_before = len(df)
    mask = ~df[label_col].astype(str).str.casefold().str.endswith(suffix.casefold())
    out = df[mask].copy()
    log.info(
        "Excluded %s rows: %d -> %d flows",
        suffix, n_before, len(out),
    )
    return out
