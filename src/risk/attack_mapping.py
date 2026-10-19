"""
src/risk/attack_mapping.py
---------------------------
ATT&CK -> CAPEC -> CWE -> CVSS mapping pipeline.

Manuscript Sec. III.D, Fig. 2.

The downstream V(S) aggregation logic lives in cvss_aggregation.py;
this module just resolves a scenario to its CWE pool and the matching
CVSS records.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


def build_threat_table(
    scenarios: list[dict],
    cwe_csv: str,
    cvss_csv: str,
) -> pd.DataFrame:
    """Build a per-scenario threat table joining CWE -> CVE -> CVSS.

    Returns
    -------
    DataFrame with columns:
        scenario_id, scenario_name, mitre_technique, cwe_id,
        cve_id, cvss, cvss_versions, cvss_vector_string
    """
    cwe_df = pd.read_csv(cwe_csv)
    cvss_df = pd.read_csv(cvss_csv)

    # cwe.csv: cve_id, cwe_id, update_date
    # cvss.csv: cve_id, cvss_vector_string, cvss, cvss_versions, update_date
    joined = cwe_df.merge(cvss_df, on="cve_id", how="inner", suffixes=("_cwe", "_cvss"))
    log.info("Threat table candidate pool: %d (CVE x CVSS x CWE) rows", len(joined))

    rows: list[dict] = []
    for sc in scenarios:
        target_cwe = sc.get("cwe")
        if not target_cwe:
            log.warning("Scenario %s has no CWE; skipping.", sc.get("id"))
            continue
        sub = joined[joined["cwe_id"].astype(str).str.upper() == str(target_cwe).upper()]
        if sub.empty:
            log.warning(
                "Scenario %s (CWE=%s) yielded no CVSS records.",
                sc.get("id"), target_cwe,
            )
            continue
        for _, r in sub.iterrows():
            rows.append({
                "scenario_id": sc["id"],
                "scenario_name": sc["name"],
                "mitre_technique": sc.get("mitre_technique"),
                "cwe_id": r["cwe_id"],
                "cve_id": r["cve_id"],
                "cvss": float(r["cvss"]),
                "cvss_versions": r.get("cvss_versions"),
                "cvss_vector_string": r.get("cvss_vector_string"),
            })

    df = pd.DataFrame(rows)
    log.info(
        "Threat table built: %d rows across %d scenarios",
        len(df), df["scenario_id"].nunique() if not df.empty else 0,
    )
    return df


def scenario_cwe_pool(threat_table: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Subset of threat table for a single scenario."""
    sub = threat_table[threat_table["scenario_id"] == scenario_id].copy()
    if sub.empty:
        log.warning("No threat-table rows for scenario %s", scenario_id)
    return sub
