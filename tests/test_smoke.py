"""
tests/test_smoke.py
-------------------
Module-level smoke tests using synthetic fixtures (no real data needed).

Run with:
    pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------- preprocessing ----------
def test_separate_benign_attack():
    from src.preprocessing.flow_filter import separate_benign_attack
    df = pd.DataFrame({
        "Src IP": ["10.0.0.1", "10.0.0.2"],
        "Dst IP": ["10.0.0.3", "10.0.0.4"],
        "Label": ["BENIGN", "SSH-BruteForce"],
    })
    benign, attack = separate_benign_attack(df, benign_label="BENIGN")
    assert len(benign) == 1 and len(attack) == 1


def test_filter_flows():
    from src.preprocessing.flow_filter import filter_flows
    df = pd.DataFrame({
        "Src IP": ["a", "b", "c"],
        "Dst IP": ["x", "y", "z"],
        "Label": ["SSH-BruteForce", "DDoS-HOIC", "Botnet Ares"],
    })
    out = filter_flows(df, ["SSH-BruteForce", "Botnet Ares"])
    assert len(out) == 2


def test_internal_only_filter():
    """RFC 1918 only on both src and dst."""
    from src.preprocessing.flow_filter import filter_internal_only
    df = pd.DataFrame({
        "Src IP": ["10.0.0.1", "172.31.69.7", "8.8.8.8", "192.168.1.5", "172.32.0.1"],
        "Dst IP": ["10.0.0.2", "172.31.69.25", "10.0.0.5", "192.168.1.6", "10.0.0.1"],
    })
    out = filter_internal_only(df)
    # Row 0: 10.x -> 10.x  OK
    # Row 1: 172.31.x -> 172.31.x  OK
    # Row 2: 8.8.8.8 not internal  REJECT
    # Row 3: 192.168.x -> 192.168.x  OK
    # Row 4: 172.32.x not in 16..31 range  REJECT
    assert len(out) == 3


def test_exclude_attempted_flows():
    from src.preprocessing.flow_filter import exclude_attempted_flows
    df = pd.DataFrame({
        "Src IP": ["a", "b", "c", "d"],
        "Dst IP": ["x", "y", "z", "w"],
        "Label": [
            "SSH-BruteForce",
            "FTP-BruteForce-Attempted",
            "DoS-Hulk-Attempted",
            "DDoS-HOIC",
        ],
    })
    out = exclude_attempted_flows(df, suffix="-Attempted")
    assert len(out) == 2
    assert set(out["Label"]) == {"SSH-BruteForce", "DDoS-HOIC"}


def test_load_scenarios_active_filter(tmp_path):
    from src.preprocessing.data_loader import load_scenarios
    p = tmp_path / "scenarios.yaml"
    p.write_text(
        """
scenarios:
  - id: A
    name: alpha
    active: true
    cic_labels: [a]
    targets: [10.0.0.1]
    mitre_technique: T1110
    cwe: CWE-1
  - id: B
    name: beta
    active: false
    cic_labels: [b]
    targets: [10.0.0.2]
    mitre_technique: T1190
    cwe: CWE-2
""",
        encoding="utf-8",
    )
    only_active = load_scenarios(p)
    assert [s["id"] for s in only_active] == ["A"]
    all_scenarios = load_scenarios(p, only_active=False)
    assert [s["id"] for s in all_scenarios] == ["A", "B"]


# ---------- graph ----------
def _toy_flows() -> pd.DataFrame:
    """Hub-and-spoke flow set."""
    rows = []
    for _ in range(5):
        rows.append({"Src IP": "10.0.0.1", "Dst IP": "10.0.0.2"})
    for _ in range(4):
        rows.append({"Src IP": "10.0.0.3", "Dst IP": "10.0.0.2"})
    for _ in range(3):
        rows.append({"Src IP": "10.0.0.2", "Dst IP": "10.0.0.4"})
    return pd.DataFrame(rows)


def test_build_graph_threshold():
    from src.graph.builder import build_communication_graph
    G = build_communication_graph(_toy_flows(), edge_threshold_tau=4)
    edges = set(G.edges())
    # only the 5-count and 4-count edges survive tau=4
    assert ("10.0.0.1", "10.0.0.2") in edges
    assert ("10.0.0.3", "10.0.0.2") in edges
    assert ("10.0.0.2", "10.0.0.4") not in edges


def test_build_graph_with_backbone():
    from src.graph.builder import build_communication_graph
    G = build_communication_graph(
        _toy_flows(), edge_threshold_tau=1,
        target_hosts=["10.0.0.2"],
        use_backbone=True,
        node_budget=10, max_hops=4,
    )
    assert G.number_of_nodes() > 0
    # Targets must always be in the graph
    assert "10.0.0.2" in G.nodes()


def test_pagerank_log1p_weight():
    from src.graph.builder import build_communication_graph
    from src.graph.centrality import compute_importance
    G = build_communication_graph(_toy_flows(), edge_threshold_tau=1)
    I = compute_importance(G, method="pagerank", weight_transform="log1p")
    vals = list(I.values())
    assert max(vals) == 1.0
    assert min(vals) == 0.0


def test_all_centralities_run():
    from src.graph.builder import build_communication_graph
    from src.graph.centrality import AVAILABLE_CENTRALITIES, compute_importance
    G = build_communication_graph(_toy_flows(), edge_threshold_tau=1)
    for m in AVAILABLE_CENTRALITIES:
        I = compute_importance(G, method=m)
        assert len(I) == G.number_of_nodes()


def test_k_hop_subgraph():
    from src.graph.builder import k_hop_subgraph
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
    reach1 = k_hop_subgraph(G, ["C"], K=1)
    assert reach1 == {"B", "C", "D"}


def test_exposure_log1p_minmax():
    from src.graph.builder import build_communication_graph
    from src.graph.exposure import compute_exposure
    G = build_communication_graph(_toy_flows(), edge_threshold_tau=1)
    E = compute_exposure(G, flows=_toy_flows(), normalize="log1p_minmax")
    assert all(0.0 <= v <= 1.0 for v in E.values())


# ---------- risk ----------
def test_cvss_aggregation_strategies():
    from src.risk.cvss_aggregation import (
        AVAILABLE_AGGREGATIONS, aggregate_cvss,
    )
    assert set(AVAILABLE_AGGREGATIONS) == {"max", "mean", "median"}
    pool = [9.8, 9.5, 7.2, 6.0, 4.0, 2.0]
    out = {m: aggregate_cvss(pool, method=m) for m in AVAILABLE_AGGREGATIONS}
    assert abs(out["max"] - 0.98) < 1e-9
    assert abs(out["median"] - 0.66) < 1e-9
    assert abs(out["mean"] - 0.6416666667) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_compute_risk_and_loss():
    from src.risk.loss import compute_loss, compute_risk
    I = {"A": 1.0, "B": 0.5}
    E = {"A": 0.5, "B": 1.0}
    risk = compute_risk(I, E, vulnerability=0.8)
    assert abs(risk["A"] - 0.4) < 1e-9
    assert abs(risk["B"] - 0.4) < 1e-9
    loss = compute_loss(risk, ["A", "B"])
    assert abs(loss - 0.8) < 1e-9


# ---------- countermeasure ----------
def test_d3fend_mapper_loads_scenarios():
    from src.countermeasure.d3fend_mapper import D3FENDMapper
    cm_path = Path(__file__).resolve().parents[1] / "config" / "countermeasures.yaml"
    mapper = D3FENDMapper(str(cm_path))
    s1 = mapper.for_scenario("S1")
    s2 = mapper.for_scenario("S2")
    # Manuscript scope: S1/S2.
    assert len(s1) >= 3 and len(s2) >= 3
    # MFA should be in S1
    assert any(c.cid == "MFA" for c in s1)


def test_apply_operation_set_v():
    from src.countermeasure.effectiveness import apply_operation
    I = {"a": 0.5}
    E = {"a": 0.7}
    I_new, E_new, V_new = apply_operation(
        "SET_V", {"V_after": 0.05}, target="a", R=["a"],
        I=I, E=E, V=0.9,
    )
    assert V_new == 0.05
    assert I_new == I
    assert E_new == E


def test_apply_operation_scale_e_target():
    from src.countermeasure.effectiveness import apply_operation
    I = {"a": 0.5}
    E = {"a": 0.6}
    I_new, E_new, V_new = apply_operation(
        "SCALE_E_TARGET", {"E_scale": 0.5}, target="a", R=["a"],
        I=I, E=E, V=0.9,
    )
    assert E_new["a"] == 0.3
    assert V_new == 0.9


def test_apply_operation_multi_target_scale_e():
    """Multi-target SCALE_E_TARGET should affect every target IP."""
    from src.countermeasure.effectiveness import apply_operation
    I = {"a": 0.5, "b": 0.5, "c": 0.5}
    E = {"a": 0.6, "b": 0.4, "c": 0.8}
    _, E_new, _ = apply_operation(
        "SCALE_E_TARGET", {"E_scale": 0.5},
        target=["a", "b", "c"],
        R=["a", "b", "c"],
        I=I, E=E, V=0.9,
    )
    assert E_new["a"] == 0.3
    assert E_new["b"] == 0.2
    assert E_new["c"] == 0.4


def test_apply_operation_multi_target_set_e():
    """Multi-target SET_E_TARGET should cap every target IP."""
    from src.countermeasure.effectiveness import apply_operation
    E = {"a": 0.4, "b": 0.05, "c": 0.7}
    _, E_new, _ = apply_operation(
        "SET_E_TARGET", {"E_after": 0.1},
        target=["a", "b", "c"],
        R=["a", "b", "c"],
        I={}, E=E, V=0.5,
    )
    # a (0.4 -> 0.1) and c (0.7 -> 0.1) capped, b (0.05) unchanged.
    assert E_new["a"] == 0.1
    assert E_new["b"] == 0.05
    assert E_new["c"] == 0.1


def test_apply_operation_set_e_target_caps():
    from src.countermeasure.effectiveness import apply_operation
    # SET_E_TARGET should cap (i.e. only lower) the value.
    E = {"a": 0.4}
    _, E_new, _ = apply_operation(
        "SET_E_TARGET", {"E_after": 0.02}, target="a", R=["a"],
        I={}, E=E, V=0.5,
    )
    assert E_new["a"] == 0.02
    # If existing already lower, value stays.
    E2 = {"a": 0.01}
    _, E2_new, _ = apply_operation(
        "SET_E_TARGET", {"E_after": 0.02}, target="a", R=["a"],
        I={}, E=E2, V=0.5,
    )
    assert E2_new["a"] == 0.01


def test_combine_residual_rules():
    from src.countermeasure.multi import combine_residual
    rs = [0.5, 0.5]
    indep = combine_residual(rs, rule="independent")
    sub_mult = combine_residual(rs, rule="sub_multiplicative", alpha=0.5)
    super_add = combine_residual(rs, rule="super_additive", beta=0.9)
    assert abs(indep - 0.25) < 1e-9
    assert sub_mult > indep
    assert super_add >= indep


# ---------- metrics ----------
def test_spearman_basic():
    from src.metrics.correlation import spearman_importance_exposure
    I = {"A": 0.1, "B": 0.5, "C": 0.9}
    E = {"A": 0.9, "B": 0.5, "C": 0.1}
    r = spearman_importance_exposure(I, E)
    assert r["rho"] == -1.0


def test_mtta_table_shape():
    from src.metrics.mtta import mtta_improvement_table
    baselines = [
        {"name": "x", "seconds": 100, "source": "s1", "relevance": "r1"},
        {"name": "y", "seconds": 1000, "source": "s2", "relevance": "r2"},
    ]
    df = mtta_improvement_table(measured_seconds=10, baselines=baselines)
    assert list(df["baseline"]) == ["x", "y"]
    assert df.loc[df["baseline"] == "y", "improvement_pct"].iloc[0] == 99.0


# ---------- experiment output structure ----------
def test_experiment_output_layout(tmp_path):
    from src.utils.results import ExperimentOutput
    exp = ExperimentOutput(
        name="test_exp", results_root=tmp_path,
        config={"dataset": {"version": "test"}},
        scenarios=["S1"],
    )
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    table_path = exp.save_table(df, "result.csv")
    assert table_path.exists() and table_path.parent.name == "tables"

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    fig_path = exp.save_figure(fig, "plot.png")
    plt.close(fig)
    assert fig_path.exists() and fig_path.parent.name == "figures"

    exp.add_summary("k", 42)
    exp.close()
    summary = tmp_path / "test_exp" / "summary.json"
    meta = tmp_path / "test_exp" / "meta.yaml"
    assert summary.exists() and meta.exists()
