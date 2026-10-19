# Loss-Centric Cyber Resilience Framework

Reproducibility package for the IEEE Access paper
*"Loss-Centric Cyber Resilience Evaluation and Automated Countermeasure Ranking Using Public Network Flow Data"*
(IEEE Access, vol. 14, pp. 89435–89447, 2026, doi:[10.1109/ACCESS.2026.3703083](https://doi.org/10.1109/ACCESS.2026.3703083)).

The package reproduces the paper pipeline (RFC1918-internal benign filtering, τ-filtered
internal communication graph, log1p-weighted PageRank, log1p+min-max exposure,
scenario CVSS proxy, operation-based countermeasure application) and the experiments
behind every results table of the paper.

---

## Dataset

**Improved CSE-CIC-IDS-2018** (Liu et al., 2022, IEEE CNS) — 62,888,908 records,
of which 59,353,486 are benign.
Hosted at: <https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html>

Suggested workflow:
1. Download the combined CSV (the variant whose name ends with `no_Attempted.csv`).
2. Run `scripts/slim_cic_columns.py` to reduce ~70 columns
   down to the 7 the framework uses, then point `--data` at the slim CSV.
3. Place the raw/slim CSV under `data/raw/` (not tracked by git) or pass its path via `--data`.

Default config flags relevant to this dataset (`config/default.yaml`):
```yaml
preprocessing:
  benign_label: "BENIGN"
  exclude_attempted: true        # no-op if the input is already *no_Attempted*
  internal_only: true            # RFC1918-internal benign flows for the graph
  internal_networks: ["172.31.0.0/16"]
graph:
  edge_threshold_tau: 3          # τ (paper Sec. IV-D)
  use_backbone: false
  impact_range_K: 3              # K (paper Sec. IV-E)
```

---

## Paper scope

| Item | Setting in the paper |
|---|---|
| Scenarios | **S1** SSH-BruteForce, **S2** DDoS-HOIC. |
| Targets | S1: `172.31.69.25`. S2: `172.31.69.28`. |
| V(S) | ATT&CK → CAPEC → CWE → CVE → CVSS with multi-version fallback (NVD v4.0→v3.1→v3.0→v2.0, then CNA v4.0→v3.1), max over the CWE pool / 10. S1 = S2 = 0.99 (scenario `cvss_proxy`). |
| CVSS aggregation sensitivity | `max` (main), `mean`, `median` over the NVD CWE pools (325 / 964 CVEs). |
| Centrality ablation | PageRank (primary) vs `degree`, `betweenness`. |
| Graph sensitivity | τ ∈ {1, 2, 3, 5, 10}; K-hop impact range K ∈ {1, 2, 3, 4}. |
| Effectiveness tiers | High [0.05, 0.25], Medium [0.30, 0.50], Low [0.50, 0.80]. |
| Countermeasure-parameter sensitivity | 5 rr values per tier, all 5⁴ = 625 combinations per scenario; D(c) not varied. |
| MTTA baseline | 300 s manual-analysis proxy (Ahn et al., IEEE Access 2024 — paper ref. [22]). |

---

## Paper tables → scripts

| Paper | Content | Script |
|---|---|---|
| Table 4, 8, 9, 10 | Graph characteristics, Loss, countermeasure ΔLoss / Score | `exp01_baseline.py` |
| Table 5 | K-hop impact range (K = 1–4) | `exp05_graph_sensitivity.py` (K sweep) |
| Table 14 | τ sweep (τ = 1, 2, 3, 5, 10) | `exp05_graph_sensitivity.py` (τ sweep) |
| Sec. V-A | Spearman ρ between I(n) and E(n) (ρ = −0.1838) | `exp07_correlation_analysis.py` |
| Table 11 | CVSS aggregation sensitivity (max / mean / median) | `exp15_cvss_live_aggregation_sensitivity.py` |
| Table 12 | Countermeasure-parameter sensitivity (625 combinations) | `exp06_countermeasure_parameter_sensitivity.py` |
| Table 13 | Centrality ablation (PageRank / Degree / Betweenness) | `exp16_pagerank_degree_betweenness_ablation.py` |
| Table 15 | MTTA (median of 50 runs after 5 warm-up runs) | `exp12_mtta_groupby_lig_scope.py` |

Notes:
* `exp15` uses the V(S) values computed from `data/cvss_pools/S1_S2_CVSS_pool_all.csv`
  (per-CVE selected CVSS after the fallback pipeline; NVD queried on 2026-05-03).
  `S1_S2_CVSS_aggregation_summary.csv` lists the resulting max / mean / median.
* `exp06` resolves ties by catalogue order in `config/countermeasures.yaml`
  (first candidate wins), identical to `np.argmax` in the original notebook.

### Original notebook

`notebooks/Experiment_v5.ipynb` is the original experiment notebook (v5: multi-version CVSS
fallback pipeline) from which this package was refactored. Its outputs are kept for reference.
Paths in its first cell are Windows-local (`D:\improveCIC-IDS-2018\...`) and must be edited
before re-running.

---

## Project layout

```
cyber_resilience_framework/
├── config/
│   ├── default.yaml           # all hyperparameters
│   ├── scenarios.yaml         # S1/S2 scenario definitions
│   └── countermeasures.yaml   # S1/S2 countermeasure catalogue (paper Tables 6, 7)
├── data/
│   ├── raw/                   # Improved CSE-CIC-IDS-2018 CSVs (not tracked)
│   ├── processed/
│   ├── mappings/              # cwe.csv, cvss.csv, d3fend.csv
│   └── cvss_pools/            # S1/S2 CWE-level CVSS pools (paper Tables 2, 11)
├── src/
│   ├── preprocessing/         # data_loader, flow_filter (incl. internal_only), augmentation
│   ├── graph/                 # builder, centrality, exposure
│   ├── risk/                  # attack_mapping, cvss_aggregation, loss
│   ├── countermeasure/        # d3fend_mapper, effectiveness (apply_operation), difficulty,
│   │                          # selector, multi (lib)
│   ├── metrics/               # mtta, correlation
│   ├── visualization/         # paper-style plotting
│   └── utils/                 # io, logging, results (ExperimentOutput)
├── experiments/               # one script per paper table (see above)
├── notebooks/                 # original Experiment_v5.ipynb
├── scripts/
│   ├── run_pipeline.py        # single-entry full pipeline runner
│   └── slim_cic_columns.py    # dataset column-slimming helper
├── tests/                     # 24 tests (smoke + integration)
└── results/<exp_name>/
    ├── tables/                # CSVs + matching XLSXs
    ├── figures/               # PNGs + matching PDFs
    ├── summary.json
    └── meta.yaml
```

---

## Running

```bash
pip install -r requirements.txt
pip install -e .
```

Paper environment: Python 3.13.5, pandas 2.2.3, NumPy 2.1.3, NetworkX 3.4.2
(AMD Ryzen 7 9700X, 96 GB RAM, Windows 11, CPU only).

Single scenario via the unified runner:

```bash
python scripts/run_pipeline.py \
  --scenario S1 \
  --data D:/improveCIC-IDS-2018/combine/All_Combined_slim.csv \
  --output results/run_S1
```

Reproduce all paper tables:

```bash
DATA=D:/improveCIC-IDS-2018/combine/All_Combined_slim.csv

python -m experiments.exp01_baseline                            --data $DATA   # Tables 4, 8-10
python -m experiments.exp05_graph_sensitivity                   --data $DATA   # Tables 5, 14
python -m experiments.exp06_countermeasure_parameter_sensitivity --data $DATA  # Table 12
python -m experiments.exp07_correlation_analysis                --data $DATA   # Spearman rho
python -m experiments.exp15_cvss_live_aggregation_sensitivity   --data $DATA   # Table 11
python -m experiments.exp16_pagerank_degree_betweenness_ablation --data $DATA  # Table 13
python -m experiments.exp12_mtta_groupby_lig_scope              --data $DATA   # Table 15
```

All hyperparameters are consolidated in `config/default.yaml`.

---

## Reproducibility note

All graph, risk and countermeasure outputs (|N|, |E_G|, I(n), E(n), V(S), Loss, ΔLoss,
Score and the resulting rankings) are deterministic.

**MTTA values are wall-clock measurements (`time.perf_counter()`) and depend on the
hardware and software environment (CPU, memory, storage, Python/library versions).**
Absolute MTTA numbers obtained on a different machine will therefore differ from those
reported in the paper (Table 15); the comparison against the 300 s manual-analysis
baseline is the quantity intended to be reproduced.

---

## Synthetic data note

`tests/test_integration.py` ships with a small synthetic CIC-IDS-style fixture
for **code-integrity testing only**. Reviewers / CI systems can verify the
pipeline glues together without the ~62M-row Improved CSE-CIC-IDS-2018
download. The synthetic data is **never** used in any manuscript figure,
table, or claim.

---

## Citation

```bibtex
@article{you2026losscentric,
  author  = {You, Hyunwook and Jang, Jisoo and Choi, Minjin and Lee, Jina and Ryu, Haneul and Shin, Dongkyoo},
  title   = {Loss-Centric Cyber Resilience Evaluation and Automated Countermeasure Ranking Using Public Network Flow Data},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {89435--89447},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3703083}
}
```
