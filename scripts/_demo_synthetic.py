"""
scripts/_demo_synthetic.py
---------------------------
Generate a small synthetic CIC-IDS-2018-like CSV in data/raw/synthetic.csv
for live-demo runs of the experiments. This is for SMOKE testing the
output structure only; never used in manuscript claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write(path: Path) -> None:
    rows = []
    benign_pairs = [
        ("10.0.0.1", "10.0.0.2"), ("10.0.0.1", "10.0.0.3"),
        ("10.0.0.2", "10.0.0.3"), ("10.0.0.3", "10.0.0.4"),
        ("10.0.0.4", "10.0.0.5"), ("10.0.0.5", "10.0.0.1"),
        ("10.0.0.2", "10.0.0.5"), ("10.0.0.6", "10.0.0.3"),
        ("10.0.0.7", "10.0.0.3"), ("10.0.0.8", "10.0.0.4"),
    ]
    for src, dst in benign_pairs:
        for _ in range(5):
            rows.append({
                "Src IP": src, "Src Port": 12345, "Dst IP": dst,
                "Dst Port": 80, "Protocol": 6,
                "Timestamp": "2018-02-14 10:00:00",
                "Label": "Benign",
            })
    # SSH attack
    for i in range(20):
        rows.append({
            "Src IP": f"172.16.0.{i+1}", "Src Port": 54321,
            "Dst IP": "10.0.0.3", "Dst Port": 22, "Protocol": 6,
            "Timestamp": "2018-02-14 11:00:00",
            "Label": "SSH-Bruteforce",
        })
    # DDoS attack
    for i in range(30):
        rows.append({
            "Src IP": f"203.0.113.{i+1}", "Src Port": 33333,
            "Dst IP": "10.0.0.4", "Dst Port": 80, "Protocol": 6,
            "Timestamp": "2018-02-15 12:00:00",
            "Label": "DDOS attack-HOIC",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out)
    print(f"wrote {out}")
