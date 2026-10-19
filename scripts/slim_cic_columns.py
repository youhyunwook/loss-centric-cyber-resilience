"""
slim_cic_columns.py
====================
Standalone preprocessing helper for the Improved CSE-CIC-IDS-2018
combined CSV.

What it does
------------
Reads `D:\\improveCIC-IDS-2018\\combine\\All_Combined_no_Attempted.csv`
(default) and writes a slimmed-down copy that keeps only the columns
the cyber-resilience framework actually consumes for graph construction.

Specifically, it keeps:
    Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp, Label

and drops every flow-statistics / ML-feature column (Flow Duration,
Total Fwd Packets, Bwd Packet Length Mean, Init Win bytes forward, ...
~70 columns in the Improved release).

Why a separate script
---------------------
The slimming is part of the manuscript's preprocessing pipeline, not
the framework's runtime. Keeping it as a one-shot standalone script
means:
  - the framework code path is not slowed down by large-file column
    pruning at every run;
  - the slimmed CSV becomes a stable, reusable input for every
    experiment (exp01..exp16) without re-reading the raw 62M-row file.

Usage (Windows)
---------------
Default:
    python slim_cic_columns.py

Custom paths:
    python slim_cic_columns.py ^
        --input  "D:\\improveCIC-IDS-2018\\combine\\All_Combined_no_Attempted.csv" ^
        --output "D:\\improveCIC-IDS-2018\\combine\\All_Combined_slim.csv"

Tunables:
    --chunksize 500000      # rows per chunk; default 1,000,000

Notes
-----
* The script tolerates the alias variants seen across CIC-IDS-2018
  releases (``Source IP`` -> ``Src IP``, ``Destination IP`` -> ``Dst IP``,
  etc.). Column names are stripped of leading/trailing whitespace.
* Output file is overwritten if it already exists.
* No dependencies beyond pandas.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# These are the canonical column names the framework consumes.
# Keep this list in sync with src/preprocessing/data_loader.py::REQUIRED_COLS.
REQUIRED_CANONICAL: list[str] = [
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
]

# Common alias variants -> canonical name.
ALIASES: dict[str, str] = {
    "Source IP": "Src IP",
    "Source Port": "Src Port",
    "Destination IP": "Dst IP",
    "Destination Port": "Dst Port",
}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def detect_columns(input_path: Path) -> tuple[list[str], dict[str, str]]:
    """Read the header line only and return (raw_cols, raw -> canonical map)."""
    with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline().rstrip("\r\n")
    raw_cols = [c.strip() for c in header_line.split(",")]
    canonical_map = {c: ALIASES.get(c, c) for c in raw_cols}
    return raw_cols, canonical_map


def slim(
    input_path: Path,
    output_path: Path,
    chunksize: int = 1_000_000,
) -> int:
    """Stream-process input_path, writing a slim CSV to output_path.

    Returns
    -------
    int
        Number of data rows written (excluding header).
    """
    raw_cols, canonical_map = detect_columns(input_path)
    # ``raw_cols`` is the *stripped* version of the header tokens.
    # Some CIC-IDS-2018 releases ship leading whitespace in column names
    # (" Src Port", " Dst Port", ...). To make ``usecols`` matching robust,
    # we tell pandas to *skip* the original header row entirely and
    # provide our cleaned names via ``names=``. That way every downstream
    # match (``usecols``, rename, ordering) operates on the stripped
    # versions and the original whitespace becomes irrelevant.

    # Decide which (cleaned) column names to actually read from disk.
    keep_clean = [c for c in raw_cols if canonical_map[c] in REQUIRED_CANONICAL]

    # Sanity check: did we find every required canonical?
    found_canonical = {canonical_map[c] for c in keep_clean}
    missing = [c for c in REQUIRED_CANONICAL if c not in found_canonical]
    if missing:
        print(
            f"!! WARNING: {len(missing)} required column(s) missing from input: "
            f"{missing}",
            file=sys.stderr,
        )

    print(f"Input file        : {input_path}")
    print(f"Output file       : {output_path}")
    print(f"Total cols in CSV : {len(raw_cols)}")
    print(f"Cols to keep      : {len(keep_clean)}  ({sorted(found_canonical)})")
    print(f"Cols to drop      : {len(raw_cols) - len(keep_clean)}")
    print(f"Chunk size        : {chunksize:,} rows")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rename_at_read = {clean: canonical_map[clean] for clean in keep_clean}
    canonical_order = [c for c in REQUIRED_CANONICAL if c in found_canonical]

    n_rows_total = 0
    t0 = time.perf_counter()
    first_chunk = True

    chunk_iter = pd.read_csv(
        input_path,
        header=0,                  # original header row exists and is skipped
        names=raw_cols,            # but we override with our stripped names
        usecols=keep_clean,        # so usecols matches the clean names
        low_memory=False,
        chunksize=chunksize,
    )

    for i, chunk in enumerate(chunk_iter):
        chunk = chunk.rename(columns=rename_at_read)
        chunk = chunk[canonical_order]
        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        n_rows_total += len(chunk)
        elapsed = time.perf_counter() - t0
        print(
            f"  chunk {i + 1:>3}: wrote {len(chunk):>10,} rows   "
            f"cumulative {n_rows_total:>12,}   ({elapsed:>6.1f} s elapsed)"
        )

    elapsed = time.perf_counter() - t0
    print()
    print(f"DONE: {n_rows_total:,} rows written in {elapsed:.1f} s")

    # Show file-size reduction.
    try:
        in_mb = input_path.stat().st_size / (1024 ** 2)
        out_mb = output_path.stat().st_size / (1024 ** 2)
        ratio = (1 - out_mb / in_mb) * 100 if in_mb > 0 else 0.0
        print(f"Size: {in_mb:,.1f} MB  ->  {out_mb:,.1f} MB  ({ratio:.1f}% reduction)")
    except OSError:
        pass

    return n_rows_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Slim the Improved CSE-CIC-IDS-2018 CSV to the 7 columns "
            "the framework consumes."
        ),
    )
    parser.add_argument(
        "--input",
        default=r"D:\improveCIC-IDS-2018\combine\All_Combined_no_Attempted.csv",
        help="Path to the input combined CSV.",
    )
    parser.add_argument(
        "--output",
        default=r"D:\improveCIC-IDS-2018\combine\All_Combined_slim.csv",
        help="Path where the slimmed CSV will be written.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per pandas chunk; lower this if RAM is tight (default 1,000,000).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    if output_path.resolve() == input_path.resolve():
        sys.exit("Output path must differ from input path (refusing to overwrite source).")

    slim(input_path, output_path, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
