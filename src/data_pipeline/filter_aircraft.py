"""
Step 1: load TrajAir day CSVs and isolate a single aircraft tail (default N13337).

Reads DATA_PATH from .env (folder of day_*_adsb directories). Writes filtered frames to
data/02_processed/ as both CSV (pandas-friendly) and Parquet (polars-friendly).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_settings(config_path: Path | None = None) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = config_path or (REPO_ROOT / "configs" / "base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_path = os.getenv("DATA_PATH", "").strip().strip('"').strip("'")
    if not data_path:
        raise SystemExit(
            "DATA_PATH is not set. Copy .env.example to .env and set DATA_PATH "
            "to your TrajAir raw_data folder."
        )

    cfg["data_path"] = Path(data_path).expanduser().resolve()
    return cfg


def list_day_csv_files(raw_root: Path, day_csv_name: str = "1.csv") -> list[Path]:
    if not raw_root.is_dir():
        raise SystemExit(f"DATA_PATH is not a directory: {raw_root}")

    files: list[Path] = []
    for day_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        csv_path = day_dir / day_csv_name
        if csv_path.is_file() and csv_path.stat().st_size > 0:
            files.append(csv_path)
    return files


def load_raw_polars(csv_files: list[Path]) -> pl.DataFrame:
    """Load all non-empty day CSVs into one Polars DataFrame.

    TrajAir day files are messy (truncated bools like ``Fals``, mixed columns),
    so we ingest as Utf8 and cast numeric fields after the union.
    """
    frames: list[pl.DataFrame] = []
    for path in csv_files:
        day_id = path.parent.name
        frame = pl.read_csv(
            path,
            infer_schema_length=0,  # all columns as Utf8; avoids bool parse errors
            ignore_errors=True,
        )
        frame = frame.with_columns(pl.lit(day_id).alias("day_folder"))
        frames.append(frame)
    if not frames:
        raise SystemExit("No readable CSV files found under DATA_PATH.")

    combined = pl.concat(frames, how="diagonal_relaxed")
    return _cast_numeric_columns(combined)


def _cast_numeric_columns(df: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = [
        "ID",
        "Altitude",
        "Speed",
        "Heading",
        "Lat",
        "Lon",
        "Age",
        "Range",
        "Bearing",
    ]
    casts = [
        pl.col(c).cast(pl.Float64, strict=False).alias(c)
        for c in numeric_cols
        if c in df.columns
    ]
    if "Tail" in df.columns:
        casts.append(pl.col("Tail").cast(pl.Utf8).str.strip_chars().alias("Tail"))
    return df.with_columns(casts) if casts else df


def isolate_tail_polars(df: pl.DataFrame, tail: str) -> pl.DataFrame:
    tail_col = pl.col("Tail").cast(pl.Utf8).str.strip_chars()
    return df.filter(tail_col == tail)


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas()


def save_outputs(
    pl_df: pl.DataFrame,
    pd_df: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd_df.to_csv(csv_path, index=False)
    pl_df.write_parquet(parquet_path)


def run(config_path: Path | None = None) -> tuple[pl.DataFrame, pd.DataFrame]:
    cfg = load_settings(config_path)
    raw_root: Path = cfg["data_path"]
    tail: str = cfg.get("aircraft_tail", "N13337")
    day_csv_name: str = cfg.get("raw", {}).get("day_csv_name", "1.csv")
    paths = cfg.get("paths", {})

    csv_out = REPO_ROOT / paths.get("n13337_csv", "data/02_processed/n13337.csv")
    parquet_out = REPO_ROOT / paths.get(
        "n13337_parquet", "data/02_processed/n13337.parquet"
    )

    csv_files = list_day_csv_files(raw_root, day_csv_name=day_csv_name)
    print(f"Reading {len(csv_files)} day files from {raw_root}")

    raw_pl = load_raw_polars(csv_files)
    print(f"Raw rows (all aircraft): {raw_pl.height:,}")

    filtered_pl = isolate_tail_polars(raw_pl, tail=tail)
    filtered_pd = to_pandas(filtered_pl)

    n_days = filtered_pl.select("day_folder").n_unique()
    print(f"Isolated Tail={tail}: {filtered_pl.height:,} rows across {n_days} days")

    save_outputs(filtered_pl, filtered_pd, csv_out, parquet_out)
    print(f"Wrote pandas CSV   -> {csv_out}")
    print(f"Wrote polars Parquet -> {parquet_out}")

    return filtered_pl, filtered_pd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Isolate aircraft TrajAir tracks (default: N13337)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to base_config.yaml",
    )
    args = parser.parse_args(argv)
    run(config_path=args.config)


if __name__ == "__main__":
    # Allow `python src/data_pipeline/filter_aircraft.py` from repo root.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
