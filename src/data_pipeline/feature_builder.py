"""
Step 2: derive onground and vertrate features for filtered N13337 tracks.

Reads data/02_processed/n13337.parquet and writes enriched CSV + Parquet outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_settings(config_path: Path | None = None) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = config_path or (REPO_ROOT / "configs" / "base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def add_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """Combine Date + Time into a sortable timestamp column."""
    return df.with_columns(
        pl.concat_str([pl.col("Date"), pl.lit(" "), pl.col("Time")])
        .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S%.f", strict=False)
        .alias("timestamp")
    )


def add_onground(
    df: pl.DataFrame,
    airport_altitude_ft: float,
    alt_tol_ft: float,
    speed_max_kt: float,
) -> pl.DataFrame:
    """
    Proposal rule: onground if |Altitude - airport| < tol AND Speed < max.
    Null speeds are treated as not onground.
    """
    near_field = (pl.col("Altitude") - airport_altitude_ft).abs() < alt_tol_ft
    slow_enough = pl.col("Speed").is_not_null() & (pl.col("Speed") < speed_max_kt)
    return df.with_columns((near_field & slow_enough).alias("onground"))


def add_vertrate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vertical rate from previous sample within the same track ID.
    Units: feet per minute. First row of each ID is null.
    """
    sorted_df = df.sort(["ID", "timestamp"])
    d_alt = pl.col("Altitude") - pl.col("Altitude").shift(1).over("ID")
    d_t_sec = (
        pl.col("timestamp") - pl.col("timestamp").shift(1).over("ID")
    ).dt.total_seconds()

    vertrate = (
        pl.when(d_t_sec > 0)
        .then((d_alt / d_t_sec) * 60.0)
        .otherwise(None)
        .alias("vertrate")
    )
    return sorted_df.with_columns(vertrate)


def build_features(df: pl.DataFrame, features_cfg: dict) -> pl.DataFrame:
    airport_alt = float(features_cfg.get("airport_altitude_ft", 1248))
    alt_tol = float(features_cfg.get("onground_alt_tol_ft", 50))
    speed_max = float(features_cfg.get("onground_speed_max_kt", 50))

    out = add_timestamp(df)
    out = add_onground(out, airport_alt, alt_tol, speed_max)
    out = add_vertrate(out)
    return out


def save_outputs(df: pl.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    df.to_pandas().to_csv(csv_path, index=False)


def print_summary(df: pl.DataFrame) -> None:
    n = df.height
    onground_n = int(df.filter(pl.col("onground")).height)
    vertrate_null = int(df.select(pl.col("vertrate").null_count()).item())
    vr = df.select(
        pl.col("vertrate").mean().alias("mean"),
        pl.col("vertrate").median().alias("median"),
        pl.col("vertrate").min().alias("min"),
        pl.col("vertrate").max().alias("max"),
    ).row(0)

    print(f"Rows: {n:,}")
    print(f"onground=true: {onground_n:,} ({100.0 * onground_n / n:.2f}%)")
    print(f"vertrate nulls: {vertrate_null:,}")
    print(
        "vertrate ft/min — "
        f"mean={vr[0]:.2f}, median={vr[1]:.2f}, min={vr[2]:.2f}, max={vr[3]:.2f}"
    )


def run(config_path: Path | None = None) -> pl.DataFrame:
    cfg = load_settings(config_path)
    paths = cfg.get("paths", {})
    features_cfg = cfg.get("features", {})

    in_parquet = REPO_ROOT / paths.get(
        "n13337_parquet", "data/02_processed/n13337.parquet"
    )
    csv_out = REPO_ROOT / paths.get(
        "n13337_features_csv", "data/02_processed/n13337_features.csv"
    )
    parquet_out = REPO_ROOT / paths.get(
        "n13337_features_parquet", "data/02_processed/n13337_features.parquet"
    )

    if not in_parquet.is_file():
        raise SystemExit(
            f"Missing input {in_parquet}. Run filter_aircraft.py first."
        )

    print(f"Reading {in_parquet}")
    raw = pl.read_parquet(in_parquet)
    featured = build_features(raw, features_cfg)

    save_outputs(featured, csv_out, parquet_out)
    print_summary(featured)
    print(f"Wrote pandas CSV    -> {csv_out}")
    print(f"Wrote polars Parquet -> {parquet_out}")
    return featured


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add onground and vertrate features to N13337 tracks."
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
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
