"""
Step 3: convert latitude, longitude, and altitude into a local NED grid.

North and east are meters from KBTP. Down is meters, positive toward the
ground, so an aircraft above the field has a negative down value.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
EARTH_RADIUS_M = 6_378_137.0
FT_TO_M = 0.3048


def load_settings(config_path: Path | None = None) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = config_path or (REPO_ROOT / "configs" / "base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def geodetic_to_ned(
    lat_deg: pl.Expr,
    lon_deg: pl.Expr,
    alt_ft: pl.Expr,
    lat0_deg: float,
    lon0_deg: float,
    alt0_ft: float,
) -> list[pl.Expr]:
    """Local tangent-plane North, East, Down in meters."""
    lat0 = math.radians(lat0_deg)
    north_m = (lat_deg - lat0_deg) * (math.pi / 180.0) * EARTH_RADIUS_M
    east_m = (lon_deg - lon0_deg) * (math.pi / 180.0) * EARTH_RADIUS_M * math.cos(lat0)
    down_m = -((alt_ft - alt0_ft) * FT_TO_M)
    return [
        north_m.alias("north_m"),
        east_m.alias("east_m"),
        down_m.alias("down_m"),
    ]


def add_ned_columns(
    df: pl.DataFrame,
    lat0_deg: float,
    lon0_deg: float,
    alt0_ft: float,
) -> pl.DataFrame:
    return df.with_columns(
        geodetic_to_ned(
            pl.col("Lat"),
            pl.col("Lon"),
            pl.col("Altitude"),
            lat0_deg,
            lon0_deg,
            alt0_ft,
        )
    )


def run(config_path: Path | None = None) -> pl.DataFrame:
    cfg = load_settings(config_path)
    paths = cfg.get("paths", {})
    features = cfg.get("features", {})
    in_parquet = REPO_ROOT / paths.get(
        "n13337_with_intent_parquet", "data/02_processed/n13337_with_intent.parquet"
    )
    if not in_parquet.is_file():
        raise SystemExit(f"Missing input {in_parquet}. Run intent_builder.py first.")

    lat0 = float(features.get("airport_latitude_deg", 40.7764))
    lon0 = float(features.get("airport_longitude_deg", -79.9497))
    alt0 = float(features.get("airport_altitude_ft", 1248))

    print(f"Reading {in_parquet}")
    print(f"NED origin: lat {lat0}, lon {lon0}, elevation {alt0} ft")
    converted = add_ned_columns(pl.read_parquet(in_parquet), lat0, lon0, alt0)
    converted.write_parquet(in_parquet)

    stats = converted.select(
        pl.col("north_m").min().alias("n_min"),
        pl.col("north_m").max().alias("n_max"),
        pl.col("east_m").min().alias("e_min"),
        pl.col("east_m").max().alias("e_max"),
        pl.col("down_m").min().alias("d_min"),
        pl.col("down_m").max().alias("d_max"),
    ).row(0)
    print(f"Rows: {converted.height:,}")
    print(f"north_m {stats[0]:.0f} to {stats[1]:.0f}")
    print(f"east_m  {stats[2]:.0f} to {stats[3]:.0f}")
    print(f"down_m  {stats[4]:.0f} to {stats[5]:.0f}")
    print(f"Updated {in_parquet}")
    return converted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add north_m, east_m, and down_m columns in a local NED grid."
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    run(config_path=args.config)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
