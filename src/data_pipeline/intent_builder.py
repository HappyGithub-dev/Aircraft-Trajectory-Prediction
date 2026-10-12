"""
Step 3: segment flights and attach initial departure-angle intent features.

Reads data/02_processed/n13337_features.parquet and writes enriched CSV + Parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_settings(config_path: Path | None = None) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = config_path or (REPO_ROOT / "configs" / "base_config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def acute_departure_angle(h_takeoff: float, h_exit: float) -> float:
    """Shortest compass arc, then acute 0..90 degrees."""
    diff = abs(h_exit - h_takeoff) % 360.0
    shortest = min(diff, 360.0 - diff)  # 0..180
    return min(shortest, 180.0 - shortest)  # 0..90


def _is_near_field(
    altitude: float | None,
    onground: bool | None,
    airport_altitude_ft: float,
    near_field_ft: float,
) -> bool:
    if onground:
        return True
    if altitude is None:
        return False
    return abs(altitude - airport_altitude_ft) < near_field_ft


def segment_closed_circuits(
    df: pl.DataFrame,
    airport_altitude_ft: float,
    near_field_ft: float,
    exit_agl_ft: float,
    gap_seconds: float,
) -> pl.DataFrame:
    """
    Keep one flight per takeoff-to-landing circuit, across every day.

    A circuit starts when the aircraft leaves the near-field band, must climb
    at least exit_agl_ft above the field, and ends when it returns to that band.
    Open segments (no landing, or a coverage gap before landing) are dropped.
    """
    sorted_df = df.sort("timestamp").with_row_index("_row")
    rows = sorted_df.select(
        ["_row", "timestamp", "day_folder", "Altitude", "Heading", "onground"]
    ).to_dicts()

    exit_alt = airport_altitude_ft + exit_agl_ft
    assignments: list[dict] = []
    flight_id = 0
    i = 0
    n = len(rows)

    while i < n:
        while i < n and _is_near_field(
            rows[i]["Altitude"],
            rows[i]["onground"],
            airport_altitude_ft,
            near_field_ft,
        ):
            i += 1
        if i >= n:
            break

        start = i
        saw_exit = False
        h_takeoff = rows[start]["Heading"]
        h_exit = None
        end = start
        j = start
        closed = False

        while j < n:
            if j > start:
                prev = rows[j - 1]
                cur = rows[j]
                gap = (cur["timestamp"] - prev["timestamp"]).total_seconds()
                day_changed = cur["day_folder"] != prev["day_folder"]
                if gap > gap_seconds or day_changed:
                    break

            cur = rows[j]
            if h_takeoff is None and cur["Heading"] is not None:
                h_takeoff = cur["Heading"]
            if cur["Altitude"] is not None and cur["Altitude"] >= exit_alt:
                saw_exit = True
                if h_exit is None and cur["Heading"] is not None:
                    h_exit = cur["Heading"]

            returned = saw_exit and _is_near_field(
                cur["Altitude"],
                cur["onground"],
                airport_altitude_ft,
                near_field_ft,
            )
            end = j
            j += 1
            if not returned:
                continue

            while j < n and _is_near_field(
                rows[j]["Altitude"],
                rows[j]["onground"],
                airport_altitude_ft,
                near_field_ft,
            ):
                gap = (rows[j]["timestamp"] - rows[j - 1]["timestamp"]).total_seconds()
                day_changed = rows[j]["day_folder"] != rows[j - 1]["day_folder"]
                if gap > gap_seconds or day_changed:
                    break
                end = j
                j += 1
            closed = True
            break

        if closed and saw_exit:
            flight_id += 1
            angle = None
            if h_takeoff is not None and h_exit is not None:
                angle = acute_departure_angle(float(h_takeoff), float(h_exit))
            for row in rows[start : end + 1]:
                assignments.append(
                    {
                        "_row": row["_row"],
                        "flight_id": flight_id,
                        "departure_angle": angle,
                    }
                )
        i = max(j, end + 1)

    if not assignments:
        raise SystemExit("No closed takeoff-to-landing circuits found.")

    labels = pl.DataFrame(assignments)
    # A null angle makes this column Null-typed; cast so the join is stable.
    labels = labels.with_columns(pl.col("departure_angle").cast(pl.Float64))
    kept = sorted_df.join(labels, on="_row", how="inner").drop("_row")
    return kept.sort(["flight_id", "timestamp"])


def add_mission_one_hots(df: pl.DataFrame) -> pl.DataFrame:
    """
    Proposal bins (gaps between ranges -> intent_mission_other):
      75-90 short, 40-60 medium, 15-35 long, 0-15 cross_country
    """
    a = pl.col("departure_angle")
    # 15° belongs only to the long bin so the flags stay mutually exclusive.
    short = (a >= 75) & (a <= 90)
    medium = (a >= 40) & (a <= 60)
    long = (a >= 15) & (a <= 35)
    cross = (a >= 0) & (a < 15)
    known = short | medium | long | cross

    return df.with_columns(
        short.cast(pl.Int8).alias("intent_mission_short"),
        medium.cast(pl.Int8).alias("intent_mission_medium"),
        long.cast(pl.Int8).alias("intent_mission_long"),
        cross.cast(pl.Int8).alias("intent_mission_cross_country"),
        (
            pl.when(a.is_null())
            .then(0)
            .when(~known)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("intent_mission_other")
        ),
    )


def build_intent(
    df: pl.DataFrame,
    gap_seconds: float,
    airport_altitude_ft: float,
    near_field_ft: float,
    exit_agl_ft: float,
) -> pl.DataFrame:
    circuits = segment_closed_circuits(
        df,
        airport_altitude_ft=airport_altitude_ft,
        near_field_ft=near_field_ft,
        exit_agl_ft=exit_agl_ft,
        gap_seconds=gap_seconds,
    )
    return add_mission_one_hots(circuits)


def save_outputs(df: pl.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    df.to_pandas().to_csv(csv_path, index=False)


def print_summary(df: pl.DataFrame, source_rows: int) -> None:
    n_flights = df.select("flight_id").n_unique()
    n_days = df.select("day_folder").n_unique()
    per_flight = df.group_by("flight_id").agg(
        pl.col("departure_angle").first().alias("departure_angle"),
        (
            (pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_seconds()
            / 60.0
        ).alias("duration_min"),
    )
    print(f"Source rows: {source_rows:,}")
    print(f"Circuit rows kept: {df.height:,}")
    print(f"Days with at least one circuit: {n_days:,}")
    print(f"Closed takeoff-to-landing flights: {n_flights:,}")
    durations = per_flight.select(
        (pl.col("duration_min") < 20).sum().alias("under_20"),
        ((pl.col("duration_min") >= 20) & (pl.col("duration_min") < 45)).sum().alias("m20_45"),
        ((pl.col("duration_min") >= 45) & (pl.col("duration_min") < 90)).sum().alias("m45_90"),
        (pl.col("duration_min") >= 90).sum().alias("over_90"),
        pl.col("duration_min").median().alias("median"),
    ).row(0)
    print(
        "Duration — "
        f"median={durations[4]:.1f} min, "
        f"<20={durations[0]}, 20-45={durations[1]}, "
        f"45-90={durations[2]}, 90+={durations[3]}"
    )
    n_valid = int(per_flight.filter(pl.col("departure_angle").is_not_null()).height)
    print(f"Flights with valid departure_angle: {n_valid:,}")

    if n_valid:
        stats = per_flight.filter(pl.col("departure_angle").is_not_null()).select(
            pl.col("departure_angle").mean().alias("mean"),
            pl.col("departure_angle").median().alias("median"),
            pl.col("departure_angle").min().alias("min"),
            pl.col("departure_angle").max().alias("max"),
        ).row(0)
        print(
            "departure_angle deg — "
            f"mean={stats[0]:.2f}, median={stats[1]:.2f}, "
            f"min={stats[2]:.2f}, max={stats[3]:.2f}"
        )

    # Bin counts at flight level (one row per flight)
    binned = per_flight.join(
        df.group_by("flight_id").agg(
            pl.col("intent_mission_short").first(),
            pl.col("intent_mission_medium").first(),
            pl.col("intent_mission_long").first(),
            pl.col("intent_mission_cross_country").first(),
            pl.col("intent_mission_other").first(),
        ),
        on="flight_id",
    )
    for name in (
        "intent_mission_short",
        "intent_mission_medium",
        "intent_mission_long",
        "intent_mission_cross_country",
        "intent_mission_other",
    ):
        count = int(binned.select(pl.col(name).sum()).item())
        print(f"{name}: {count}")


def run(config_path: Path | None = None) -> pl.DataFrame:
    cfg = load_settings(config_path)
    paths = cfg.get("paths", {})
    features_cfg = cfg.get("features", {})
    intent_cfg = cfg.get("intent", {})

    in_parquet = REPO_ROOT / paths.get(
        "n13337_features_parquet", "data/02_processed/n13337_features.parquet"
    )
    csv_out = REPO_ROOT / paths.get(
        "n13337_with_intent_csv", "data/02_processed/n13337_with_intent.csv"
    )
    parquet_out = REPO_ROOT / paths.get(
        "n13337_with_intent_parquet", "data/02_processed/n13337_with_intent.parquet"
    )

    if not in_parquet.is_file():
        raise SystemExit(
            f"Missing input {in_parquet}. Run feature_builder.py first."
        )

    gap_seconds = float(intent_cfg.get("flight_gap_seconds", 600))
    exit_agl_ft = float(intent_cfg.get("exit_agl_ft", 500))
    near_field_ft = float(intent_cfg.get("near_field_ft", 150))
    airport_alt = float(features_cfg.get("airport_altitude_ft", 1248))

    print(f"Reading {in_parquet}")
    raw = pl.read_parquet(in_parquet)
    intent_df = build_intent(
        raw,
        gap_seconds=gap_seconds,
        airport_altitude_ft=airport_alt,
        near_field_ft=near_field_ft,
        exit_agl_ft=exit_agl_ft,
    )

    save_outputs(intent_df, csv_out, parquet_out)
    print_summary(intent_df, source_rows=raw.height)
    print(f"Wrote pandas CSV    -> {csv_out}")
    print(f"Wrote polars Parquet -> {parquet_out}")
    return intent_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add departure-angle intent features to N13337 tracks."
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
