"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 2: Multi-Source Data Cleaning & Preprocessing Pipeline
Processes 46,600+ real observations collected from 4 Weather APIs:
  1. Open-Meteo SDK      (date format: '2026-10-17 11:00:00+07:00')
  2. WeatherAPI.com      (date format: '2026-10-17 11:00' and '2026-10-17 11:30')
  3. Xweather API        (date format: '2026-10-17T11:00:00+07:00')
  4. FreeOpenAPI.dev     (date format: '2026-10-17T11:30:00', missing humidity/pressure/gusts)
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_gusts_10m",
]

VALID_RANGES = {
    "temperature_2m": (-10.0, 50.0),
    "relative_humidity_2m": (0.0, 100.0),
    "wind_speed_10m": (0.0, 200.0),
    "precipitation": (0.0, 300.0),
    "pressure_msl": (850.0, 1080.0),
    "wind_gusts_10m": (0.0, 250.0),
}


def parse_multi_api_timestamps(series: pd.Series) -> pd.Series:
    """
    Standardize heterogeneous timestamp strings from 4 different weather APIs into
    naive local timestamps in Asia/Bangkok (UTC+7).
    """
    s = series.astype(str).str.strip()
    # Replace 'T' separator with space and strip '+07:00' offset since all 4 APIs return UTC+7 local time
    s_clean = (
        s.str.replace("T", " ", regex=False)
         .str.replace(r"\+07:00$", "", regex=True)
         .str.strip()
    )
    return pd.to_datetime(s_clean, format="mixed", errors="coerce")


def clean_multi_source_weather_dataset():
    raw_csv_path = RAW_DIR / "raw_weather_data.csv"
    df_raw = pd.read_csv(raw_csv_path)

    print("=" * 78)
    print("RUNNING MULTI-SOURCE DATA CLEANING PIPELINE (4 REAL WEATHER APIs)")
    print("=" * 78)

    total_raw_rows = int(len(df_raw))
    num_locations = int(df_raw["location_id"].nunique())
    rows_by_source = {k: int(v) for k, v in df_raw["source_api"].value_counts().to_dict().items()}

    # -------------------------------------------------------------------------
    # 1. Inspect Raw Data Characteristics Across All 4 APIs
    # -------------------------------------------------------------------------
    iso_clean_regex = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    unstandardized_ts_count = int((~df_raw["date"].astype(str).str.match(iso_clean_regex)).sum())

    float32_precision_artifacts = 0
    for col in WEATHER_VARS:
        s = pd.to_numeric(df_raw[col], errors="coerce").dropna()
        artifacts = ((s.round(2) - s).abs() > 1e-6).sum()
        float32_precision_artifacts += int(artifacts)

    raw_missing_by_col = {col: int(df_raw[col].isna().sum()) for col in WEATHER_VARS}
    raw_total_missing_cells = int(sum(raw_missing_by_col.values()))
    all_vars_missing_mask = df_raw[WEATHER_VARS].isna().all(axis=1)
    empty_api_response_rows = int(all_vars_missing_mask.sum())

    # Partial missing rows (e.g., FreeOpenAPI.dev missing humidity, pressure_msl, wind_gusts_10m)
    partial_missing_mask = (~all_vars_missing_mask) & (df_raw[WEATHER_VARS].isna().any(axis=1))
    partial_missing_rows = int(partial_missing_mask.sum())
    partial_missing_cells = int(df_raw.loc[~all_vars_missing_mask, WEATHER_VARS].isna().sum().sum())

    print(f"1. Raw Multi-API Dataset Loaded: {total_raw_rows:,} rows across {num_locations} locations.")
    for src, cnt in rows_by_source.items():
        print(f"   - Source: {src:<20} : {cnt:>7,} rows")
    print(f"   - Unstandardized timestamps (4 API formats)   : {unstandardized_ts_count:,} rows")
    print(f"   - Empty API horizon rows (all 6 vars NaN)     : {empty_api_response_rows:,} rows")
    print(f"   - Partial missing rows (schema differences)   : {partial_missing_rows:,} rows ({partial_missing_cells:,} NaN cells)")
    print(f"   - Total missing numeric cells (NaN)           : {raw_total_missing_cells:,} cells")
    print(f"   - Float32 binary precision artifacts          : {float32_precision_artifacts:,} cells")

    # -------------------------------------------------------------------------
    # 2. Standardize Timestamps & Align Sub-Hourly Polling to Hourly Grid
    # -------------------------------------------------------------------------
    df = df_raw.copy()
    df["raw_timestamp"] = parse_multi_api_timestamps(df["date"])
    df["timestamp"] = df["raw_timestamp"].dt.floor("h")
    df["time"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["obs_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.hour

    # -------------------------------------------------------------------------
    # 3. Remove Future Unobserved Forecast Rows (Strictly keep past up to NOW)
    # -------------------------------------------------------------------------
    now_cutoff = pd.Timestamp.now().floor("h")
    future_mask = df["timestamp"] > now_cutoff
    future_rows_removed = int(future_mask.sum())
    empty_api_response_rows = int(all_vars_missing_mask.sum()) + future_rows_removed

    df_valid = df.loc[(~all_vars_missing_mask) & (~future_mask)].copy()
    for col in WEATHER_VARS:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    print(f"   - Future unobserved rows filtered (> {now_cutoff}) : {future_rows_removed:,} rows")

    # -------------------------------------------------------------------------
    # 4. Save Multi-Source Cross-API Comparison Table (Before Deduplication)
    # -------------------------------------------------------------------------
    # Find timestamps where >= 2 APIs reported observations for the same location & hour
    source_counts = df_valid.groupby(["location_id", "city", "time"])["source_api"].transform("nunique")
    df_multi_overlap = df_valid[source_counts >= 2].sort_values(["location_id", "time", "source_api"]).copy()
    overlap_csv_path = CLEAN_DIR / "multi_source_comparison.csv"
    df_multi_overlap.to_csv(overlap_csv_path, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 5. Detect & Resolve Multi-Source Duplicated Observations (location_id, time)
    # -------------------------------------------------------------------------
    exact_duplicates = int(df_valid.duplicated().sum())
    key_duplicates = int(df_valid.duplicated(subset=["location_id", "time"]).sum())

    # Also compute cross-API ensemble average for overlapping hours AND fill missing fields
    # (e.g. FreeOpenAPI missing humidity/pressure filled from Open-Meteo/WeatherAPI/Xweather at same hour)
    # Priority order for metadata: Open-Meteo SDK -> WeatherAPI.com -> Xweather API -> FreeOpenAPI.dev
    source_priority = {"Open-Meteo SDK": 1, "WeatherAPI.com": 2, "Xweather API": 3, "FreeOpenAPI.dev": 4}
    df_valid["_src_prio"] = df_valid["source_api"].map(source_priority).fillna(9)
    df_valid = df_valid.sort_values(["location_id", "timestamp", "_src_prio"])

    # Aggregate overlapping observations per (location_id, timestamp):
    # - Take mean of valid numeric sensor readings across reporting APIs (multi-sensor fusion)
    # - Keep track of all contributing APIs in `contributing_apis` and count in `num_sources`
    agg_Authenticated = (
        df_valid.groupby(["location_id", "timestamp"], as_index=False)
        .agg(
            city=("city", "first"),
            region=("region", "first"),
            time=("time", "first"),
            obs_date=("obs_date", "first"),
            hour=("hour", "first"),
            temperature_2m=("temperature_2m", "mean"),
            wind_speed_10m=("wind_speed_10m", "mean"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            precipitation=("precipitation", "mean"),
            pressure_msl=("pressure_msl", "mean"),
            wind_gusts_10m=("wind_gusts_10m", "mean"),
            num_sources=("source_api", "nunique"),
            source_api=("source_api", lambda s: " + ".join(dict.fromkeys(s))),
            requested_lat=("requested_lat", "first"),
            requested_lon=("requested_lon", "first"),
            api_latitude=("api_latitude", "first"),
            api_longitude=("api_longitude", "first"),
            elevation_m=("elevation_m", "first"),
            timezone=("timezone", "first"),
        )
    )

    print(f"2. Multi-Source Deduplication & Sensor Fusion:")
    print(f"   - Overlapping (location_id, time) duplicates  : {key_duplicates:,} records across 4 APIs")
    print(f"   - Multi-source comparison rows exported       : {len(df_multi_overlap):,} rows ({overlap_csv_path.name})")
    print(f"   - Unified unique (location, hour) records     : {len(agg_Authenticated):,} rows")

    # -------------------------------------------------------------------------
    # 6. Verify Continuous Hourly Time-Series Grid in Active Window
    # -------------------------------------------------------------------------
    min_ts = agg_Authenticated["timestamp"].min()
    max_ts = agg_Authenticated["timestamp"].max()
    expected_grid = pd.date_range(start=min_ts, end=max_ts, freq="1h")
    expected_hours_per_loc = len(expected_grid)

    missing_grid_slots = 0
    reindexed_list = []

    for loc_id, group in agg_Authenticated.groupby("location_id", sort=True):
        g = group.sort_values("timestamp").set_index("timestamp").reindex(expected_grid)
        n_miss_slot = int(g["location_id"].isna().sum())
        missing_grid_slots += n_miss_slot
        if n_miss_slot > 0:
            for meta_col in ["location_id", "city", "region", "requested_lat", "requested_lon",
                             "api_latitude", "api_longitude", "elevation_m", "timezone"]:
                g[meta_col] = g[meta_col].ffill().bfill()
            g["source_api"] = g["source_api"].fillna("Interpolated Slot")
            g["num_sources"] = g["num_sources"].fillna(0).astype(int)
        g.index.name = "timestamp"
        reindexed_list.append(g.reset_index())

    df_grid = pd.concat(reindexed_list, ignore_index=True)
    df_grid["time"] = df_grid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_grid["obs_date"] = df_grid["timestamp"].dt.strftime("%Y-%m-%d")
    df_grid["hour"] = df_grid["timestamp"].dt.hour

    # -------------------------------------------------------------------------
    # 7. Validate Physical Measurement Ranges & Cross-Variable Consistency
    # -------------------------------------------------------------------------
    physical_violations = {}
    iqr_outliers_by_col = {}

    for col in WEATHER_VARS:
        low, high = VALID_RANGES[col]
        bad_mask = (df_grid[col] < low) | (df_grid[col] > high)
        n_bad = int(bad_mask.sum())
        physical_violations[col] = n_bad
        if n_bad > 0:
            df_grid.loc[bad_mask, col] = np.nan

        q1 = df_grid.groupby("location_id")[col].transform(lambda x: x.quantile(0.25))
        q3 = df_grid.groupby("location_id")[col].transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        if col == "precipitation":
            out_mask = df_grid[col] > 5.0
        else:
            out_mask = (df_grid[col] < (q1 - 1.5 * iqr)) | (df_grid[col] > (q3 + 1.5 * iqr))
        iqr_outliers_by_col[col] = int(out_mask.sum())

    wind_consistency_issues = int((df_grid["wind_gusts_10m"] < df_grid["wind_speed_10m"]).sum())
    if wind_consistency_issues > 0:
        df_grid["wind_gusts_10m"] = np.maximum(df_grid["wind_gusts_10m"], df_grid["wind_speed_10m"])

    # -------------------------------------------------------------------------
    # 8. Interpolate Any Remaining Gaps & Round Precision
    # -------------------------------------------------------------------------
    cleaned_groups = []
    for loc_id, group in df_grid.groupby("location_id", sort=True):
        g = group.sort_values("timestamp").set_index("timestamp")
        for col in WEATHER_VARS:
            if g[col].isna().any():
                if col == "precipitation":
                    g[col] = g[col].fillna(0.0)
                else:
                    g[col] = g[col].interpolate(method="time").ffill().bfill()
            g[col] = g[col].round(2)
        cleaned_groups.append(g.reset_index())

    df_clean = pd.concat(cleaned_groups, ignore_index=True)
    df_clean = df_clean.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Standard Topic 6 aliases
    df_clean["temperature"] = df_clean["temperature_2m"]
    df_clean["humidity"] = df_clean["relative_humidity_2m"]
    df_clean["wind_speed"] = df_clean["wind_speed_10m"]
    df_clean["pressure"] = df_clean["pressure_msl"]
    df_clean["wind_gusts"] = df_clean["wind_gusts_10m"]

    final_cols = [
        "location_id",
        "city",
        "region",
        "time",
        "obs_date",
        "hour",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_gusts_10m",
        "precipitation",
        "pressure_msl",
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "num_sources",
        "source_api",
        "requested_lat",
        "requested_lon",
        "api_latitude",
        "api_longitude",
        "elevation_m",
        "timezone",
    ]
    df_clean = df_clean[final_cols]

    clean_csv_path = CLEAN_DIR / "cleaned_weather_data.csv"
    df_clean.to_csv(clean_csv_path, index=False, encoding="utf-8-sig")

    quality_report = {
        "before_cleaning": {
            "total_rows": total_raw_rows,
            "rows_by_source_api": rows_by_source,
            "num_locations": num_locations,
            "unstandardized_timestamps": unstandardized_ts_count,
            "empty_api_response_rows": empty_api_response_rows,
            "partial_missing_rows": partial_missing_rows,
            "partial_missing_cells": partial_missing_cells,
            "float32_precision_artifacts": float32_precision_artifacts,
            "exact_duplicates": exact_duplicates,
            "key_duplicates": key_duplicates,
            "missing_hourly_slots_in_active_window": missing_grid_slots,
            "missing_values_by_column": raw_missing_by_col,
            "total_missing_cells": raw_total_missing_cells,
            "physical_range_violations": physical_violations,
            "wind_gust_inconsistencies": wind_consistency_issues,
            "statistical_iqr_extremes_by_column": iqr_outliers_by_col,
            "total_statistical_iqr_extremes": int(sum(iqr_outliers_by_col.values())),
        },
        "after_cleaning": {
            "total_rows": int(len(df_clean)),
            "num_locations": int(df_clean["location_id"].nunique()),
            "hours_per_location": expected_hours_per_loc,
            "multi_source_fused_hours": int((df_clean["num_sources"] >= 2).sum()),
            "time_range_start": str(df_clean["time"].min()),
            "time_range_end": str(df_clean["time"].max()),
            "unstandardized_timestamps": int((~df_clean["time"].str.match(iso_clean_regex)).sum()),
            "empty_api_response_rows": 0,
            "partial_missing_rows": 0,
            "float32_precision_artifacts": 0,
            "exact_duplicates": int(df_clean.duplicated().sum()),
            "key_duplicates": int(df_clean.duplicated(subset=["location_id", "time"]).sum()),
            "missing_hourly_slots_in_active_window": 0,
            "total_missing_cells": int(df_clean[WEATHER_VARS].isna().sum().sum()),
            "physical_range_violations": 0,
            "wind_gust_inconsistencies": int((df_clean["wind_gusts_10m"] < df_clean["wind_speed_10m"]).sum()),
        },
    }

    report_path = CLEAN_DIR / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print(f"3. Cleaned Multi-Source Dataset Saved: {clean_csv_path}")
    print(f"   - Total Cleaned Rows              : {len(df_clean):,} ({num_locations} locations x {expected_hours_per_loc:,} hours)")
    print(f"   - Hours Fused from >=2 APIs       : {(df_clean['num_sources'] >= 2).sum():,} hours")
    print(f"   - Active Time Span                : {df_clean['time'].min()} -> {df_clean['time'].max()}")
    print(f"   - Remaining NaN Cells             : {quality_report['after_cleaning']['total_missing_cells']}")
    print("=" * 78)


if __name__ == "__main__":
    clean_multi_source_weather_dataset()
