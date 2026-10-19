"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 4: Multi-Year ERA5 Archive Collection & Cleaning (Modeling Extension Dataset)

Why this step exists:
  - The core Topic 6 dataset (steps 1-3) covers ~3 months, which never contains a full seasonal cycle.
  - Weather has strong yearly seasonality plus inter-annual variability (ENSO: El Nino / La Nina),
    so a trustworthy model evaluation needs walk-forward validation across many years.

What it does:
  1. Calls `https://archive-api.open-meteo.com/v1/archive` (ERA5 / ECMWF Reanalysis) with our own code,
     year by year, for the same 20 locations as step 1 (2016-01-01 -> ~6 days ago, ERA5 publication delay).
  2. Respects the Open-Meteo free-tier limits (600 calls/min, 5,000 calls/hour, 10,000 calls/day).
     Requests longer than 2 weeks are billed as multiple calls, so one year x 20 locations ~= 521 calls.
     A client-side rolling budget waits automatically before each request.
  3. Saves one resumable parquet chunk per year (`data/archive/chunks/era5_<year>.parquet`), so a re-run
     only downloads what is missing.
  4. Cleans the combined series with the same rules as step 2 (range validation, gust consistency,
     interpolation of short gaps, continuous hourly grid, 2-decimal rounding) and writes:
       - data/archive/era5_multiyear_cleaned.parquet
       - data/archive/archive_quality_report.json
"""

import importlib.util
import json
import sys
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
from retry_requests import retry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = BASE_DIR / "data" / "archive"
CHUNK_DIR = ARCHIVE_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_YEAR = 2016
ERA5_DELAY_DAYS = 6
TIMEZONE = "Asia/Bangkok"

HOURLY_VARS = [
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

# Stay safely below the free-tier limits (600 / minute, 5,000 / hour)
MINUTE_BUDGET = 560.0
HOUR_BUDGET = 4700.0


def load_locations() -> list[dict]:
    """Reuse the exact 20-location list from step 1 so both datasets share the same coordinates."""
    spec = importlib.util.spec_from_file_location(
        "collect_step1", str(BASE_DIR / "src" / "01_collect_weather_data.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [dict(loc) for loc in module.LOCATION_METADATA]


def api_call_weight(n_locations: int, n_days: int, n_vars: int) -> float:
    """Open-Meteo fractional call counting: >2 weeks or >10 variables per location count as multiple calls."""
    return n_locations * max(1.0, n_days / 14.0) * max(1.0, n_vars / 10.0)


class RollingRateBudget:
    """Client-side rolling-window limiter for the minutely and hourly API budgets."""

    def __init__(self, minute_budget: float, hour_budget: float):
        self.minute_budget = minute_budget
        self.hour_budget = hour_budget
        self.history: deque[tuple[float, float]] = deque()

    def _used(self, window_sec: float, now: float) -> float:
        return sum(w for t, w in self.history if now - t < window_sec)

    def wait_for(self, weight: float) -> None:
        while True:
            now = time.time()
            while self.history and now - self.history[0][0] >= 3600:
                self.history.popleft()
            used_min = self._used(60, now)
            used_hour = self._used(3600, now)
            if used_min + weight <= self.minute_budget and used_hour + weight <= self.hour_budget:
                return
            if used_hour + weight > self.hour_budget:
                oldest_t = self.history[0][0]
                wait_sec = max(5.0, 3600 - (now - oldest_t) + 2)
                reason = f"hourly budget ({used_hour:,.0f} + {weight:,.0f} > {self.hour_budget:,.0f})"
            else:
                wait_sec = 62.0
                reason = f"minutely budget ({used_min:,.0f} + {weight:,.0f} > {self.minute_budget:,.0f})"
            print(f"      [rate-limit] Waiting {wait_sec / 60:.1f} min for {reason}...", flush=True)
            time.sleep(min(wait_sec, 300.0))

    def record(self, weight: float) -> None:
        self.history.append((time.time(), weight))


def fetch_archive_chunk(client, locations: list[dict], start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": [loc["req_lat"] for loc in locations],
        "longitude": [loc["req_lon"] for loc in locations],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARS,
        "timezone": TIMEZONE,
    }
    responses = client.weather_api(ARCHIVE_URL, params=params)
    frames = []
    for idx, response in enumerate(responses):
        meta = locations[idx]
        hourly = response.Hourly()
        timestamps = (
            pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
            .tz_convert(TIMEZONE)
            .tz_localize(None)
        )
        data = {
            "location_id": meta["location_id"],
            "city": meta["city"],
            "region": meta["region"],
            "requested_lat": meta["req_lat"],
            "requested_lon": meta["req_lon"],
            "elevation_m": round(float(response.Elevation()), 1),
            "timestamp": timestamps,
        }
        for v_idx, var in enumerate(HOURLY_VARS):
            data[var] = hourly.Variables(v_idx).ValuesAsNumpy().astype(np.float64)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def collect_all_chunks(locations: list[dict]) -> list[Path]:
    session = retry(requests.Session(), retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=session)
    budget = RollingRateBudget(MINUTE_BUDGET, HOUR_BUDGET)

    end_date = date.today() - timedelta(days=ERA5_DELAY_DAYS)
    chunk_paths = []
    total_weight = 0.0

    for year in range(START_YEAR, end_date.year + 1):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), end_date)
        chunk_path = CHUNK_DIR / f"era5_{year}.parquet"
        chunk_paths.append(chunk_path)

        if chunk_path.exists():
            stored_max = pd.read_parquet(chunk_path, columns=["timestamp"])["timestamp"].max()
            if year < end_date.year or stored_max.date() >= end:
                print(f"  [{year}] Chunk already collected -> skip ({chunk_path.name})")
                continue

        n_days = (end - start).days + 1
        weight = api_call_weight(len(locations), n_days, len(HOURLY_VARS))
        print(f"  [{year}] {start} -> {end} | {len(locations)} locations | billed ~{weight:,.0f} API calls")
        budget.wait_for(weight)

        df_chunk = None
        for attempt in range(1, 7):
            try:
                t0 = time.perf_counter()
                df_chunk = fetch_archive_chunk(client, locations, start, end)
                print(f"      Received {len(df_chunk):,} hourly rows in {time.perf_counter() - t0:.1f}s")
                break
            except Exception as exc:  # Open-Meteo returns the reason string for limit errors
                msg = str(exc)
                if "Hourly" in msg:
                    wait_sec = 600
                elif "Minutely" in msg or "limit" in msg.lower():
                    wait_sec = 70
                else:
                    wait_sec = 20 * attempt
                print(f"      Attempt {attempt} failed: {msg[:160]} -> retry in {wait_sec}s", flush=True)
                time.sleep(wait_sec)
        if df_chunk is None:
            raise RuntimeError(f"Could not collect ERA5 archive chunk for {year}. Re-run later to resume.")

        budget.record(weight)
        total_weight += weight
        df_chunk.to_parquet(chunk_path, index=False)

    print(f"  Billed API calls in this run: ~{total_weight:,.0f}")
    return chunk_paths


def clean_multiyear_archive(chunk_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    df = pd.concat([pd.read_parquet(p) for p in chunk_paths], ignore_index=True)
    df = df.sort_values(["location_id", "timestamp"])

    raw_rows = int(len(df))
    duplicates = int(df.duplicated(subset=["location_id", "timestamp"]).sum())
    df = df.drop_duplicates(subset=["location_id", "timestamp"], keep="last")

    raw_missing = {v: int(df[v].isna().sum()) for v in HOURLY_VARS}

    # ERA5 tail: the last days may not be published yet (all variables NaN) -> truncate to a common end
    valid_mask = df[HOURLY_VARS].notna().any(axis=1)
    last_valid_per_loc = df[valid_mask].groupby("location_id")["timestamp"].max()
    common_end = last_valid_per_loc.min()
    tail_rows_removed = int((df["timestamp"] > common_end).sum())
    df = df[df["timestamp"] <= common_end].copy()

    range_violations = {}
    for var in HOURLY_VARS:
        low, high = VALID_RANGES[var]
        bad = (df[var] < low) | (df[var] > high)
        range_violations[var] = int(bad.sum())
        df.loc[bad, var] = np.nan

    cleaned_groups = []
    missing_grid_slots = 0
    for loc_id, g in df.groupby("location_id", sort=True):
        g = g.set_index("timestamp").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq="1h")
        missing_grid_slots += int(len(grid) - len(g))
        g = g.reindex(grid)
        g.index.name = "timestamp"
        for meta_col in ["location_id", "city", "region", "requested_lat", "requested_lon", "elevation_m"]:
            g[meta_col] = g[meta_col].ffill().bfill()
        for var in HOURLY_VARS:
            if var == "precipitation":
                g[var] = g[var].fillna(0.0)
            else:
                g[var] = g[var].interpolate(method="time", limit=6).ffill().bfill()
        cleaned_groups.append(g.reset_index())

    df_clean = pd.concat(cleaned_groups, ignore_index=True)
    gust_issues = int((df_clean["wind_gusts_10m"] < df_clean["wind_speed_10m"]).sum())
    df_clean["wind_gusts_10m"] = np.maximum(df_clean["wind_gusts_10m"], df_clean["wind_speed_10m"])
    for var in HOURLY_VARS:
        df_clean[var] = df_clean[var].round(2)
    df_clean["location_id"] = df_clean["location_id"].astype(int)
    df_clean = df_clean.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    report = {
        "source": ARCHIVE_URL,
        "model": "ERA5 / ECMWF Reanalysis (Open-Meteo best_match)",
        "collected_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "before_cleaning": {
            "total_rows": raw_rows,
            "duplicate_location_hours": duplicates,
            "missing_values_by_column": raw_missing,
            "unpublished_era5_tail_rows": tail_rows_removed,
            "physical_range_violations": range_violations,
            "missing_hourly_slots": missing_grid_slots,
            "wind_gust_inconsistencies": gust_issues,
        },
        "after_cleaning": {
            "total_rows": int(len(df_clean)),
            "num_locations": int(df_clean["location_id"].nunique()),
            "hours_per_location": int(df_clean.groupby("location_id").size().iloc[0]),
            "time_range_start": str(df_clean["timestamp"].min()),
            "time_range_end": str(df_clean["timestamp"].max()),
            "total_missing_cells": int(df_clean[HOURLY_VARS].isna().sum().sum()),
            "wind_gust_inconsistencies": int((df_clean["wind_gusts_10m"] < df_clean["wind_speed_10m"]).sum()),
        },
    }
    return df_clean, report


def main():
    print("=" * 78)
    print("STEP 4: MULTI-YEAR ERA5 ARCHIVE COLLECTION (2016 -> NOW) FOR WALK-FORWARD VALIDATION")
    print(f"        Endpoint: {ARCHIVE_URL}")
    print("=" * 78)

    locations = load_locations()
    chunk_paths = collect_all_chunks(locations)

    print("\nCleaning combined multi-year archive...")
    df_clean, report = clean_multiyear_archive(chunk_paths)

    out_path = ARCHIVE_DIR / "era5_multiyear_cleaned.parquet"
    df_clean.to_parquet(out_path, index=False)
    report_path = ARCHIVE_DIR / "archive_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    a = report["after_cleaning"]
    b = report["before_cleaning"]
    print("=" * 78)
    print(f"Raw rows                    : {b['total_rows']:,}")
    print(f"Unpublished ERA5 tail rows  : {b['unpublished_era5_tail_rows']:,}")
    print(f"Range violations            : {sum(b['physical_range_violations'].values()):,}")
    print(f"Gust inconsistencies fixed  : {b['wind_gust_inconsistencies']:,}")
    print(f"Cleaned rows                : {a['total_rows']:,} ({a['num_locations']} locations x {a['hours_per_location']:,} hours)")
    print(f"Time range                  : {a['time_range_start']} -> {a['time_range_end']}")
    print(f"Saved cleaned archive       : {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved quality report        : {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
