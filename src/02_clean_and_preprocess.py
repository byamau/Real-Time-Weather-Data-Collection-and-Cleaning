"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 2: Data cleaning, standardization and multi-source integration

Input : data/raw/raw_weather_data.csv   (+ data/raw/api_request_log.csv when it exists)
Output: data/cleaned/cleaned_weather_data.csv      continuous hourly series per location
        data/cleaned/multi_source_comparison.csv   hours reported by >= 2 source families
        data/cleaned/api_coverage_report.csv       expected vs returned records per source x location
        data/cleaned/data_quality_report.json      before / after indicators of every step

Cleaning steps (each one is counted in the quality report)
  1. Detect missing API responses   : request log (status != ok) + coverage of every
                                      source x location (records returned vs. expected).
  2. Standardize timestamps         : 4+ string formats, with or without UTC offset, are parsed
                                      time-zone aware and converted to Asia/Ho_Chi_Minh (UTC+7),
                                      then aligned to the nearest hour.
  3. Remove empty records           : rows where every weather variable is missing.
  4. Remove not-yet-observed values : forecast hours later than the moment they were collected.
  5. Validate measurement ranges    : physical limits per variable (out of range -> missing).
  6. Detect duplicated observations : (a) exact duplicates, (b) same source family / location /
                                      hour (archive vs. recent model, current vs. hourly,
                                      repeated polls), (c) different sources at the same hour.
  7. Integrate sources              : one value per (location, hour) = mean over source families,
                                      with a cross-source spread and conflict flag.
  8. Combine observations over time : continuous hourly grid per location; gaps of <= 6 h are
                                      interpolated and flagged, longer gaps stay missing (NaN).
  9. Cross-variable consistency     : wind gust >= mean wind speed; rounding to 2 decimals.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "Asia/Ho_Chi_Minh"

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

# Lower rank wins when one source family reports the same location-hour more than once
RECORD_TYPE_RANK = {
    "reanalysis_hourly": 1,
    "conditions_hourly": 2,
    "current_conditions": 3,
    "model_hourly": 4,
    "forecast_hourly": 5,
}

# Cross-source disagreement thresholds used to flag inconsistent hours
CONFLICT_THRESHOLDS = {"temperature_2m": 3.0, "relative_humidity_2m": 20.0, "pressure_msl": 5.0}

MAX_INTERPOLATION_GAP_H = 6

# Expected records per request for runs collected before the request log existed
LEGACY_EXPECTED = {
    "Open-Meteo Archive": None,          # one batch request; expected = rows returned for that location
    "Open-Meteo Forecast": 6 * 24,       # past_days=5 + today
    "WeatherAPI.com": 1 + 3 * 24,        # current + 3 days hourly
    "Xweather API": 24,                  # last 24 hourly periods
    "FreeOpenAPI.dev": 1,                # current snapshot
}


# =============================================================================
# 0. Load raw data (and label rows collected by the first version of step 1)
# =============================================================================
def _contiguous_blocks(loc_ids: pd.Series) -> pd.Series:
    return (loc_ids != loc_ids.shift()).cumsum()


def upgrade_legacy_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    The first collector version wrote neither run_id, collected_at nor record_type, and labelled
    both Open-Meteo endpoints "Open-Meteo SDK". It always wrote rows in a fixed order:
    archive (20 location blocks) -> forecast past_days (20 blocks) -> WeatherAPI (per location:
    1 current row, then hourly rows) -> Xweather -> FreeOpenAPI. That order lets us recover the
    labels exactly without guessing.
    """
    for col in ["run_id", "collected_at", "source_family", "record_type", "endpoint"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(object)
    legacy = df["run_id"].isna()
    n_legacy = int(legacy.sum())
    if n_legacy == 0:
        return df, 0

    leg = df.loc[legacy].copy()
    om = leg["source_api"] == "Open-Meteo SDK"
    om_blocks = _contiguous_blocks(leg.loc[om, "location_id"])
    n_locations = leg["location_id"].nunique()
    is_archive = om_blocks <= n_locations
    leg.loc[om_blocks.index[is_archive], ["source_api", "record_type", "endpoint"]] = [
        "Open-Meteo Archive", "reanalysis_hourly", "https://archive-api.open-meteo.com/v1/archive"]
    leg.loc[om_blocks.index[~is_archive], ["source_api", "record_type", "endpoint"]] = [
        "Open-Meteo Forecast", "model_hourly", "https://api.open-meteo.com/v1/forecast"]

    wa = leg["source_api"] == "WeatherAPI.com"
    wa_blocks = _contiguous_blocks(leg.loc[wa, "location_id"])
    first_in_block = ~wa_blocks.duplicated()
    leg.loc[wa_blocks.index[first_in_block], "record_type"] = "current_conditions"
    leg.loc[wa_blocks.index[~first_in_block], "record_type"] = "forecast_hourly"
    leg.loc[wa, "endpoint"] = "https://api.weatherapi.com/v1/forecast.json"

    xw = leg["source_api"] == "Xweather API"
    leg.loc[xw, ["record_type", "endpoint"]] = ["conditions_hourly", "https://data.api.xweather.com/conditions"]
    fo = leg["source_api"] == "FreeOpenAPI.dev"
    leg.loc[fo, ["record_type", "endpoint"]] = ["current_conditions", "https://www.freeopenapi.dev/api/v1/weather/current"]

    leg["source_family"] = leg["source_api"].replace(
        {"Open-Meteo Archive": "Open-Meteo", "Open-Meteo Forecast": "Open-Meteo"})
    leg["run_id"] = "legacy-run"
    df.loc[legacy] = leg
    return df, n_legacy


# =============================================================================
# 2. Timestamp standardization
# =============================================================================
TS_FORMAT_PATTERNS = {
    "YYYY-MM-DD HH:MM:SS+07:00": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",
    "YYYY-MM-DDTHH:MM:SS+07:00": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",
    "YYYY-MM-DDTHH:MM:SSZ (UTC)": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
    "YYYY-MM-DDTHH:MM:SS (no offset)": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
    "YYYY-MM-DDTHH:MM (no offset)": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$",
    "YYYY-MM-DD HH:MM (no offset)": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$",
    "YYYY-MM-DD HH:MM:SS (standard)": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
}


def classify_timestamp_formats(s: pd.Series) -> dict:
    s = s.astype(str).str.strip()
    counts = {}
    matched = pd.Series(False, index=s.index)
    for name, pat in TS_FORMAT_PATTERNS.items():
        m = s.str.match(pat) & ~matched
        if m.any():
            counts[name] = int(m.sum())
        matched |= m
    if (~matched).any():
        counts["other / unparseable"] = int((~matched).sum())
    return counts


def parse_timestamps_to_local(dates: pd.Series, zones: pd.Series) -> tuple[pd.Series, int, int]:
    """
    Strings with an explicit UTC offset are converted from that offset; strings without one are
    interpreted in the time zone the API reported for that row. Result: naive local UTC+7 time.
    """
    s = dates.astype(str).str.strip().str.replace("T", " ", regex=False)
    has_offset = s.str.contains(r"[+-]\d{2}:\d{2}$|Z$", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    aware = pd.to_datetime(s[has_offset], format="mixed", errors="coerce", utc=True)
    out.loc[has_offset] = aware.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).astype("datetime64[ns]")

    naive_idx = s.index[~has_offset]
    naive = pd.to_datetime(s[naive_idx], format="mixed", errors="coerce")
    zone = zones.loc[naive_idx].fillna(LOCAL_TZ).astype(str)
    for tz in zone.unique():
        idx = zone.index[zone == tz]
        try:
            localized = naive[idx].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            out.loc[idx] = localized.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).astype("datetime64[ns]")
        except Exception:
            out.loc[idx] = naive[idx]
    return out, int(has_offset.sum()), int((~has_offset).sum())


def align_to_hour(ts: pd.Series) -> pd.Series:
    """Nearest full hour (xx:30 rounds up), e.g. a 17:45 'current' snapshot -> 18:00."""
    return (ts + pd.Timedelta(minutes=30)).dt.floor("h")


# =============================================================================
# 1. Missing API responses: request log + coverage
# =============================================================================
def build_request_table(df: pd.DataFrame) -> pd.DataFrame:
    """One line per (run, source_api, location) with expected vs. returned records."""
    log_path = RAW_DIR / "api_request_log.csv"
    frames = []
    logged_runs = set()
    if log_path.exists():
        log = pd.read_csv(log_path)
        logged_runs = set(log["run_id"].unique())
        agg = (
            log.groupby(["run_id", "source_api", "location_id"], as_index=False)
            .agg(status=("status", lambda s: "ok" if (s == "ok").all() else ",".join(sorted(set(s) - {"ok"}))),
                 http_status=("http_status", "last"),
                 records_expected=("records_expected", "sum"),
                 records_returned=("records_returned", "sum"))
        )
        agg["evidence"] = "request_log"
        frames.append(agg)

    # Runs without a request log (first collector version): infer from the raw rows
    all_locations = sorted(df["location_id"].dropna().astype(int).unique())
    for run_id, g in df[~df["run_id"].isin(logged_runs)].groupby("run_id"):
        for src, expected in LEGACY_EXPECTED.items():
            sub = g[g["source_api"] == src]
            for loc in all_locations:
                rows = sub[sub["location_id"] == loc]
                returned = int(rows[WEATHER_VARS].notna().any(axis=1).sum())
                exp = len(rows) if expected is None else expected
                frames.append(pd.DataFrame([{
                    "run_id": run_id, "source_api": src, "location_id": loc,
                    "status": "ok" if returned > 0 else "missing",
                    "http_status": np.nan, "records_expected": exp, "records_returned": returned,
                    "evidence": "inferred_from_raw_rows",
                }]))
    req = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not req.empty:
        req["coverage_pct"] = (100 * req["records_returned"] / req["records_expected"].replace(0, np.nan)).round(1)
        req["incomplete"] = (req["status"] == "ok") & (req["records_returned"] < req["records_expected"])
    return req


# =============================================================================
# Main cleaning pipeline
# =============================================================================
def clean_multi_source_weather_dataset():
    df_raw = pd.read_csv(RAW_DIR / "raw_weather_data.csv", low_memory=False)
    print("=" * 78)
    print("STEP 2: CLEANING & INTEGRATING MULTI-SOURCE WEATHER DATA")
    print("=" * 78)

    df, n_legacy = upgrade_legacy_rows(df_raw.copy())
    for col in WEATHER_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    total_raw_rows = int(len(df))
    rows_by_source = {k: int(v) for k, v in df["source_api"].value_counts().items()}
    rows_by_record_type = {k: int(v) for k, v in df["record_type"].value_counts().items()}
    print(f"Raw rows: {total_raw_rows:,} ({n_legacy:,} legacy rows labelled from collection order)")

    # --- raw-data characteristics -------------------------------------------------
    ts_formats = classify_timestamp_formats(df["date"])
    float32_artifacts = int(sum(((df[c].dropna().round(2) - df[c].dropna()).abs() > 1e-6).sum() for c in WEATHER_VARS))
    raw_missing_by_col = {c: int(df[c].isna().sum()) for c in WEATHER_VARS}
    missing_by_source_var = (
        df.groupby("source_api")[WEATHER_VARS].apply(lambda g: g.isna().sum()).astype(int)
    )
    schema_gaps = {
        src: [v for v in WEATHER_VARS if missing_by_source_var.loc[src, v] == rows_by_source[src]]
        for src in missing_by_source_var.index
    }

    # --- Step 2: timestamps -------------------------------------------------------
    df["timestamp_exact"], n_with_offset, n_naive = parse_timestamps_to_local(df["date"], df["timezone"])
    unparseable = int(df["timestamp_exact"].isna().sum())
    df = df.dropna(subset=["timestamp_exact"]).copy()
    df["timestamp"] = align_to_hour(df["timestamp_exact"])
    sub_hourly_aligned = int((df["timestamp_exact"] != df["timestamp"]).sum())

    # collection time: legacy rows get the latest 'current' snapshot hour (cannot be later than collection)
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")
    for run_id, g in df[df["collected_at"].isna()].groupby("run_id"):
        cur = g.loc[g["record_type"] == "current_conditions", "timestamp"]
        df.loc[g.index, "collected_at"] = cur.max() if len(cur) else g["timestamp"].max()
    df["collected_hour"] = df["collected_at"].dt.floor("h")

    # --- Step 1: missing API responses -------------------------------------------
    req = build_request_table(df)
    req.to_csv(CLEAN_DIR / "api_coverage_report.csv", index=False, encoding="utf-8-sig")
    missing_req = req[req["status"] != "ok"]
    missing_pairs = (
        missing_req.groupby("source_api")["location_id"].apply(lambda s: sorted(int(x) for x in s)).to_dict()
        if not missing_req.empty else {}
    )

    # --- Step 3: empty records ----------------------------------------------------
    empty_mask = df[WEATHER_VARS].isna().all(axis=1)
    empty_rows = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # --- Step 4: values that were still forecasts when they were collected --------
    future_mask = df["timestamp"] > df["collected_hour"]
    future_rows = int(future_mask.sum())
    future_by_source = {k: int(v) for k, v in df.loc[future_mask, "source_api"].value_counts().items()}
    df = df[~future_mask].copy()

    # --- Step 5: physical ranges (record level) ------------------------------------
    range_violations = {}
    for col, (low, high) in VALID_RANGES.items():
        bad = (df[col] < low) | (df[col] > high)
        range_violations[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    gust_record_issues = int((df["wind_gusts_10m"] < df["wind_speed_10m"]).sum())

    # --- Step 6a: exact duplicates ------------------------------------------------
    # identical values for the same provider, location and hour (e.g. the archive endpoint and the
    # forecast endpoint of Open-Meteo returning the same numbers, or the same hour re-collected)
    df["_rank"] = df["record_type"].map(RECORD_TYPE_RANK).fillna(9)
    df = df.sort_values(["location_id", "timestamp", "source_family", "_rank", "collected_at"],
                        ascending=[True, True, True, True, False])
    dup_key = ["source_family", "location_id", "timestamp", *WEATHER_VARS]
    exact_dup_mask = df.duplicated(subset=dup_key, keep="first")
    exact_duplicates = int(exact_dup_mask.sum())
    exact_dup_by_source = {k: int(v) for k, v in df.loc[exact_dup_mask, "source_api"].value_counts().items()}
    df = df[~exact_dup_mask].copy()

    # --- Step 6b: same family, same location-hour ---------------------------------
    family_key = ["location_id", "timestamp", "source_family"]
    within_family_dups = int(df.duplicated(subset=family_key).sum())
    within_family_by_family = {k: int(v) for k, v in df.loc[df.duplicated(subset=family_key), "source_family"].value_counts().items()}
    meta_cols = ["city", "region", "requested_lat", "requested_lon", "api_latitude", "api_longitude",
                 "elevation_m", "timezone", "source_api", "record_type"]
    # groupby.first() takes the first NON-missing value per column, i.e. the best-ranked record
    # that actually has the variable (e.g. archive value, else the recent model value).
    df_family = df.groupby(family_key, as_index=False)[meta_cols + WEATHER_VARS].first()

    # --- Step 6c / 7: cross-source integration --------------------------------------
    n_fam = df_family.groupby(["location_id", "timestamp"])["source_family"].transform("nunique")
    cross_source_dups = int(df_family.duplicated(subset=["location_id", "timestamp"]).sum())
    df_overlap = df_family[n_fam >= 2].sort_values(["location_id", "timestamp", "source_family"]).copy()
    df_overlap["time"] = df_overlap["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_overlap.drop(columns=["timestamp"]).to_csv(CLEAN_DIR / "multi_source_comparison.csv", index=False, encoding="utf-8-sig")

    # Elevation/coordinates come from Open-Meteo (grid-cell metadata) when available
    om_meta = (df_family[df_family["source_family"] == "Open-Meteo"]
               .groupby("location_id")[["api_latitude", "api_longitude", "elevation_m", "timezone"]].first())

    g = df_family.groupby(["location_id", "timestamp"])
    fused = g[WEATHER_VARS].mean()
    fused["num_sources"] = g["source_family"].nunique()
    fused["source_api"] = g["source_api"].agg(lambda s: " + ".join(sorted(set(s))))
    fused["record_types"] = g["record_type"].agg(lambda s: " + ".join(sorted(set(s))))
    for var, thr in CONFLICT_THRESHOLDS.items():
        fused[f"spread_{var}"] = (g[var].max() - g[var].min()).round(2)
    conflict = np.zeros(len(fused), dtype=bool)
    for var, thr in CONFLICT_THRESHOLDS.items():
        conflict |= (fused[f"spread_{var}"] > thr).to_numpy()
    fused["cross_source_conflict"] = conflict
    fused = fused.reset_index()
    conflict_hours = int(fused["cross_source_conflict"].sum())

    # systematic level differences of each provider against Open-Meteo at the same location-hour
    ref = df_family[df_family["source_family"] == "Open-Meteo"].set_index(["location_id", "timestamp"])[WEATHER_VARS]
    source_bias = {}
    for fam, gf in df_family[df_family["source_family"] != "Open-Meteo"].groupby("source_family"):
        j = gf.set_index(["location_id", "timestamp"])[WEATHER_VARS].join(ref, rsuffix="_ref", how="inner")
        source_bias[fam] = {"matched_hours": int(len(j))}
        for v in ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m"]:
            d = (j[v] - j[f"{v}_ref"]).dropna()
            if len(d):
                source_bias[fam][v] = {"mean_diff": round(float(d.mean()), 2), "mean_abs_diff": round(float(d.abs().mean()), 2)}
    conflict_by_var = {var: int((fused[f"spread_{var}"] > thr).sum()) for var, thr in CONFLICT_THRESHOLDS.items()}

    loc_meta = (df_family.groupby("location_id")[["city", "region", "requested_lat", "requested_lon"]].first()
                .join(om_meta, how="left"))

    # --- Step 8: continuous hourly grid & flagged imputation ------------------------
    start, end = fused["timestamp"].min(), fused["timestamp"].max()
    grid = pd.date_range(start, end, freq="1h")
    pieces = []
    missing_slots = 0
    imputed_cells = {v: 0 for v in WEATHER_VARS}
    for loc_id, gl in fused.groupby("location_id"):
        gl = gl.set_index("timestamp").reindex(grid)
        gl.index.name = "timestamp"
        gl["location_id"] = loc_id
        slot_missing = gl["num_sources"].isna()
        missing_slots += int(slot_missing.sum())
        gl["num_sources"] = gl["num_sources"].fillna(0).astype(int)
        gl["source_api"] = gl["source_api"].fillna("none (imputed)")
        gl["record_types"] = gl["record_types"].fillna("none")
        gl["cross_source_conflict"] = gl["cross_source_conflict"].fillna(False).astype(bool)
        imputed = pd.DataFrame(False, index=gl.index, columns=WEATHER_VARS)
        for v in WEATHER_VARS:
            miss = gl[v].isna()
            if not miss.any():
                continue
            # only gaps of at most MAX_INTERPOLATION_GAP_H consecutive hours are filled; longer gaps stay missing
            run_id_ = (miss != miss.shift()).cumsum()
            run_len = miss.groupby(run_id_).transform("sum")
            short_gap = miss & (run_len <= MAX_INTERPOLATION_GAP_H)
            interp = gl[v].interpolate(method="time", limit_area="inside")
            filled = gl[v].where(~short_gap, interp)
            imputed[v] = short_gap & filled.notna()
            gl[v] = filled
            imputed_cells[v] += int(imputed[v].sum())
        gl["imputed_vars"] = imputed.apply(lambda r: ",".join(c for c in WEATHER_VARS if r[c]), axis=1) if imputed.any().any() else ""
        pieces.append(gl.reset_index())
    df_grid = pd.concat(pieces, ignore_index=True)
    df_grid = df_grid.join(loc_meta, on="location_id")
    remaining_nan = {v: int(df_grid[v].isna().sum()) for v in WEATHER_VARS}

    # --- Step 9: cross-variable consistency & rounding ------------------------------
    gust_issues = int((df_grid["wind_gusts_10m"] < df_grid["wind_speed_10m"]).sum())
    df_grid["wind_gusts_10m"] = np.fmax(df_grid["wind_gusts_10m"], df_grid["wind_speed_10m"])
    for v in WEATHER_VARS:
        df_grid[v] = df_grid[v].round(2)

    # descriptive extremes (kept, not removed: they are real weather within physical limits)
    iqr_extremes = {}
    for v in WEATHER_VARS:
        if v == "precipitation":
            iqr_extremes[v] = int((df_grid[v] > 5.0).sum())
            continue
        q1 = df_grid.groupby("location_id")[v].transform(lambda x: x.quantile(0.25))
        q3 = df_grid.groupby("location_id")[v].transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        iqr_extremes[v] = int(((df_grid[v] < q1 - 1.5 * iqr) | (df_grid[v] > q3 + 1.5 * iqr)).sum())

    df_grid["time"] = df_grid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_grid["obs_date"] = df_grid["timestamp"].dt.strftime("%Y-%m-%d")
    df_grid["hour"] = df_grid["timestamp"].dt.hour
    df_grid["is_imputed"] = df_grid["imputed_vars"].astype(str).str.len() > 0
    # Topic 6 attribute names
    df_grid["temperature"] = df_grid["temperature_2m"]
    df_grid["humidity"] = df_grid["relative_humidity_2m"]
    df_grid["wind_speed"] = df_grid["wind_speed_10m"]
    df_grid["pressure"] = df_grid["pressure_msl"]
    df_grid["wind_gusts"] = df_grid["wind_gusts_10m"]

    final_cols = [
        "location_id", "city", "region", "time", "obs_date", "hour",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation", "pressure_msl",
        "temperature", "humidity", "wind_speed", "pressure",
        "num_sources", "source_api", "record_types",
        "spread_temperature_2m", "spread_relative_humidity_2m", "spread_pressure_msl", "cross_source_conflict",
        "is_imputed", "imputed_vars",
        "requested_lat", "requested_lon", "api_latitude", "api_longitude", "elevation_m", "timezone",
    ]
    df_clean = df_grid[final_cols].sort_values(["location_id", "time"]).reset_index(drop=True)
    df_clean.to_csv(CLEAN_DIR / "cleaned_weather_data.csv", index=False, encoding="utf-8-sig")

    # --- Quality report ----------------------------------------------------------
    std_regex = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    ok_requests = int((req["status"] == "ok").sum())
    report = {
        "collection": {
            "runs": sorted(df_raw.get("run_id", pd.Series(["legacy-run"])).fillna("legacy-run").unique().tolist()),
            "num_locations": int(df["location_id"].nunique()),
            "rows_by_source_api": rows_by_source,
            "rows_by_record_type": rows_by_record_type,
            "legacy_rows_labelled_from_collection_order": n_legacy,
        },
        "missing_api_responses": {
            "requests_checked": int(len(req)),
            "requests_ok": ok_requests,
            "requests_missing_or_failed": int(len(missing_req)),
            "missing_by_source": {k: len(v) for k, v in missing_pairs.items()},
            "missing_location_ids_by_source": missing_pairs,
            "incomplete_responses": int(req["incomplete"].sum()) if "incomplete" in req else 0,
            "evidence": sorted(req["evidence"].unique().tolist()) if not req.empty else [],
        },
        "before_cleaning": {
            "total_rows": total_raw_rows,
            "num_locations": int(df_raw["location_id"].nunique()),
            "timestamp_formats": ts_formats,
            "unstandardized_timestamps": int((~df_raw["date"].astype(str).str.match(std_regex)).sum()),
            "timestamps_with_utc_offset": n_with_offset,
            "timestamps_without_offset": n_naive,
            "unparseable_timestamps": unparseable,
            "sub_hourly_records_aligned": sub_hourly_aligned,
            "empty_records_all_vars_missing": empty_rows,
            "not_yet_observed_forecast_rows": future_rows,
            "not_yet_observed_by_source": future_by_source,
            "missing_values_by_column": raw_missing_by_col,
            "total_missing_cells": int(sum(raw_missing_by_col.values())),
            "variables_not_provided_by_source": {k: v for k, v in schema_gaps.items() if v},
            "physical_range_violations": range_violations,
            "wind_gust_inconsistencies_record_level": gust_record_issues,
            "exact_duplicates": exact_duplicates,
            "exact_duplicates_by_source": exact_dup_by_source,
            "same_source_family_duplicates": within_family_dups,
            "same_source_family_duplicates_by_family": within_family_by_family,
            "cross_source_duplicates": cross_source_dups,
            "float32_precision_artifacts": float32_artifacts,
        },
        "integration": {
            "fused_location_hours": int(len(fused)),
            "multi_source_location_hours": int((fused["num_sources"] >= 2).sum()),
            "cross_source_conflict_hours": conflict_hours,
            "cross_source_conflict_by_variable": conflict_by_var,
            "conflict_thresholds": CONFLICT_THRESHOLDS,
            "difference_vs_open_meteo": source_bias,
        },
        "after_cleaning": {
            "total_rows": int(len(df_clean)),
            "num_locations": int(df_clean["location_id"].nunique()),
            "hours_per_location": int(len(grid)),
            "time_range_start": str(df_clean["time"].min()),
            "time_range_end": str(df_clean["time"].max()),
            "missing_hourly_slots_filled": missing_slots,
            "imputed_cells_by_variable": imputed_cells,
            "rows_with_imputed_values": int(df_clean["is_imputed"].sum()),
            "total_missing_cells": int(sum(remaining_nan.values())),
            "missing_cells_left_in_long_gaps": remaining_nan,
            "unstandardized_timestamps": int((~df_clean["time"].str.match(std_regex)).sum()),
            "exact_duplicates": int(df_clean.duplicated().sum()),
            "key_duplicates": int(df_clean.duplicated(subset=["location_id", "time"]).sum()),
            "physical_range_violations": int(sum(
                ((df_clean[c] < lo) | (df_clean[c] > hi)).sum() for c, (lo, hi) in VALID_RANGES.items())),
            "wind_gust_inconsistencies_fixed": gust_issues,
            "wind_gust_inconsistencies": int((df_clean["wind_gusts_10m"] < df_clean["wind_speed_10m"]).sum()),
            "float32_precision_artifacts": 0,
            "statistical_iqr_extremes_kept": iqr_extremes,
        },
    }
    with open(CLEAN_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=int)

    b, a, m = report["before_cleaning"], report["after_cleaning"], report["missing_api_responses"]
    print(f"1. Missing API responses   : {m['requests_missing_or_failed']} of {m['requests_checked']} source x location requests "
          f"-> {m['missing_by_source']}")
    print(f"2. Timestamps              : formats {b['timestamp_formats']}, {b['sub_hourly_records_aligned']:,} sub-hourly aligned")
    print(f"3. Empty records removed   : {b['empty_records_all_vars_missing']:,}")
    print(f"4. Not-yet-observed rows   : {b['not_yet_observed_forecast_rows']:,} {b['not_yet_observed_by_source']}")
    print(f"5. Range violations        : {sum(b['physical_range_violations'].values())}")
    print(f"6. Duplicates              : exact {b['exact_duplicates']:,} | same family {b['same_source_family_duplicates']:,} "
          f"| cross-source {b['cross_source_duplicates']:,}")
    print(f"7. Multi-source hours      : {report['integration']['multi_source_location_hours']:,} "
          f"(conflicts: {report['integration']['cross_source_conflict_hours']})")
    print(f"8. Grid                    : {a['total_rows']:,} rows = {a['num_locations']} x {a['hours_per_location']:,} h "
          f"({a['time_range_start']} -> {a['time_range_end']}), imputed cells {sum(a['imputed_cells_by_variable'].values())}, "
          f"remaining NaN {a['total_missing_cells']}")
    print(f"9. Gust < speed fixed      : {a['wind_gust_inconsistencies_fixed']}")
    print("=" * 78)


if __name__ == "__main__":
    clean_multi_source_weather_dataset()
