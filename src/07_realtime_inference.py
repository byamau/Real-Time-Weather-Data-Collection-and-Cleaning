"""
Topic 6 — Step 7: live API collection + on-the-fly cleaning + 7-day forecast (CLI and dashboard)

  1. Calls the 4 providers for one location (Open-Meteo, WeatherAPI.com, Xweather, FreeOpenAPI.dev).
  2. Cleans the live records the same way as step 2 (time-zone aware timestamps aligned to the
     hour, range validation, multi-source integration) for the audit table in the dashboard.
  3. Model input = Open-Meteo hourly history only. The model was trained on Open-Meteo data, so
     feeding it a mix of station data and other providers would shift the input distribution
     (several providers differ systematically, see data_quality_report.json).
  4. Forecast = recursive ML forecast (damped towards climatology), blended with the
     Open-Meteo NWP forecast with a weight that grows with lead time. The blend weights below
     are a HEURISTIC and have not been validated (that needs archived NWP forecasts, see README).
     Both components are returned so the dashboard can show them separately.
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_core as fc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
LOCAL_TZ = "Asia/Ho_Chi_Minh"
TARGET_VARS = fc.TARGET_VARS
VALID_RANGES = fc.CLIP

# NWP blending heuristic (unvalidated): weight of the Open-Meteo forecast at lead h
NWP_WEIGHT_START = 0.15
NWP_WEIGHT_SLOPE_PER_H = 0.003
NWP_WEIGHT_MAX = 0.55
PRECIP_NWP_WEIGHT = 0.65          # rain amount mostly from NWP when NWP predicts rain
PRECIP_NWP_DRY_THRESHOLD = 0.10   # mm/h; below this the NWP hour is treated as dry


def load_env_keys() -> dict:
    env = dict(os.environ)
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def load_bundle(path: Path | None = None) -> dict:
    return joblib.load(path or MODEL_DIR / "weather_forecast_weights.joblib")


def http_get_json(url: str, headers: dict | None = None, timeout: int = 10):
    req_headers = {"User-Agent": "VN-Live-Weather-Topic6/4.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _status(ok: bool, n: int, t0: float, err=None) -> dict:
    return {"status": "ONLINE" if ok else f"OFFLINE: {err}", "records": n,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def _to_local_hour(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip().str.replace("T", " ", regex=False)
    has_off = s.str.contains(r"[+-]\d{2}:\d{2}$|Z$", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[has_off] = pd.to_datetime(s[has_off], utc=True, errors="coerce", format="mixed").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    out[~has_off] = pd.to_datetime(s[~has_off], errors="coerce", format="mixed")
    return (out + pd.Timedelta(minutes=30)).dt.floor("h")


def fetch_and_clean_live_city_data(loc_meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (df_raw_live, df_clean_live, api_status).
    df_clean_live.attrs["model_context"]  : Open-Meteo hourly history (model input)
    df_clean_live.attrs["synoptic_future"]: Open-Meteo hourly forecast after now
    """
    env = load_env_keys()
    lat, lon = loc_meta["requested_lat"], loc_meta["requested_lon"]
    rows, api_status = [], {}

    def add(src, rtype, ts, obj_map):
        rows.append({"source_api": src, "record_type": rtype, "raw_date": ts, **obj_map})

    t0 = time.perf_counter()
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly={','.join(TARGET_VARS)}&past_days=5&forecast_days=8&timezone=Asia%2FBangkok")
        h = http_get_json(url).get("hourly", {})
        for i, ts in enumerate(h.get("time", [])):
            add("Open-Meteo", "model_hourly", ts, {v: h[v][i] for v in TARGET_VARS})
        api_status["Open-Meteo API"] = _status(True, len(h.get("time", [])), t0)
    except Exception as e:
        api_status["Open-Meteo API"] = _status(False, 0, t0, e)

    t0 = time.perf_counter()
    try:
        key = env.get("WEATHERAPI_KEY", "")
        j = http_get_json(f"https://api.weatherapi.com/v1/forecast.json?key={key}&q={lat},{lon}&days=3&aqi=no&alerts=no")
        mapping = lambda o: {"temperature_2m": o.get("temp_c"), "relative_humidity_2m": o.get("humidity"),
                             "wind_speed_10m": o.get("wind_kph"), "precipitation": o.get("precip_mm"),
                             "pressure_msl": o.get("pressure_mb")}
        n = 0
        cur = j.get("current", {})
        if cur.get("last_updated"):
            add("WeatherAPI.com", "current_conditions", cur["last_updated"], mapping(cur)); n += 1
        for day in j.get("forecast", {}).get("forecastday", []):
            for hr in day.get("hour", []):
                add("WeatherAPI.com", "forecast_hourly", hr.get("time"), mapping(hr)); n += 1
        api_status["WeatherAPI.com"] = _status(True, n, t0)
    except Exception as e:
        api_status["WeatherAPI.com"] = _status(False, 0, t0, e)

    t0 = time.perf_counter()
    try:
        cid, sec = env.get("XWEATHER_CLIENT_ID", ""), env.get("XWEATHER_CLIENT_SECRET", "")
        j = http_get_json(f"https://data.api.xweather.com/conditions/{lat},{lon}?client_id={cid}&client_secret={sec}"
                          f"&from=-24hours&to=now&plimit=24")
        periods = (j.get("response") or [{}])[0].get("periods", [])
        for p in periods:
            add("Xweather API", "conditions_hourly", p.get("dateTimeISO"),
                {"temperature_2m": p.get("tempC"), "relative_humidity_2m": p.get("humidity"),
                 "wind_speed_10m": p.get("windSpeedKPH"), "precipitation": p.get("precipMM"),
                 "pressure_msl": p.get("pressureMB")})
        api_status["Xweather API"] = _status(True, len(periods), t0)
    except Exception as e:
        api_status["Xweather API"] = _status(False, 0, t0, e)

    t0 = time.perf_counter()
    try:
        j = http_get_json(f"https://www.freeopenapi.dev/api/v1/weather/current?lat={lat}&lon={lon}",
                          headers={"X-Api-Key": env.get("FREE_OPENAPI_DEV_KEY", "")})
        cur = j.get("current") or {}
        n = 0
        if cur.get("timestamp"):
            add("FreeOpenAPI.dev", "current_conditions", cur["timestamp"],
                {"temperature_2m": cur.get("temperatureC"), "relative_humidity_2m": cur.get("humidity"),
                 "wind_speed_10m": cur.get("windSpeedKmh"), "precipitation": cur.get("precipitationMm"),
                 "pressure_msl": cur.get("pressureHpa")}); n = 1
        api_status["FreeOpenAPI.dev"] = _status(True, n, t0)
    except Exception as e:
        api_status["FreeOpenAPI.dev"] = _status(False, 0, t0, e)

    now_hr = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).floor("h")
    df = pd.DataFrame(rows)
    if df.empty:
        return _offline_fallback(loc_meta, api_status)

    df["timestamp"] = _to_local_hour(df["raw_date"])
    df = df.dropna(subset=["timestamp"])
    for v in TARGET_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        lo, hi = VALID_RANGES[v]
        df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan
    df["location_id"], df["city"], df["region"] = loc_meta["location_id"], loc_meta["city"], loc_meta["region"]

    past = df[df["timestamp"] <= now_hr]
    df_raw_live = past.drop(columns=["timestamp"]).copy()

    # integrated audit table (one row per hour, mean over providers)
    fam = past.sort_values("record_type").groupby(["timestamp", "source_api"], as_index=False)[TARGET_VARS].first()
    g = fam.groupby("timestamp")
    fused = g[TARGET_VARS].mean()
    fused["num_sources"] = g["source_api"].nunique()
    fused["sources"] = g["source_api"].agg(lambda s: " + ".join(sorted(set(s))))
    fused = fused.reindex(pd.date_range(fused.index.min(), fused.index.max(), freq="1h"))
    fused.index.name = "timestamp"
    fused["num_sources"] = fused["num_sources"].fillna(0).astype(int)
    fused["sources"] = fused["sources"].fillna("none")
    for v in TARGET_VARS:
        fused[v] = fused[v].interpolate(method="time", limit=6, limit_area="inside").round(2)
    df_clean_live = fused.reset_index()
    df_clean_live["location_id"], df_clean_live["city"] = loc_meta["location_id"], loc_meta["city"]
    df_clean_live["time"] = df_clean_live["timestamp"].dt.strftime("%d/%m/%Y %H:%M")

    om = df[df["source_api"] == "Open-Meteo"].groupby("timestamp")[TARGET_VARS].first().sort_index()
    context = om[om.index <= now_hr]
    if len(context) < fc.BUFFER_HOURS:
        context = df_clean_live.set_index("timestamp")[TARGET_VARS]  # degraded mode: Open-Meteo unavailable
    df_clean_live.attrs["model_context"] = context.reset_index()
    df_clean_live.attrs["synoptic_future"] = om[om.index > now_hr].reset_index()
    return df_raw_live, df_clean_live, api_status


def _offline_fallback(loc_meta, api_status):
    df_local = pd.read_csv(CLEAN_DIR / "cleaned_weather_data.csv")
    d = df_local[df_local["location_id"] == loc_meta["location_id"]].tail(120).copy()
    d["timestamp"] = pd.to_datetime(d["time"])
    d["num_sources"], d["sources"] = d["num_sources"], d["source_api"]
    d.attrs["model_context"] = d[["timestamp", *TARGET_VARS]].copy()
    d.attrs["synoptic_future"] = pd.DataFrame()
    return d, d, api_status


def nwp_weight(step: int) -> float:
    return min(NWP_WEIGHT_MAX, NWP_WEIGHT_START + NWP_WEIGHT_SLOPE_PER_H * step)


def forecast_7_days_via_weight_inference(bundle: dict, df_clean_history: pd.DataFrame, loc_meta: dict,
                                         horizon_hours: int = 168):
    t_start = time.perf_counter()
    model: fc.WeatherForecaster = bundle["forecaster"]
    ctx = df_clean_history.attrs.get("model_context")
    if ctx is None or len(ctx) == 0:
        ctx = df_clean_history[["timestamp", *TARGET_VARS]]
    ctx = ctx.sort_values("timestamp").set_index("timestamp")[TARGET_VARS]
    ctx = ctx.reindex(pd.date_range(ctx.index.max() - pd.Timedelta(hours=fc.BUFFER_HOURS - 1), ctx.index.max(), freq="1h"))
    ctx = ctx.interpolate(limit_direction="both")
    last_ts = ctx.index.max()
    buffers = {v: ctx[v].to_numpy(dtype=float)[None, :] for v in TARGET_VARS}
    ml = model.forecast(buffers, pd.DatetimeIndex([last_ts]), np.array([loc_meta["location_id"]]), horizon=horizon_hours)

    syn = df_clean_history.attrs.get("synoptic_future", pd.DataFrame())
    syn = syn.set_index("timestamp") if isinstance(syn, pd.DataFrame) and not syn.empty else pd.DataFrame()

    rows = []
    for step in range(1, horizon_hours + 1):
        ts = last_ts + pd.Timedelta(hours=step)
        row = {"timestamp": ts, "step_hour": step}
        w = nwp_weight(step)
        for v in TARGET_VARS:
            ml_val = float(ml[v][0, step - 1])
            nwp_val = float(syn.at[ts, v]) if (not syn.empty and ts in syn.index and pd.notna(syn.at[ts, v])) else np.nan
            if np.isnan(nwp_val):
                val = ml_val
            elif v == "precipitation":
                val = 0.0 if nwp_val <= PRECIP_NWP_DRY_THRESHOLD else PRECIP_NWP_WEIGHT * nwp_val + (1 - PRECIP_NWP_WEIGHT) * max(0.0, ml_val)
            else:
                val = (1 - w) * ml_val + w * nwp_val
            lo, hi = VALID_RANGES[v]
            row[v] = round(float(np.clip(val, lo, hi)), 2)
            row[f"{v}_ml"] = round(ml_val, 2)
            row[f"{v}_nwp"] = None if np.isnan(nwp_val) else round(nwp_val, 2)
        row["nwp_weight"] = round(w, 3)
        row["time"] = ts.strftime("%d/%m/%Y %H:%M")
        row["date_sort"], row["date"], row["hour"] = ts.strftime("%Y-%m-%d"), ts.strftime("%d/%m/%Y"), ts.hour
        row["city"], row["region"] = loc_meta["city"], loc_meta["region"]
        rows.append(row)
    df_hourly = pd.DataFrame(rows)

    df_daily = (
        df_hourly.groupby(["date_sort", "date"], as_index=False)
        .agg(temp_mean=("temperature_2m", "mean"), temp_min=("temperature_2m", "min"), temp_max=("temperature_2m", "max"),
             humidity_mean=("relative_humidity_2m", "mean"), wind_mean=("wind_speed_10m", "mean"),
             wind_max=("wind_speed_10m", "max"), precip_total=("precipitation", "sum"), pressure_mean=("pressure_msl", "mean"))
        .sort_values("date_sort").head(7).round(2)
    )

    def classify(r):
        if r["precip_total"] >= 15.0:
            return "Mưa lớn"
        if r["precip_total"] >= 2.5:
            return "Mưa rào"
        if r["humidity_mean"] >= 84.0:
            return "Nhiều mây"
        if r["temp_max"] >= 33.0:
            return "Nắng nóng"
        return "Nắng đẹp"

    df_daily["condition"] = df_daily.apply(classify, axis=1)
    return df_hourly, df_daily, round((time.perf_counter() - t_start) * 1000, 1)


if __name__ == "__main__":
    bundle = load_bundle()
    meta = next(l for l in bundle["locations"] if l["city"] == "Ho Chi Minh City")
    _, clean, status = fetch_and_clean_live_city_data(meta)
    print(json.dumps(status, ensure_ascii=False, indent=1))
    hourly, daily, ms = forecast_7_days_via_weight_inference(bundle, clean, meta)
    print(f"7-day forecast for {meta['city']} in {ms} ms")
    print(daily.to_string(index=False))
