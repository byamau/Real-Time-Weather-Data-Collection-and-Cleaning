"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 1: Multi-source weather data collection through web APIs (our own collection code)

Two collection modes
--------------------
  --mode full      (default) One complete collection run:
                     1A. Open-Meteo Archive API  (ERA5 reanalysis, last ~92 days, hourly)
                     1B. Open-Meteo Forecast API (past_days=5: recent hourly model values)
                     2.  WeatherAPI.com /v1/forecast.json (current conditions + hourly values of 3 days)
                     3.  Vaisala Xweather /conditions      (hourly conditions of the last 24 h)
                     4.  FreeOpenAPI.dev /weather/current  (current snapshot)
  --mode realtime  Poll only the *current conditions* of every source, optionally repeated
                   (--repeat N --interval-min M). Each poll is a separate run, so the raw file
                   accumulates real observations over time ("combine observations over time").

What is recorded for every run
------------------------------
  data/raw/raw_weather_data.csv     raw rows, APPENDED across runs (use --fresh to start over).
                                    Each row carries run_id, collected_at, source_api,
                                    source_family and record_type.
  data/raw/api_request_log.csv      one line per API request: status (ok / empty / http_error /
                                    network_error / parse_error / missing_key), HTTP code,
                                    latency, records expected vs. returned. This is how
                                    "missing API responses" are detected in step 2.
  data/raw/responses/<run_id>.json  JSON payload excerpts per request (no API keys).

API keys are read from .env (WEATHERAPI_KEY, XWEATHER_CLIENT_ID, XWEATHER_CLIENT_SECRET,
FREE_OPENAPI_DEV_KEY) and are masked in every log.
"""

import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RESP_DIR = RAW_DIR / "responses"
RAW_CSV = RAW_DIR / "raw_weather_data.csv"
REQUEST_LOG_CSV = RAW_DIR / "api_request_log.csv"

LOCAL_TZ = "Asia/Ho_Chi_Minh"
ARCHIVE_DAYS = 92
RECENT_PAST_DAYS = 5

HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_gusts_10m",
]

RAW_COLUMNS = [
    "run_id", "collected_at", "date", "location_id", "city", "region",
    "source_api", "source_family", "record_type", "endpoint",
    "requested_lat", "requested_lon", "api_latitude", "api_longitude", "elevation_m", "timezone",
    *HOURLY_VARS,
]

# record_type vocabulary (used by step 2 to decide which record wins inside one source family)
#   reanalysis_hourly  : ERA5 reanalysis (Open-Meteo Archive)
#   model_hourly       : hourly model values for past hours (Open-Meteo Forecast past_days)
#   conditions_hourly  : Xweather hourly conditions (station-assimilated analysis)
#   current_conditions : "current" snapshot of a provider at request time
#   forecast_hourly    : hourly values from a forecast endpoint (WeatherAPI forecast.json)

LOCATION_METADATA = [
    {"location_id": 1,  "city": "Phu Loc - Hue",      "region": "Central",           "req_lat": 16.1667, "req_lon": 107.8333},
    {"location_id": 2,  "city": "Da Lat - Lam Dong",  "region": "Central Highlands", "req_lat": 11.4111, "req_lon": 107.8444},
    {"location_id": 3,  "city": "Nam Dinh",           "region": "North",             "req_lat": 20.4737, "req_lon": 106.0229},
    {"location_id": 4,  "city": "Ho Chi Minh City",   "region": "South",             "req_lat": 10.8230, "req_lon": 106.6296},
    {"location_id": 5,  "city": "Nha Trang",          "region": "South Central",     "req_lat": 12.2451, "req_lon": 109.1943},
    {"location_id": 6,  "city": "Da Nang",            "region": "Central",           "req_lat": 16.0678, "req_lon": 108.2208},
    {"location_id": 7,  "city": "Hanoi",              "region": "North",             "req_lat": 21.2598, "req_lon": 105.8045},
    {"location_id": 8,  "city": "Can Tho",            "region": "Mekong Delta",      "req_lat": 10.0371, "req_lon": 105.7883},
    {"location_id": 9,  "city": "Phu Quoc",           "region": "Islands",           "req_lat": 10.2872, "req_lon": 104.0105},
    {"location_id": 10, "city": "Ha Long",            "region": "North",             "req_lat": 20.9505, "req_lon": 107.0734},
    {"location_id": 11, "city": "Sa Pa",              "region": "North Mountain",    "req_lat": 22.3402, "req_lon": 103.8441},
    {"location_id": 12, "city": "Vinh",               "region": "North Central",     "req_lat": 18.6734, "req_lon": 105.6923},
    {"location_id": 13, "city": "Hue City",           "region": "Central",           "req_lat": 16.4619, "req_lon": 107.5955},
    {"location_id": 14, "city": "Phan Thiet",         "region": "South Central",     "req_lat": 10.9289, "req_lon": 108.1021},
    {"location_id": 15, "city": "Vung Tau",           "region": "South",             "req_lat": 10.3460, "req_lon": 107.0843},
    {"location_id": 16, "city": "Ca Mau",             "region": "Mekong Delta",      "req_lat": 9.1768,  "req_lon": 105.1524},
    {"location_id": 17, "city": "Con Dao",            "region": "Islands",           "req_lat": 8.7029,  "req_lon": 106.6489},
    {"location_id": 18, "city": "Pleiku",             "region": "Central Highlands", "req_lat": 13.9833, "req_lon": 108.0000},
    {"location_id": 19, "city": "Buon Ma Thuot",      "region": "Central Highlands", "req_lat": 12.6675, "req_lon": 108.0378},
    {"location_id": 20, "city": "Dien Bien Phu",      "region": "North Mountain",    "req_lat": 21.3860, "req_lon": 103.0230},
]


# =============================================================================
# Helpers: environment, HTTP with status logging
# =============================================================================
def load_env_file(env_path: Path) -> None:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


SECRET_PARAM_RE = re.compile(r"(key|client_id|client_secret|apikey|api_key)=([^&]+)", re.IGNORECASE)


def mask_url(url: str) -> str:
    return SECRET_PARAM_RE.sub(lambda m: f"{m.group(1)}=***", url)


class RequestLogger:
    """Collects one log line per API request so missing responses can be audited later."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.rows: list[dict] = []
        self.payloads: list[dict] = []

    def log(self, source_api, endpoint, location_id, status, http_status=None, latency_ms=0.0,
            records_expected=0, records_returned=0, error=""):
        self.rows.append({
            "run_id": self.run_id,
            "requested_at": now_local().strftime("%Y-%m-%d %H:%M:%S"),
            "source_api": source_api,
            "endpoint": mask_url(endpoint),
            "location_id": location_id,
            "status": status,
            "http_status": http_status,
            "latency_ms": round(latency_ms, 1),
            "records_expected": records_expected,
            "records_returned": records_returned,
            "error_message": str(error)[:300],
        })

    def keep_payload(self, source_api, location_id, payload):
        self.payloads.append({"source_api": source_api, "location_id": location_id, "payload": payload})


def now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)


def http_get_json(url: str, headers: dict | None = None, timeout: int = 15):
    """Return (payload, http_status, status, error, latency_ms). Never raises."""
    req_headers = {"User-Agent": "VN-Weather-Topic6-Collector/4.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            code = resp.status
        latency = (time.perf_counter() - t0) * 1000
        try:
            return json.loads(body), code, "ok", "", latency
        except json.JSONDecodeError as exc:
            return None, code, "parse_error", exc, latency
    except urllib.error.HTTPError as exc:
        return None, exc.code, "http_error", exc.reason, (time.perf_counter() - t0) * 1000
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        return None, None, "network_error", exc, (time.perf_counter() - t0) * 1000


def base_row(meta: dict, run_id: str, collected_at: str, source_api: str, family: str,
             record_type: str, endpoint: str) -> dict:
    return {
        "run_id": run_id,
        "collected_at": collected_at,
        "location_id": meta["location_id"],
        "city": meta["city"],
        "region": meta["region"],
        "source_api": source_api,
        "source_family": family,
        "record_type": record_type,
        "endpoint": endpoint,
        "requested_lat": meta["req_lat"],
        "requested_lon": meta["req_lon"],
    }


# =============================================================================
# Source 1: Open-Meteo (Archive ERA5 + Forecast past_days) via the official SDK
# =============================================================================
def collect_open_meteo_hourly(logger: RequestLogger, run_id: str, collected_at: str) -> list[pd.DataFrame]:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(str(BASE_DIR / ".cache_archive"), expire_after=3600)
    client = openmeteo_requests.Client(session=retry(cache_session, retries=5, backoff_factor=0.3))

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=ARCHIVE_DAYS)
    jobs = [
        (
            "Open-Meteo Archive", "reanalysis_hourly",
            "https://archive-api.open-meteo.com/v1/archive",
            {"start_date": start_dt.strftime("%Y-%m-%d"), "end_date": end_dt.strftime("%Y-%m-%d")},
            (ARCHIVE_DAYS + 1) * 24,
        ),
        (
            "Open-Meteo Forecast", "model_hourly",
            "https://api.open-meteo.com/v1/forecast",
            {"past_days": RECENT_PAST_DAYS, "forecast_days": 1},
            (RECENT_PAST_DAYS + 1) * 24,
        ),
    ]
    frames = []
    for source_api, record_type, url, extra, expected in jobs:
        print(f"[{source_api}] {url}")
        params = {
            "latitude": [loc["req_lat"] for loc in LOCATION_METADATA],
            "longitude": [loc["req_lon"] for loc in LOCATION_METADATA],
            "hourly": HOURLY_VARS,
            "timezone": "Asia/Bangkok",
            **extra,
        }
        t0 = time.perf_counter()
        try:
            responses = client.weather_api(url, params=params)
        except Exception as exc:  # the SDK raises on HTTP / limit errors
            latency = (time.perf_counter() - t0) * 1000
            for meta in LOCATION_METADATA:
                logger.log(source_api, url, meta["location_id"], "network_error", None, latency, expected, 0, exc)
            print(f"   FAILED for all locations: {exc}")
            continue
        latency = (time.perf_counter() - t0) * 1000 / max(1, len(LOCATION_METADATA))

        for idx, response in enumerate(responses):
            meta = LOCATION_METADATA[idx]
            tz_name = response.Timezone()
            tz_name = tz_name.decode("utf-8") if isinstance(tz_name, bytes) else str(tz_name)
            hourly = response.Hourly()
            times = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ).tz_convert(tz_name)
            df = pd.DataFrame({"date": times.astype(str)})
            for k, v in base_row(meta, run_id, collected_at, source_api, "Open-Meteo", record_type, url).items():
                df[k] = v
            df["api_latitude"] = round(float(response.Latitude()), 6)
            df["api_longitude"] = round(float(response.Longitude()), 6)
            df["elevation_m"] = round(float(response.Elevation()), 1)
            df["timezone"] = tz_name
            for v_idx, var in enumerate(HOURLY_VARS):
                df[var] = hourly.Variables(v_idx).ValuesAsNumpy()
            meta["elevation_m"] = df["elevation_m"].iloc[0]

            n_valid = int(df[HOURLY_VARS].notna().any(axis=1).sum())
            status = "ok" if n_valid > 0 else "empty"
            logger.log(source_api, url, meta["location_id"], status, 200, latency, expected, n_valid)
            frames.append(df)
            print(f"   {meta['city']:<18} -> {len(df):,} rows ({n_valid:,} with values)")
    return frames


# =============================================================================
# Source 2: WeatherAPI.com
# =============================================================================
def weatherapi_row(meta, run_id, collected_at, record_type, endpoint, loc_info, obj, time_key):
    row = base_row(meta, run_id, collected_at, "WeatherAPI.com", "WeatherAPI.com", record_type, endpoint)
    row.update({
        "date": obj.get(time_key),
        "api_latitude": loc_info.get("lat", meta["req_lat"]),
        "api_longitude": loc_info.get("lon", meta["req_lon"]),
        "elevation_m": meta.get("elevation_m", np.nan),
        "timezone": loc_info.get("tz_id", LOCAL_TZ),
        "temperature_2m": obj.get("temp_c"),
        "wind_speed_10m": obj.get("wind_kph"),
        "relative_humidity_2m": obj.get("humidity"),
        "precipitation": obj.get("precip_mm"),
        "pressure_msl": obj.get("pressure_mb"),
        "wind_gusts_10m": obj.get("gust_kph"),
    })
    return row


def collect_weatherapi(logger, run_id, collected_at, current_only: bool) -> list[dict]:
    key = os.environ.get("WEATHERAPI_KEY", "")
    endpoint = "https://api.weatherapi.com/v1/current.json" if current_only else "https://api.weatherapi.com/v1/forecast.json"
    expected = 1 if current_only else 1 + 3 * 24
    print(f"[WeatherAPI.com] {endpoint}")
    rows = []
    for meta in LOCATION_METADATA:
        q = f"{meta['req_lat']},{meta['req_lon']}"
        url = f"{endpoint}?key={key}&q={q}&aqi=no" + ("" if current_only else "&days=3&alerts=no")
        if not key:
            logger.log("WeatherAPI.com", url, meta["location_id"], "missing_key", None, 0, expected, 0, "WEATHERAPI_KEY not set")
            continue
        payload, code, status, err, latency = http_get_json(url)
        n = 0
        if status == "ok":
            loc_info = payload.get("location", {})
            curr = payload.get("current", {})
            if curr and "last_updated" in curr:
                rows.append(weatherapi_row(meta, run_id, collected_at, "current_conditions", endpoint, loc_info, curr, "last_updated"))
                n += 1
            if not current_only:
                for day in payload.get("forecast", {}).get("forecastday", []):
                    for h in day.get("hour", []):
                        rows.append(weatherapi_row(meta, run_id, collected_at, "forecast_hourly", endpoint, loc_info, h, "time"))
                        n += 1
            status = "ok" if n > 0 else "empty"
            logger.keep_payload("WeatherAPI.com", meta["location_id"], {"location": loc_info, "current": curr})
        logger.log("WeatherAPI.com", url, meta["location_id"], status, code, latency, expected, n, err)
        print(f"   {meta['city']:<18} -> {status} ({n} records)")
        time.sleep(0.1)
    return rows


# =============================================================================
# Source 3: Vaisala Xweather /conditions
# =============================================================================
def collect_xweather(logger, run_id, collected_at, hours_back: int) -> list[dict]:
    cid = os.environ.get("XWEATHER_CLIENT_ID", "")
    secret = os.environ.get("XWEATHER_CLIENT_SECRET", "")
    print("[Xweather API] https://data.api.xweather.com/conditions")
    rows = []
    for meta in LOCATION_METADATA:
        lat, lon = meta["req_lat"], meta["req_lon"]
        url = (
            f"https://data.api.xweather.com/conditions/{lat},{lon}"
            f"?client_id={cid}&client_secret={secret}&from=-{hours_back}hours&to=now&plimit={hours_back}"
        )
        if not (cid and secret):
            logger.log("Xweather API", url, meta["location_id"], "missing_key", None, 0, hours_back, 0, "XWEATHER credentials not set")
            continue
        payload, code, status, err, latency = http_get_json(url)
        n = 0
        if status == "ok":
            if payload.get("success") is False:
                status, err = "http_error", (payload.get("error") or {}).get("description", "success=false")
            resp_list = payload.get("response") or []
            first = resp_list[0] if resp_list else {}
            place_loc, prof = first.get("loc", {}), first.get("profile", {})
            for p in first.get("periods", []):
                row = base_row(meta, run_id, collected_at, "Xweather API", "Xweather API", "conditions_hourly",
                               "https://data.api.xweather.com/conditions")
                row.update({
                    "date": p.get("dateTimeISO"),
                    "api_latitude": place_loc.get("lat", lat),
                    "api_longitude": place_loc.get("long", lon),
                    "elevation_m": prof.get("elevM", meta.get("elevation_m", np.nan)),
                    "timezone": prof.get("tz", LOCAL_TZ),
                    "temperature_2m": p.get("tempC"),
                    "wind_speed_10m": p.get("windSpeedKPH"),
                    "relative_humidity_2m": p.get("humidity"),
                    "precipitation": p.get("precipMM"),
                    "pressure_msl": p.get("pressureMB"),
                    "wind_gusts_10m": p.get("windGustKPH"),
                })
                rows.append(row)
                n += 1
            if status == "ok" and n == 0:
                status = "empty"
            logger.keep_payload("Xweather API", meta["location_id"], {"loc": place_loc, "profile": prof, "periods": n})
        logger.log("Xweather API", url, meta["location_id"], status, code, latency, hours_back, n, err)
        print(f"   {meta['city']:<18} -> {status} ({n} records)")
        time.sleep(0.12)
    return rows


# =============================================================================
# Source 4: FreeOpenAPI.dev current weather
# =============================================================================
def collect_freeopenapi(logger, run_id, collected_at) -> list[dict]:
    key = os.environ.get("FREE_OPENAPI_DEV_KEY", "")
    endpoint = "https://www.freeopenapi.dev/api/v1/weather/current"
    print(f"[FreeOpenAPI.dev] {endpoint}")
    rows = []
    for meta in LOCATION_METADATA:
        url = f"{endpoint}?lat={meta['req_lat']}&lon={meta['req_lon']}"
        if not key:
            logger.log("FreeOpenAPI.dev", url, meta["location_id"], "missing_key", None, 0, 1, 0, "FREE_OPENAPI_DEV_KEY not set")
            continue
        payload, code, status, err, latency = http_get_json(url, headers={"X-Api-Key": key})
        n = 0
        if status == "ok":
            curr = payload.get("current") or {}
            if "timestamp" in curr:
                row = base_row(meta, run_id, collected_at, "FreeOpenAPI.dev", "FreeOpenAPI.dev", "current_conditions", endpoint)
                row.update({
                    "date": curr.get("timestamp"),
                    "api_latitude": payload.get("latitude", meta["req_lat"]),
                    "api_longitude": payload.get("longitude", meta["req_lon"]),
                    "elevation_m": meta.get("elevation_m", np.nan),
                    "timezone": payload.get("timezone", LOCAL_TZ),
                    "temperature_2m": curr.get("temperatureC"),
                    "wind_speed_10m": curr.get("windSpeedKmh"),
                    "relative_humidity_2m": curr.get("humidity"),
                    "precipitation": curr.get("precipitationMm"),
                    "pressure_msl": curr.get("pressureHpa"),
                    "wind_gusts_10m": curr.get("windGustsKmh"),
                })
                rows.append(row)
                n = 1
            else:
                status = "empty"
            logger.keep_payload("FreeOpenAPI.dev", meta["location_id"], payload)
        logger.log("FreeOpenAPI.dev", url, meta["location_id"], status, code, latency, 1, n, err)
        print(f"   {meta['city']:<18} -> {status}")
        time.sleep(0.15)
    return rows


# =============================================================================
# Source 1C (realtime mode): Open-Meteo current conditions, parsed from plain JSON
# =============================================================================
def collect_open_meteo_current(logger, run_id, collected_at) -> list[dict]:
    endpoint = "https://api.open-meteo.com/v1/forecast"
    query = urllib.parse.urlencode({
        "latitude": ",".join(str(m["req_lat"]) for m in LOCATION_METADATA),
        "longitude": ",".join(str(m["req_lon"]) for m in LOCATION_METADATA),
        "current": ",".join(HOURLY_VARS),
        "timezone": "Asia/Bangkok",
    })
    url = f"{endpoint}?{query}"
    print(f"[Open-Meteo current] {endpoint}")
    payload, code, status, err, latency = http_get_json(url)
    rows = []
    items = payload if isinstance(payload, list) else ([payload] if payload else [])
    for idx, meta in enumerate(LOCATION_METADATA):
        item = items[idx] if idx < len(items) else None
        cur = (item or {}).get("current") or {}
        n = 0
        if status == "ok" and cur.get("time"):
            row = base_row(meta, run_id, collected_at, "Open-Meteo Forecast", "Open-Meteo", "current_conditions", endpoint)
            row.update({
                "date": cur["time"],
                "api_latitude": item.get("latitude"),
                "api_longitude": item.get("longitude"),
                "elevation_m": item.get("elevation"),
                "timezone": item.get("timezone", "Asia/Bangkok"),
                **{var: cur.get(var) for var in HOURLY_VARS},
            })
            rows.append(row)
            n = 1
        st = status if status != "ok" else ("ok" if n else "empty")
        logger.log("Open-Meteo Forecast", url, meta["location_id"], st, code, latency / len(LOCATION_METADATA), 1, n, err)
    print(f"   {len(rows)}/{len(LOCATION_METADATA)} locations returned current conditions")
    return rows


# =============================================================================
# Run orchestration
# =============================================================================
def write_outputs(frames: list[pd.DataFrame], logger: RequestLogger, fresh: bool) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESP_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True) if frames else pd.DataFrame()
    df_new = df_new.reindex(columns=RAW_COLUMNS)

    if RAW_CSV.exists() and not fresh:
        df_old = pd.read_csv(RAW_CSV, low_memory=False)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    df_log = pd.DataFrame(logger.rows)
    if REQUEST_LOG_CSV.exists() and not fresh:
        df_log = pd.concat([pd.read_csv(REQUEST_LOG_CSV), df_log], ignore_index=True)
    df_log.to_csv(REQUEST_LOG_CSV, index=False, encoding="utf-8-sig")

    with open(RESP_DIR / f"{logger.run_id}.json", "w", encoding="utf-8") as f:
        json.dump(logger.payloads, f, ensure_ascii=False, indent=1, default=str)
    return df_new


def run_once(mode: str, fresh: bool) -> None:
    collected = now_local()
    run_id = f"{mode}-{collected.strftime('%Y%m%dT%H%M%S')}"
    collected_at = collected.strftime("%Y-%m-%d %H:%M:%S")
    logger = RequestLogger(run_id)
    print("=" * 78)
    print(f"COLLECTION RUN {run_id} (mode={mode})")
    print("=" * 78)

    frames: list[pd.DataFrame] = []
    if mode == "full":
        frames += collect_open_meteo_hourly(logger, run_id, collected_at)
        frames.append(pd.DataFrame(collect_weatherapi(logger, run_id, collected_at, current_only=False)))
        frames.append(pd.DataFrame(collect_xweather(logger, run_id, collected_at, hours_back=24)))
    else:
        frames.append(pd.DataFrame(collect_open_meteo_current(logger, run_id, collected_at)))
        frames.append(pd.DataFrame(collect_weatherapi(logger, run_id, collected_at, current_only=True)))
        frames.append(pd.DataFrame(collect_xweather(logger, run_id, collected_at, hours_back=2)))
    frames.append(pd.DataFrame(collect_freeopenapi(logger, run_id, collected_at)))

    df_new = write_outputs(frames, logger, fresh)
    log = pd.DataFrame(logger.rows)
    print("-" * 78)
    print(f"Rows collected in this run : {len(df_new):,}")
    if not log.empty:
        print(log.groupby(["source_api", "status"]).size().to_string())
    print(f"Raw CSV (appended)         : {RAW_CSV}")
    print(f"Request log                : {REQUEST_LOG_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Topic 6 weather data collector")
    parser.add_argument("--mode", choices=["full", "realtime"], default="full")
    parser.add_argument("--repeat", type=int, default=1, help="number of polling runs (realtime mode)")
    parser.add_argument("--interval-min", type=float, default=60.0, help="minutes between polling runs")
    parser.add_argument("--fresh", action="store_true", help="overwrite raw CSV and request log instead of appending")
    args = parser.parse_args()

    load_env_file(BASE_DIR / ".env")
    for i in range(args.repeat):
        run_once(args.mode, fresh=args.fresh and i == 0)
        if i < args.repeat - 1:
            print(f"Sleeping {args.interval_min:.0f} min before next poll ({i + 2}/{args.repeat})...")
            time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
