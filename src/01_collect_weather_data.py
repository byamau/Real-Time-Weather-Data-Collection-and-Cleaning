"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 1: Multi-Source Real-Time & Time-Series Weather Data Collection
Integrates 4 Real Weather APIs across 20 Locations in Vietnam:
  1. Open-Meteo Official Python SDK (openmeteo_requests, past_days=92, forecast_days=1)
  2. WeatherAPI.com (/v1/forecast.json & current, 3 days hourly + realtime)
  3. Vaisala Xweather API (/conditions & /observations, 24h hourly + station observations)
  4. FreeOpenAPI.dev (/api/v1/weather/current, real-time conditions)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Path) -> dict:
    """Load API credentials from .env file into environment variables."""
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
                os.environ.setdefault(k.strip(), v.strip())
    return env_vars


def http_get_json(url: str, headers: dict = None, timeout: int = 15) -> dict:
    req_headers = {"User-Agent": "VN-MultiAPI-WeatherCollector/2.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# 20 Locations matching the exact latitude & longitude arrays provided by the user
LOCATION_METADATA = [
    {"location_id": 1,  "city": "Phu Loc (Hue)",       "region": "Central",           "req_lat": 16.1667, "req_lon": 107.8333},
    {"location_id": 2,  "city": "Da Lat (Lam Dong)",   "region": "Central Highlands", "req_lat": 11.4111, "req_lon": 107.8444},
    {"location_id": 3,  "city": "Nam Dinh",            "region": "North",             "req_lat": 20.4737, "req_lon": 106.0229},
    {"location_id": 4,  "city": "Ho Chi Minh City",    "region": "South",             "req_lat": 10.8230, "req_lon": 106.6296},
    {"location_id": 5,  "city": "Nha Trang",           "region": "South Central",     "req_lat": 12.2451, "req_lon": 109.1943},
    {"location_id": 6,  "city": "Da Nang",             "region": "Central",           "req_lat": 16.0678, "req_lon": 108.2208},
    {"location_id": 7,  "city": "Hanoi",               "region": "North",             "req_lat": 21.2598, "req_lon": 105.8045},
    {"location_id": 8,  "city": "Can Tho",             "region": "Mekong Delta",      "req_lat": 10.0371, "req_lon": 105.7883},
    {"location_id": 9,  "city": "Phu Quoc",            "region": "Islands",           "req_lat": 10.2872, "req_lon": 104.0105},
    {"location_id": 10, "city": "Ha Long",             "region": "North",             "req_lat": 20.9505, "req_lon": 107.0734},
    {"location_id": 11, "city": "Sa Pa",               "region": "North Mountain",    "req_lat": 22.3402, "req_lon": 103.8441},
    {"location_id": 12, "city": "Vinh",                "region": "North Central",     "req_lat": 18.6734, "req_lon": 105.6923},
    {"location_id": 13, "city": "Hue City",            "region": "Central",           "req_lat": 16.4619, "req_lon": 107.5955},
    {"location_id": 14, "city": "Phan Thiet",          "region": "South Central",     "req_lat": 10.9289, "req_lon": 108.1021},
    {"location_id": 15, "city": "Vung Tau",            "region": "South",             "req_lat": 10.3460, "req_lon": 107.0843},
    {"location_id": 16, "city": "Ca Mau",              "region": "Mekong Delta",      "req_lat": 9.1768,  "req_lon": 105.1524},
    {"location_id": 17, "city": "Con Dao",             "region": "Islands",           "req_lat": 8.7029,  "req_lon": 106.6489},
    {"location_id": 18, "city": "Pleiku",              "region": "Central Highlands", "req_lat": 13.9833, "req_lon": 108.0000},
    {"location_id": 19, "city": "Buon Ma Thuot",       "region": "Central Highlands", "req_lat": 12.6675, "req_lon": 108.0378},
    {"location_id": 20, "city": "Dien Bien Phu",       "region": "North Mountain",    "req_lat": 21.3860, "req_lon": 103.0230},
]


def collect_all_four_weather_apis():
    load_env_file(BASE_DIR / ".env")
    weatherapi_key = os.environ.get("WEATHERAPI_KEY", "")
    freeopenapi_key = os.environ.get("FREE_OPENAPI_DEV_KEY", "")
    xweather_id = os.environ.get("XWEATHER_CLIENT_ID", "")
    xweather_secret = os.environ.get("XWEATHER_CLIENT_SECRET", "")

    all_frames = []
    raw_json_archive = {
        "open_meteo_sdk": [],
        "weatherapi_com": [],
        "xweather_api": [],
        "freeopenapi_dev": [],
    }

    # =========================================================================
    # SOURCE 1: Open-Meteo Official Python SDK (openmeteo_requests)
    # =========================================================================
    print("=" * 78)
    print("[SOURCE 1/4] COLLECTING FROM OPEN-METEO PYTHON SDK (openmeteo_requests)")
    print("=" * 78)

    cache_path = str(BASE_DIR / ".cache")
    cache_session = requests_cache.CachedSession(cache_path, expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [loc["req_lat"] for loc in LOCATION_METADATA],
        "longitude": [loc["req_lon"] for loc in LOCATION_METADATA],
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_gusts_10m"
        ],
        "timezone": "Asia/Bangkok",
        "past_days": 92,
        "forecast_days": 1,
    }
    responses = openmeteo.weather_api(url, params=params)

    for idx, response in enumerate(responses):
        meta = LOCATION_METADATA[idx]
        tz_name = response.Timezone().decode("utf-8") if isinstance(response.Timezone(), bytes) else str(response.Timezone())
        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ).tz_convert(tz_name).astype(str),
            "location_id": meta["location_id"],
            "city": meta["city"],
            "region": meta["region"],
            "source_api": "Open-Meteo SDK",
            "requested_lat": meta["req_lat"],
            "requested_lon": meta["req_lon"],
            "api_latitude": round(float(response.Latitude()), 6),
            "api_longitude": round(float(response.Longitude()), 6),
            "elevation_m": round(float(response.Elevation()), 1),
            "timezone": tz_name,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(2).ValuesAsNumpy(),
            "precipitation": hourly.Variables(3).ValuesAsNumpy(),
            "pressure_msl": hourly.Variables(4).ValuesAsNumpy(),
            "wind_gusts_10m": hourly.Variables(5).ValuesAsNumpy(),
        }
        df_om = pd.DataFrame(data=hourly_data)
        all_frames.append(df_om)

        # Update elevation in metadata for other APIs that don't return elevation
        meta["elevation_m"] = round(float(response.Elevation()), 1)

        raw_json_archive["open_meteo_sdk"].append({
            "location_id": meta["location_id"],
            "city": meta["city"],
            "api_latitude": float(response.Latitude()),
            "api_longitude": float(response.Longitude()),
            "elevation_m": float(response.Elevation()),
            "timezone": tz_name,
            "rows_collected": len(df_om),
        })
        print(f"  [Open-Meteo {idx+1:02d}/20] {meta['city']:<18} -> {len(df_om):,} hourly records")

    # =========================================================================
    # SOURCE 2: WeatherAPI.com (/v1/forecast.json - 3 Days Hourly + Current)
    # =========================================================================
    print("\n" + "=" * 78)
    print("[SOURCE 2/4] COLLECTING FROM WEATHERAPI.COM (/v1/forecast.json)")
    print("=" * 78)

    wapi_rows = []
    for idx, meta in enumerate(LOCATION_METADATA):
        lat, lon = meta["req_lat"], meta["req_lon"]
        w_url = f"http://api.weatherapi.com/v1/forecast.json?key={weatherapi_key}&q={lat},{lon}&days=3&aqi=no&alerts=no"
        try:
            payload = http_get_json(w_url)
            loc_info = payload.get("location", {})
            curr = payload.get("current", {})
            f_days = payload.get("forecast", {}).get("forecastday", [])

            raw_json_archive["weatherapi_com"].append({
                "location_id": meta["location_id"],
                "city": meta["city"],
                "location_response": loc_info,
                "current_response": curr,
                "forecast_days_count": len(f_days),
            })

            # Append current observation (timestamp format: 'YYYY-MM-DD HH:MM')
            if curr and "last_updated" in curr:
                wapi_rows.append({
                    "date": curr.get("last_updated"),
                    "location_id": meta["location_id"],
                    "city": meta["city"],
                    "region": meta["region"],
                    "source_api": "WeatherAPI.com",
                    "requested_lat": lat,
                    "requested_lon": lon,
                    "api_latitude": loc_info.get("lat", lat),
                    "api_longitude": loc_info.get("lon", lon),
                    "elevation_m": meta.get("elevation_m", np.nan),
                    "timezone": loc_info.get("tz_id", "Asia/Ho_Chi_Minh"),
                    "temperature_2m": curr.get("temp_c"),
                    "wind_speed_10m": curr.get("wind_kph"),
                    "relative_humidity_2m": curr.get("humidity"),
                    "precipitation": curr.get("precip_mm"),
                    "pressure_msl": curr.get("pressure_mb"),
                    "wind_gusts_10m": curr.get("gust_kph"),
                })

            # Append 72 hourly observations across 3 days
            hour_cnt = 0
            for day_obj in f_days:
                for h_obj in day_obj.get("hour", []):
                    hour_cnt += 1
                    wapi_rows.append({
                        "date": h_obj.get("time"),
                        "location_id": meta["location_id"],
                        "city": meta["city"],
                        "region": meta["region"],
                        "source_api": "WeatherAPI.com",
                        "requested_lat": lat,
                        "requested_lon": lon,
                        "api_latitude": loc_info.get("lat", lat),
                        "api_longitude": loc_info.get("lon", lon),
                        "elevation_m": meta.get("elevation_m", np.nan),
                        "timezone": loc_info.get("tz_id", "Asia/Ho_Chi_Minh"),
                        "temperature_2m": h_obj.get("temp_c"),
                        "wind_speed_10m": h_obj.get("wind_kph"),
                        "relative_humidity_2m": h_obj.get("humidity"),
                        "precipitation": h_obj.get("precip_mm"),
                        "pressure_msl": h_obj.get("pressure_mb"),
                        "wind_gusts_10m": h_obj.get("gust_kph"),
                    })
            print(f"  [WeatherAPI {idx+1:02d}/20] {meta['city']:<18} -> {hour_cnt} hourly + 1 current obs")
        except Exception as e:
            print(f"  [WeatherAPI {idx+1:02d}/20] {meta['city']:<18} -> ERROR: {e}")
        time.sleep(0.1)

    if wapi_rows:
        all_frames.append(pd.DataFrame(wapi_rows))

    # =========================================================================
    # SOURCE 3: Vaisala Xweather API (/conditions 24h & /observations station)
    # =========================================================================
    print("\n" + "=" * 78)
    print("[SOURCE 3/4] COLLECTING FROM XWEATHER API (/conditions & /observations)")
    print("=" * 78)

    xw_rows = []
    for idx, meta in enumerate(LOCATION_METADATA):
        lat, lon = meta["req_lat"], meta["req_lon"]
        cond_url = (
            f"https://data.api.xweather.com/conditions/{lat},{lon}"
            f"?client_id={xweather_id}&client_secret={xweather_secret}&from=-24hours&to=now&plimit=24"
        )
        try:
            payload = http_get_json(cond_url)
            resp_list = payload.get("response", [])
            periods = resp_list[0].get("periods", []) if resp_list else []
            place_loc = resp_list[0].get("loc", {}) if resp_list else {}
            prof = resp_list[0].get("profile", {}) if resp_list else {}

            raw_json_archive["xweather_api"].append({
                "location_id": meta["location_id"],
                "city": meta["city"],
                "loc": place_loc,
                "profile": prof,
                "periods_count": len(periods),
                "sample_period": periods[0] if periods else None,
            })

            for p in periods:
                xw_rows.append({
                    "date": p.get("dateTimeISO"),
                    "location_id": meta["location_id"],
                    "city": meta["city"],
                    "region": meta["region"],
                    "source_api": "Xweather API",
                    "requested_lat": lat,
                    "requested_lon": lon,
                    "api_latitude": place_loc.get("lat", lat),
                    "api_longitude": place_loc.get("long", lon),
                    "elevation_m": prof.get("elevM", meta.get("elevation_m", np.nan)),
                    "timezone": prof.get("tz", "Asia/Ho_Chi_Minh"),
                    "temperature_2m": p.get("tempC"),
                    "wind_speed_10m": p.get("windSpeedKPH"),
                    "relative_humidity_2m": p.get("humidity"),
                    "precipitation": p.get("precipMM"),
                    "pressure_msl": p.get("pressureMB"),
                    "wind_gusts_10m": p.get("windGustKPH"),
                })
            print(f"  [Xweather   {idx+1:02d}/20] {meta['city']:<18} -> {len(periods)} hourly condition periods")
        except Exception as e:
            print(f"  [Xweather   {idx+1:02d}/20] {meta['city']:<18} -> ERROR: {e}")
        time.sleep(0.12)

    if xw_rows:
        all_frames.append(pd.DataFrame(xw_rows))

    # =========================================================================
    # SOURCE 4: FreeOpenAPI.dev (/api/v1/weather/current)
    # =========================================================================
    print("\n" + "=" * 78)
    print("[SOURCE 4/4] COLLECTING FROM FREEOPENAPI.DEV (/api/v1/weather/current)")
    print("=" * 78)

    foa_rows = []
    for idx, meta in enumerate(LOCATION_METADATA):
        lat, lon = meta["req_lat"], meta["req_lon"]
        foa_url = f"https://www.freeopenapi.dev/api/v1/weather/current?lat={lat}&lon={lon}"
        try:
            payload = http_get_json(foa_url, headers={"X-Api-Key": freeopenapi_key})
            curr = payload.get("current", {})
            raw_json_archive["freeopenapi_dev"].append({
                "location_id": meta["location_id"],
                "city": meta["city"],
                "response": payload,
            })
            if curr and "timestamp" in curr:
                foa_rows.append({
                    "date": curr.get("timestamp"),
                    "location_id": meta["location_id"],
                    "city": meta["city"],
                    "region": meta["region"],
                    "source_api": "FreeOpenAPI.dev",
                    "requested_lat": lat,
                    "requested_lon": lon,
                    "api_latitude": payload.get("latitude", lat),
                    "api_longitude": payload.get("longitude", lon),
                    "elevation_m": meta.get("elevation_m", np.nan),
                    "timezone": payload.get("timezone", "Asia/Ho_Chi_Minh"),
                    "temperature_2m": curr.get("temperatureC"),
                    "wind_speed_10m": curr.get("windSpeedKmh"),
                    "relative_humidity_2m": curr.get("humidity"),       # None in FreeOpenAPI /current schema
                    "precipitation": curr.get("precipitationMm"),
                    "pressure_msl": curr.get("pressureHpa"),            # None in FreeOpenAPI /current schema
                    "wind_gusts_10m": curr.get("windGustsKmh"),         # None in FreeOpenAPI /current schema
                })
            print(f"  [FreeOpenAPI {idx+1:02d}/20] {meta['city']:<18} -> 1 current obs (timestamp={curr.get('timestamp')})")
        except Exception as e:
            print(f"  [FreeOpenAPI {idx+1:02d}/20] {meta['city']:<18} -> ERROR: {e}")
        time.sleep(0.15)

    if foa_rows:
        all_frames.append(pd.DataFrame(foa_rows))

    # =========================================================================
    # Combine All 4 Authentic API Outputs into Raw Dataset
    # =========================================================================
    df_raw = pd.concat(all_frames, ignore_index=True)

    raw_csv_path = RAW_DIR / "raw_weather_data.csv"
    df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

    raw_json_path = RAW_DIR / "raw_weather_responses.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_json_archive, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("MULTI-SOURCE REAL WEATHER DATA COLLECTION SUMMARY")
    print("=" * 78)
    for src, cnt in df_raw["source_api"].value_counts().items():
        print(f"  - {src:<20}: {cnt:>7,} rows")
    print(f"  - {'TOTAL RAW ROWS':<20}: {len(df_raw):>7,} rows across {df_raw['location_id'].nunique()} locations")
    print(f"Saved Raw CSV  : {raw_csv_path}")
    print(f"Saved Raw JSON : {raw_json_path}")
    print("=" * 78)


if __name__ == "__main__":
    collect_all_four_weather_apis()
