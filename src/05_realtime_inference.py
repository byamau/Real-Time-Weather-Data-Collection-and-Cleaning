"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 5: Real-Time API Fetching, Live Data Cleaning & 7-Day Weight Inference Engine
This module is used both via CLI and inside the Streamlit Live Demo (`app.py`):
  1. Calls all 4 Real-Time APIs (`Open-Meteo SDK`, `WeatherAPI.com`, `Xweather API`, `FreeOpenAPI.dev`)
     for a selected location to obtain the latest 96 hours of real-time context plus synoptic guidance.
  2. Executes the Real-Time Cleaning Pipeline (standardize timestamps, deduplicate, fuse sensors, validate ranges).
  3. Loads the pre-trained weights from `models/weather_forecast_weights.joblib` and runs
     Hybrid + XGBoost Residual Weight Inference combined with synoptic NWP dynamics for the next 7 days (168 hours).
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"

TARGET_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "pressure_msl",
]

VALID_RANGES = {
    "temperature_2m": (-10.0, 50.0),
    "relative_humidity_2m": (0.0, 100.0),
    "wind_speed_10m": (0.0, 200.0),
    "precipitation": (0.0, 300.0),
    "pressure_msl": (850.0, 1080.0),
}


def load_env_keys() -> dict:
    env_vars = {}
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    return env_vars


def http_get_json(url: str, headers: dict = None, timeout: int = 10) -> dict:
    req_headers = {"User-Agent": "VN-Live-Weather-Demo/3.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_and_clean_live_city_data(loc_meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Call all 4 live APIs in real-time for a single city, return:
      - df_raw_live: Uncleaned multi-API records just fetched (up to current hour)
      - df_clean_live: Cleaned, standardized, sensor-fused hourly history (plus synoptic future stored in attrs)
      - api_status: Dict summarizing status & latency of each of the 4 APIs
    """
    env = load_env_keys()
    lat = loc_meta["requested_lat"]
    lon = loc_meta["requested_lon"]
    loc_id = loc_meta["location_id"]
    city = loc_meta["city"]
    region = loc_meta["region"]

    raw_rows = []
    api_status = {}

    # 1. Open-Meteo Live API (past_days=4 + forecast_days=8 for synoptic coupling)
    t0 = time.perf_counter()
    try:
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,wind_speed_10m,relative_humidity_2m,precipitation,pressure_msl,wind_gusts_10m"
            f"&past_days=4&forecast_days=8&timezone=Asia%2FBangkok"
        )
        om_json = http_get_json(om_url)
        hourly = om_json.get("hourly", {})
        times = hourly.get("time", [])
        for i, ts_str in enumerate(times):
            raw_rows.append({
                "source_api": "Open-Meteo SDK",
                "raw_date": ts_str,
                "location_id": loc_id,
                "city": city,
                "region": region,
                "temperature_2m": hourly["temperature_2m"][i],
                "relative_humidity_2m": hourly["relative_humidity_2m"][i],
                "wind_speed_10m": hourly["wind_speed_10m"][i],
                "precipitation": hourly["precipitation"][i],
                "pressure_msl": hourly["pressure_msl"][i],
            })
        api_status["Open-Meteo API"] = {
            "status": "ONLINE",
            "records": len(times),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        api_status["Open-Meteo API"] = {"status": f"OFFLINE: {e}", "records": 0, "latency_ms": 0}

    # 2. WeatherAPI.com Live API
    t0 = time.perf_counter()
    try:
        wkey = env.get("WEATHERAPI_KEY", "")
        w_url = f"http://api.weatherapi.com/v1/forecast.json?key={wkey}&q={lat},{lon}&days=3&aqi=no&alerts=no"
        w_json = http_get_json(w_url)
        curr = w_json.get("current", {})
        cnt = 0
        if curr and "last_updated" in curr:
            cnt += 1
            raw_rows.append({
                "source_api": "WeatherAPI.com",
                "raw_date": curr.get("last_updated"),
                "location_id": loc_id,
                "city": city,
                "region": region,
                "temperature_2m": curr.get("temp_c"),
                "relative_humidity_2m": curr.get("humidity"),
                "wind_speed_10m": curr.get("wind_kph"),
                "precipitation": curr.get("precip_mm"),
                "pressure_msl": curr.get("pressure_mb"),
            })
        for fday in w_json.get("forecast", {}).get("forecastday", []):
            for h in fday.get("hour", []):
                cnt += 1
                raw_rows.append({
                    "source_api": "WeatherAPI.com",
                    "raw_date": h.get("time"),
                    "location_id": loc_id,
                    "city": city,
                    "region": region,
                    "temperature_2m": h.get("temp_c"),
                    "relative_humidity_2m": h.get("humidity"),
                    "wind_speed_10m": h.get("wind_kph"),
                    "precipitation": h.get("precip_mm"),
                    "pressure_msl": h.get("pressure_mb"),
                })
        api_status["WeatherAPI.com"] = {
            "status": "ONLINE",
            "records": cnt,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        api_status["WeatherAPI.com"] = {"status": f"OFFLINE: {e}", "records": 0, "latency_ms": 0}

    # 3. Vaisala Xweather Live API
    t0 = time.perf_counter()
    try:
        xid = env.get("XWEATHER_CLIENT_ID", "")
        xsec = env.get("XWEATHER_CLIENT_SECRET", "")
        x_url = f"https://data.api.xweather.com/conditions/{lat},{lon}?client_id={xid}&client_secret={xsec}&from=-24hours&to=now&plimit=24"
        x_json = http_get_json(x_url)
        resp_list = x_json.get("response", [])
        periods = resp_list[0].get("periods", []) if resp_list else []
        for p in periods:
            raw_rows.append({
                "source_api": "Xweather API",
                "raw_date": p.get("dateTimeISO"),
                "location_id": loc_id,
                "city": city,
                "region": region,
                "temperature_2m": p.get("tempC"),
                "relative_humidity_2m": p.get("humidity"),
                "wind_speed_10m": p.get("windSpeedKPH"),
                "precipitation": p.get("precipMM"),
                "pressure_msl": p.get("pressureMB"),
            })
        api_status["Xweather API"] = {
            "status": "ONLINE",
            "records": len(periods),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        api_status["Xweather API"] = {"status": f"OFFLINE: {e}", "records": 0, "latency_ms": 0}

    # 4. FreeOpenAPI.dev Live API
    t0 = time.perf_counter()
    try:
        fkey = env.get("FREE_OPENAPI_DEV_KEY", "")
        f_url = f"https://www.freeopenapi.dev/api/v1/weather/current?lat={lat}&lon={lon}"
        f_json = http_get_json(f_url, headers={"X-Api-Key": fkey})
        curr = f_json.get("current", {})
        if curr and "timestamp" in curr:
            raw_rows.append({
                "source_api": "FreeOpenAPI.dev",
                "raw_date": curr.get("timestamp"),
                "location_id": loc_id,
                "city": city,
                "region": region,
                "temperature_2m": curr.get("temperatureC"),
                "relative_humidity_2m": np.nan,
                "wind_speed_10m": curr.get("windSpeedKmh"),
                "precipitation": curr.get("precipitationMm"),
                "pressure_msl": np.nan,
            })
        api_status["FreeOpenAPI.dev"] = {
            "status": "ONLINE",
            "records": 1 if curr else 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        api_status["FreeOpenAPI.dev"] = {"status": f"OFFLINE: {e}", "records": 0, "latency_ms": 0}

    df_all_live = pd.DataFrame(raw_rows)

    if df_all_live.empty:
        df_local = pd.read_csv(CLEAN_DIR / "cleaned_weather_data.csv")
        df_city = df_local[df_local["location_id"] == loc_id].tail(120).copy()
        df_city["timestamp"] = pd.to_datetime(df_city["time"])
        df_city["time_vn"] = df_city["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
        return df_city, df_city, api_status

    # Live Data Cleaning Pipeline
    df = df_all_live.copy()
    s_clean = (
        df["raw_date"].astype(str)
        .str.replace("T", " ", regex=False)
        .str.replace(r"\+07:00$", "", regex=True)
        .str.strip()
    )
    df["timestamp"] = pd.to_datetime(s_clean, format="mixed", errors="coerce").dt.floor("h")
    df = df.dropna(subset=["timestamp"]).copy()

    for col in TARGET_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        low, high = VALID_RANGES[col]
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    df_fused_all = (
        df.groupby("timestamp", as_index=False)
        .agg(
            location_id=("location_id", "first"),
            city=("city", "first"),
            region=("region", "first"),
            temperature_2m=("temperature_2m", "mean"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            wind_speed_10m=("wind_speed_10m", "mean"),
            precipitation=("precipitation", "mean"),
            pressure_msl=("pressure_msl", "mean"),
            num_sources=("source_api", "nunique"),
            sources=("source_api", lambda s: " + ".join(dict.fromkeys(s))),
        )
        .sort_values("timestamp")
    )

    grid = pd.date_range(df_fused_all["timestamp"].min(), df_fused_all["timestamp"].max(), freq="1h")
    df_fused_all = df_fused_all.set_index("timestamp").reindex(grid)
    df_fused_all.index.name = "timestamp"
    df_fused_all["location_id"] = loc_id
    df_fused_all["city"] = city
    df_fused_all["region"] = region
    for col in TARGET_VARS:
        if col == "precipitation":
            df_fused_all[col] = df_fused_all[col].fillna(0.0).round(2)
        else:
            df_fused_all[col] = df_fused_all[col].interpolate(method="time").ffill().bfill().round(2)
    df_fused_all["num_sources"] = df_fused_all["num_sources"].fillna(1).astype(int)
    df_fused_all["sources"] = df_fused_all["sources"].fillna("Interpolated")
    df_fused_all = df_fused_all.reset_index()

    now_hr = pd.Timestamp.now().floor("h")
    # Split strictly into observed history (<= now_hr) and synoptic NWP guidance (> now_hr)
    df_raw_live = df[df["timestamp"] <= now_hr].drop(columns=["timestamp"]).copy()
    df_clean_live = df_fused_all[df_fused_all["timestamp"] <= now_hr].copy()
    df_synoptic_future = df_fused_all[df_fused_all["timestamp"] > now_hr].copy()

    df_clean_live["time"] = df_clean_live["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
    df_clean_live.attrs["synoptic_future"] = df_synoptic_future

    return df_raw_live, df_clean_live, api_status


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def gru_predict_from_weights(gru_w: dict, X_scaled: np.ndarray) -> np.ndarray:
    """
    Execute the exact mathematical PyTorch GRU forward pass using the saved NumPy weight matrices.
    """
    seq_idx_arr = np.array(gru_w["seq_indices"], dtype=np.int64)
    X_seq = X_scaled[:, seq_idx_arr]  # (N, 8, 5)
    N, T, _ = X_seq.shape
    H = gru_w["hidden_dim"]

    W_ih = gru_w["W_ih"]  # (3H, 5)
    W_hh = gru_w["W_hh"]  # (3H, H)
    b_ih = gru_w["b_ih"]  # (3H,)
    b_hh = gru_w["b_hh"]  # (3H,)

    h = np.zeros((N, H), dtype=np.float64)
    for t in range(T):
        x_t = X_seq[:, t, :]
        gi = x_t @ W_ih.T + b_ih
        gh = h @ W_hh.T + b_hh

        i_r, i_z, i_n = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
        h_r, h_z, h_n = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:]

        r_t = _sigmoid(i_r + h_r)
        z_t = _sigmoid(i_z + h_z)
        n_t = np.tanh(i_n + r_t * h_n)
        h = (1.0 - z_t) * n_t + z_t * h

    combined = np.hstack([h, X_scaled])  # (N, H + 58)
    out_norm = (combined @ gru_w["fc_W"].T + gru_w["fc_b"]).ravel()
    return out_norm * gru_w["y_std"] + gru_w["y_mean"]


def forecast_7_days_via_weight_inference(
    weights_bundle: dict,
    df_clean_history: pd.DataFrame,
    loc_meta: dict,
    horizon_hours: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Execute 7-Day Recursive Weight Inference using the saved Hybrid (GRU + Ridge + HistGB) + XGBoost Residual weights
    coupled with multi-API synoptic NWP dynamics so day-to-day weather transitions are captured accurately.
    """
    t_start = time.perf_counter()

    feature_cols = weights_bundle["feature_cols"]
    lags = weights_bundle["lags"]
    hgb_models = weights_bundle["hgb_models"]
    ridge_models = weights_bundle["ridge_models"]
    gru_weights = weights_bundle.get("gru_weights", {})
    xgb_residual_models = weights_bundle.get("xgb_residual_models", {})
    scaler = weights_bundle["scaler"]

    loc_id = loc_meta["location_id"]
    lat = loc_meta["requested_lat"]
    lon = loc_meta["requested_lon"]
    elev = loc_meta["elevation_m"]

    synoptic_df = df_clean_history.attrs.get("synoptic_future", pd.DataFrame())
    synoptic_lookup = {}
    if isinstance(synoptic_df, pd.DataFrame) and not synoptic_df.empty:
        for _, srow in synoptic_df.iterrows():
            synoptic_lookup[srow["timestamp"]] = srow

    now_hr = pd.Timestamp.now().floor("h")
    hist_upto_now = df_clean_history[df_clean_history["timestamp"] <= now_hr]
    if len(hist_upto_now) >= 48:
        hist = hist_upto_now.sort_values("timestamp").tail(96).copy()
    else:
        hist = df_clean_history.sort_values("timestamp").tail(96).copy()
    last_ts = hist["timestamp"].max()

    buf_arrays = {var: list(hist[var].to_numpy(dtype=np.float64)) for var in TARGET_VARS}
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    forecast_rows = []

    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        hr = next_ts.hour
        doy = next_ts.dayofyear

        feat_dict = {
            "location_id": float(loc_id),
            "requested_lat": float(lat),
            "requested_lon": float(lon),
            "elevation_m": float(elev),
            "hour_sin": float(np.sin(2 * np.pi * hr / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * hr / 24.0)),
            "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
            "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        }

        n_buf = len(buf_arrays["temperature_2m"])
        for var in TARGET_VARS:
            s_list = buf_arrays[var]
            for lag in lags:
                idx = max(0, n_buf - lag)
                feat_dict[f"{var}_lag_{lag}h"] = float(s_list[idx])
            recent_24 = s_list[max(0, n_buf - 24):]
            feat_dict[f"{var}_roll24_mean"] = float(np.mean(recent_24))
            feat_dict[f"{var}_roll24_std"] = float(np.std(recent_24)) if len(recent_24) > 1 else 0.0

        X_step = np.array([[feat_dict[c] for c in feature_cols]], dtype=np.float64)
        X_step_scaled = (X_step - scaler_mean) / scaler_scale

        step_pred = {"timestamp": next_ts}
        syn_row = synoptic_lookup.get(next_ts, None)

        # Progressive hybrid coupling: short-range relies heavily on local autoregressive + GRU + XGBoost residual weights,
        # while medium-range (days 2-7) incorporates synoptic air-mass transitions so daily weather evolves naturally
        syn_weight = min(0.55, 0.15 + 0.003 * step)
        ml_weight = 1.0 - syn_weight

        for var in TARGET_VARS:
            pred_hgb = float(hgb_models[var].predict(X_step)[0])
            pred_ridge = float(ridge_models[var].predict(X_step_scaled)[0])
            pred_gru = (
                float(gru_predict_from_weights(gru_weights[var], X_step_scaled)[0])
                if var in gru_weights
                else pred_ridge
            )
            pred_xgb_res = (
                float(xgb_residual_models[var].predict(X_step)[0])
                if var in xgb_residual_models
                else 0.0
            )
            base_pred = 0.15 * pred_gru + 0.10 * pred_ridge + 0.75 * pred_hgb
            raw_ml_pred = base_pred + 0.50 * pred_xgb_res

            if syn_row is not None and pd.notna(syn_row.get(var)):
                syn_val = float(syn_row[var])
                if var == "precipitation":
                    # Combine synoptic rain occurrence with ML intensity correction
                    if syn_val <= 0.10:
                        val = 0.0
                    else:
                        val = 0.65 * syn_val + 0.35 * max(0.0, raw_ml_pred)
                else:
                    val = ml_weight * raw_ml_pred + syn_weight * syn_val
            else:
                val = raw_ml_pred
                if var == "precipitation" and val < 0.55:
                    val = 0.0

            low, high = VALID_RANGES[var]
            val = float(np.clip(val, low, high))
            step_pred[var] = round(val, 2)
            buf_arrays[var].append(step_pred[var])

        step_pred["step_hour"] = step
        step_pred["time"] = next_ts.strftime("%d/%m/%Y %H:%M")
        step_pred["date_sort"] = next_ts.strftime("%Y-%m-%d")
        step_pred["date"] = next_ts.strftime("%d/%m/%Y")
        step_pred["hour"] = hr
        step_pred["city"] = loc_meta["city"]
        step_pred["region"] = loc_meta["region"]
        forecast_rows.append(step_pred)

    df_hourly_forecast = pd.DataFrame(forecast_rows)

    df_daily_forecast = (
        df_hourly_forecast.groupby(["date_sort", "date"], as_index=False)
        .agg(
            temp_mean=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            humidity_mean=("relative_humidity_2m", "mean"),
            wind_mean=("wind_speed_10m", "mean"),
            wind_max=("wind_speed_10m", "max"),
            precip_total=("precipitation", "sum"),
            pressure_mean=("pressure_msl", "mean"),
        )
        .sort_values("date_sort")
        .head(7)
        .round(2)
    )

    def classify_weather(row):
        if row["precip_total"] >= 15.0:
            return "Mưa lớn"
        elif row["precip_total"] >= 2.5:
            return "Mưa rào"
        elif row["humidity_mean"] >= 84.0:
            return "Nhiều mây"
        elif row["temp_max"] >= 33.0:
            return "Nắng nóng"
        else:
            return "Nắng đẹp"

    df_daily_forecast["condition"] = df_daily_forecast.apply(classify_weather, axis=1)
    inference_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return df_hourly_forecast, df_daily_forecast, inference_ms


if __name__ == "__main__":
    weights_path = MODEL_DIR / "weather_forecast_weights.joblib"
    bundle = joblib.load(weights_path)
    hcm_meta = [loc for loc in bundle["locations"] if loc["city"] == "Ho Chi Minh City"][0]
    raw_df, clean_df, status = fetch_and_clean_live_city_data(hcm_meta)
    hr_fc, day_fc, ms = forecast_7_days_via_weight_inference(bundle, clean_df, hcm_meta, horizon_hours=168)
    print(f"Live 7-Day Hybrid + XGBoost Residual Weight Inference completed in {ms} ms for {hcm_meta['city']}!")
    print(day_fc.to_string(index=False))
