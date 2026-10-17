"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 4: Time-Series Feature Engineering, Hybrid (GRU + Ridge + HistGB) + XGBoost Residual Training & Weight Serialization
Follows ML Best Practices:
  - Stationarity (ADF) and Seasonality (24h Autocorrelation) testing
  - Strict Chronological Split (70% Train / 15% Validation / 15% Test) BEFORE fitting scalers
  - Compares 5 Models:
      1. Seasonal Naive
      2. Ridge Autoregressive
      3. GRU Neural Network
      4. HistGradientBoosting
      5. Hybrid (GRU + Ridge + HistGB) + XGBoost Residual
  - Saves trained model weights & preprocessing pipeline to `models/weather_forecast_weights.joblib`
"""

import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "pressure_msl",
]

LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
CHRONO_LAGS = [72, 48, 24, 12, 6, 3, 2, 1]  # Ordered from oldest (t-72h) to newest (t-1h) for GRU sequence


def build_supervised_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct autoregressive lag features, rolling window statistics, cyclical temporal features,
    and spatial coordinates for each location's time series.
    """
    df = df.sort_values(["location_id", "timestamp"]).copy()

    hour = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    feature_cols = [
        "location_id",
        "requested_lat",
        "requested_lon",
        "elevation_m",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    grouped = df.groupby("location_id", sort=False)

    for var in TARGET_VARS:
        for lag in LAGS:
            col_name = f"{var}_lag_{lag}h"
            df[col_name] = grouped[var].shift(lag)
            feature_cols.append(col_name)

        shifted_1h = grouped[var].shift(1)
        roll24_mean = (
            shifted_1h.groupby(df["location_id"])
            .rolling(window=24, min_periods=6)
            .mean()
            .reset_index(level=0, drop=True)
        )
        roll24_std = (
            shifted_1h.groupby(df["location_id"])
            .rolling(window=24, min_periods=6)
            .std()
            .fillna(0.0)
            .reset_index(level=0, drop=True)
        )
        df[f"{var}_roll24_mean"] = roll24_mean
        df[f"{var}_roll24_std"] = roll24_std
        feature_cols.extend([f"{var}_roll24_mean", f"{var}_roll24_std"])

    df_model = df.dropna(subset=feature_cols + TARGET_VARS).reset_index(drop=True)
    return df_model, feature_cols


def get_gru_sequence_indices(feature_cols: list[str]) -> list[list[int]]:
    """
    Return a 2D index map of shape (8, 5) corresponding to chronological lags
    [t-72h, t-48h, t-24h, t-12h, t-6h, t-3h, t-2h, t-1h] across the 5 weather variables.
    """
    seq_indices = []
    for lag in CHRONO_LAGS:
        step_cols = [feature_cols.index(f"{var}_lag_{lag}h") for var in TARGET_VARS]
        seq_indices.append(step_cols)
    return seq_indices


class WeatherGRUNet(nn.Module):
    """
    PyTorch Gated Recurrent Unit (GRU) Network that processes the 8-step chronological lag tensor
    (t-72h -> ... -> t-1h across 5 variables) and combines the GRU hidden state with full covariates.
    """

    def __init__(self, seq_input_dim: int, full_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=seq_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + full_dim, 1)

    def forward(self, x_seq: torch.Tensor, x_full: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x_seq)
        h_last = h_n.squeeze(0)
        combined = torch.cat([h_last, x_full], dim=1)
        return self.fc(combined).squeeze(1)


def train_pytorch_gru_and_export_weights(
    X_scaled: np.ndarray,
    y: np.ndarray,
    seq_indices: list[list[int]],
    hidden_dim: int = 32,
    epochs: int = 18,
    batch_size: int = 512,
    lr: float = 0.004,
) -> dict:
    """
    Train PyTorch WeatherGRUNet on chronological training data and export its exact gate weights
    as pure NumPy matrices for fast, portable Weight Inference.
    """
    torch.manual_seed(42)
    seq_idx_arr = np.array(seq_indices, dtype=np.int64)  # shape (8, 5)
    X_seq = X_scaled[:, seq_idx_arr]  # shape (N, 8, 5)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) if np.std(y) > 1e-6 else 1.0
    y_norm = (y - y_mean) / y_std

    ds = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_norm, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = WeatherGRUNet(seq_input_dim=len(TARGET_VARS), full_dim=X_scaled.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for b_seq, b_full, b_y in loader:
            optimizer.zero_grad()
            pred = model(b_seq, b_full)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    sd = model.state_dict()
    return {
        "W_ih": sd["gru.weight_ih_l0"].detach().cpu().numpy().astype(np.float64),
        "W_hh": sd["gru.weight_hh_l0"].detach().cpu().numpy().astype(np.float64),
        "b_ih": sd["gru.bias_ih_l0"].detach().cpu().cpu().numpy().astype(np.float64),
        "b_hh": sd["gru.bias_hh_l0"].detach().cpu().numpy().astype(np.float64),
        "fc_W": sd["fc.weight"].detach().cpu().numpy().astype(np.float64),
        "fc_b": sd["fc.bias"].detach().cpu().numpy().astype(np.float64),
        "hidden_dim": hidden_dim,
        "seq_indices": seq_indices,
        "y_mean": y_mean,
        "y_std": y_std,
    }


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


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def fit_hybrid_gru_ridge_hgb_xgboost_residual(
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    y: np.ndarray,
    seq_indices: list[list[int]],
) -> tuple[dict, Ridge, HistGradientBoostingRegressor, XGBRegressor]:
    """
    Two-stage Hybrid Residual Learning with GRU + Ridge + HistGradientBoosting:
      - Stage 1 (Base Trend & Structure Ensemble):
          * GRU Neural Network captures sequential temporal dynamics across the 8 lag steps
          * Ridge Regression captures global linear autoregressive trend
          * HistGradientBoosting captures primary non-linear tree structure
          y_base = 0.15 * y_gru + 0.10 * y_ridge + 0.75 * y_hgb
      - Stage 2 (XGBoost Residual Learner):
          * XGBRegressor learns the residual r = y - y_base
      - Final Hybrid Prediction:
          y_hat = y_base + 0.50 * y_xgb_residual
    """
    gru_w = train_pytorch_gru_and_export_weights(X_scaled, y, seq_indices, hidden_dim=32, epochs=16)
    y_gru_pred = gru_predict_from_weights(gru_w, X_scaled)

    ridge_full = Ridge(alpha=10.0)
    ridge_full.fit(X_scaled, y)
    y_ridge_pred = ridge_full.predict(X_scaled)

    hgb_full = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=42,
    )
    hgb_full.fit(X_raw, y)
    y_hgb_pred = hgb_full.predict(X_raw)

    # Stage 1 Base prediction combining GRU + Ridge + HistGB
    y_base = 0.15 * y_gru_pred + 0.10 * y_ridge_pred + 0.75 * y_hgb_pred

    # Stage 2 Residual r = y - y_base
    base_residuals = y - y_base

    xgb_residual = XGBRegressor(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=4,
    )
    xgb_residual.fit(X_raw, base_residuals)

    return gru_w, ridge_full, hgb_full, xgb_residual


def predict_hybrid(
    gru_w: dict,
    ridge_model: Ridge,
    hgb_model: HistGradientBoostingRegressor,
    xgb_res_model: XGBRegressor,
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
) -> np.ndarray:
    y_gru = gru_predict_from_weights(gru_w, X_scaled)
    y_ridge = ridge_model.predict(X_scaled)
    y_hgb = hgb_model.predict(X_raw)
    base_pred = 0.15 * y_gru + 0.10 * y_ridge + 0.75 * y_hgb
    res_pred = xgb_res_model.predict(X_raw)
    return base_pred + 0.50 * res_pred


def train_and_save_weights():
    print("=" * 78)
    print("STEP 4: HYBRID (GRU + RIDGE + HISTGB) + XGBOOST RESIDUAL TRAINING")
    print("=" * 78)

    df_clean = pd.read_csv(CLEAN_DIR / "cleaned_weather_data.csv")
    df_clean["timestamp"] = pd.to_datetime(df_clean["time"])

    # 1. Stationarity & Seasonality Check
    hcm_series = (
        df_clean[df_clean["city"] == "Ho Chi Minh City"]
        .sort_values("timestamp")["temperature_2m"]
        .dropna()
    )
    adf_stat, adf_pval, _, _, _, _ = adfuller(hcm_series, maxlag=24)
    autocorr_24h = float(hcm_series.autocorr(lag=24))
    print("1. Time-Series Diagnostics:")
    print(f"   - ADF Statistic : {adf_stat:.4f} (p-value: {adf_pval:.4e})")
    print(f"   - 24h Diurnal Autocorrelation : {autocorr_24h:.4f}")

    # 2. Build Features & Perform Strict Chronological Split
    df_feat, feature_cols = build_supervised_features(df_clean)
    seq_indices = get_gru_sequence_indices(feature_cols)

    unique_times = np.sort(df_feat["timestamp"].unique())
    n_times = len(unique_times)
    train_end_ts = unique_times[int(n_times * 0.70)]
    val_end_ts = unique_times[int(n_times * 0.85)]

    train_mask = df_feat["timestamp"] <= train_end_ts
    val_mask = (df_feat["timestamp"] > train_end_ts) & (df_feat["timestamp"] <= val_end_ts)
    test_mask = df_feat["timestamp"] > val_end_ts

    df_train = df_feat[train_mask].copy()
    df_val = df_feat[val_mask].copy()
    df_test = df_feat[test_mask].copy()

    print(f"\n2. Chronological Data Split ({len(feature_cols)} Features):")
    print(f"   - Train Set (70%) : {len(df_train):>6,} rows ({df_train['timestamp'].min()} -> {df_train['timestamp'].max()})")
    print(f"   - Val Set   (15%) : {len(df_val):>6,} rows ({df_val['timestamp'].min()} -> {df_val['timestamp'].max()})")
    print(f"   - Test Set  (15%) : {len(df_test):>6,} rows ({df_test['timestamp'].min()} -> {df_test['timestamp'].max()})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[feature_cols])
    X_val_scaled = scaler.transform(df_val[feature_cols])
    X_test_scaled = scaler.transform(df_test[feature_cols])

    X_train_raw = df_train[feature_cols].to_numpy(dtype=np.float64)
    X_val_raw = df_val[feature_cols].to_numpy(dtype=np.float64)
    X_test_raw = df_test[feature_cols].to_numpy(dtype=np.float64)

    # 3. Compare Baseline vs Ridge vs GRU vs HistGB vs Hybrid (GRU + Ridge + HistGB) + XGBoost Residual
    y_train_temp = df_train["temperature_2m"].to_numpy()
    y_val_temp = df_val["temperature_2m"].to_numpy()
    y_test_temp = df_test["temperature_2m"].to_numpy()

    # Baseline: Seasonal Naive
    pred_val_naive = df_val["temperature_2m_lag_24h"].to_numpy()
    pred_test_naive = df_test["temperature_2m_lag_24h"].to_numpy()

    # Model 1: Ridge Autoregressive
    t0 = time.perf_counter()
    ridge_temp = Ridge(alpha=10.0)
    ridge_temp.fit(X_train_scaled, y_train_temp)
    ridge_train_time = time.perf_counter() - t0
    pred_val_ridge = ridge_temp.predict(X_val_scaled)
    pred_test_ridge = ridge_temp.predict(X_test_scaled)

    # Model 2: GRU Neural Network (PyTorch)
    t0 = time.perf_counter()
    gru_temp_w = train_pytorch_gru_and_export_weights(X_train_scaled, y_train_temp, seq_indices, hidden_dim=32, epochs=16)
    gru_train_time = time.perf_counter() - t0
    pred_val_gru = gru_predict_from_weights(gru_temp_w, X_val_scaled)
    pred_test_gru = gru_predict_from_weights(gru_temp_w, X_test_scaled)

    # Model 3: HistGradientBoosting
    t0 = time.perf_counter()
    hgb_temp = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=7,
        min_samples_leaf=30,
        l2_regularization=1.5,
        random_state=42,
    )
    hgb_temp.fit(X_train_raw, y_train_temp)
    hgb_train_time = time.perf_counter() - t0
    pred_val_hgb = hgb_temp.predict(X_val_raw)
    pred_test_hgb = hgb_temp.predict(X_test_raw)

    # Model 4: Hybrid (GRU + Ridge + HistGB) + XGBoost Residual
    t0 = time.perf_counter()
    hyb_gru, hyb_ridge, hyb_hgb, hyb_xgb_res = fit_hybrid_gru_ridge_hgb_xgboost_residual(
        X_train_raw, X_train_scaled, y_train_temp, seq_indices
    )
    hyb_train_time = time.perf_counter() - t0
    pred_val_hybrid = predict_hybrid(hyb_gru, hyb_ridge, hyb_hgb, hyb_xgb_res, X_val_raw, X_val_scaled)
    pred_test_hybrid = predict_hybrid(hyb_gru, hyb_ridge, hyb_hgb, hyb_xgb_res, X_test_raw, X_test_scaled)

    model_comparison = {}
    for name, y_v_pred, y_t_pred, tr_time in [
        ("Seasonal Naive", pred_val_naive, pred_test_naive, 0.001),
        ("Ridge Autoregressive", pred_val_ridge, pred_test_ridge, ridge_train_time),
        ("GRU Neural Network", pred_val_gru, pred_test_gru, gru_train_time),
        ("HistGradientBoosting", pred_val_hgb, pred_test_hgb, hgb_train_time),
        ("Hybrid GRU + XGBoost Residual", pred_val_hybrid, pred_test_hybrid, hyb_train_time),
    ]:
        val_mae = float(mean_absolute_error(y_val_temp, y_v_pred))
        val_rmse = float(root_mean_squared_error(y_val_temp, y_v_pred))
        val_r2 = float(r2_score(y_val_temp, y_v_pred))

        test_mae = float(mean_absolute_error(y_test_temp, y_t_pred))
        test_rmse = float(root_mean_squared_error(y_test_temp, y_t_pred))
        test_r2 = float(r2_score(y_test_temp, y_t_pred))
        ci_low, ci_high = bootstrap_ci(y_test_temp, y_t_pred, mean_absolute_error)

        model_comparison[name] = {
            "val_mae": round(val_mae, 4),
            "val_rmse": round(val_rmse, 4),
            "val_r2": round(val_r2, 4),
            "test_mae": round(test_mae, 4),
            "test_mae_95ci": [round(ci_low, 4), round(ci_high, 4)],
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "train_time_sec": round(tr_time, 4),
        }

    print("\n3. Model Comparison on Temperature_2m Target:")
    for m_name, metrics in model_comparison.items():
        print(
            f"   - {m_name:<30} | Val MAE: {metrics['val_mae']:.3f}°C | "
            f"Test MAE: {metrics['test_mae']:.3f}°C | "
            f"Test RMSE: {metrics['test_rmse']:.3f}°C | Test R²: {metrics['test_r2']:.4f}"
        )

    # 4. Retrain Production Multi-Target Hybrid (GRU + Ridge + HistGB) + XGBoost Residual Models on Train + Val
    print("\n4. Training Production Hybrid (GRU + Ridge + HistGB) + XGBoost Residual Models for All 5 Variables...")
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    X_train_val_raw = df_train_val[feature_cols].to_numpy(dtype=np.float64)

    production_scaler = StandardScaler()
    X_train_val_scaled = production_scaler.fit_transform(df_train_val[feature_cols])
    X_test_prod_scaled = production_scaler.transform(df_test[feature_cols])

    trained_gru_weights = {}
    trained_hgb_models = {}
    trained_ridge_models = {}
    trained_xgb_residual_models = {}
    multi_target_metrics = {}

    for target in TARGET_VARS:
        y_tv = df_train_val[target].to_numpy(dtype=np.float64)
        y_te = df_test[target].to_numpy(dtype=np.float64)

        g_w, r_mod, h_mod, xgb_res_mod = fit_hybrid_gru_ridge_hgb_xgboost_residual(
            X_train_val_raw, X_train_val_scaled, y_tv, seq_indices
        )
        trained_gru_weights[target] = g_w
        trained_ridge_models[target] = r_mod
        trained_hgb_models[target] = h_mod
        trained_xgb_residual_models[target] = xgb_res_mod

        preds_te = predict_hybrid(g_w, r_mod, h_mod, xgb_res_mod, X_test_raw, X_test_prod_scaled)
        if target in ["relative_humidity_2m", "wind_speed_10m", "precipitation"]:
            preds_te = np.clip(preds_te, 0.0, 100.0 if target == "relative_humidity_2m" else None)

        mae = float(mean_absolute_error(y_te, preds_te))
        rmse = float(root_mean_squared_error(y_te, preds_te))
        r2 = float(r2_score(y_te, preds_te))
        multi_target_metrics[target] = {
            "test_mae": round(mae, 4),
            "test_rmse": round(rmse, 4),
            "test_r2": round(r2, 4),
        }
        print(f"   - Target [{target:<21}] -> Test MAE: {mae:>6.3f} | Test RMSE: {rmse:>6.3f} | Test R²: {r2:>6.4f}")

    hourly_climatology = (
        df_clean.groupby(["location_id", "hour"], as_index=False)[TARGET_VARS]
        .mean()
        .round(2)
    )
    city_metadata_df = (
        df_clean.groupby(["location_id", "city", "region"], as_index=False)[
            ["requested_lat", "requested_lon", "elevation_m"]
        ]
        .first()
        .sort_values("location_id")
    )

    # 5. Serialize Model Weights & Metadata Bundle to Disk (.joblib)
    weights_bundle = {
        "model_name": "Hybrid (GRU + Ridge + HistGradientBoosting) + XGBoost Residual Forecaster",
        "version": "4.0",
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_cols": feature_cols,
        "target_vars": TARGET_VARS,
        "lags": LAGS,
        "seq_indices": seq_indices,
        "gru_weights": trained_gru_weights,
        "hgb_models": trained_hgb_models,
        "ridge_models": trained_ridge_models,
        "xgb_residual_models": trained_xgb_residual_models,
        "scaler": production_scaler,
        "hourly_climatology": hourly_climatology.to_dict(orient="records"),
        "locations": city_metadata_df.to_dict(orient="records"),
        "model_comparison": model_comparison,
        "multi_target_test_metrics": multi_target_metrics,
        "stationarity_diagnostics": {
            "adf_statistic": round(float(adf_stat), 4),
            "adf_pvalue": float(adf_pval),
            "lag24_autocorrelation": round(autocorr_24h, 4),
        },
    }

    weights_path = MODEL_DIR / "weather_forecast_weights.joblib"
    joblib.dump(weights_bundle, weights_path, compress=3)

    eval_report_path = MODEL_DIR / "model_evaluation_report.json"
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": weights_bundle["model_name"],
                "trained_at": weights_bundle["trained_at"],
                "train_rows": int(len(df_train)),
                "val_rows": int(len(df_val)),
                "test_rows": int(len(df_test)),
                "num_features": len(feature_cols),
                "stationarity_diagnostics": weights_bundle["stationarity_diagnostics"],
                "temperature_model_comparison": model_comparison,
                "multi_target_test_metrics": multi_target_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 6. Plot Figure 7: Large High-Resolution Evaluation Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.0, 6.4), dpi=180)

    m_names = list(model_comparison.keys())
    maes = [model_comparison[k]["test_mae"] for k in m_names]
    rmses = [model_comparison[k]["test_rmse"] for k in m_names]
    x = np.arange(len(m_names))
    w = 0.36
    b1 = ax1.bar(x - w / 2, maes, w, label="Test MAE - °C", color="#2b8cbe", edgecolor="black")
    b2 = ax1.bar(x + w / 2, rmses, w, label="Test RMSE - °C", color="#f46d43", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_names, fontsize=9.5, rotation=12, ha="right")
    ax1.set_ylabel("Sai số - °C", fontsize=11, fontweight="bold")
    ax1.set_title("So sánh sai số các mô hình trên tập Test", fontsize=12.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=10)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.annotate(
            f"{h:.3f}°C",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(rmses) * 1.24)

    # Right panel: Actual vs Predicted Temperature on Test Set for Ho Chi Minh City
    hcm_test = df_test[df_test["city"] == "Ho Chi Minh City"].sort_values("timestamp").copy()
    hcm_X_raw = hcm_test[feature_cols].to_numpy(dtype=np.float64)
    hcm_X_scl = production_scaler.transform(hcm_test[feature_cols])
    hcm_pred_hybrid = predict_hybrid(
        trained_gru_weights["temperature_2m"],
        trained_ridge_models["temperature_2m"],
        trained_hgb_models["temperature_2m"],
        trained_xgb_residual_models["temperature_2m"],
        hcm_X_raw,
        hcm_X_scl,
    )
    hcm_pred_gru = gru_predict_from_weights(trained_gru_weights["temperature_2m"], hcm_X_scl)

    ax2.plot(hcm_test["timestamp"], hcm_test["temperature_2m"], color="black", linewidth=2.2, label="Thực tế")
    ax2.plot(hcm_test["timestamp"], hcm_pred_hybrid, color="#d73027", linestyle="--", linewidth=1.9, label="Hybrid GRU + XGBoost Residual")
    ax2.plot(hcm_test["timestamp"], hcm_pred_gru, color="#2b8cbe", linestyle=":", linewidth=1.6, label="GRU Neural Network")
    ax2.set_title("Nhiệt độ Thực tế và Dự báo trên tập Test tại Ho Chi Minh City", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Ngày/Tháng/Năm", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Nhiệt độ - °C", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    plt.setp(ax2.get_xticklabels(), rotation=12, ha="right", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig7_path = FIG_DIR / "fig7_model_training_and_forecast_eval.png"
    fig.savefig(fig7_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

    print("=" * 78)
    print(f"Saved Trained Model Weights to : {weights_path} ({weights_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved Evaluation Report to     : {eval_report_path}")
    print(f"Saved Model Evaluation Plot to : {fig7_path}")
    print("=" * 78)


if __name__ == "__main__":
    train_and_save_weights()
