"""
Shared forecasting core (extension of Topic 6) used by:
  05_walk_forward_evaluation.py, 06_train_production_model.py, 07_realtime_inference.py, app.py

Design decisions (see README, Chapter 4)
  * One hybrid model per weather variable predicts the value 1 hour ahead. A 7-day forecast is
    produced recursively (168 steps), and it is EVALUATED recursively by lead time; the
    1-hour-ahead error alone says nothing about day 2-7.
  * The model predicts the anomaly from a climatology fitted on the training data only
    (location x hour-of-day mean for < 1 year of data; smooth harmonics of day-of-year and
    hour-of-day for >= 1 year). This lets long recursive forecasts relax towards climatology.
  * Features are built by ONE function for training and for live inference (identical lag,
    rolling-window and ddof conventions), and location_id is not used as a numeric feature.
  * Recursive errors compound, so the reported forecast is damped towards an anchor with a
    lead-dependent factor exp(-lead / tau). Per variable, the anchor (training climatology or
    seasonal naive = the last 24 observed hours repeated) and tau are chosen on the last 15 % of
    the training window (an inner model is fitted on the first 85 %). Never on test data.
  * Hybrid = weighted members (Ridge, HistGradientBoosting, optional GRU) + XGBoost residual.
    Member weights (non-negative least squares) and the residual shrinkage are chosen on
    chronologically later inner-validation blocks, and XGBoost learns OUT-OF-FOLD residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "pressure_msl",
]
LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
ROLL_WINDOW = 24
BUFFER_HOURS = 96
STATIC_COLS = ["requested_lat", "requested_lon", "elevation_m"]
CLIP = {
    "temperature_2m": (-10.0, 50.0),
    "relative_humidity_2m": (0.0, 100.0),
    "wind_speed_10m": (0.0, 200.0),
    "precipitation": (0.0, 300.0),
    "pressure_msl": (850.0, 1080.0),
}


# =============================================================================
# Climatology (fitted on training data only)
# =============================================================================
def _harmonic_design(ts: pd.DatetimeIndex) -> np.ndarray:
    doy = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float)
    cols = [np.ones(len(ts))]
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * hour / 24.0), np.cos(2 * np.pi * k * hour / 24.0)]
    sd, cd = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    sh, ch = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    cols += [sd * sh, sd * ch, cd * sh, cd * ch]
    return np.column_stack(cols)


class Climatology:
    """Per-location expected value of every variable at a given date-hour."""

    def fit(self, df: pd.DataFrame) -> "Climatology":
        ts = pd.DatetimeIndex(df["timestamp"])
        span_days = (ts.max() - ts.min()).days
        self.mode = "harmonic" if span_days >= 365 else "hourly_mean"
        self.locations = np.sort(df["location_id"].unique())
        self._loc_index = {int(l): i for i, l in enumerate(self.locations)}
        if self.mode == "hourly_mean":
            tab = df.assign(_h=ts.hour).groupby(["location_id", "_h"])[TARGET_VARS].mean()
            self.table = np.full((len(self.locations), 24, len(TARGET_VARS)), np.nan)
            for (loc, h), row in tab.iterrows():
                self.table[self._loc_index[int(loc)], int(h)] = row.to_numpy()
            # hours never seen for a location fall back to that location's mean
            for i in range(len(self.locations)):
                loc_mean = np.nanmean(self.table[i], axis=0)
                miss = np.isnan(self.table[i])
                self.table[i][miss] = np.take(loc_mean, np.where(miss)[1])
        else:
            n_coef = _harmonic_design(ts[:1]).shape[1]
            self.coef = np.zeros((len(self.locations), len(TARGET_VARS), n_coef))
            for loc, g in df.groupby("location_id"):
                X = _harmonic_design(pd.DatetimeIndex(g["timestamp"]))
                for j, var in enumerate(TARGET_VARS):
                    y = g[var].to_numpy(dtype=float)
                    ok = ~np.isnan(y)
                    self.coef[self._loc_index[int(loc)], j] = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        return self

    def predict(self, loc_ids, timestamps) -> dict[str, np.ndarray]:
        ts = pd.DatetimeIndex(timestamps)
        idx = np.array([self._loc_index[int(l)] for l in np.asarray(loc_ids)])
        if self.mode == "hourly_mean":
            vals = self.table[idx, ts.hour.to_numpy()]
        else:
            X = _harmonic_design(ts)
            vals = np.einsum("nc,nvc->nv", X, self.coef[idx])
        return {var: vals[:, j] for j, var in enumerate(TARGET_VARS)}


# =============================================================================
# Features: one assembly function for training and inference
# =============================================================================
def feature_names(use_doy: bool) -> list[str]:
    names = list(STATIC_COLS) + ["hour_sin", "hour_cos"]
    if use_doy:
        names += ["doy_sin", "doy_cos"]
    for var in TARGET_VARS:
        names += [f"{var}_lag_{lag}h" for lag in LAGS]
        names += [f"{var}_roll24_mean", f"{var}_roll24_std", f"{var}_clim", f"{var}_anom_lag_1h", f"{var}_anom_lag_24h"]
    return names


def _assemble(static, ts, lag_vals, roll_mean, roll_std, clim_t, clim_t1, clim_t24, use_doy) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    hour = ts.hour.to_numpy(dtype=float)
    cols = [static[:, 0], static[:, 1], static[:, 2],
            np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)]
    if use_doy:
        doy = ts.dayofyear.to_numpy(dtype=float)
        cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    for var in TARGET_VARS:
        lv = lag_vals[var]
        cols += [lv[:, i] for i in range(len(LAGS))]
        cols += [roll_mean[var], roll_std[var], clim_t[var],
                 lv[:, LAGS.index(1)] - clim_t1[var], lv[:, LAGS.index(24)] - clim_t24[var]]
    return np.column_stack(cols).astype(np.float32)


def build_training_matrix(df: pd.DataFrame, clim: Climatology, use_doy: bool):
    """
    df: continuous hourly grid per location (NaN allowed in gaps) with location_id, timestamp,
    static columns and TARGET_VARS. Returns X, Y (dict of anomaly targets), clim_t (dict),
    timestamps and location ids of usable rows.
    """
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("location_id", sort=False)
    ts = pd.DatetimeIndex(df["timestamp"])
    loc = df["location_id"].to_numpy()
    lag_vals, roll_mean, roll_std = {}, {}, {}
    for var in TARGET_VARS:
        lag_vals[var] = np.column_stack([g[var].shift(lag).to_numpy(dtype=float) for lag in LAGS])
        prev = g[var].shift(1)
        roll = prev.groupby(df["location_id"]).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW)
        roll_mean[var] = roll.mean().reset_index(level=0, drop=True).sort_index().to_numpy()
        roll_std[var] = roll.std(ddof=1).reset_index(level=0, drop=True).sort_index().to_numpy()
    clim_t = clim.predict(loc, ts)
    clim_t1 = clim.predict(loc, ts - pd.Timedelta(hours=1))
    clim_t24 = clim.predict(loc, ts - pd.Timedelta(hours=24))
    X = _assemble(df[STATIC_COLS].to_numpy(dtype=float), ts, lag_vals, roll_mean, roll_std,
                  clim_t, clim_t1, clim_t24, use_doy)
    Y = {var: df[var].to_numpy(dtype=float) - clim_t[var] for var in TARGET_VARS}
    ok = ~np.isnan(X).any(axis=1)
    for var in TARGET_VARS:
        ok &= ~np.isnan(Y[var])
    return X[ok], {v: Y[v][ok] for v in TARGET_VARS}, ts[ok], loc[ok]


def build_step_features(buffers: dict[str, np.ndarray], target_ts, static: np.ndarray, loc_ids,
                        clim: Climatology, use_doy: bool) -> tuple[np.ndarray, dict]:
    """buffers[var]: (N, L>=BUFFER_HOURS); the last column is the value at target_ts - 1h."""
    ts = pd.DatetimeIndex(target_ts)
    lag_vals = {v: np.column_stack([buffers[v][:, -lag] for lag in LAGS]) for v in TARGET_VARS}
    last24 = {v: buffers[v][:, -ROLL_WINDOW:] for v in TARGET_VARS}
    roll_mean = {v: last24[v].mean(axis=1) for v in TARGET_VARS}
    roll_std = {v: last24[v].std(axis=1, ddof=1) for v in TARGET_VARS}
    clim_t = clim.predict(loc_ids, ts)
    clim_t1 = clim.predict(loc_ids, ts - pd.Timedelta(hours=1))
    clim_t24 = clim.predict(loc_ids, ts - pd.Timedelta(hours=24))
    X = _assemble(static, ts, lag_vals, roll_mean, roll_std, clim_t, clim_t1, clim_t24, use_doy)
    return X, clim_t


def seasonal_naive_anchor(buffer: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last 24 observed hours: (N, L) -> (N, horizon)."""
    last24 = buffer[:, -24:]
    return np.tile(last24, (1, horizon // 24 + 1))[:, :horizon]


# =============================================================================
# GRU member (PyTorch for training, exported to NumPy for inference)
# =============================================================================
def gru_sequence_indices(names: list[str]) -> list[list[int]]:
    return [[names.index(f"{v}_lag_{lag}h") for v in TARGET_VARS] for lag in sorted(LAGS, reverse=True)]


def train_gru(X_scaled: np.ndarray, y: np.ndarray, seq_idx, hidden: int = 32, epochs: int = 16,
              batch: int = 512, lr: float = 0.004, seed: int = 42) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self, seq_dim, full_dim):
            super().__init__()
            self.gru = nn.GRU(seq_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden + full_dim, 1)

        def forward(self, xs, xf):
            _, h = self.gru(xs)
            return self.fc(torch.cat([h.squeeze(0), xf], dim=1)).squeeze(1)

    idx = np.array(seq_idx)
    y_mean, y_std = float(np.mean(y)), float(np.std(y)) or 1.0
    ds = TensorDataset(torch.tensor(X_scaled[:, idx], dtype=torch.float32),
                       torch.tensor(X_scaled, dtype=torch.float32),
                       torch.tensor((y - y_mean) / y_std, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=gen)
    net = Net(len(TARGET_VARS), X_scaled.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        for xs, xf, yb in loader:
            opt.zero_grad()
            loss_fn(net(xs, xf), yb).backward()
            opt.step()
    sd = net.state_dict()
    to_np = lambda k: sd[k].detach().cpu().numpy().astype(np.float64)
    return {"W_ih": to_np("gru.weight_ih_l0"), "W_hh": to_np("gru.weight_hh_l0"),
            "b_ih": to_np("gru.bias_ih_l0"), "b_hh": to_np("gru.bias_hh_l0"),
            "fc_W": to_np("fc.weight"), "fc_b": to_np("fc.bias"),
            "hidden": hidden, "seq_idx": seq_idx, "y_mean": y_mean, "y_std": y_std}


def gru_predict(w: dict, X_scaled: np.ndarray) -> np.ndarray:
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
    seq = X_scaled[:, np.array(w["seq_idx"])].astype(np.float64)
    H = w["hidden"]
    h = np.zeros((len(X_scaled), H))
    for t in range(seq.shape[1]):
        gi = seq[:, t] @ w["W_ih"].T + w["b_ih"]
        gh = h @ w["W_hh"].T + w["b_hh"]
        r = sig(gi[:, :H] + gh[:, :H])
        z = sig(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - z) * n + z * h
    out = np.hstack([h, X_scaled]) @ w["fc_W"].T + w["fc_b"]
    return out.ravel() * w["y_std"] + w["y_mean"]


# =============================================================================
# Hybrid model for one variable
# =============================================================================
class HybridModel:
    MEMBER_NAMES_BASE = ["ridge", "hgb"]

    def __init__(self, use_gru: bool, seq_idx=None, gru_epochs: int = 16, seed: int = 42):
        self.use_gru = use_gru
        self.seq_idx = seq_idx
        self.gru_epochs = gru_epochs
        self.seed = seed
        self.member_names = self.MEMBER_NAMES_BASE + (["gru"] if use_gru else [])

    # -- members -------------------------------------------------------------
    def _fit_members(self, X, y) -> dict:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        m = {"scaler": scaler, "ridge": Ridge(alpha=10.0).fit(Xs, y)}
        m["hgb"] = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_depth=8, min_samples_leaf=25,
            l2_regularization=1.0, early_stopping=False, random_state=self.seed).fit(X, y)
        if self.use_gru:
            m["gru"] = train_gru(Xs, y, self.seq_idx, epochs=self.gru_epochs, seed=self.seed)
        return m

    def _member_preds(self, m, X) -> np.ndarray:
        Xs = m["scaler"].transform(X)
        cols = [m["ridge"].predict(Xs), m["hgb"].predict(X)]
        if self.use_gru:
            cols.append(gru_predict(m["gru"], Xs))
        return np.column_stack(cols)

    def _new_xgb(self):
        return XGBRegressor(n_estimators=180, learning_rate=0.03, max_depth=5, min_child_weight=25,
                            subsample=0.85, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=4.0,
                            random_state=self.seed, n_jobs=2)

    # -- fit with chronological inner validation --------------------------------
    def fit(self, X, y, t) -> "HybridModel":
        ut = np.unique(t)
        c1, c2 = ut[int(len(ut) * 0.70)], ut[int(len(ut) * 0.85)]
        A, B, C = t <= c1, (t > c1) & (t <= c2), t > c2

        m_a = self._fit_members(X[A], y[A])
        P_b = self._member_preds(m_a, X[B])
        w, _ = nnls(P_b, y[B])
        self.weights = w / w.sum() if w.sum() > 0 else np.full(P_b.shape[1], 1.0 / P_b.shape[1])

        m_ab = self._fit_members(X[A | B], y[A | B])
        P_c = self._member_preds(m_ab, X[C])
        base_b, base_c = P_b @ self.weights, P_c @ self.weights
        r_b, r_c = y[B] - base_b, y[C] - base_c  # out-of-fold residuals

        xgb_b = self._new_xgb().fit(X[B], r_b)
        res_c = xgb_b.predict(X[C])
        grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mse = [np.mean((y[C] - base_c - a * res_c) ** 2) for a in grid]
        self.residual_alpha = float(grid[int(np.argmin(mse))])
        self.inner_validation = {
            "members_mae_C": {n: float(np.mean(np.abs(y[C] - P_c[:, i]))) for i, n in enumerate(self.member_names)},
            "base_mae_C": float(np.mean(np.abs(y[C] - base_c))),
            "hybrid_mae_C": float(np.mean(np.abs(y[C] - base_c - self.residual_alpha * res_c))),
        }
        r = np.full(len(y), np.nan)
        r[B], r[C] = r_b, r_c
        BC = B | C
        self.xgb = self._new_xgb().fit(X[BC], r[BC])
        self.members = self._fit_members(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        base = self._member_preds(self.members, X) @ self.weights
        if self.residual_alpha > 0:
            base = base + self.residual_alpha * self.xgb.predict(X)
        return base

    def describe(self) -> dict:
        return {"weights": {n: round(float(w), 3) for n, w in zip(self.member_names, self.weights)},
                "residual_alpha": self.residual_alpha,
                "inner_validation": {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()})
                                     for k, v in self.inner_validation.items()}}


# =============================================================================
# Multi-variable recursive forecaster
# =============================================================================
class WeatherForecaster:
    def __init__(self, use_gru: bool = True, gru_epochs: int = 16, seed: int = 42):
        self.use_gru = use_gru
        self.gru_epochs = gru_epochs
        self.seed = seed

    DAMPING_TAUS = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, np.inf]

    def fit(self, df: pd.DataFrame, train_stride: int = 1, verbose: bool = True,
            calibrate_damping: bool = True, horizon: int = 168) -> "WeatherForecaster":
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.damping_tau = {v: np.inf for v in TARGET_VARS}
        self.damping_anchor = {v: "climatology" for v in TARGET_VARS}
        self.damping_info = {"calibrated": False}
        if calibrate_damping:
            self._calibrate_damping(df, train_stride, horizon, verbose)
        self.climatology = Climatology().fit(df)
        span_days = (df["timestamp"].max() - df["timestamp"].min()).days
        self.use_doy = span_days >= 365
        self.feature_names = feature_names(self.use_doy)
        self.static = df.groupby("location_id")[STATIC_COLS].first()
        X, Y, ts, _ = build_training_matrix(df, self.climatology, self.use_doy)
        if train_stride > 1:  # keep every k-th hour (same hours for all locations) to bound training cost
            hours_since_epoch = ts.asi8 // 3_600_000_000_000
            keep = hours_since_epoch % train_stride == 0
            X, ts = X[keep], ts[keep]
            Y = {v: Y[v][keep] for v in TARGET_VARS}
        t = ts.to_numpy()
        seq_idx = gru_sequence_indices(self.feature_names)
        self.models = {}
        for var in TARGET_VARS:
            self.models[var] = HybridModel(self.use_gru, seq_idx, self.gru_epochs, self.seed).fit(X, Y[var], t)
            if verbose:
                d = self.models[var].describe()
                print(f"      {var:<21} weights={d['weights']} residual_alpha={d['residual_alpha']}")
        self.train_info = {"rows": int(len(X)), "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max()),
                           "climatology": self.climatology.mode, "use_doy": self.use_doy,
                           "n_features": len(self.feature_names), "use_gru": self.use_gru}
        return self

    def _calibrate_damping(self, df: pd.DataFrame, train_stride: int, horizon: int, verbose: bool) -> None:
        """
        Lead-time damping towards climatology, chosen on the LAST 15 % of the training period:
        an inner forecaster is fitted on the first 85 %, issues recursive forecasts from origins
        (00 h / 12 h) inside the last 15 %, and for each variable we pick tau minimising MAE of
            clim + exp(-lead / tau) * (forecast - clim).
        tau = inf means no damping. Nothing from the test period is used.
        """
        ut = np.sort(df["timestamp"].unique())
        cut = pd.Timestamp(ut[int(len(ut) * 0.85)])
        inner = WeatherForecaster(self.use_gru, self.gru_epochs, self.seed).fit(
            df[df["timestamp"] <= cut], train_stride=train_stride, verbose=False, calibrate_damping=False)
        times = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h")
        locs = np.sort(df["location_id"].unique())
        cube = {v: df.pivot(index="location_id", columns="timestamp", values=v).reindex(index=locs, columns=times).to_numpy(float)
                for v in TARGET_VARS}
        origins = [i for i, t in enumerate(times) if t >= cut and t.hour in (0, 12) and i >= BUFFER_HOURS and i < len(times) - 1]
        if not origins:
            return
        o = np.repeat(origins, len(locs))
        li = np.tile(np.arange(len(locs)), len(origins))
        hist = o[:, None] + np.arange(-BUFFER_HOURS + 1, 1)[None, :]
        fut = o[:, None] + np.arange(1, horizon + 1)[None, :]
        fut_ok = fut < len(times)
        fut = np.minimum(fut, len(times) - 1)
        bufs = {v: cube[v][li[:, None], hist] for v in TARGET_VARS}
        ok = np.all([~np.isnan(bufs[v]).any(axis=1) for v in TARGET_VARS], axis=0)
        if ok.sum() == 0:
            return
        bufs = {v: b[ok] for v, b in bufs.items()}
        o, li, fut, fut_ok = o[ok], li[ok], fut[ok], fut_ok[ok]
        last = times[o]
        loc_ids = locs[li]
        raw = inner.forecast(bufs, last, loc_ids, horizon=horizon)
        lead_ts = pd.DatetimeIndex(np.repeat(last.to_numpy(), horizon)) + pd.to_timedelta(np.tile(np.arange(1, horizon + 1), len(last)), unit="h")
        clim = inner.climatology.predict(np.repeat(loc_ids, horizon), lead_ts)
        leads = np.arange(1, horizon + 1)[None, :]
        n_pairs = int(fut_ok.sum())
        for v in TARGET_VARS:
            y = np.where(fut_ok, cube[v][li[:, None], fut], np.nan)
            anchors = {"climatology": clim[v].reshape(len(last), horizon),
                       "seasonal_naive": seasonal_naive_anchor(bufs[v], horizon)}
            best, best_mae = ("climatology", np.inf), np.inf
            for name, c in anchors.items():
                for tau in self.DAMPING_TAUS:
                    lam = np.exp(-leads / tau) if np.isfinite(tau) else np.ones_like(leads, dtype=float)
                    mae = np.nanmean(np.abs(c + lam * (raw[v] - c) - y))
                    if mae < best_mae - 1e-9:
                        best, best_mae = (name, tau), mae
            self.damping_anchor[v], self.damping_tau[v] = best[0], float(best[1])
        self.damping_info = {"calibrated": True, "inner_train_end": str(cut), "origins": len(origins),
                             "trajectories": int(len(last)), "scored_pairs": n_pairs,
                             "anchor": dict(self.damping_anchor),
                             "tau_hours": {v: (None if np.isinf(t) else t) for v, t in self.damping_tau.items()}}
        if verbose:
            print(f"      damping: {self.damping_info['anchor']} tau(h)={self.damping_info['tau_hours']}")

    def static_for(self, loc_ids) -> np.ndarray:
        return self.static.loc[np.asarray(loc_ids), STATIC_COLS].to_numpy(dtype=float)

    def forecast(self, buffers: dict[str, np.ndarray], last_obs_ts, loc_ids, horizon: int = 168) -> dict[str, np.ndarray]:
        """
        buffers[var]: (N, L>=96) observed history ending at last_obs_ts (one per trajectory).
        Returns dict var -> (N, horizon) recursive forecasts for last_obs_ts + 1h ... + horizon h.
        """
        bufs = {v: np.asarray(buffers[v], dtype=float).copy() for v in TARGET_VARS}
        last = pd.DatetimeIndex(last_obs_ts)
        static = self.static_for(loc_ids)
        out = {v: np.zeros((len(last), horizon)) for v in TARGET_VARS}
        taus = getattr(self, "damping_tau", {v: np.inf for v in TARGET_VARS})
        anchor_kind = getattr(self, "damping_anchor", {v: "climatology" for v in TARGET_VARS})
        naive = {v: seasonal_naive_anchor(bufs[v], horizon) for v in TARGET_VARS}
        for step in range(1, horizon + 1):
            ts = last + pd.Timedelta(hours=step)
            X, clim_t = build_step_features(bufs, ts, static, loc_ids, self.climatology, self.use_doy)
            for v in TARGET_VARS:
                lo, hi = CLIP[v]
                raw = np.clip(self.models[v].predict(X) + clim_t[v], lo, hi)
                # the recursion continues on the raw prediction (as during calibration);
                # only the reported value is damped towards climatology
                bufs[v] = np.concatenate([bufs[v][:, 1:], raw[:, None]], axis=1)
                lam = np.exp(-step / taus[v]) if np.isfinite(taus[v]) else 1.0
                anchor = clim_t[v] if anchor_kind[v] == "climatology" else naive[v][:, step - 1]
                out[v][:, step - 1] = np.clip(anchor + lam * (raw - anchor), lo, hi)
        return out

    def describe(self) -> dict:
        return {"train": self.train_info, "damping": getattr(self, "damping_info", {}),
                "models": {v: m.describe() for v, m in self.models.items()}}
