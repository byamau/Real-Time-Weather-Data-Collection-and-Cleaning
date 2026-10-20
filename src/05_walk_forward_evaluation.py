"""
Topic 6 — Extension: walk-forward (rolling-origin) evaluation of the 7-day forecaster

Why
  A single fixed 70/15/15 split of a ~3-month series tests one 2-week window, and the old
  report evaluated only the 1-hour-ahead error. This script instead:
    * slides the test window forward (expanding training window, refit every k folds),
    * issues recursive 168-hour forecasts from many origins inside every test window,
    * reports error BY LEAD TIME (1 h ... 168 h) against three baselines
      (persistence, seasonal naive = repeat the last 24 h, training-fold climatology),
    * summarises skill per fold, with bootstrap confidence intervals over folds, and
      stratified by season and (when available) ENSO phase.

Data
  --data cleaned  (default) data/cleaned/cleaned_weather_data.csv — the Topic 6 dataset.
                  ~3 months, so folds are short (7 days) and the result is indicative only.
  --data archive  data/archive/era5_multiyear_cleaned.parquet from 04_collect_multiyear_archive.py
                  (optional, many years). Use this to test robustness across years / ENSO phases.

No embargo is needed between training and test because the model forecasts recursively
1 hour at a time and every training target lies strictly before the test window. A direct
h-step-ahead model would need an embargo of at least h hours.

Outputs (models/walk_forward/)
  folds.csv, errors_by_fold_lead.csv, errors_by_fold_region_bucket.csv, summary.json
  figures/fig8_walk_forward_mae_by_lead.png, figures/fig9_walk_forward_fold_skill.png
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_core as fc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "models" / "walk_forward"
FIG_DIR = BASE_DIR / "figures"
# NOAA CPC uses the Relative ONI (RONI) for official ENSO monitoring since 1 Feb 2026
# (NWS Public Information Statement 26-05); the legacy ONI file is still updated and used as fallback.
ENSO_URLS = [("RONI", "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"),
             ("ONI", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt")]
ONI_CACHE = BASE_DIR / "data" / "external" / "enso_index.csv"

BUCKETS = {"h1": (1, 1), "day1": (1, 24), "day2-3": (25, 72), "day4-7": (73, 168)}
MODELS = ["hybrid", "persistence", "seasonal_naive", "climatology"]
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


# =============================================================================
# Data
# =============================================================================
def load_data(kind: str) -> pd.DataFrame:
    if kind == "cleaned":
        df = pd.read_csv(BASE_DIR / "data" / "cleaned" / "cleaned_weather_data.csv")
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        path = BASE_DIR / "data" / "archive" / "era5_multiyear_cleaned.parquet"
        if not path.exists():
            sys.exit(f"{path} not found — run src/04_collect_multiyear_archive.py first.")
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    cols = ["location_id", "city", "region", "timestamp", *fc.STATIC_COLS, *fc.TARGET_VARS]
    return df[cols].sort_values(["location_id", "timestamp"]).reset_index(drop=True)


def to_cube(df: pd.DataFrame):
    """Arrays [location, hour] per variable on a common hourly grid."""
    times = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h")
    locs = np.sort(df["location_id"].unique())
    cube = {}
    for v in fc.TARGET_VARS:
        piv = df.pivot(index="location_id", columns="timestamp", values=v).reindex(index=locs, columns=times)
        cube[v] = piv.to_numpy(dtype=float)
    return times, locs, cube


# =============================================================================
# ENSO (NOAA CPC Relative ONI / ONI) — optional, cached
# =============================================================================
ONI_SEASON_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def load_oni() -> pd.DataFrame | None:
    """Seasonal ENSO index (RONI, else ONI) with NOAA episode phases; cached in data/external/."""
    if ONI_CACHE.exists():
        return pd.read_csv(ONI_CACHE)
    rows, used = [], None
    for name, url in ENSO_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                text = r.read().decode("utf-8")
        except Exception as exc:
            print(f"   {name} download failed ({exc})")
            continue
        for line in text.splitlines():
            parts = line.split()
            # lines look like "SEAS YR [TOTAL] ANOM": season code, year, ..., anomaly last
            if len(parts) >= 3 and parts[0] in ONI_SEASON_MONTH and parts[1].isdigit():
                try:
                    rows.append({"year": int(parts[1]), "month": ONI_SEASON_MONTH[parts[0]], "oni": float(parts[-1])})
                except ValueError:
                    continue
        if rows:
            used = name
            break
    if not rows:
        print("   ENSO index unavailable; ENSO stratification skipped.")
        return None
    oni = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    oni["index_name"] = used
    # NOAA convention: El Nino / La Nina period = >= 5 consecutive overlapping seasons beyond +/-0.5
    oni["phase"] = "Neutral"
    for sign, label in ((1, "El Nino"), (-1, "La Nina")):
        hit = (sign * oni["oni"]) >= 0.5
        run = (hit != hit.shift()).cumsum()
        length = hit.groupby(run).transform("sum")
        oni.loc[hit & (length >= 5), "phase"] = label
    ONI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    oni.to_csv(ONI_CACHE, index=False)
    return oni


def enso_for(ts: pd.Timestamp, oni: pd.DataFrame | None) -> tuple[str, float]:
    if oni is None:
        return "unknown", np.nan
    row = oni[(oni["year"] == ts.year) & (oni["month"] == ts.month)]
    if row.empty:
        return "unknown", np.nan
    return row["phase"].iloc[0], float(row["oni"].iloc[0])


# =============================================================================
# Walk-forward loop
# =============================================================================
def run(args):
    t_start = time.perf_counter()
    df = load_data(args.data)
    times, locs, cube = to_cube(df)
    regions = df.groupby("location_id")["region"].first().reindex(locs).to_numpy()
    data_start, data_end = times[0], times[-1]
    H = args.horizon
    fold_days = args.fold_days or (7 if args.data == "cleaned" else 30)
    refit_every = args.refit_every or (1 if args.data == "cleaned" else 3)
    stride = args.train_stride or (1 if args.data == "cleaned" else 3)
    first_test = pd.Timestamp(args.first_test) if args.first_test else (
        data_start + pd.Timedelta(days=42) if args.data == "cleaned" else pd.Timestamp("2019-01-01"))
    first_test = first_test.normalize()
    last_origin = data_end - pd.Timedelta(hours=H)
    fold_starts = pd.date_range(first_test, last_origin, freq=f"{fold_days}D")
    origin_hours = [int(h) for h in args.origin_hours.split(",")]
    oni = load_oni() if args.data == "archive" or args.enso else None

    print("=" * 78)
    print(f"WALK-FORWARD EVALUATION  data={args.data}  {data_start} -> {data_end}")
    print(f"  folds={len(fold_starts)} x {fold_days} d | refit every {refit_every} fold(s) | horizon={H} h | "
          f"origins at {origin_hours} h | GRU={'off' if args.no_gru else 'on'} | train stride={stride}")
    print("=" * 78)
    if len(fold_starts) == 0:
        sys.exit("Not enough data for a single fold.")

    lead_rows, region_rows, fold_rows = [], [], []
    model = None
    for k, fs in enumerate(fold_starts):
        fe = fs + pd.Timedelta(days=fold_days)
        if model is None or k % refit_every == 0:
            train_lo = fs - pd.DateOffset(years=args.max_train_years) if args.max_train_years else data_start
            train_df = df[(df["timestamp"] < fs) & (df["timestamp"] >= train_lo)]
            t0 = time.perf_counter()
            model = fc.WeatherForecaster(use_gru=not args.no_gru, gru_epochs=args.gru_epochs).fit(
                train_df, train_stride=stride, verbose=False)
            fit_sec = time.perf_counter() - t0
            refit = True
        else:
            fit_sec, refit = 0.0, False

        origins = [o for o in pd.date_range(fs, min(fe - pd.Timedelta(hours=1), last_origin), freq="1h")
                   if o.hour in origin_hours and o - pd.Timedelta(hours=fc.BUFFER_HOURS) >= data_start]
        if not origins:
            continue
        o_idx = np.array([times.get_loc(o) for o in origins])
        # trajectories = origins x locations
        traj_o = np.repeat(o_idx, len(locs))
        traj_l = np.tile(np.arange(len(locs)), len(o_idx))
        hist_idx = traj_o[:, None] + np.arange(-fc.BUFFER_HOURS + 1, 1)[None, :]
        fut_idx = traj_o[:, None] + np.arange(1, H + 1)[None, :]
        buffers = {v: cube[v][traj_l[:, None], hist_idx] for v in fc.TARGET_VARS}
        truth = {v: cube[v][traj_l[:, None], fut_idx] for v in fc.TARGET_VARS}
        valid = np.all([~np.isnan(buffers[v]).any(axis=1) for v in fc.TARGET_VARS], axis=0)
        buffers = {v: b[valid] for v, b in buffers.items()}
        truth = {v: t[valid] for v, t in truth.items()}
        traj_o, traj_l = traj_o[valid], traj_l[valid]
        last_ts = times[traj_o]
        loc_ids = locs[traj_l]

        preds = model.forecast(buffers, last_ts, loc_ids, horizon=H)
        lead_ts = pd.DatetimeIndex(np.repeat(last_ts.to_numpy(), H)) + pd.to_timedelta(np.tile(np.arange(1, H + 1), len(last_ts)), unit="h")
        clim = model.climatology.predict(np.repeat(loc_ids, H), lead_ts)

        mid = fs + (fe - fs) / 2
        phase, oni_val = enso_for(mid, oni)
        fold_rows.append({"fold": k, "test_start": str(fs), "test_end": str(fe), "n_origins": len(origins),
                          "n_trajectories": int(valid.sum()), "refit": refit, "fit_seconds": round(fit_sec, 1),
                          "train_rows": model.train_info["rows"], "train_end": model.train_info["end"],
                          "season": SEASON[mid.month], "enso_phase": phase, "oni": oni_val})

        for v in fc.TARGET_VARS:
            y = truth[v]
            cand = {
                "hybrid": preds[v],
                "persistence": np.repeat(buffers[v][:, -1:], H, axis=1),
                "seasonal_naive": np.tile(buffers[v][:, -24:], (1, H // 24 + 1))[:, :H],
                "climatology": clim[v].reshape(len(last_ts), H),
            }
            for name, p in cand.items():
                err = p - y
                ok = ~np.isnan(err)
                n = ok.sum(axis=0)
                abs_sum = np.where(ok, np.abs(err), 0).sum(axis=0)
                sq_sum = np.where(ok, err ** 2, 0).sum(axis=0)
                for lead in range(H):
                    lead_rows.append((k, v, name, lead + 1, int(n[lead]), abs_sum[lead], sq_sum[lead]))
                for bname, (a, b) in BUCKETS.items():
                    sl = slice(a - 1, b)
                    for reg in np.unique(regions):
                        rm = regions[traj_l] == reg
                        e = err[rm, sl]
                        okr = ~np.isnan(e)
                        region_rows.append((k, v, name, bname, reg, int(okr.sum()),
                                            float(np.abs(e[okr]).sum()), float((e[okr] ** 2).sum())))
        hyb_t = np.nanmean(np.abs(preds["temperature_2m"] - truth["temperature_2m"]), axis=0)
        naive_t = np.tile(buffers["temperature_2m"][:, -24:], (1, H // 24 + 1))[:, :H]
        sn_t = np.nanmean(np.abs(naive_t - truth["temperature_2m"]), axis=0)
        print(f"  fold {k:>3} {fs.date()} | {'refit' if refit else '     '} {fit_sec:6.1f}s | traj {int(valid.sum()):>5} | "
              f"T MAE h1 {hyb_t[0]:.2f} | day1 {hyb_t[:24].mean():.2f} (naive {sn_t[:24].mean():.2f}) | "
              f"day4-7 {hyb_t[72:].mean():.2f} (naive {sn_t[72:].mean():.2f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    leads = pd.DataFrame(lead_rows, columns=["fold", "variable", "model", "lead_h", "n", "abs_sum", "sq_sum"])
    leads["mae"] = leads["abs_sum"] / leads["n"]
    leads["rmse"] = np.sqrt(leads["sq_sum"] / leads["n"])
    regs = pd.DataFrame(region_rows, columns=["fold", "variable", "model", "bucket", "region", "n", "abs_sum", "sq_sum"])
    regs["mae"] = regs["abs_sum"] / regs["n"]
    folds.to_csv(OUT_DIR / "folds.csv", index=False)
    leads.drop(columns=["abs_sum", "sq_sum"]).to_csv(OUT_DIR / "errors_by_fold_lead.csv", index=False)
    regs.drop(columns=["abs_sum", "sq_sum"]).to_csv(OUT_DIR / "errors_by_fold_region_bucket.csv", index=False)

    summary = summarise(leads, regs, folds, args, fold_days, refit_every, stride, model)
    summary["runtime_minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    plot(leads, folds, summary)
    print_summary(summary)


# =============================================================================
# Summaries
# =============================================================================
def fold_bucket_mae(leads: pd.DataFrame) -> pd.DataFrame:
    out = []
    for bname, (a, b) in BUCKETS.items():
        sub = leads[(leads["lead_h"] >= a) & (leads["lead_h"] <= b)]
        g = sub.groupby(["fold", "variable", "model"], as_index=False)[["abs_sum", "n"]].sum()
        g["bucket"] = bname
        g["mae"] = g["abs_sum"] / g["n"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def summarise(leads, regs, folds, args, fold_days, refit_every, stride, last_model) -> dict:
    fb = fold_bucket_mae(leads)
    wide = fb.pivot_table(index=["fold", "variable", "bucket"], columns="model", values="mae").reset_index()
    wide = wide.merge(folds[["fold", "season", "enso_phase"]], on="fold", how="left")
    res = {
        "config": {"data": args.data, "fold_days": fold_days, "refit_every_folds": refit_every,
                   "horizon_h": args.horizon, "origin_hours": args.origin_hours, "gru": not args.no_gru,
                   "train_stride": stride, "max_train_years": args.max_train_years,
                   "n_folds": int(len(folds)), "test_period": [folds["test_start"].min(), folds["test_end"].max()],
                   "n_trajectories_total": int(folds["n_trajectories"].sum())},
        "buckets_h": BUCKETS,
        "by_variable": {},
        "by_season": {},
        "by_enso_phase": {},
        "last_model": last_model.describe() if last_model is not None else None,
    }
    for v in fc.TARGET_VARS:
        res["by_variable"][v] = {}
        for bname in BUCKETS:
            w = wide[(wide["variable"] == v) & (wide["bucket"] == bname)]
            entry = {}
            for m in MODELS:
                vals = w[m].to_numpy()
                entry[f"{m}_mae"] = round(float(np.nanmean(vals)), 4)
                entry[f"{m}_mae_fold_std"] = round(float(np.nanstd(vals, ddof=1)), 4) if len(vals) > 1 else None
                entry[f"{m}_mae_ci95_over_folds"] = bootstrap_ci(vals)
            for ref in ["seasonal_naive", "climatology", "persistence"]:
                skill = 1 - w["hybrid"] / w[ref]
                entry[f"skill_vs_{ref}"] = round(float(skill.mean()), 4)
                entry[f"skill_vs_{ref}_ci95"] = bootstrap_ci(skill.to_numpy())
                entry[f"share_folds_beating_{ref}"] = round(float((w["hybrid"] < w[ref]).mean()), 3)
            res["by_variable"][v][bname] = entry
    for key, col in (("by_season", "season"), ("by_enso_phase", "enso_phase")):
        for grp, w in wide[(wide["variable"] == "temperature_2m")].groupby(col):
            res[key][grp] = {
                b: {"n_folds": int((w["bucket"] == b).sum()),
                    "hybrid_mae": round(float(w.loc[w["bucket"] == b, "hybrid"].mean()), 4),
                    "seasonal_naive_mae": round(float(w.loc[w["bucket"] == b, "seasonal_naive"].mean()), 4),
                    "climatology_mae": round(float(w.loc[w["bucket"] == b, "climatology"].mean()), 4),
                    "skill_vs_seasonal_naive": round(float((1 - w.loc[w["bucket"] == b, "hybrid"]
                                                            / w.loc[w["bucket"] == b, "seasonal_naive"]).mean()), 4)}
                for b in BUCKETS}
    rb = regs.groupby(["variable", "model", "bucket", "region"], as_index=False)[["abs_sum", "n"]].sum()
    rb["mae"] = (rb["abs_sum"] / rb["n"]).round(4)
    res["temperature_by_region"] = {
        reg: {b: {m: float(g[(g["bucket"] == b) & (g["model"] == m)]["mae"].iloc[0]) for m in MODELS} for b in BUCKETS}
        for reg, g in rb[rb["variable"] == "temperature_2m"].groupby("region")}
    return res


def print_summary(s: dict) -> None:
    print("-" * 78)
    print("MAE averaged over folds (temperature_2m, °C)        hybrid | seas.naive | climatology | skill vs naive [95% CI]")
    for b, e in s["by_variable"]["temperature_2m"].items():
        print(f"  {b:<8} {e['hybrid_mae']:>6.3f} | {e['seasonal_naive_mae']:>10.3f} | {e['climatology_mae']:>11.3f} | "
              f"{e['skill_vs_seasonal_naive']:+.3f} {e['skill_vs_seasonal_naive_ci95']}")
    print(f"Saved: {OUT_DIR}  (runtime {s.get('runtime_minutes', '?')} min)")


def plot(leads: pd.DataFrame, folds: pd.DataFrame, summary: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    styles = {"hybrid": ("#d73027", "-", "Hybrid (recursive)"), "seasonal_naive": ("#4575b4", "--", "Seasonal naive (repeat last 24 h)"),
              "persistence": ("#999999", ":", "Persistence (last value)"), "climatology": ("#1a9850", "-.", "Climatology (training folds)")}
    units = {"temperature_2m": "°C", "relative_humidity_2m": "%", "wind_speed_10m": "km/h", "pressure_msl": "hPa"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (v, unit) in zip(axes.flat, units.items()):
        sub = leads[leads["variable"] == v]
        for m, (col, ls, lbl) in styles.items():
            g = sub[sub["model"] == m].groupby("lead_h")["mae"]
            mean = g.mean()
            ax.plot(mean.index, mean.values, color=col, linestyle=ls, linewidth=2 if m == "hybrid" else 1.5, label=lbl)
            if m == "hybrid" and sub["fold"].nunique() > 1:
                lo, hi = g.quantile(0.1), g.quantile(0.9)
                ax.fill_between(mean.index, lo.values, hi.values, color=col, alpha=0.15, label="Hybrid: 10–90% of folds")
        for d in range(24, 169, 24):
            ax.axvline(d, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_title(f"{v}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Lead time (hours after the last observation)")
        ax.set_ylabel(f"MAE ({unit})")
        ax.set_xlim(1, 168)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0, 0].legend(fontsize=8.5, loc="lower right")
    cfg = summary["config"]
    fig.suptitle(f"Walk-forward error by lead time — {cfg['n_folds']} folds, {cfg['n_trajectories_total']:,} forecasts "
                 f"({cfg['data']} data, {cfg['test_period'][0][:10]} → {cfg['test_period'][1][:10]})",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_walk_forward_mae_by_lead.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fb = fold_bucket_mae(leads)
    t = fb[fb["variable"] == "temperature_2m"].pivot_table(index=["fold", "bucket"], columns="model", values="mae").reset_index()
    t["skill"] = 1 - t["hybrid"] / t["seasonal_naive"]
    t = t.merge(folds[["fold", "test_start", "enso_phase"]], on="fold")
    t["test_start"] = pd.to_datetime(t["test_start"])
    colors = {"El Nino": "#d73027", "La Nina": "#4575b4", "Neutral": "#7f7f7f", "unknown": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for b, mk in (("day1", "o"), ("day2-3", "s"), ("day4-7", "^")):
        s = t[t["bucket"] == b].sort_values("test_start")
        ax.plot(s["test_start"], s["skill"], color="#555555", linewidth=0.8, alpha=0.6)
        ax.scatter(s["test_start"], s["skill"], c=[colors.get(p, "#7f7f7f") for p in s["enso_phase"]],
                   marker=mk, s=45, edgecolors="black", linewidths=0.4, label=b, zorder=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Skill vs. seasonal naive\n(1 − MAE_model / MAE_naive)")
    ax.set_title("Temperature skill per walk-forward fold (marker = lead bucket, colour = ENSO phase if available)",
                 fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_walk_forward_fold_skill.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Walk-forward evaluation of the recursive 7-day forecaster")
    p.add_argument("--data", choices=["cleaned", "archive"], default="cleaned")
    p.add_argument("--fold-days", type=int, default=None, help="test window per fold (default 7 cleaned / 30 archive)")
    p.add_argument("--refit-every", type=int, default=None, help="refit model every k folds (default 1 cleaned / 3 archive)")
    p.add_argument("--first-test", type=str, default=None, help="first test date (default: start+42 d / 2019-01-01)")
    p.add_argument("--max-train-years", type=int, default=None, help="sliding instead of expanding training window")
    p.add_argument("--horizon", type=int, default=168)
    p.add_argument("--origin-hours", type=str, default="0,12")
    p.add_argument("--train-stride", type=int, default=None, help="keep every k-th training hour (default 1 / 3)")
    p.add_argument("--no-gru", action="store_true")
    p.add_argument("--gru-epochs", type=int, default=16)
    p.add_argument("--enso", action="store_true", help="also try ENSO stratification for --data cleaned")
    run(p.parse_args())


if __name__ == "__main__":
    main()
