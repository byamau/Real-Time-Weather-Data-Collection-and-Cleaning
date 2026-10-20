"""
Topic 6 — Extension: train the production forecaster used by the dashboard

Fits forecast_core.WeatherForecaster on ALL available cleaned data (the walk-forward script
05_walk_forward_evaluation.py is where the model is evaluated) and saves:
  models/weather_forecast_weights.joblib   fitted forecaster + location metadata + versions
  models/model_evaluation_report.json      model description + walk-forward summary (if present)

Usage
  python src/06_train_production_model.py                 # Topic 6 cleaned dataset (default)
  python src/06_train_production_model.py --data archive  # optional multi-year ERA5 archive
  python src/06_train_production_model.py --no-gru        # without the PyTorch GRU member
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_core as fc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["cleaned", "archive"], default="cleaned")
    p.add_argument("--no-gru", action="store_true")
    p.add_argument("--train-stride", type=int, default=None)
    args = p.parse_args()

    if args.data == "cleaned":
        df = pd.read_csv(BASE_DIR / "data" / "cleaned" / "cleaned_weather_data.csv")
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        df = pd.read_parquet(BASE_DIR / "data" / "archive" / "era5_multiyear_cleaned.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    stride = args.train_stride or (1 if args.data == "cleaned" else 3)

    print("=" * 78)
    print(f"STEP 6: TRAINING PRODUCTION FORECASTER on {args.data} data "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()}, {df['location_id'].nunique()} locations)")
    print("=" * 78)
    t0 = time.perf_counter()
    model = fc.WeatherForecaster(use_gru=not args.no_gru).fit(df, train_stride=stride, verbose=True)
    fit_sec = time.perf_counter() - t0

    locations = (
        df.groupby(["location_id", "city", "region"], as_index=False)[fc.STATIC_COLS].first()
        .sort_values("location_id").to_dict(orient="records")
    )
    trained_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    bundle = {
        "forecaster": model,
        "locations": locations,
        "trained_at": trained_at,
        "data": args.data,
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "pandas": pd.__version__},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "weather_forecast_weights.joblib"
    joblib.dump(bundle, path, compress=3)

    wf_path = MODEL_DIR / "walk_forward" / "summary.json"
    walk_forward = json.loads(wf_path.read_text(encoding="utf-8")) if wf_path.exists() else None
    report = {
        "model_name": "Recursive hybrid forecaster: weighted (Ridge + HistGradientBoosting"
                      + ("" if args.no_gru else " + GRU") + ") + out-of-fold XGBoost residual, damped towards climatology",
        "trained_at": trained_at,
        "fit_seconds": round(fit_sec, 1),
        "production_model": model.describe(),
        "versions": bundle["versions"],
        "walk_forward_evaluation": walk_forward,
        "note": "Production model is fitted on all data. Its expected accuracy is the walk-forward result, "
                "which is computed with models that never saw the test windows.",
    }
    with open(MODEL_DIR / "model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB) in {fit_sec:.0f} s")
    if walk_forward is None:
        print("Note: run src/05_walk_forward_evaluation.py to add the evaluation to the report.")


if __name__ == "__main__":
    main()
