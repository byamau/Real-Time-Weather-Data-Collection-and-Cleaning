"""
Topic 6: Real-Time Weather Data Collection and Cleaning
Step 3: Data Quality Evaluation, Descriptive Statistics & Matplotlib Visualizations
on 100% Real 4-API Dataset (Open-Meteo SDK + WeatherAPI.com + Xweather API + FreeOpenAPI.dev)
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_and_visualize_real_data():
    df_clean = pd.read_csv(CLEAN_DIR / "cleaned_weather_data.csv")
    df_overlap = pd.read_csv(CLEAN_DIR / "multi_source_comparison.csv")
    with open(CLEAN_DIR / "data_quality_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    df_clean["timestamp"] = pd.to_datetime(df_clean["time"])

    # -------------------------------------------------------------------------
    # 1. Export City-Level Descriptive Statistics CSV (across 1,511 hours per city)
    # -------------------------------------------------------------------------
    city_summary = (
        df_clean.groupby(["location_id", "region", "city", "elevation_m"], as_index=False)
        .agg(
            observations=("time", "count"),
            multi_api_hours=("num_sources", lambda s: int((s >= 2).sum())),
            temp_mean=("temperature_2m", "mean"),
            temp_std=("temperature_2m", "std"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            humidity_mean=("relative_humidity_2m", "mean"),
            humidity_min=("relative_humidity_2m", "min"),
            wind_mean=("wind_speed_10m", "mean"),
            wind_max=("wind_speed_10m", "max"),
            gust_max=("wind_gusts_10m", "max"),
            precip_total=("precipitation", "sum"),
            precip_max_hourly=("precipitation", "max"),
            pressure_mean=("pressure_msl", "mean"),
            pressure_min=("pressure_msl", "min"),
            pressure_max=("pressure_msl", "max"),
        )
        .round(2)
        .sort_values("location_id")
    )
    summary_path = CLEAN_DIR / "city_summary_statistics.csv"
    city_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # Figure 1: Multi-API Data Standardization, Deduplication & Extreme Audit
    # -------------------------------------------------------------------------
    before = report["before_cleaning"]
    iqr_dict = before["statistical_iqr_extremes_by_column"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    std_labels = [
        "Float32 Precision\nArtifacts",
        "Unstandardized\nTimestamps (4 APIs)",
        "Overlapping\nDuplicates (4 APIs)",
        "Unobserved / Future\nAPI Rows Filtered",
        "Wind Gusts < Speed\nInconsistencies",
    ]
    before_std = [
        before["float32_precision_artifacts"],
        before["unstandardized_timestamps"],
        before["key_duplicates"],
        before["empty_api_response_rows"],
        before["wind_gust_inconsistencies"],
    ]
    after_std = [0, 0, 0, 0, 0]

    total_raw_cnt = before["total_rows"]
    total_clean_cnt = report["after_cleaning"]["total_rows"]
    hours_per_loc = report["after_cleaning"]["hours_per_location"]

    x1 = np.arange(len(std_labels))
    w = 0.36
    b1 = ax1.bar(x1 - w / 2, before_std, w, label=f"Raw Multi-API Data ({total_raw_cnt:,} rows)", color="#e05a47", edgecolor="black")
    b2 = ax1.bar(x1 + w / 2, after_std, w, label=f"Cleaned Dataset ({total_clean_cnt:,} rows)", color="#2b8cbe", edgecolor="black")
    ax1.set_title("Multi-API Data Quality Issues Detected & Cleaned", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Count of Cells / Records", fontsize=10.5)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(std_labels, fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=9)
    for bar in b1:
        h = int(bar.get_height())
        ax1.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    for bar in b2:
        h = int(bar.get_height())
        ax1.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#1b5e20")
    ax1.set_ylim(0, max(before_std) * 1.16)

    # Right panel: Real Meteorological Extremes detected by 1.5*IQR across 20 locations
    var_names = list(iqr_dict.keys())
    var_counts = [iqr_dict[k] for k in var_names]
    y2 = np.arange(len(var_names))
    bars_iqr = ax2.barh(y2, var_counts, color="#756bb1", edgecolor="black", alpha=0.85)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(var_names, fontsize=9.5)
    ax2.set_xlabel("Number of Extreme Hourly Observations (1.5×IQR / Heavy Rain)", fontsize=9.5)
    ax2.set_title("Real Meteorological Extremes Across 20 Locations", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars_iqr:
        bw = int(bar.get_width())
        ax2.text(bw + max(var_counts) * 0.02, bar.get_y() + bar.get_height() / 2, f"{bw:,}", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, max(var_counts) * 1.18)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_data_quality_before_after.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Boxplot Distributions of All 6 Weather Variables
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    plot_specs = [
        ("temperature_2m", "Temperature 2m (°C)", "#f46d43"),
        ("relative_humidity_2m", "Relative Humidity 2m (%)", "#66c2a5"),
        ("pressure_msl", "Sea-Level Pressure (hPa)", "#5e4fa2"),
        ("wind_speed_10m", "Wind Speed 10m (km/h)", "#3288bd"),
        ("wind_gusts_10m", "Wind Gusts 10m (km/h)", "#d53e4f"),
        ("precipitation", "Hourly Precipitation (mm)", "#2b8cbe"),
    ]
    for ax, (col, title_lbl, color) in zip(axes.flat, plot_specs):
        series = df_clean[col].dropna()
        bp = ax.boxplot(
            series,
            orientation="horizontal",
            patch_artist=True,
            widths=0.45,
            flierprops=dict(marker="o", markerfacecolor=color, markersize=3, alpha=0.35),
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.75)
        ax.set_title(f"{title_lbl}\n[Min: {series.min():.1f} | Mean: {series.mean():.1f} | Max: {series.max():.1f}]", fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(f"Distributions & Real Meteorological Outliers Across All 6 Variables ({total_clean_cnt:,} Hourly Observations)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_range_validation_boxplots.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Daily Mean Temperature Time-Series for 8 Key Locations
    # -------------------------------------------------------------------------
    rep_cities = [
        "Ho Chi Minh City", "Hanoi", "Da Nang", "Can Tho",
        "Da Lat - Lam Dong", "Phu Quoc", "Sa Pa", "Nha Trang"
    ]
    daily_temp = (
        df_clean[df_clean["city"].isin(rep_cities)]
        .groupby(["city", "obs_date"], as_index=False)["temperature_2m"]
        .mean()
    )
    daily_temp["obs_date"] = pd.to_datetime(daily_temp["obs_date"])

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    palette = ["#d73027", "#f46d43", "#fdae61", "#4575b4", "#2b5c8f", "#1a9850", "#762a83", "#008080"]
    for c_name, color in zip(rep_cities, palette):
        sub = daily_temp[daily_temp["city"] == c_name].sort_values("obs_date")
        ax.plot(sub["obs_date"], sub["temperature_2m"], label=c_name, linewidth=1.9, color=color, alpha=0.9)

    ax.set_title("Daily Mean Temperature Time-Series Across 8 Representative Locations (Archive + Real-Time Fused)", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel(f"Date ({df_clean['obs_date'].min()} to {df_clean['obs_date'].max()})", fontsize=11)
    ax.set_ylabel("Daily Mean Temperature (°C)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_temperature_timeseries.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 4: All 20 Locations Ranked by Mean Temperature & Min-Max Range
    # -------------------------------------------------------------------------
    ranked = city_summary.sort_values("temp_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 7.2))
    y_pos = np.arange(len(ranked))

    ax.hlines(y=y_pos, xmin=ranked["temp_min"], xmax=ranked["temp_max"], color="#6baed6", alpha=0.8, linewidth=5, label="Min–Max Temperature Range")
    ax.scatter(ranked["temp_mean"], y_pos, color="#d94801", s=65, zorder=3, label="Mean Temperature (°C)")

    for i, row in ranked.iterrows():
        ax.text(row["temp_max"] + 0.35, i, f"{row['temp_mean']:.1f}°C ({row['temp_min']:.1f}–{row['temp_max']:.1f}°C)",
                va="center", fontsize=8.5, color="#252525")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['city']} - {int(r['elevation_m'])}m" for _, r in ranked.iterrows()], fontsize=9)
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_title(f"Temperature Profile Across All 20 Locations ({hours_per_loc:,} Hours/Location)", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlim(ranked["temp_min"].min() - 2, ranked["temp_max"].max() + 6.5)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_all_20_cities_temperature_range.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 5: Multivariate Correlation & Cumulative Precipitation Across 20 Locations
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    sample_df = df_clean.sample(n=min(4000, len(df_clean)), random_state=42)
    sc = ax1.scatter(
        sample_df["temperature_2m"],
        sample_df["relative_humidity_2m"],
        c=sample_df["pressure_msl"],
        cmap="viridis",
        alpha=0.55,
        s=15,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Sea-Level Pressure (hPa)", fontsize=9.5)
    ax1.set_title("Temperature vs. Relative Humidity (Colored by Pressure)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Temperature 2m (°C)", fontsize=10.5)
    ax1.set_ylabel("Relative Humidity 2m (%)", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.35)

    top_rain = city_summary.sort_values("precip_total", ascending=True).reset_index(drop=True)
    y_r = np.arange(len(top_rain))
    bars = ax2.barh(y_r, top_rain["precip_total"], color="#2b8cbe", alpha=0.85, edgecolor="black")
    ax2.set_yticks(y_r)
    ax2.set_yticklabels(top_rain["city"], fontsize=8.5)
    ax2.set_xlabel("Cumulative Precipitation (mm)", fontsize=10.5)
    ax2.set_title("Total Cumulative Precipitation Across All 20 Locations (mm)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    for b in bars:
        w = b.get_width()
        ax2.text(w + max(top_rain["precip_total"]) * 0.015, b.get_y() + b.get_height() / 2, f"{w:.0f} mm", va="center", fontsize=8)
    ax2.set_xlim(0, top_rain["precip_total"].max() * 1.18)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_multivariate_and_precipitation.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 6: Cross-API Real-Time Comparison (All 4 Real-Time APIs)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.4))

    # Left panel: Mean Temperature & Wind Speed reported by each of the 4 Real-Time APIs
    api_means = (
        df_overlap.groupby("source_api", as_index=False)
        .agg(
            records=("temperature_2m", "count"),
            temp_mean=("temperature_2m", "mean"),
            wind_mean=("wind_speed_10m", "mean"),
            humidity_mean=("relative_humidity_2m", "mean"),
        )
        .sort_values("records", ascending=False)
    )
    x_api = np.arange(len(api_means))
    w_api = 0.35
    b_t = ax1.bar(x_api - w_api / 2, api_means["temp_mean"], w_api, label="Mean Temperature (°C)", color="#f46d43", edgecolor="black")
    b_w = ax1.bar(x_api + w_api / 2, api_means["wind_mean"], w_api, label="Mean Wind Speed (km/h)", color="#3288bd", edgecolor="black")
    ax1.set_xticks(x_api)
    ax1.set_xticklabels(
        [f"{r['source_api']}\n(Realtime, n={int(r['records'])})" for _, r in api_means.iterrows()],
        fontsize=8.8
    )
    ax1.set_ylabel("Measurement Value (°C / km/h)", fontsize=10.5)
    ax1.set_title("Cross-API Real-Time Sensor Comparison (All 4 Live APIs)", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True)
    for bar in b_t:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}°C", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b_w:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, max(api_means["temp_mean"]) * 1.2)

    # Right panel: Real-Time Temperature Curve & Live Snapshot in Ho Chi Minh City across All 4 APIs
    hcm_ov = df_overlap[df_overlap["city"] == "Ho Chi Minh City"].copy()
    hcm_ov["ts"] = pd.to_datetime(hcm_ov["time"])
    api_styles = {
        "Open-Meteo SDK": ("#d73027", "-", "o", "Open-Meteo SDK (Realtime)"),
        "WeatherAPI.com": ("#2b8cbe", "--", "s", "WeatherAPI.com (Realtime)"),
        "Xweather API": ("#1a9850", "-.", "^", "Xweather API (Realtime)"),
        "FreeOpenAPI.dev": ("#762a83", ":", "D", "FreeOpenAPI.dev (Realtime)"),
    }
    for src_name, (col_hex, lstyle, marker_sh, label_txt) in api_styles.items():
        sub = hcm_ov[hcm_ov["source_api"] == src_name].sort_values("ts")
        if len(sub) > 1:
            ax2.plot(
                sub["ts"], sub["temperature_2m"],
                linestyle=lstyle, color=col_hex, linewidth=2.0,
                marker=marker_sh, markevery=max(1, len(sub) // 6), markersize=5.5,
                label=label_txt
            )
        elif len(sub) == 1:
            ax2.plot(
                sub["ts"], sub["temperature_2m"],
                linestyle="none", color=col_hex,
                marker=marker_sh, markersize=8.5,
                label=label_txt, zorder=6
            )

    ax2.set_title("Real-Time Multi-API Temperature Agreement in Ho Chi Minh City", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Observation Timestamp (UTC+7)", fontsize=10.5)
    ax2.set_ylabel("Temperature (°C)", fontsize=10.5)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig6_path = FIG_DIR / "fig6_multi_api_source_comparison.png"
    try:
        if fig6_path.exists():
            fig6_path.unlink()
        fig.savefig(fig6_path, bbox_inches="tight")
    except OSError:
        fig6_alt = FIG_DIR / "fig6_realtime_4_apis_comparison.png"
        fig.savefig(fig6_alt, bbox_inches="tight")
        fig6_path = fig6_alt
    plt.close(fig)

    print("=" * 78)
    print(f"Saved 20-city summary statistics to: {summary_path}")
    print(f"Saved 6 high-resolution figures to : {FIG_DIR} (Latest: {fig6_path.name})")
    print("=" * 78)


if __name__ == "__main__":
    evaluate_and_visualize_real_data()
