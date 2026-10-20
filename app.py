r"""
Topic 6: Real-Time Weather Data Collection, Cleaning & 7-Day Forecast Dashboard
Run command: .\.venv\Scripts\python.exe -m streamlit run app.py
"""

import importlib.util
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"

sys.path.insert(0, str(BASE_DIR / "src"))  # forecast_core must be importable to load the model bundle
spec = importlib.util.spec_from_file_location(
    "realtime_inference", str(BASE_DIR / "src" / "07_realtime_inference.py")
)
rt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt_module)

st.set_page_config(
    page_title="Real-Time Weather Data Collection, Cleaning & Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_saved_weights():
    return rt_module.load_bundle(MODEL_DIR / "weather_forecast_weights.joblib")


@st.cache_data
def load_offline_datasets():
    df_clean = pd.read_csv(CLEAN_DIR / "cleaned_weather_data.csv")
    df_summary = pd.read_csv(CLEAN_DIR / "city_summary_statistics.csv")
    df_overlap = pd.read_csv(CLEAN_DIR / "multi_source_comparison.csv")
    with open(CLEAN_DIR / "data_quality_report.json", "r", encoding="utf-8") as f:
        q_report = json.load(f)
    with open(MODEL_DIR / "model_evaluation_report.json", "r", encoding="utf-8") as f:
        m_report = json.load(f)
    return df_clean, df_summary, df_overlap, q_report, m_report


weights_bundle = load_saved_weights()
df_clean_all, df_city_summary, df_overlap_all, quality_report, model_report = load_offline_datasets()
locations_list = weights_bundle["locations"]
city_names = [loc["city"] for loc in locations_list]

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## Bảng Điều Khiển")
    st.caption("Thu Thập, Làm Sạch Và Dự Báo Thời Tiết Thời Gian Thực")

    selected_label = st.selectbox("Chọn Địa Điểm:", city_names, index=3)
    selected_idx = city_names.index(selected_label)
    selected_loc = locations_list[selected_idx]

    forecast_days = st.slider("Số Ngày Dự Báo Tương Lai:", min_value=3, max_value=7, value=7, step=1)
    horizon_hours = forecast_days * 24

    st.markdown("---")
    live_btn = st.button(
        "GỌI 4 API REAL-TIME, LÀM SẠCH VÀ DỰ BÁO NGAY",
        type="primary",
        use_container_width=True,
    )

# =============================================================================
# EXECUTE REAL-TIME API FETCH + LIVE CLEANING + WEIGHT INFERENCE
# =============================================================================
state_key = f"live_result_v3_{selected_loc['location_id']}_{horizon_hours}"
if live_btn or state_key not in st.session_state:
    with st.spinner(f"Đang gọi trực tiếp 4 API thời gian thực cho {selected_loc['city']}, làm sạch dữ liệu và dự báo {forecast_days} ngày..."):
        df_raw_live, df_clean_live, api_status = rt_module.fetch_and_clean_live_city_data(selected_loc)
        df_hr_fc, df_day_fc, infer_ms = rt_module.forecast_7_days_via_weight_inference(
            weights_bundle, df_clean_live, selected_loc, horizon_hours=horizon_hours
        )
        st.session_state[state_key] = {
            "df_raw_live": df_raw_live,
            "df_clean_live": df_clean_live,
            "api_status": api_status,
            "df_hr_fc": df_hr_fc,
            "df_day_fc": df_day_fc,
            "infer_ms": infer_ms,
        }

res = st.session_state[state_key]
df_raw_live = res["df_raw_live"]
df_clean_live = res["df_clean_live"]
api_status = res["api_status"]
df_hr_fc = res["df_hr_fc"]
df_day_fc = res["df_day_fc"]
infer_ms = res["infer_ms"]

# =============================================================================
# HEADER & CURRENT REAL-TIME KPI METRICS
# =============================================================================
st.title(f"Hệ Thống Thu Thập, Làm Sạch Và Dự Báo Thời Tiết Thời Gian Thực — {selected_loc['city']}")
st.markdown(
    f"**Tọa độ:** `{selected_loc['requested_lat']}°N, {selected_loc['requested_lon']}°E` | "
    f"**Độ cao:** `{selected_loc['elevation_m']} m` | "
    f"**Mô hình:** `Hybrid (Ridge + HistGB + GRU) + XGBoost residual, trộn NWP Open-Meteo` | "
    f"**Thời gian phân tích:** `{infer_ms} ms`"
)

latest_obs = df_clean_live.ffill().iloc[-1]  # a provider may lack some variables in the latest hour
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Nhiệt Độ Hiện Tại", f"{latest_obs['temperature_2m']:.1f} °C", f"Dự báo 24h tới: {df_hr_fc.iloc[23]['temperature_2m']:.1f} °C")
kpi2.metric("Độ Ẩm Tương Đối", f"{latest_obs['relative_humidity_2m']:.0f} %", f"TB {forecast_days} ngày: {df_hr_fc['relative_humidity_2m'].mean():.0f} %")
kpi3.metric("Tốc Độ Gió", f"{latest_obs['wind_speed_10m']:.1f} km/h", f"Max {forecast_days} ngày: {df_hr_fc['wind_speed_10m'].max():.1f} km/h")
kpi4.metric("Lượng Mưa Giờ Gần Nhất", f"{latest_obs['precipitation']:.2f} mm", f"Tổng {forecast_days} ngày: {df_hr_fc['precipitation'].sum():.1f} mm")
kpi5.metric("Áp Suất Khí Quyển", f"{latest_obs['pressure_msl']:.1f} hPa", f"TB: {df_hr_fc['pressure_msl'].mean():.1f} hPa")
kpi6.metric("Thời Gian Phân Tích", f"{infer_ms} ms", f"{horizon_hours} bước dự báo")

# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    f"1. Dự Báo {forecast_days} Ngày Tới",
    "2. Gọi 4 API Trực Tiếp Và Làm Sạch Dữ Liệu",
    "3. Đánh Giá Dữ Liệu",
    "4. Đánh Giá Mô Hình",
])

# -----------------------------------------------------------------------------
# TAB 1: 7-DAY FORECAST
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Bảng Tóm Tắt Dự Báo {forecast_days} Ngày Tới Tại {selected_loc['city']}")
    cols_day = st.columns(len(df_day_fc))
    for idx, (_, d_row) in enumerate(df_day_fc.iterrows()):
        with cols_day[idx]:
            st.markdown(
                f"""
                <div style="padding:14px 10px; border-radius:10px; border:1px solid #cbd5e1; background-color:#f8fafc; text-align:center;">
                    <div style="font-size:17px; font-weight:800; color:#0f172a; margin-bottom:6px;">{d_row['date']}</div>
                    <div style="font-size:17px; font-weight:700; color:#1d4ed8; margin:6px 0;">{d_row['condition']}</div>
                    <div style="font-size:23px; font-weight:800; color:#d94801; margin:6px 0;">{d_row['temp_mean']:.1f}°C</div>
                    <div style="font-size:13.5px; font-weight:600; color:#334155; margin:4px 0;">Thấp: <b>{d_row['temp_min']:.1f}°C</b> | Cao: <b>{d_row['temp_max']:.1f}°C</b></div>
                    <div style="font-size:13.5px; font-weight:600; color:#0369a1; margin:4px 0;">Độ ẩm: {d_row['humidity_mean']:.0f}% | Mưa: {d_row['precip_total']:.1f}mm</div>
                    <div style="font-size:13.5px; font-weight:600; color:#1e293b; margin:4px 0;">Gió: {d_row['wind_mean']:.1f} km/h | {d_row['pressure_mean']:.0f} hPa</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Biểu Đồ Chuỗi Thời Gian Thực Tế Và Dự Báo Tương Lai")

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 9.5), sharex=True)
    fc_start_ts = df_hr_fc["timestamp"].min()
    hist_tail = df_clean_live[df_clean_live["timestamp"] < fc_start_ts].tail(72).copy()
    if hist_tail.empty:
        hist_tail = df_clean_live.tail(72).copy()

    # Subplot 1: Temperature
    axes[0].plot(hist_tail["timestamp"], hist_tail["temperature_2m"], color="#1e293b", linewidth=2.2, label="Lịch sử thực tế 72 giờ qua")
    axes[0].plot(df_hr_fc["timestamp"], df_hr_fc["temperature_2m"], color="#d94801", linewidth=2.2, linestyle="--", marker="o", markevery=12, label=f"Dự báo {forecast_days} ngày tới (ML + NWP)")
    axes[0].plot(df_hr_fc["timestamp"], df_hr_fc["temperature_2m_ml"], color="#7c3aed", linewidth=1.2, linestyle=":", label="Thành phần ML (chưa trộn NWP)")
    axes[0].axvline(hist_tail["timestamp"].max(), color="#dc2626", linestyle=":", linewidth=1.8, label="Thời điểm hiện tại")
    axes[0].set_ylabel("Nhiệt độ - °C", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Chuỗi Thời Gian Thực Tế Và Dự Báo {forecast_days} Ngày Tới — {selected_loc['city']}", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=10)

    # Subplot 2: Relative Humidity & Precipitation
    ax2_rain = axes[1].twinx()
    axes[1].plot(hist_tail["timestamp"], hist_tail["relative_humidity_2m"], color="#0284c7", linewidth=2.0, label="Độ ẩm thực tế")
    axes[1].plot(df_hr_fc["timestamp"], df_hr_fc["relative_humidity_2m"], color="#0284c7", linewidth=2.0, linestyle="--", label="Độ ẩm dự báo")
    ax2_rain.bar(hist_tail["timestamp"], hist_tail["precipitation"], width=0.03, color="#0284c7", alpha=0.35, label="Lượng mưa thực tế")
    ax2_rain.bar(df_hr_fc["timestamp"], df_hr_fc["precipitation"], width=0.03, color="#38bdf8", alpha=0.55, label="Lượng mưa dự báo")
    axes[1].axvline(hist_tail["timestamp"].max(), color="#dc2626", linestyle=":", linewidth=1.8)
    axes[1].set_ylabel("Độ ẩm - %", color="#0284c7", fontsize=11, fontweight="bold")
    ax2_rain.set_ylabel("Lượng mưa - mm/h", color="#0369a1", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=10)

    # Subplot 3: Wind Speed & Sea-Level Pressure
    ax3_pres = axes[2].twinx()
    axes[2].plot(hist_tail["timestamp"], hist_tail["wind_speed_10m"], color="#16a34a", linewidth=2.0, label="Tốc độ gió thực tế")
    axes[2].plot(df_hr_fc["timestamp"], df_hr_fc["wind_speed_10m"], color="#16a34a", linewidth=2.0, linestyle="--", label="Tốc độ gió dự báo")
    ax3_pres.plot(hist_tail["timestamp"], hist_tail["pressure_msl"], color="#7c3aed", linewidth=1.6, alpha=0.8, label="Áp suất thực tế")
    ax3_pres.plot(df_hr_fc["timestamp"], df_hr_fc["pressure_msl"], color="#7c3aed", linewidth=1.6, linestyle=":", alpha=0.9, label="Áp suất dự báo")
    axes[2].axvline(hist_tail["timestamp"].max(), color="#dc2626", linestyle=":", linewidth=1.8)
    axes[2].set_ylabel("Tốc độ gió - km/h", color="#16a34a", fontsize=11, fontweight="bold")
    ax3_pres.set_ylabel("Áp suất - hPa", color="#7c3aed", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Ngày/Tháng/Năm", fontsize=11, fontweight="bold")
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    plt.setp(axes[2].get_xticklabels(), fontsize=10, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl1:
        st.markdown("#### Bảng Chi Tiết Dự Báo Theo Từng Giờ")
        st.dataframe(
            df_hr_fc[["step_hour", "time", "city", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "pressure_msl"]],
            use_container_width=True,
            height=260,
        )
    with col_dl2:
        st.markdown("#### Tải Kết Quả Dự Báo")
        csv_bytes = df_hr_fc.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label=f"Tải CSV Dự Báo {forecast_days} Ngày - {selected_loc['city']}",
            data=csv_bytes,
            file_name=f"forecast_{forecast_days}days_{selected_loc['city'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: LIVE API FETCH & REAL-TIME DATA CLEANING INSPECTOR
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"Trạng Thái Gọi Trực Tiếp 4 Web API Thời Gian Thực Tại {selected_loc['city']}")
    api_cols = st.columns(4)
    for idx, (api_name, info) in enumerate(api_status.items()):
        with api_cols[idx]:
            st.metric(
                label=api_name,
                value=info["status"],
                delta=f"{info['records']} dòng | {info['latency_ms']} ms",
            )

    c_raw, c_clean = st.columns(2)
    with c_raw:
        st.markdown("#### Dữ Liệu Thô Vừa Gọi Từ 4 API")
        st.caption(
            f"Tổng số dòng thô: `{len(df_raw_live)}` dòng | "
            f"Số ô bị khuyết: `{int(df_raw_live[['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation', 'pressure_msl']].isna().sum().sum())}` ô | "
            f"Định dạng thời gian hỗn hợp giữa 4 API."
        )
        df_raw_display = df_raw_live.drop(columns=["location_id", "region"], errors="ignore")
        st.dataframe(df_raw_display.tail(25), use_container_width=True, height=340)

    with c_clean:
        st.markdown("#### Dữ Liệu Đã Chuẩn Hóa Và Hợp Nhất Đa Nguồn")
        st.caption(
            f"Tổng số giờ liên tục: `{len(df_clean_live)}` giờ | "
            f"Số ô khuyết: `0` ô | "
            f"Số giờ hợp nhất từ nhiều API: `{int((df_clean_live['num_sources'] >= 2).sum())}` giờ."
        )
        st.dataframe(
            df_clean_live[["time", "city", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "pressure_msl", "num_sources", "sources"]].tail(25),
            use_container_width=True,
            height=340,
        )

# -----------------------------------------------------------------------------
# TAB 3: DATA EVALUATION DASHBOARD
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Đánh Giá Dữ Liệu")
    b_info = quality_report["before_cleaning"]
    a_info = quality_report["after_cleaning"]

    m_info = quality_report["missing_api_responses"]
    q_df = pd.DataFrame({
        "Chỉ Số Đánh Giá Chất Lượng": [
            "Tổng số dòng",
            "Số lượng địa điểm",
            "Response API bị thiếu (nguồn × địa điểm)",
            "Timestamp chưa chuẩn hóa",
            "Bản ghi dự báo chưa xảy ra lúc thu thập",
            "Trùng lặp hoàn toàn (cùng nhà cung cấp)",
            "Trùng lặp cùng nguồn, cùng giờ",
            "Trùng lặp giữa các nguồn (đã hợp nhất)",
            "Ô dữ liệu bị khuyết",
            "Bất nhất vật lý gió giật < gió trung bình",
        ],
        "Trước Khi Làm Sạch": [
            f"{b_info['total_rows']:,}",
            f"{b_info['num_locations']}",
            f"{m_info['requests_missing_or_failed']} / {m_info['requests_checked']}",
            f"{b_info['unstandardized_timestamps']:,}",
            f"{b_info['not_yet_observed_forecast_rows']:,}",
            f"{b_info['exact_duplicates']:,}",
            f"{b_info['same_source_family_duplicates']:,}",
            f"{b_info['cross_source_duplicates']:,}",
            f"{b_info['total_missing_cells']:,}",
            f"{a_info['wind_gust_inconsistencies_fixed']:,}",
        ],
        "Sau Khi Làm Sạch": [
            f"{a_info['total_rows']:,}",
            f"{a_info['num_locations']}",
            "phát hiện & ghi nhận (bù bằng nguồn khác)",
            f"{a_info['unstandardized_timestamps']}",
            "0",
            f"{a_info['exact_duplicates']}",
            "0",
            f"{a_info['key_duplicates']}",
            f"{a_info['total_missing_cells']}",
            f"{a_info['wind_gust_inconsistencies']}",
        ],
    })
    st.table(q_df)

    st.markdown("### Thống Kê Mô Tả Toàn Bộ 20 Địa Điểm Khảo Sát")
    df_summary_display = df_city_summary.drop(columns=["location_id", "region"], errors="ignore")
    st.dataframe(df_summary_display, use_container_width=True, height=360)

    st.markdown("### Bộ Biểu Đồ Phân Tích Chất Lượng Dữ Liệu Và Khí Tượng")
    g1, g2 = st.columns(2)
    with g1:
        st.image(str(FIG_DIR / "fig1_data_quality_before_after.png"), caption="Hình 1: Đối sánh lỗi chất lượng Trước và Sau khi làm sạch", use_container_width=True)
        st.image(str(FIG_DIR / "fig3_temperature_timeseries.png"), caption="Hình 3: Chuỗi thời gian nhiệt độ trung bình ngày của 8 địa điểm đại diện", use_container_width=True)
        st.image(str(FIG_DIR / "fig5_multivariate_and_precipitation.png"), caption="Hình 5: Tương quan đa biến Nhiệt độ - Độ ẩm - Áp suất và Tổng lượng mưa", use_container_width=True)
    with g2:
        st.image(str(FIG_DIR / "fig6_multi_api_source_comparison.png"), caption="Hình 6: Độ lệch hệ thống giữa các nguồn tại cùng giờ", use_container_width=True)
        st.image(str(FIG_DIR / "fig2_range_validation_boxplots.png"), caption="Hình 2: Phân phối Boxplot 6 biến khí tượng sau làm sạch", use_container_width=True)
        st.image(str(FIG_DIR / "fig4_all_20_cities_temperature_range.png"), caption="Hình 4: So sánh nhiệt độ trung bình và biên độ nhiệt của 20 địa điểm", use_container_width=True)
    st.image(str(FIG_DIR / "fig7_api_coverage_heatmap.png"), caption="Hình 7: Độ phủ response của từng API theo địa điểm (phát hiện response bị thiếu)", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: MODEL EVALUATION (WALK-FORWARD, BY LEAD TIME)
# -----------------------------------------------------------------------------
with tab4:
    wf = model_report.get("walk_forward_evaluation")
    st.subheader("Đánh Giá Walk-Forward Theo Thời Hạn Dự Báo (Lead Time)")
    st.markdown(
        "Mô hình dự báo 1 giờ/lần và được chạy **cuốn chiếu 168 bước** cho dự báo 7 ngày, nên được đánh giá "
        "theo từng lead time trên nhiều cửa sổ kiểm thử trượt dần (walk-forward), so với 3 baseline: "
        "persistence, seasonal naive (lặp lại 24 giờ gần nhất) và climatology của tập train."
    )
    if wf is None:
        st.warning("Chưa có kết quả walk-forward. Chạy `python src/05_walk_forward_evaluation.py`.")
    else:
        cfg = wf["config"]
        st.caption(f"{cfg['n_folds']} fold × {cfg['fold_days']} ngày, {cfg['n_trajectories_total']:,} lượt dự báo 168 giờ, "
                   f"dữ liệu `{cfg['data']}` ({cfg['test_period'][0][:10]} → {cfg['test_period'][1][:10]}).")
        st.image(str(FIG_DIR / "fig8_walk_forward_mae_by_lead.png"), caption="Hình 8: MAE theo lead time", use_container_width=True)
        names = {"h1": "1 giờ", "day1": "Ngày 1", "day2-3": "Ngày 2–3", "day4-7": "Ngày 4–7"}
        units = {"temperature_2m": "°C", "relative_humidity_2m": "%", "wind_speed_10m": "km/h", "precipitation": "mm/h", "pressure_msl": "hPa"}
        var_sel = st.selectbox("Biến:", list(wf["by_variable"].keys()), index=0)
        rows = []
        for bk, e in wf["by_variable"][var_sel].items():
            rows.append({
                "Thời hạn": names.get(bk, bk),
                f"Hybrid MAE ({units[var_sel]})": e["hybrid_mae"],
                "Seasonal naive": e["seasonal_naive_mae"],
                "Persistence": e["persistence_mae"],
                "Climatology": e["climatology_mae"],
                "Skill vs naive [95% CI]": f"{e['skill_vs_seasonal_naive']:+.3f} {e['skill_vs_seasonal_naive_ci95']}",
                "Tỷ lệ fold thắng naive": e["share_folds_beating_seasonal_naive"],
            })
        st.table(pd.DataFrame(rows))
        st.image(str(FIG_DIR / "fig9_walk_forward_fold_skill.png"), caption="Hình 9: Skill theo từng fold", use_container_width=True)

    st.markdown("---")
    st.subheader("Cấu Hình Mô Hình Production")
    prod = model_report["production_model"]
    st.json({"train": prod["train"], "damping": prod["damping"],
             "weights_per_variable": {v: m["weights"] for v, m in prod["models"].items()},
             "residual_alpha": {v: m["residual_alpha"] for v, m in prod["models"].items()}})
    st.info(
        "Dự báo trên Tab 1 = thành phần ML ở trên trộn với dự báo NWP của Open-Meteo, trọng số NWP tăng dần theo lead time "
        f"({rt_module.NWP_WEIGHT_START:.2f} → tối đa {rt_module.NWP_WEIGHT_MAX:.2f}). Trọng số trộn này là heuristic, "
        "chưa được kiểm định vì cần dữ liệu dự báo NWP lưu trữ tại từng thời điểm phát dự báo."
    )
