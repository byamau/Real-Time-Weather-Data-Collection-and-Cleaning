# Topic 6: Real-Time Weather Data Collection, Cleaning & 7-Day Forecast

Dự án thu thập dữ liệu thời tiết thời gian thực từ **4 Web API**, làm sạch & hợp nhất dữ liệu chuỗi thời gian đa nguồn cho **20 tỉnh/thành phố tại Việt Nam**, huấn luyện mô hình dự báo **Hybrid + Học phần dư XGBoost (Residual Learning)**, lưu trọng số (`weather_forecast_weights.joblib`) và triển khai **Web Dashboard bằng Streamlit** để dự báo thời tiết 7 ngày tới (168 giờ).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
Đồ Án Clean Data/
├── app.py                              # Giao diện Web Dashboard Streamlit (Demo trực tiếp)
├── run_dashboard.bat                   # File chạy nhanh Streamlit 1-click trên Windows
├── requirements.txt                    # Danh sách thư viện Python cần cài đặt
├── .env                                # Chứa API Keys của các dịch vụ thời tiết
├── src/
│   ├── 01_collect_weather_data.py      # Bước 1: Gọi 4 API thu thập 46,600 dòng dữ liệu thô
│   ├── 02_clean_and_preprocess.py      # Bước 2: Làm sạch, chuẩn hóa & hợp nhất thành 30,220 dòng sạch
│   ├── 03_evaluate_and_visualize.py    # Bước 3: Thống kê & xuất các biểu đồ phân tích chất lượng dữ liệu (fig1 -> fig6)
│   ├── 04_train_and_save_weights.py    # Bước 4: Huấn luyện mô hình Hybrid + XGBoost Residual & lưu trọng số (.joblib)
│   └── 05_realtime_inference.py        # Bước 5: Gọi 4 API thời gian thực, làm sạch trực tiếp & dự báo 7 ngày bằng Weight Inference
├── data/
│   ├── raw/
│   │   ├── raw_weather_data.csv        # Dữ liệu thô từ 4 API (46,600 dòng)
│   │   └── raw_weather_responses.json  # Metadata phản hồi JSON từ 4 API
│   └── cleaned/
│       ├── cleaned_weather_data.csv    # Dữ liệu sạch hoàn chỉnh (30,220 dòng = 20 tỉnh × 1,511 giờ)
│       ├── multi_source_comparison.csv # Dữ liệu đối sánh các khung giờ giao nhau giữa 4 API
│       ├── city_summary_statistics.csv # Thống kê mô tả khí tượng của 20 tỉnh/thành phố
│       └── data_quality_report.json    # Báo cáo kiểm toán chất lượng dữ liệu Trước & Sau khi làm sạch
├── models/
│   ├── weather_forecast_weights.joblib # File trọng số mô hình đã huấn luyện (dùng để suy luận trực tiếp)
│   └── model_evaluation_report.json    # Kết quả đánh giá MAE, RMSE, R2 trên tập Validation & Test
└── figures/                            # Toàn bộ biểu đồ trực quan hóa chất lượng dữ liệu & đánh giá mô hình (fig1 -> fig7)
```

---

## 2. Hướng Dẫn Cài Đặt Môi Trường (Dành Cho Thành Viên Trong Nhóm)

### Yêu cầu hệ thống
* **Python:** Phiên bản `3.10` trở lên.
* **Hệ điều hành:** Windows / macOS / Linux.

### Bước 1: Mở Terminal (PowerShell hoặc Command Prompt) tại thư mục dự án
Di chuyển vào thư mục gốc của đồ án:
```powershell
cd "đường_dẫn_tới_thư_mục\Đồ Án Clean Data"
```

### Bước 2: Tạo môi trường ảo (Khuyến nghị nếu chạy trên máy mới) và cài đặt thư viện
* **Nếu máy bạn chưa có thư viện, chạy lệnh cài đặt từ `requirements.txt`:**
  ```powershell
  pip install -r requirements.txt
  ```
* **Hoặc nếu muốn tạo môi trường ảo `.venv` riêng biệt trên máy của bạn:**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

### Bước 3: Kiểm tra file cấu hình `.env`
Đảm bảo trong thư mục gốc của dự án có file `.env` chứa khóa kết nối của 3 API có xác thực (Open-Meteo không cần key):
```env
WEATHERAPI_KEY=4cccdf3aea344818bbc42453261710
FREE_OPENAPI_DEV_KEY=foa_tPQbxE66tIgUviOPSbVvZvi37z3IXw0o
XWEATHER_CLIENT_ID=sUiRPGi9vppWKE7rO3LjQ
XWEATHER_CLIENT_SECRET=w4hTUpmqywBxRuFJH7IRhcgBSCaNLrC7XNXQxh0g
```

---

## 3. Cách Chạy Nhanh Web Dashboard Streamlit (Để Check & Demo)

Trong dự án đã **có sẵn dữ liệu sạch (`data/cleaned/`)** và **trọng số mô hình đã huấn luyện (`models/weather_forecast_weights.joblib`)**, nên các thành viên có thể mở ngay giao diện Web Dashboard mà không cần chờ train lại:

### Cách 1: Chạy bằng lệnh chuẩn trong Terminal / PowerShell
* Nếu dùng Python mặc định của máy:
  ```powershell
  python -m streamlit run app.py
  ```
* Nếu dùng môi trường `.venv` có sẵn trong thư mục:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```

### Cách 2: Chạy bằng 1-Click trên Windows
* Nhấn đúp chuột vào file **`run_dashboard.bat`** trong thư mục dự án.

Sau khi chạy, trình duyệt sẽ tự động mở địa chỉ:
👉 **`http://localhost:8501`**

---

## 4. Hướng Dẫn Chạy Từng Bước Trong Pipeline (Từ Thu Thập $\rightarrow$ Làm Sạch $\rightarrow$ Train Model)

Nếu thành viên muốn kiểm tra hoặc chạy lại từng bước xử lý từ đầu bằng code Python:

### Bước 1: Thu thập dữ liệu từ 4 Web API (`01_collect_weather_data.py`)
Gọi trực tiếp 4 API (`Open-Meteo SDK`, `WeatherAPI.com`, `Xweather API`, `FreeOpenAPI.dev`) cho 20 địa điểm tại Việt Nam và lưu vào `data/raw/`:
```powershell
python src/01_collect_weather_data.py
```

### Bước 2: Làm sạch và tiền xử lý dữ liệu (`02_clean_and_preprocess.py`)
Thực hiện 6 bước làm sạch dữ liệu thực tế:
1. Chuẩn hóa 4 định dạng thời gian khác nhau về `YYYY-MM-DD HH:MM:SS` (`UTC+7`).
2. Loại bỏ `15,380` dòng phản hồi rỗng hoàn toàn (`NaN` ở cả 6 biến vượt quá giới hạn lưu trữ 61 ngày của endpoint forecast).
3. Nội suy và bù khuyết chéo `20` dòng khuyết cục bộ (`60` ô `NaN`) do khác biệt cấu trúc JSON của `FreeOpenAPI.dev`.
4. Xử lý `1,000` bản ghi trùng khóa thời gian `(location_id, timestamp)` bằng kỹ thuật hợp nhất đa cảm biến (Multi-API Sensor Fusion).
5. Kiểm tra miền giá trị vật lý hợp lệ và sửa `94` bản ghi bất nhất logic (`wind_gusts_10m < wind_speed_10m`).
6. Làm tròn `96,525` ô bị nhiễu dấu chấm động nhị phân `Float32` và đồng bộ lưới thời gian liên tục `1,511` giờ/địa điểm (`30,220` dòng sạch).

```powershell
python src/02_clean_and_preprocess.py
```

### Bước 3: Xuất thống kê và biểu đồ phân tích chất lượng dữ liệu (`03_evaluate_and_visualize.py`)
Tạo bảng thống kê 20 tỉnh/thành và xuất các biểu đồ từ `fig1` đến `fig6` vào thư mục `figures/`:
```powershell
python src/03_evaluate_and_visualize.py
```

### Bước 4: Huấn luyện mô hình Hybrid + Học phần dư XGBoost & Lưu trọng số (`04_train_and_save_weights.py`)
* Tạo 58 đặc trưng chuỗi thời gian (`Lag 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h`, trung bình trượt 24h, mã hóa chu kỳ ngày/năm `sin/cos`, tọa độ & độ cao).
* Chia tập dữ liệu nghiêm ngặt theo thứ tự thời gian: `70% Train` (`20,160` dòng) / `15% Validation` (`4,320` dòng) / `15% Test` (`4,300` dòng).
* Huấn luyện kiến trúc **Hybrid + XGBoost Residual**:
  * **Tầng 1 (Mô hình cơ sở):** Kết hợp `Ridge Autoregressive` và `HistGradientBoosting` để dự báo xu hướng chính $\hat{y}_{\text{base}}$.
  * **Tầng 2 (Học phần dư bằng XGBoost):** Dùng `XGBRegressor` học trực tiếp chuỗi phần dư $r_t = y_t - \hat{y}_{\text{base}}(X_t)$.
  * **Tầng 3 (Tổng hợp Hybrid):** $\hat{y}_{\text{Hybrid}} = \hat{y}_{\text{base}} + 0.50 \cdot \hat{r}_{\text{XGBoost}}$ (đạt **Test MAE = `0.363°C`**, **Test RMSE = `0.553°C`**, **$R^2 = 0.9751$**).
* Lưu toàn bộ trọng số vào `models/weather_forecast_weights.joblib` và xuất biểu đồ đánh giá `figures/fig7_model_training_and_forecast_eval.png`.

```powershell
python src/04_train_and_save_weights.py
```

### Bước 5: Kiểm thử suy luận thời gian thực 7 ngày trên CLI (`05_realtime_inference.py`)
Gọi trực tiếp 4 API lấy dữ liệu mới nhất, làm sạch tại chỗ và nạp file trọng số `weather_forecast_weights.joblib` để dự báo 168 giờ (7 ngày tới):
```powershell
python src/05_realtime_inference.py
```

---

## 5. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

1. **Lỗi `'streamlit' is not recognized as the name of a cmdlet`:**
   * Nguyên nhân: Đường dẫn `Scripts` của Python chưa được thêm vào biến môi trường `PATH`.
   * Cách khắc phục: Luôn chạy bằng cú pháp `python -m streamlit run app.py` (hoặc `.\.venv\Scripts\python.exe -m streamlit run app.py`).
2. **Lỗi thiếu thư viện (`ModuleNotFoundError: No module named 'xgboost'` hoặc `'openmeteo_requests'`):**
   * Cách khắc phục: Chạy lệnh `pip install -r requirements.txt` bằng đúng trình thông dịch Python bạn đang dùng để chạy Streamlit.
3. **Khi demo ở nơi không có kết nối Internet hoặc API bị quá hạn mức (Rate Limit):**
   * Hệ thống trong `src/05_realtime_inference.py` đã có cơ chế **Tự động chuyển đổi dự phòng (Automatic Fallback)**: nếu 1 trong 4 API bị lỗi mạng, hệ thống vẫn hợp nhất từ các API còn hoạt động; nếu mất mạng hoàn toàn, hệ thống tự động nạp chuỗi lịch sử sạch gần nhất trong `data/cleaned/cleaned_weather_data.csv` để đảm bảo buổi demo trên lớp không bao giờ bị gián đoạn.
