# BÁO CÁO ĐỒ ÁN CUỐI KỲ
## TOPIC 6: REAL-TIME WEATHER DATA COLLECTION AND CLEANING
### (Tích hợp Đa nguồn Thực tế từ 4 Web API: Open-Meteo SDK, WeatherAPI.com, Xweather API, FreeOpenAPI.dev)

---

### TRANG BÌA (TITLE PAGE)
- **Trường / Đại học:** *(Điền tên trường của bạn)*
- **Khoa / Bộ môn:** Khoa Công nghệ Thông tin / Khoa học Dữ liệu
- **Học phần:** Thu thập và Tiền xử lý Dữ liệu (Data Collection and Cleaning)
- **Đề tài:** **Topic 6 – Real-Time Weather Data Collection and Cleaning**
- **Danh sách thành viên thực hiện:**
  1. Họ và tên: `........................................` – MSSV: `................`
  2. Họ và tên: `........................................` – MSSV: `................`
  3. Họ và tên: `........................................` – MSSV: `................`

---

## TÓM TẮT & GIỚI THIỆU (ABSTRACT / INTRODUCTION)

### 1. Mục đích của đồ án (Purpose of the Project)
Đồ án xây dựng một hệ thống thu thập, làm sạch và tích hợp dữ liệu thời tiết chuỗi thời gian (**Time-Series Weather Dataset**) theo thời gian thực từ **4 nguồn Web API độc lập** tại **20 địa điểm trải dài khắp Việt Nam**. Việc kết hợp đa nguồn API thực tế giúp giải quyết trọn vẹn các thách thức điển hình trong Khoa học Dữ liệu:
- Đồng bộ **4 định dạng thời gian khác nhau** (`YYYY-MM-DD HH:MM:SS+07:00`, `YYYY-MM-DD HH:MM`, `YYYY-MM-DDTHH:MM:SS+07:00`, `YYYY-MM-DDTHH:MM:SS`).
- Xử lý **dữ liệu khuyết toàn cục** (`15.380` dòng rỗng vượt giới hạn lưu trữ 61 ngày của Open-Meteo `/v1/forecast`) và **dữ liệu khuyết cục bộ do khác biệt Schema** (`FreeOpenAPI.dev` khuyết 3 trường `humidity`, `pressure`, `wind_gusts` ở cả 20 địa điểm).
- Xử lý **1.000 bản ghi trùng lặp mốc thời gian `(location_id, time)`** giữa 4 API bằng phương pháp hợp nhất đa cảm biến (**Multi-Source Sensor Fusion**) và khử nhiễu dấu chấm động nhị phân `Float32` trên **96.525 ô dữ liệu**.

### 2. Bốn Nguồn Web API Thực tế Đã Thu thập (Selected Data Sources)
1. **Open-Meteo Official Python SDK (`openmeteo_requests`):** Thu thập chuỗi quan trắc dài hạn (`past_days=92, forecast_days=1`) cho 20 tọa độ $\rightarrow$ **`44.640` bản ghi**.
2. **WeatherAPI.com (`/v1/forecast.json` & `current`):** Thu thập chuỗi 72 giờ (3 ngày) và thời tiết hiện tại cho 20 tọa độ $\rightarrow$ **`1.460` bản ghi**.
3. **Vaisala Xweather API (`/conditions`):** Thu thập chuỗi quan trắc khí tượng 24 giờ gần nhất (`from=-24hours&to=now`) cho 20 tọa độ $\rightarrow$ **`480` bản ghi**.
4. **FreeOpenAPI.dev (`/api/v1/weather/current`):** Thu thập quan trắc thời gian thực hiện tại cho 20 tọa độ $\rightarrow$ **`20` bản ghi**.

- **Tổng số bản ghi thô thu thập (`raw_weather_data.csv`):** **`46.600` dòng** (gấp **77 lần** yêu cầu tối thiểu 600 quan trắc).
- **Tổng số bản ghi sạch sau hợp nhất (`cleaned_weather_data.csv`):** **`30.220` dòng** (`20 địa điểm × 1.511 giờ liên tục`, `0%` lỗi/khuyết).

---

## CHƯƠNG 1: THU THẬP DỮ LIỆU VÀ MÔ TẢ TẬP DỮ LIỆU (CHAPTER 1: DATA COLLECTION AND DATASET DESCRIPTION)

### 1.1. Kiến trúc Thu thập Đa nguồn (`src/01_collect_weather_data.py`)
Script [`src/01_collect_weather_data.py`](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/src/01_collect_weather_data.py) tải khóa xác thực từ file `.env` bảo mật và lần lượt truy vấn cả 4 API cho 20 tọa độ địa lý tại Việt Nam:

| Nguồn Web API | Phương thức kết nối | Định dạng thời gian gốc (`date`) | Số dòng thu được | Đặc điểm kỹ thuật & Schema gốc |
| :--- | :--- | :--- | :-: | :--- |
| **1. Open-Meteo SDK** | `openmeteo_requests` + `requests_cache` + `retry_requests` | `2026-10-17 11:00:00+07:00` | `44,640` | Mảng nhị phân `Float32`; 32 ngày đầu (`15.380` dòng) rỗng `NaN` |
| **2. WeatherAPI.com** | REST JSON (`key=...&days=3`) | `2026-10-17 11:00` / `11:30` | `1,460` | Gồm khối `current` và `forecast.forecastday[].hour[]` |
| **3. Xweather API** | REST JSON (`client_id` & `client_secret`) | `2026-10-17T11:00:00+07:00` | `480` | Khối `response[].periods[]` (`tempC`, `windSpeedKPH`, `pressureMB`...) |
| **4. FreeOpenAPI.dev** | REST JSON (`Header X-Api-Key`) | `2026-10-17T11:30:00` | `20` | Khối `current` khuyết 3 cột `humidity`, `pressure`, `wind_gusts` (`60` ô `NaN`) |
| **TỔNG CỘNG** | **4 Web APIs** | **4 định dạng khác nhau** | **`46,600`** | Lưu trữ đầy đủ trong `raw_weather_data.csv` & `raw_weather_responses.json` |

### 1.2. Danh sách 20 Địa điểm Khảo sát tại Việt Nam
1. `Phu Loc (Hue)` (`16.1667, 107.8333` – Độ cao `1002m`)
2. `Da Lat (Lam Dong)` (`11.4111, 107.8444` – Độ cao `707m`)
3. `Nam Dinh` (`20.4737, 106.0229` – Độ cao `3m`)
4. `Ho Chi Minh City` (`10.8230, 106.6296` – Độ cao `6m`)
5. `Nha Trang` (`12.2451, 109.1943` – Độ cao `10m`)
6. `Da Nang` (`16.0678, 108.2208` – Độ cao `11m`)
7. `Hanoi` (`21.2598, 105.8045` – Độ cao `20m`)
8. `Can Tho` (`10.0371, 105.7883` – Độ cao `9m`)
9. `Phu Quoc` (`10.2872, 104.0105` – Độ cao `180m`)
10. `Ha Long` (`20.9505, 107.0734` – Độ cao `7m`)
11. `Sa Pa` (`22.3402, 103.8441` – Độ cao `1545m`)
12. `Vinh` (`18.6734, 105.6923` – Độ cao `8m`)
13. `Hue City` (`16.4619, 107.5955` – Độ cao `8m`)
14. `Phan Thiet` (`10.9289, 108.1021` – Độ cao `9m`)
15. `Vung Tau` (`10.3460, 107.0843` – Độ cao `13m`)
16. `Ca Mau` (`9.1768, 105.1524` – Độ cao `7m`)
17. `Con Dao` (`8.7029, 106.6489` – Độ cao `92m`)
18. `Pleiku` (`13.9833, 108.0000` – Độ cao `761m`)
19. `Buon Ma Thuot` (`12.6675, 108.0378` – Độ cao `462m`)
20. `Dien Bien Phu` (`21.3860, 103.0230` – Độ cao `482m`)

---

## CHƯƠNG 2: LÀM SẠCH VÀ TIỀN XỬ LÝ DỮ LIỆU (CHAPTER 2: DATA CLEANING AND PREPROCESSING)

Pipeline [`src/02_clean_and_preprocess.py`](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/src/02_clean_and_preprocess.py) thực hiện 6 bước chuẩn hóa và hợp nhất trên 46.600 dòng dữ liệu thô:

1. **Chuẩn hóa đa định dạng thời gian (`Standardize timestamps`):** Chuyển đổi cả 4 định dạng thời gian của 4 API về chuẩn `YYYY-MM-DD HH:MM:SS` theo múi giờ địa phương `UTC+7` (`Asia/Bangkok`), đồng thời quy chuẩn các bản ghi lấy mẫu ở phút lẻ (`11:30` từ `WeatherAPI` và `FreeOpenAPI`) về khung giờ tròn (`dt.floor('h')`).
2. **Phát hiện và xử lý phản hồi API bị khuyết (`Detect missing API responses`):**
   - Loại bỏ **15.380 dòng rỗng hoàn toàn** (`all 6 vars == NaN`) ở đầu chuỗi Open-Meteo (`2026-07-17` đến `2026-08-18`).
   - Đối với **20 dòng khuyết cục bộ (`60` ô `NaN`)** của `FreeOpenAPI.dev` (thiếu `humidity`, `pressure_msl`, `wind_gusts_10m`), hệ thống tự động bù đắp bằng giá trị thực tế tại đúng mốc giờ đó từ 3 API còn lại (`Open-Meteo`, `WeatherAPI`, `Xweather`) trong bước hợp nhất đa nguồn!
3. **Phát hiện trùng lặp và Hợp nhất đa nguồn (`Detect duplicated observations & Multi-API Fusion`):**
   - Phát hiện **1.000 bản ghi trùng khóa `(location_id, time)`** tại **740 mốc giờ giao thoa** (tổng cộng `1.740` dòng giao thoa được xuất riêng ra [`multi_source_comparison.csv`](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/data/cleaned/multi_source_comparison.csv) để phục vụ đối sánh chéo).
   - Hợp nhất các bản ghi cùng `(location_id, timestamp)` bằng trung bình cộng đa cảm biến (ensemble mean) và ghi nhận danh sách các nguồn đóng góp trong cột `source_api` (ví dụ: `Open-Meteo SDK + WeatherAPI.com + Xweather API + FreeOpenAPI.dev`).
4. **Kiểm tra miền giá trị vật lý & tính nhất quán (`Validate measurement ranges`):**
   - Kiểm tra 100% quan trắc nằm trong ngưỡng vật lý hợp lệ.
   - Phát hiện và sửa **94 bản ghi bất nhất vật lý** (`wind_gusts_10m < wind_speed_10m`).
   - Thống kê **1.537 quan trắc thời tiết cực đoan thực tế** theo quy tắc $1.5 \times \text{IQR}$.
5. **Khử nhiễu dấu chấm động `Float32`:** Chuẩn hóa **96.525 ô số thực** về đúng 2 chữ số thập phân.

---

## CHƯƠNG 3: KẾT QUẢ VÀ ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU (CHAPTER 3: RESULTS AND DATA QUALITY EVALUATION)

### 3.1. Bảng Đối sánh Chất lượng Dữ liệu Trước và Sau khi Làm sạch

| Chỉ số đánh giá chất lượng (Data Quality Indicator) | Trước khi làm sạch (`Raw 4-API Data`) | Sau khi làm sạch (`Cleaned Data`) | Đánh giá cải thiện |
| :--- | :-: | :-: | :--- |
| **Tổng số bản ghi (Total Records)** | 46.600 | **30.220** | Chuẩn hóa đúng $20 \text{ địa điểm} \times 1.511 \text{ giờ}$ |
| **Số nguồn Web API tích hợp** | 4 API rời rạc | **4 API hợp nhất** | `740` mốc giờ được tổng hợp từ $\ge 2$ API |
| **Bản ghi chưa đồng nhất định dạng `time`** | 46.600 | **0** | 100% đạt chuẩn `YYYY-MM-DD HH:MM:SS` |
| **Dòng API rỗng hoàn toàn (`>61d Horizon`)** | 15.380 | **0** | Đã lọc bỏ vùng không có dữ liệu lưu trữ |
| **Dòng khuyết cục bộ do khác Schema (`FreeOpenAPI`)** | 20 (60 ô `NaN`) | **0** | Được bù đắp hoàn toàn khi hợp nhất với 3 API kia |
| **Tổng số ô bị khuyết (`NaN` cells)** | 92.340 | **0** | Đạt độ đầy đủ 100% (Completeness = 100%) |
| **Bản ghi trùng khóa `(location_id, time)`** | 1.000 | **0** | Hợp nhất đa nguồn (Multi-sensor fusion) |
| **Ô bị nhiễu thập phân `Float32` (>2 chữ số lẻ)** | 96.525 | **0** | Làm tròn chuẩn xác về 2 chữ số thập phân |
| **Bản ghi bất nhất (`wind_gusts < wind_speed`)** | 94 | **0** | Đảm bảo tính nhất quán vật lý 100% |

### 3.2. Hệ thống 6 Biểu đồ Trực quan hóa (`figures/`)
1. **[fig1_data_quality_before_after.png](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/figures/fig1_data_quality_before_after.png):** Đối sánh các lỗi chất lượng dữ liệu Trước/Sau khi làm sạch trên 46.600 dòng từ 4 API và thống kê các giờ thời tiết cực đoan thực tế.
2. **[fig2_range_validation_boxplots.png](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/figures/fig2_range_validation_boxplots.png):** Boxplot phân phối của cả 6 biến khí tượng trên 30.220 quan trắc sạch.
3. **[fig3_temperature_timeseries.png](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/figures/fig3_temperature_timeseries.png):** Chuỗi thời gian nhiệt độ trung bình ngày của 8 địa điểm đại diện từ tháng 8 đến tháng 10/2026.
4. **[fig4_all_20_cities_temperature_range.png](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/figures/fig4_all_20_cities_temperature_range.png):** So sánh nhiệt độ trung bình và dải Min–Max của toàn bộ 20 địa điểm (`1.511` giờ/địa điểm).
5. **[fig5_multivariate_and_precipitation.png](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/figures/fig5_multivariate_and_precipitation.png):** Tương quan đa biến Nhiệt độ – Độ ẩm – Áp suất và tổng lượng mưa tích lũy của 20 địa điểm.
6. **[fig6_multi_api_source_comparison.png](file:///c:/Users/Windows/Downloads/%C4%90%E1%BB%93%20%C3%81n%20Clean%20Data/figures/fig6_multi_api_source_comparison.png):** **Biểu đồ đối sánh chéo 4 nguồn Web API** (`Open-Meteo SDK`, `WeatherAPI.com`, `Xweather API`, `FreeOpenAPI.dev`) tại các mốc thời gian giao thoa.

---

## KẾT LUẬN VÀ TÀI LIỆU THAM KHẢO (CONCLUSION & REFERENCES)
- **Kết luận:** Đồ án đã kết hợp thành công **4 nguồn Web API thời tiết thực tế** trên **20 địa điểm tại Việt Nam**, thu thập **46.600 bản ghi thô** và xây dựng pipeline làm sạch đa nguồn chuẩn mực để tạo ra bộ dữ liệu chuỗi thời gian **30.220 bản ghi** hoàn chỉnh.
- **Tài liệu tham khảo:**
  1. Open-Meteo API & Python SDK: `https://open-meteo.com/en/docs`
  2. WeatherAPI.com Documentation: `https://www.weatherapi.com/docs/`
  3. Vaisala Xweather API Documentation: `https://www.xweather.com/docs/weather-api`
  4. FreeOpenAPI.dev Weather API: `https://www.freeopenapi.dev/`
