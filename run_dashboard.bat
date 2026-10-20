@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================================
echo   HE THONG DU BAO THOI TIET THOI GIAN THUC - STREAMLIT DASHBOARD
echo ============================================================================

:: 1. Tu dong nhan dien trinh thong dich Python (.venv -> python -> py)
set "PY_CMD="

if exist ".\.venv\Scripts\python.exe" (
    ".\.venv\Scripts\python.exe" --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=.\.venv\Scripts\python.exe"
    )
)

if "%PY_CMD%"=="" (
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    py --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Khong tim thay Python tren may! Vui long cai dat Python 3.10+ truoc khi chay.
    pause
    exit /b 1
)

echo [INFO] Dang su dung Python: %PY_CMD%

:: 2. Kiem tra va tu dong cai dat thu vien neu thieu tren may thanh vien
"%PY_CMD%" -c "import streamlit, xgboost, torch, openmeteo_requests, sklearn, scipy, pyarrow" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Dang cai dat cac thu vien can thiet tu requirements.txt...
    "%PY_CMD%" -m pip install -r requirements.txt
)

:: 3. Neu chua co file trong so hoac nguoi dung truyen tham so --pipeline thi chay Pipeline
if "%~1"=="--pipeline" goto :run_pipeline
if not exist "models\weather_forecast_weights.joblib" goto :run_pipeline
goto :launch_app

:run_pipeline
echo.
echo [BUOC 1/6] Thu thap du lieu tu 4 nha cung cap API (ghi log tung request)...
"%PY_CMD%" src\01_collect_weather_data.py
echo [BUOC 2/6] Lam sach, chuan hoa va hop nhat du lieu da nguon...
"%PY_CMD%" src\02_clean_and_preprocess.py
echo [BUOC 3/6] Thong ke mo ta va ve bieu do chat luong du lieu (fig1 -^> fig7)...
"%PY_CMD%" src\03_evaluate_and_visualize.py
echo [BUOC 4/6] (Tuy chon) Thu thap ERA5 nhieu nam: chay rieng src\04_collect_multiyear_archive.py neu can.
echo [BUOC 5/6] Danh gia walk-forward theo lead time (fig8, fig9)...
"%PY_CMD%" src\05_walk_forward_evaluation.py
echo [BUOC 6/6] Huan luyen mo hinh production va luu trong so...
"%PY_CMD%" src\06_train_production_model.py
:launch_app
echo.
echo [INFO] Dang khoi dong giao dien Web Dashboard Streamlit tai http://localhost:8501 ...
"%PY_CMD%" -m streamlit run app.py
pause
