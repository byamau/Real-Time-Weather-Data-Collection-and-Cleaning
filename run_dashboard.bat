@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================================
echo   KHOI DONG HE THONG REAL-TIME WEATHER DATA COLLECTION ^& WEIGHT INFERENCE
echo ============================================================================
".\.venv\Scripts\python.exe" -m streamlit run app.py
pause
