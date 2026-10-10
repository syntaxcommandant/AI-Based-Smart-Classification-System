@echo off
title EcoSort AI - Smart Waste Classification System
color 0A
cls
echo ======================================================================
echo          ECOSORT AI - SMART WASTE CLASSIFICATION SYSTEM
echo                Review 2 Live Evaluation Demo
echo ======================================================================
echo.
echo [1/3] Activating Python Virtual Environment...
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in %~dp0venv!
    echo Please make sure the venv folder exists.
    pause
    exit /b 1
)

echo [2/3] Opening Browser at http://127.0.0.1:5000 ...
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:5000"

echo [3/3] Starting Flask Application Server...
echo.
echo ======================================================================
echo   * Web App URL:      http://127.0.0.1:5000
echo   * Review Report:    http://127.0.0.1:5000/progress_report.html (open file)
echo   * Sample Images:    Folder 'sample_demo_images' in project
echo.
echo   [!] IMPORTANT: Keep this window OPEN during your presentation!
echo   [!] To close the app later, just close this window or press Ctrl+C.
echo ======================================================================
echo.

.\venv\Scripts\python.exe app.py

echo.
echo Server stopped.
pause
