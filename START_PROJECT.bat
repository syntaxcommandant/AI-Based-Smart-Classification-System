@echo off
title EcoSort AI - Smart Waste Classification System
color 0A
cls
echo ======================================================================
echo          ECOSORT AI - SMART WASTE CLASSIFICATION SYSTEM
echo                Review 2 Live Evaluation Demo
echo ======================================================================
echo.
cd /d "%~dp0"

echo [1/2] Opening Web Browser at http://127.0.0.1:5000 ...
start http://127.0.0.1:5000

echo [2/2] Starting EcoSort AI Flask Server...
echo.
echo ======================================================================
echo   * Web App:       http://127.0.0.1:5000
echo   * Sample Images: Folder 'sample_demo_images'
echo.
echo   [!] Keep this window OPEN while presenting!
echo   [!] To stop, press Ctrl+C or close this window.
echo ======================================================================
echo.

.\venv\Scripts\python.exe app.py
pause
