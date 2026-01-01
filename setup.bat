@echo off
echo ================================
echo Smart Traffic System - Setup
echo ================================

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b
)

REM --- Create venv if not exists ---
if not exist .venv (
    echo ▶ Creating virtual environment...
    python -m venv .venv
) else (
    echo ✔ Virtual environment already exists.
)

REM --- Activate venv ---
echo ▶ Activating virtual environment...
call .venv\Scripts\activate.bat

REM --- Install dependencies ---
if not exist requirements.txt (
    echo ❌ requirements.txt not found!
    pause
    exit /b
)

echo ▶ Installing dependencies...
pip install -r requirements.txt

echo.
echo ================================
echo ✅ Setup complete!
echo ================================
echo To run the app:
echo.
echo   1. Activate venv:
echo      .venv\Scripts\Activate.ps1
echo.
echo   2. Run:
echo      python tkinter_ui.py
echo.
pause
