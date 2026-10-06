@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
if errorlevel 1 (
    echo.
    echo The app could not start. The error above explains what went wrong.
    echo For missing packages, run: python -m pip install -r requirements.txt
    pause
)
