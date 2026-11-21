@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [1/3] Virtual environment .venv already exists.
)

echo [2/3] Updating pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/3] Installing dependencies ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Done. Run "run.bat" to start the application.
exit /b 0

:error
echo.
echo Installation failed. See the messages above.
exit /b 1
