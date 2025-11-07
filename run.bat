@echo off
REM Batch script to run the Fingerprint Recognition System

echo ==================================
echo Fingerprint Recognition System
echo ==================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

echo Checking required packages...
python -c "import numpy, cv2, PIL, sklearn, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
    echo Packages installed successfully!
) else (
    echo All required packages are installed!
)

echo.
echo Starting Fingerprint Recognition System...
echo.

python src/main.py

if errorlevel 1 (
    echo.
    echo ERROR: Application encountered an error
    pause
    exit /b 1
)
