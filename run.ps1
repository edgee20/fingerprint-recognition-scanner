# PowerShell script to run the Fingerprint Recognition System

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Fingerprint Recognition System" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCmd) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://www.python.org/" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "Python found: $($pythonCmd.Source)" -ForegroundColor Green
Write-Host ""

# Check if required packages are installed
Write-Host "Checking required packages..." -ForegroundColor Yellow
$packages = @("numpy", "opencv-python", "Pillow", "scikit-learn", "matplotlib")
$missingPackages = @()

foreach ($package in $packages) {
    $installed = python -c "import $($package.Replace('-', '_').Split('-')[0])" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "Missing packages detected: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "Packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "All required packages are installed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Fingerprint Recognition System..." -ForegroundColor Cyan
Write-Host ""

# Run the main application
python src/main.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Application encountered an error" -ForegroundColor Red
    pause
    exit 1
}
