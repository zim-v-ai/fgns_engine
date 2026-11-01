$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   FGNS v0.5.1: WINDOWS TRITON & RTX 5070 PROVISIONER   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$VenvPath = ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[1/5] Creating Python Virtual Environment (.venv)..." -ForegroundColor Yellow
    python -m venv $VenvPath
} else {
    Write-Host "[1/5] Existing Virtual Environment detected (.venv)." -ForegroundColor Green
}

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    Write-Host "[2/5] Activating Virtual Environment..." -ForegroundColor Yellow
    & $ActivateScript
} else {
    Write-Error "Activation script not found at $ActivateScript"
}

Write-Host "[3/5] Upgrading Pip and Core Tooling..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

Write-Host "[4/5] Installing PyTorch with CUDA 12.6 Compute Wheels..." -ForegroundColor Yellow
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "[5/5] Installing Windows Triton and FGNS in Editable Mode..." -ForegroundColor Yellow
python -m pip install triton-windows
python -m pip install -e .

$VSPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
if (Test-Path $VSPath) {
    $MSVCVersion = Get-ChildItem $VSPath | Select-Object -First 1
    if ($MSVCVersion) {
        $CLPath = Join-Path $MSVCVersion.FullName "bin\Hostx64\x64"
        if (Test-Path $CLPath) {
            Write-Host "Discovered Visual Studio MSVC cl.exe at: $CLPath" -ForegroundColor Green
            $env:PATH = "$CLPath;" + $env:PATH
        }
    }
}

New-Item -ItemType Directory -Force -Path "target\triton_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

Write-Host "============================================================" -ForegroundColor Green
Write-Host "   FGNS v0.5.1 ENVIRONMENT SUCCESSFULLY PROVISIONED!    " -ForegroundColor Green
Write-Host "   Hardware Target: NVIDIA GeForce RTX 5070 (Blackwell)     " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green