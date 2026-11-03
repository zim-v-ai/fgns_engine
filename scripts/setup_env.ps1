$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   FGNS v4.2.0: WINDOWS CUDA 12.8 & RTX 5070 PROVISIONER    " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[1/5] Creating Python Virtual Environment (.venv)..." -ForegroundColor Yellow
    python -m venv $VenvPath
} else {
    Write-Host "[1/5] Existing Virtual Environment detected (.venv)." -ForegroundColor Green
}

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (Test-Path $ActivateScript) {
    Write-Host "[2/5] Activating Virtual Environment ($ActivateScript)..." -ForegroundColor Yellow
    & $ActivateScript
} else {
    Write-Error "Activation script not found at $ActivateScript"
}

Write-Host "[3/5] Upgrading Pip and Core Tooling..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[4/5] Installing PyTorch with CUDA 12.8 (cu128) for RTX 5070 Blackwell (sm_120)..." -ForegroundColor Yellow
& $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

Write-Host "[5/5] Installing Windows Triton and FGNS in Editable Mode (preserving cu128 torch)..." -ForegroundColor Yellow
& $VenvPython -m pip install triton-windows
& $VenvPython -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128

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

if (Test-Path "target\cuda_cache") {
    Remove-Item -Recurse -Force "target\cuda_cache" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force -Path "target\triton_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "target\cuda_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

Write-Host "============================================================" -ForegroundColor Green
Write-Host "   FGNS v4.2.0 CUDA 12.8 ENVIRONMENT SUCCESSFULLY READY!    " -ForegroundColor Green
Write-Host "   Hardware Target: NVIDIA GeForce RTX 5070 (Blackwell)     " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green