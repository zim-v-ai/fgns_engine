$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   FGNS v0.6.5: SOTA OFFLINE META-LEARNING CORTEX       " -ForegroundColor Cyan
Write-Host "   Architecture    : Dual-State Cauchy Flow (TRM-ACCRO N=4)  " -ForegroundColor Cyan
Write-Host "   Memory Pipeline : Zero-RAM Sharded Streaming (<250MB)    " -ForegroundColor Cyan
Write-Host "   Hardware Target : NVIDIA GeForce RTX 5070 (Blackwell)    " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.8"
$env:CUDA_DEVICE_MAX_CONNECTIONS = "1"
$env:TORCH_LOGS = "-all"

$TargetCache = "target\triton_cache"
if (-not (Test-Path $TargetCache)) {
    New-Item -ItemType Directory -Force -Path $TargetCache | Out-Null
}
$env:TRITON_CACHE_DIR = (Resolve-Path $TargetCache).Path

$VenvActivate = ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
    Write-Host "[OK] Virtual Environment Activated: .venv" -ForegroundColor Green
} else {
    Write-Host "[WARN] .venv activation script not found. Using system python." -ForegroundColor Yellow
}

$VS_Search_Roots = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
)

$CL_Path_Discovered = $null
foreach ($Root in $VS_Search_Roots) {
    if (Test-Path $Root) {
        $LatestVersion = Get-ChildItem -Path $Root -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($LatestVersion) {
            $HostX64 = Join-Path $LatestVersion.FullName "bin\Hostx64\x64"
            if (Test-Path (Join-Path $HostX64 "cl.exe")) {
                $CL_Path_Discovered = $HostX64
                break
            }
        }
    }
}

if ($CL_Path_Discovered) {
    Write-Host "[OK] Discovered Visual Studio MSVC cl.exe compiler at:" -ForegroundColor Green
    Write-Host "     $CL_Path_Discovered" -ForegroundColor Gray
    $env:PATH = "$CL_Path_Discovered;" + $env:PATH
} else {
    Write-Host "[WARN] Visual Studio cl.exe not located in standard paths. Triton JIT may use fallback." -ForegroundColor Yellow
}

$TrainPath = "data\arc-agi_training_challenges.json"
if (Test-Path "data\synthetic\nvarc_full") {
    $TrainPath = "data\synthetic\nvarc_full"
    Write-Host "[DATA FABRIC] Streaming Corpus Discovered: NVARC Full (103k+ Puzzles)" -ForegroundColor Green
} elseif (Test-Path "data\synthetic\nvarc_training") {
    $TrainPath = "data\synthetic\nvarc_training"
    Write-Host "[DATA FABRIC] Streaming Corpus Discovered: NVARC Training (55k+ Puzzles)" -ForegroundColor Green
} else {
    Write-Host "[DATA FABRIC] Using Benchmark Challenges: $TrainPath" -ForegroundColor Yellow
}

$ValPath = "data\arc-agi_training_challenges.json"
if (-not (Test-Path $ValPath)) {
    $ValPath = "data\arc-agi_evaluation_challenges.json"
}
Write-Host "[BENCHMARK] Frozen Gold Benchmark Target: $ValPath" -ForegroundColor Green

python -m fgns.train.meta_train.pipeline `
    --data $TrainPath `
    --val-data $ValPath `
    --val-size 400 `
    --val-interval 2500 `
    --mode full `
    --epochs 25 `
    --lr 0.0003