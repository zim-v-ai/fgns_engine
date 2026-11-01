$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Green
Write-Host "   FGNS v0.6.5: SOTA COMBAT TEST-TIME TRAINING SOLVER   " -ForegroundColor Green
Write-Host "   Reasoning Swarm : 4-Step Cellular TRM-ACCRO Flow         " -ForegroundColor Green
Write-Host "   Information Cap : Strict Singular LoRA Bottleneck (R=2)  " -ForegroundColor Green
Write-Host "   Dynamic Bank    : Recirculation Active (333s - 480s)     " -ForegroundColor Green
Write-Host "   Verification    : Augmented Invariance LOOCV under D4xS10" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

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

$TargetChallenges = "arc-agi_test_challenges.json"

$Discovered = python -c "from fgns.solver.submit.path_resolver import DatasetPathResolver; print(DatasetPathResolver.resolve_path('$TargetChallenges'))"

if (-not (Test-Path $Discovered)) {
    $TargetChallenges = "arc-agi_training_challenges.json"
    Write-Host "[INFO] Official competition test file not found." -ForegroundColor Yellow
    Write-Host "[INFO] Commencing Combat Benchmark on: $TargetChallenges" -ForegroundColor Yellow
} else {
    Write-Host "[COMBAT] Resolved Target Challenge Manifest: $Discovered" -ForegroundColor Green
    $TargetChallenges = $Discovered
}

python -m fgns.solver.submit.eval_runner `
    --data $TargetChallenges `
    --time 333 `
    --log artifacts\logs\solver.log