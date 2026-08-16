# Full reproduction of the paper's experiments (~2-3 hours on a desktop-class
# machine; the reference machine was an i9-14900K with 64 GB RAM).
#
#   scripts\reproduce_all.ps1           (after scripts\build.ps1)
#
# Every dataset is generated with a fixed, recorded seed (1000 + 10n + d) and
# every query experiment uses query seed 42 with 200 queries, matching the
# paper.  Outputs:
#   <build>\aof_bench_results.csv   construction/ADS/query measurements
#   <build>\cert_bench_results.csv  verifiable-construction measurements
#   <build>\itree_results.csv       I-tree baseline (parsed from its logs)
#   <build>\cert_deviations.csv     constructor deviations detected by the
#                                   certificate replay (parsed from its logs)
#   figs\*.png + figs\summary.txt   the paper's result figures
param(
    [string]$BuildDir = "out/build/release"
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot

$exeDir = Join-Path $repo $BuildDir
if (-not (Test-Path (Join-Path $exeDir "AofBench.exe"))) {
    $exeDir = Join-Path $exeDir "Release"
}
if (-not (Test-Path (Join-Path $exeDir "AofBench.exe"))) {
    throw "Executables not found. Run scripts\build.ps1 first."
}
Set-Location $exeDir
New-Item -ItemType Directory -Force runs, results | Out-Null
Remove-Item aof_bench_results.csv, cert_bench_results.csv -ErrorAction SilentlyContinue

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    $line | Out-File -Append -Encoding utf8 sweep.log
    Write-Output $line
}
function Gen($n, $d) {
    $seed = 1000 + 10 * $n + $d
    if (-not (Test-Path "${n}_functions_${d}d.bin")) {
        & .\FunctionGenerator.exe $n $d $seed | Out-Null
    }
}

Log "=== FULL REPRODUCTION START ==="

# --- AOF-tree: construction, ADS, authenticated top-k/range queries -------
$aofGrid = @(
    @(10,2),@(50,2),@(100,2),@(150,2),@(200,2),@(250,2),@(300,2),@(350,2),@(400,2),@(450,2),@(500,2),
    @(10,3),@(25,3),@(50,3),@(75,3),@(100,3),@(125,3),@(150,3),@(200,3),@(250,3),
    @(10,4),@(20,4),@(30,4),@(40,4),@(50,4),@(60,4)
)
foreach ($cfg in $aofGrid) {
    $n = $cfg[0]; $d = $cfg[1]
    Gen $n $d
    Log "AofBench n=$n d=$d"
    & .\AofBench.exe $n $d 200 42 *> "runs\aof_${n}_${d}.txt"
    Log "AofBench n=$n d=$d exit=$LASTEXITCODE"
}

# --- Verifiable construction: certificates -------------------------------
$certGrid = @(
    @(10,2),@(50,2),@(100,2),@(150,2),@(200,2),@(250,2),
    @(10,3),@(25,3),@(50,3),@(75,3),@(100,3),
    @(10,4),@(20,4),@(30,4)
)
foreach ($cfg in $certGrid) {
    $n = $cfg[0]; $d = $cfg[1]
    Gen $n $d
    Log "CertBench n=$n d=$d"
    & .\CertBench.exe $n $d *> "runs\cert_${n}_${d}.txt"
    Log "CertBench n=$n d=$d exit=$LASTEXITCODE"
}

# --- I-tree (Simplex) baseline: the scales it can reach ------------------
$itreeGrid = @(
    @(10,2),@(20,2),@(30,2),@(40,2),@(50,2),
    @(10,3),@(15,3),@(20,3),@(25,3),
    @(10,4),@(12,4),@(14,4),@(16,4)
)
foreach ($cfg in $itreeGrid) {
    $n = $cfg[0]; $d = $cfg[1]
    Gen $n $d
    Log "ITreeSimplex n=$n d=$d"
    & .\ITreeSimplex.exe $n $d 1 *> "runs\itree_${n}_${d}.txt"
    Log "ITreeSimplex n=$n d=$d exit=$LASTEXITCODE"
}

# --- Parse the logs and draw every figure ---------------------------------
Log "parse + plot"
python (Join-Path $repo "scripts\parse_itree.py") $exeDir
python (Join-Path $repo "scripts\parse_cert_deviations.py") $exeDir
python (Join-Path $repo "scripts\plot_results.py") $exeDir (Join-Path $repo "figs")

Log "=== FULL REPRODUCTION DONE ==="
Write-Host "`nFigures and summary.txt are in figs\ ; raw CSVs are in $exeDir"
