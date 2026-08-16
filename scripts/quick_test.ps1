# Smoke test (~2-3 minutes): runs every program once on small inputs, checks
# their self-verification gates, and produces sample figures.  Use this to
# validate the build before launching the full reproduction.
#
#   scripts\quick_test.ps1              (after scripts\build.ps1)
param(
    [string]$BuildDir = "out/build/release"
)
$ErrorActionPreference = 'Stop'
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

function Gen($n, $d) {
    # Same seeding rule as the full reproduction (and the paper's datasets).
    $seed = 1000 + 10 * $n + $d
    & .\FunctionGenerator.exe $n $d $seed | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "FunctionGenerator failed at n=$n d=$d" }
}

$failures = 0
function Run($name, $block) {
    Write-Host "== $name"
    & $block
    if ($LASTEXITCODE -ne 0) { Write-Host "   FAILED (exit $LASTEXITCODE)"; $script:failures++ }
    else { Write-Host "   ok" }
}

foreach ($cfg in @(@(10,2),@(30,2),@(10,3),@(10,4))) {
    $n = $cfg[0]; $d = $cfg[1]
    Gen $n $d
    Run "AofBench n=$n d=$d (construction + ADS + 200 authenticated queries + tamper tests)" {
        & .\AofBench.exe $n $d 200 42 *> "runs\aof_${n}_${d}.txt"
    }
}
foreach ($cfg in @(@(10,2),@(10,3))) {
    $n = $cfg[0]; $d = $cfg[1]
    Gen $n $d
    Run "CertBench n=$n d=$d (certificate generation + owner verification + falsification tests)" {
        & .\CertBench.exe $n $d *> "runs\cert_${n}_${d}.txt"
    }
}
foreach ($cfg in @(@(10,2),@(10,3))) {
    $n = $cfg[0]; $d = $cfg[1]
    Gen $n $d
    Run "ITreeSimplex baseline n=$n d=$d" {
        & .\ITreeSimplex.exe $n $d 1 *> "runs\itree_${n}_${d}.txt"
    }
}

python (Join-Path $repo "scripts\parse_itree.py") $exeDir
python (Join-Path $repo "scripts\parse_cert_deviations.py") $exeDir
python (Join-Path $repo "scripts\plot_results.py") $exeDir (Join-Path $repo "figs")

if ($failures -eq 0) {
    Write-Host "`nQUICK TEST PASSED: all programs ran, all self-verification gates held."
    Write-Host "Sample figures written to figs\ (small-n shapes only; run scripts\reproduce_all.ps1 for the paper's grids)."
} else {
    Write-Host "`nQUICK TEST FAILED: $failures program run(s) reported failure; see runs\*.txt"
    exit 1
}
