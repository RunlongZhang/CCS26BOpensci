# Configure and build all four programs with CMake + vcpkg (manifest mode).
# Run from a "Developer PowerShell for VS" so that the MSVC compiler is on PATH.
#
#   scripts\build.ps1 -VcpkgRoot C:\path\to\vcpkg
#
# If -VcpkgRoot is omitted, the VCPKG_ROOT environment variable is used.
# Dependencies (OpenSSL, HiGHS, Eigen) are installed automatically by vcpkg
# on first configure; no paths need to be edited anywhere.
param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$BuildDir  = "out/build/release"
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

if (-not $VcpkgRoot) {
    throw "vcpkg not specified. Pass -VcpkgRoot <path> or set the VCPKG_ROOT environment variable. (To install vcpkg: git clone https://github.com/microsoft/vcpkg ; .\vcpkg\bootstrap-vcpkg.bat)"
}
$toolchain = Join-Path $VcpkgRoot "scripts/buildsystems/vcpkg.cmake"
if (-not (Test-Path $toolchain)) {
    throw "vcpkg toolchain file not found at: $toolchain"
}

$dest = Join-Path $repo $BuildDir

# Prefer Ninja (single-config, executables land directly in the build dir).
# Fall back to the default Visual Studio generator when no compiler is on
# PATH; that produces executables under <build>/Release instead.
$haveCl = $null -ne (Get-Command cl.exe -ErrorAction SilentlyContinue)
if ($haveCl) {
    cmake -S $repo -B $dest -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_TOOLCHAIN_FILE=$toolchain"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
    cmake --build $dest
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
} else {
    Write-Host "cl.exe not on PATH; using the Visual Studio generator (multi-config)."
    cmake -S $repo -B $dest "-DCMAKE_TOOLCHAIN_FILE=$toolchain"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
    cmake --build $dest --config Release
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
}

Write-Host "`nBuild complete. Executables are in:"
if (Test-Path (Join-Path $dest "AofBench.exe")) { Write-Host "  $dest" }
elseif (Test-Path (Join-Path $dest "Release/AofBench.exe")) { Write-Host "  $dest\Release" }
