param(
    [ValidateRange(1, 10000)]
    [int]$Repeats = 25
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $projectRoot "data\Logic"
$outputRoot = Join-Path $projectRoot "out\logic-consolidation"
$env:PYTHONPATH = Join-Path $projectRoot "python"
$env:PYTHONDONTWRITEBYTECODE = "1"

$symbolic = Join-Path $dataRoot "logical_problems_symbolic.csv"
if (-not (Test-Path -LiteralPath $symbolic)) {
    throw "Symbolic Logic dataset was not found: $symbolic"
}

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Build Tools discovery utility was not found."
}
$visualStudio = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) {
    throw "A Visual Studio installation with C++ tools was not found."
}

$vcvars = Join-Path $visualStudio "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $visualStudio `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = Join-Path $visualStudio `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$build = Join-Path $projectRoot "out\logic-consolidation-build"
$nativeLibrary = Join-Path $build "ptm.dll"

$commands = @(
    "call `"$vcvars`" >nul",
    ("`"$cmake`" -S `"$projectRoot`" -B `"$build`" -G Ninja " +
        "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release " +
        "-DPTM_BUILD_TESTS=OFF -DPTM_BUILD_BENCHMARKS=OFF"),
    "`"$cmake`" --build `"$build`""
) -join " && "
cmd.exe /d /c $commands
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$env:PTM_NATIVE_LIBRARY = $nativeLibrary
python (Join-Path $projectRoot "examples\logic_class_ii_consolidate.py") `
    --data-dir $dataRoot `
    --output-dir $outputRoot `
    --repeats $Repeats
exit $LASTEXITCODE
