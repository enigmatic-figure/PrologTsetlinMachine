param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 5,
    [ValidateRange(1, 1000000)]
    [int]$Repeats = 25,
    [ValidateRange(2, 1000000)]
    [int]$Clauses = 100,
    [string]$TrainingData,
    [string]$TestData
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $TrainingData) {
    $TrainingData = Join-Path $projectRoot `
        "data\NoisyXOR\NoisyXORTrainingData.txt"
}
if (-not $TestData) {
    $TestData = Join-Path $projectRoot `
        "data\NoisyXOR\NoisyXORTestData.txt"
}
if (-not (Test-Path -LiteralPath $TrainingData)) {
    throw "Training dataset was not found: $TrainingData"
}
if (-not (Test-Path -LiteralPath $TestData)) {
    throw "Test dataset was not found: $TestData"
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
$build = Join-Path $projectRoot "out\logic-benchmark"
$benchmark = Join-Path $build "ptm_logic_benchmark.exe"

$commands = @(
    "call `"$vcvars`" >nul",
    ("`"$cmake`" -S `"$projectRoot`" -B `"$build`" -G Ninja " +
        "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release " +
        "-DPTM_BUILD_TESTS=OFF -DPTM_BUILD_BENCHMARKS=ON"),
    "`"$cmake`" --build `"$build`"",
    ("`"$benchmark`" `"$TrainingData`" `"$TestData`" " +
        "$Epochs $Repeats $Clauses")
) -join " && "

cmd.exe /d /c $commands
exit $LASTEXITCODE
