param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 5,
    [ValidateRange(1, 1000000)]
    [int]$Repeats = 5,
    [ValidateRange(2, 1000000)]
    [int]$Clauses = 100,
    [ValidateSet(
        "token_presence",
        "token_count_threshold",
        "position_one_hot",
        "ast_relational"
    )]
    [string[]]$Encodings = @(
        "token_presence",
        "token_count_threshold",
        "position_one_hot",
        "ast_relational"
    )
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $projectRoot "data\Logic"
$outputRoot = Join-Path $projectRoot "out\logic-dataset"
$env:PYTHONPATH = Join-Path $projectRoot "python"
$env:PYTHONDONTWRITEBYTECODE = "1"

$natural = Join-Path $dataRoot "logical_problems_dataset.csv"
$symbolic = Join-Path $dataRoot "logical_problems_symbolic.csv"
if (-not (Test-Path -LiteralPath $natural)) {
    throw "Natural Logic dataset was not found: $natural"
}
if (-not (Test-Path -LiteralPath $symbolic)) {
    throw "Symbolic Logic dataset was not found: $symbolic"
}

$prepareArguments = @(
    (Join-Path $projectRoot "examples\logic_dataset_prepare.py"),
    "--data-dir", $dataRoot,
    "--output-dir", $outputRoot,
    "--encodings"
) + $Encodings
python @prepareArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
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
    "`"$cmake`" --build `"$build`""
) -join " && "
cmd.exe /d /c $commands
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

foreach ($encoding in $Encodings) {
    Write-Output ""
    Write-Output "=== $encoding ==="
    $training = Join-Path $outputRoot "${encoding}_train.txt"
    $evaluation = Join-Path $outputRoot "${encoding}_evaluation.txt"
    & $benchmark $training $evaluation $Epochs $Repeats $Clauses
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
