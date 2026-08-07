param(
    [string[]]$Clauses = "20",
    [string[]]$Features = "256",
    [string[]]$Densities = "0.02",
    [string[]]$ResidentPages = "1",
    [int]$Repeats = 5000,
    [int]$Warmup = 100,
    [int]$Samples = 5,
    [string[]]$Backend = "all",
    [switch]$Sweep,
    [switch]$GpuSweep,
    [switch]$JsonLines
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
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
$build = Join-Path $projectRoot "out\packed-tm-benchmark-build"
$cache = Join-Path $build "CMakeCache.txt"
if (Test-Path -LiteralPath $cache) {
    $cachedHomeLine = Get-Content -LiteralPath $cache |
        Where-Object { $_ -like "CMAKE_HOME_DIRECTORY:INTERNAL=*" } |
        Select-Object -First 1
    $cachedHome = if ($cachedHomeLine) {
        ($cachedHomeLine -split "=", 2)[1].Replace('/', '\')
    }
    if ($cachedHome -and -not $cachedHome.Equals(
            $projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha.ComputeHash(
                [Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant()))
            $pathTag = -join ($digest[0..5] | ForEach-Object {
                $_.ToString("x2")
            })
        } finally {
            $sha.Dispose()
        }
        $build = Join-Path $projectRoot "out\packed-tm-benchmark-$pathTag"
    }
}
$benchmark = Join-Path $build "ptm_packed_tm_benchmark.exe"

$commands = @(
    "call `"$vcvars`" >nul",
    ("`"$cmake`" -S `"$projectRoot`" -B `"$build`" -G Ninja " +
        "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release " +
        "-DPTM_BUILD_TESTS=OFF -DPTM_BUILD_EXAMPLES=OFF " +
        "-DPTM_BUILD_BENCHMARKS=ON -DPTM_ENABLE_CUDA=OFF"),
    "`"$cmake`" --build `"$build`" --target ptm_packed_tm_benchmark"
) -join " && "
$buildOutput = cmd.exe /d /c $commands 2>&1
$buildStatus = $LASTEXITCODE
if ($JsonLines) {
    foreach ($line in $buildOutput) {
        [Console]::Error.WriteLine($line)
    }
} else {
    $buildOutput
}
if ($buildStatus -ne 0) {
    exit $buildStatus
}

$benchmarkArguments = @(
    "--repeats", $Repeats,
    "--warmup", $Warmup,
    "--samples", $Samples,
    "--backend", ($Backend -join ",")
)
if ($GpuSweep) {
    $benchmarkArguments += "--gpu-sweep"
} elseif ($Sweep) {
    $benchmarkArguments += "--sweep"
} else {
    $benchmarkArguments += @(
        "--clauses", ($Clauses -join ","),
        "--features", ($Features -join ","),
        "--densities", ($Densities -join ","),
        "--resident-pages", ($ResidentPages -join ",")
    )
}
if ($JsonLines) {
    $benchmarkArguments += "--jsonl"
}
& $benchmark @benchmarkArguments
exit $LASTEXITCODE
