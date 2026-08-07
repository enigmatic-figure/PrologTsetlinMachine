$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "python"
$env:PYTHONDONTWRITEBYTECODE = "1"

$pythonArguments = @()
$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pythonLauncher) {
    $python = $pythonLauncher.Source
    $pythonArguments = @("-3")
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Install Python 3 or the Windows py launcher."
    }
    $python = $pythonCommand.Source
}

$gprolog = Get-Command gprolog -ErrorAction SilentlyContinue
if ($gprolog) {
    $env:PTM_GPROLOG = $gprolog.Source
} elseif (Test-Path -LiteralPath "C:\GNU-Prolog\bin\gprolog.exe") {
    $env:PTM_GPROLOG = "C:\GNU-Prolog\bin\gprolog.exe"
} else {
    throw "GNU Prolog was not found. Set PTM_GPROLOG before running verification."
}
$gprologBin = Split-Path -Parent $env:PTM_GPROLOG
$gplc = Join-Path $gprologBin "gplc.exe"
$env:PATH = $gprologBin + [IO.Path]::PathSeparator + $env:PATH

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
$ctest = Join-Path $visualStudio `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
$build = Join-Path $projectRoot "build"
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
        $build = Join-Path $projectRoot "out\windows-verify-$pathTag"
    }
}

$nativeBuild = @(
    "call `"$vcvars`" >nul",
    ("`"$cmake`" -S `"$projectRoot`" -B `"$build`" -G Ninja " +
        "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DPTM_BUILD_TESTS=ON"),
    "`"$cmake`" --build `"$build`"",
    "`"$ctest`" --test-dir `"$build`" --output-on-failure"
) -join " && "
cmd.exe /d /c $nativeBuild
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $gplc -c --no-susp-warn `
    -o (Join-Path $build "bounded_threshold_search.obj") `
    (Join-Path $projectRoot "prolog\bounded_threshold_search.pl")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$env:PTM_NATIVE_LIBRARY = Join-Path $build "ptm.dll"
Push-Location $projectRoot
try {
    & $python @pythonArguments -m unittest discover -s tests/python -v
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $python @pythonArguments examples/tabular_xor.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $python @pythonArguments examples/prolog_threshold.py
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
