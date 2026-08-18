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

$configuredGProlog = $env:PTM_GPROLOG
$gprolog = Get-Command gprolog -ErrorAction SilentlyContinue
if ($configuredGProlog -and
    (Test-Path -LiteralPath $configuredGProlog -PathType Leaf)) {
    $env:PTM_GPROLOG = (Resolve-Path -LiteralPath $configuredGProlog).Path
} elseif ($gprolog) {
    $env:PTM_GPROLOG = $gprolog.Source
} else {
    throw "GNU Prolog was not found on PATH. Set PTM_GPROLOG to its executable."
}
$gprologBin = Split-Path -Parent $env:PTM_GPROLOG
$gplc = Join-Path $gprologBin "gplc.exe"
$env:PATH = $gprologBin + [IO.Path]::PathSeparator + $env:PATH

$programFilesX86 = [Environment]::GetFolderPath(
    [Environment+SpecialFolder]::ProgramFilesX86)
$vswhere = Join-Path $programFilesX86 `
    "Microsoft Visual Studio\Installer\vswhere.exe"
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
        "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release " +
        "-DPTM_BUILD_TESTS=ON"),
    "`"$cmake`" --build `"$build`"",
    "`"$ctest`" --test-dir `"$build`" --output-on-failure"
) -join " && "
cmd.exe /d /c $nativeBuild
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$installPrefix = Join-Path $build "install-root"
$consumerBuild = Join-Path $build "consumer-smoke"
$installConsumer = @(
    "call `"$vcvars`" >nul",
    "`"$cmake`" --install `"$build`" --prefix `"$installPrefix`"",
    ("`"$cmake`" -S `"$(Join-Path $projectRoot 'tests\consumer')`" " +
        "-B `"$consumerBuild`" -G Ninja " +
        "-DCMAKE_MAKE_PROGRAM=`"$ninja`" " +
        "-DCMAKE_BUILD_TYPE=Release " +
        "-DCMAKE_PREFIX_PATH=`"$installPrefix`""),
    "`"$cmake`" --build `"$consumerBuild`"",
    "`"$(Join-Path $consumerBuild 'ptm_consumer_smoke.exe')`"",
    "`"$(Join-Path $installPrefix 'bin\ptmrt.exe')`" --help >nul"
) -join " && "
cmd.exe /d /c $installConsumer
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $gplc -c --no-susp-warn `
    -o (Join-Path $build "bounded_threshold_search.obj") `
    (Join-Path $projectRoot "prolog\bounded_threshold_search.pl")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& $gplc -c --no-susp-warn `
    -o (Join-Path $build "bounded_structure_search.obj") `
    (Join-Path $projectRoot "prolog\bounded_structure_search.pl")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$env:PTM_NATIVE_LIBRARY = Join-Path $build "ptm.dll"
Push-Location $projectRoot
try {
    & $python @pythonArguments -m pytest tests/python -q
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $python @pythonArguments examples/tabular_xor.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $python @pythonArguments examples/prolog_threshold.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $python @pythonArguments examples/prolog_structures.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $preprocessingArtifact = Join-Path $build "preprocessing-demo.ptm"
    & $python @pythonArguments examples/export_preprocessing_demo.py `
        $preprocessingArtifact
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $ptmrt = Join-Path $build "ptmrt.exe"
    & $ptmrt verify $preprocessingArtifact
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $recordOutput = & $ptmrt run-record $preprocessingArtifact `
        age:int=30 status:string=ready active:bool=true
    if ($LASTEXITCODE -ne 0 -or
        $recordOutput -notmatch '"features":\[1,1,1,1,1,0\]') {
        throw "Raw-record preprocessing verification failed: $recordOutput"
    }
    exit 0
} finally {
    Pop-Location
}
