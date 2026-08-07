param(
    [string[]]$Clauses = "256",
    [string[]]$Features = "1024",
    [string[]]$Densities = "0.02",
    [string[]]$ResidentPages = "1,16,256",
    [int]$Repeats = 100,
    [int]$Warmup = 20,
    [int]$Samples = 5,
    [string[]]$Backend = "scalar,avx2,cuda_sparse,cuda_warp_tile,cuda_dense_bitset," +
        "cuda_sparse_fused_vote,cuda_warp_tile_fused_vote," +
        "cuda_dense_bitset_fused_vote",
    [string]$Distro = "Ubuntu-22.04",
    [string]$CudaRoot = "/usr/local/cuda",
    [string]$CudaArchitectures = "75",
    [switch]$Sweep,
    [switch]$GpuSweep,
    [switch]$JsonLines
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($projectRoot -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "The CUDA WSL wrapper requires a drive-rooted Windows project path."
}
$drive = $Matches[1].ToLowerInvariant()
$relativePath = $Matches[2].Replace('\', '/')
$wslProjectRoot = "/mnt/$drive/$relativePath"
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $digest = $sha.ComputeHash(
        [Text.Encoding]::UTF8.GetBytes($wslProjectRoot))
    $pathTag = -join ($digest[0..5] | ForEach-Object {
        $_.ToString("x2")
    })
} finally {
    $sha.Dispose()
}
$nvcc = "$CudaRoot/bin/nvcc"
& wsl.exe -d $Distro -- test -x $nvcc
if ($LASTEXITCODE -ne 0) {
    throw "CUDA compiler was not found at $nvcc in $Distro."
}

$build = "$wslProjectRoot/out/wsl-cuda-$pathTag"
$benchmark = "$build/ptm_packed_tm_benchmark"
$cudaPath = "$CudaRoot/bin:/usr/bin:/bin"
$configureArguments = @(
    "-d", $Distro, "--", "env", "PATH=$cudaPath",
    "cmake", "-S", $wslProjectRoot, "-B", $build, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DPTM_BUILD_TESTS=OFF",
    "-DPTM_BUILD_EXAMPLES=OFF",
    "-DPTM_BUILD_BENCHMARKS=ON",
    "-DPTM_ENABLE_CUDA=ON",
    "-DCMAKE_CUDA_COMPILER=$nvcc",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
)
$buildArguments = @(
    "-d", $Distro, "--", "cmake", "--build", $build,
    "--target", "ptm_packed_tm_benchmark", "-j2"
)
$buildOutput = @(& wsl.exe @configureArguments 2>&1)
$configureStatus = $LASTEXITCODE
if ($configureStatus -eq 0) {
    $buildOutput += @(& wsl.exe @buildArguments 2>&1)
    $buildStatus = $LASTEXITCODE
} else {
    $buildStatus = $configureStatus
}
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
    "-d", $Distro, "--", $benchmark,
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
& wsl.exe @benchmarkArguments
exit $LASTEXITCODE
