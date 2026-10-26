<#
.SYNOPSIS
    Builds a standalone, offline Windows release package for UTriage using Nuitka.

.DESCRIPTION
    Creates an isolated venv, installs CPU-only PyTorch, runs test verification,
    compiles the launcher with Nuitka standalone, bundles configurations and assets,
    generates SHA256 checksums, and produces a distributable zip archive.

.PARAMETER Version
    Release version string (e.g., '0.0.1').

.PARAMETER OutDir
    Destination directory for unpacked release assets.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Version = "0.0.1",

    [Parameter(Mandatory = $false)]
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $Executable $($Arguments -join ' ')"
    }
}

function New-BuildVenv {
    param([string]$VenvPath)
    if (Test-Path -Path $VenvPath) {
        Remove-Item -Path $VenvPath -Recurse -Force
    }
    Invoke-CheckedCommand -Executable "python" -Arguments @("-m", "venv", $VenvPath)
}

function Install-PackageAndDeps {
    param(
        [string]$PythonExe,
        [string]$RepoRoot
    )
    Invoke-CheckedCommand -Executable $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    $torchArgs = @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    Invoke-CheckedCommand -Executable $PythonExe -Arguments $torchArgs
    $pkgArgs = @("-m", "pip", "install", "$RepoRoot[dev,ml,app]", "nuitka")
    Invoke-CheckedCommand -Executable $PythonExe -Arguments $pkgArgs
}

function Invoke-TestGate {
    param([string]$PythonExe)
    Invoke-CheckedCommand -Executable $PythonExe -Arguments @("-m", "pytest")
}

function Invoke-NuitkaCompile {
    param(
        [string]$PythonExe,
        [string]$LauncherPath,
        [string]$OutputDir
    )
    $args = @(
        "-m", "nuitka",
        "--standalone",
        "--output-dir=$OutputDir",
        $LauncherPath
    )
    Invoke-CheckedCommand -Executable $PythonExe -Arguments $args
}

function Copy-ReleaseAssets {
    param(
        [string]$RepoRoot,
        [string]$NuitkaBuildDir,
        [string]$TargetDir
    )
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    $distDir = Get-ChildItem -Path $NuitkaBuildDir -Directory -Filter "*.dist" | Select-Object -First 1
    if ($distDir) {
        Copy-Item -Path (Join-Path $distDir.FullName "*") -Destination $TargetDir -Recurse -Force
    }

    Copy-Item -Path (Join-Path $RepoRoot "configs") -Destination (Join-Path $TargetDir "configs") -Recurse -Force

    $modelsDir = Join-Path $RepoRoot "models"
    if (Test-Path -Path $modelsDir) {
        Copy-Item -Path $modelsDir -Destination (Join-Path $TargetDir "models") -Recurse -Force
    }

    Copy-Item -Path (Join-Path $RepoRoot "app") -Destination (Join-Path $TargetDir "app") -Recurse -Force

    $streamlitDir = Join-Path $TargetDir ".streamlit"
    New-Item -ItemType Directory -Path $streamlitDir -Force | Out-Null
    Copy-Item -Path (Join-Path $RepoRoot ".streamlit\config.toml") -Destination (Join-Path $streamlitDir "config.toml") -Force

    Copy-Item -Path (Join-Path $RepoRoot "docs\install.md") -Destination (Join-Path $TargetDir "install.md") -Force
}

function Write-Sha256Checksums {
    param([string]$TargetDir)
    $checksumPath = Join-Path $TargetDir "SHA256SUMS.txt"
    if (Test-Path -Path $checksumPath) {
        Remove-Item -Path $checksumPath -Force
    }

    $files = Get-ChildItem -Path $TargetDir -Recurse -File |
        Where-Object { $_.FullName -ne $checksumPath }

    $lines = foreach ($file in $files) {
        $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $rel = $file.FullName.Substring($TargetDir.Length).TrimStart('\', '/') -replace '\\', '/'
        "$hash  $rel"
    }

    $lines | Set-Content -Path $checksumPath -Encoding UTF8
}

function Compress-ReleaseArchive {
    param(
        [string]$TargetDir,
        [string]$Version,
        [string]$OutDirParent
    )
    $zipPath = Join-Path $OutDirParent "utriage-$Version-windows-x64.zip"
    if (Test-Path -Path $zipPath) {
        Remove-Item -Path $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $TargetDir "*") -DestinationPath $zipPath -Force
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutDir = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir
} else {
    Join-Path $repoRoot $OutDir
}

$tempBase = Join-Path $env:TEMP ("utriage-build-" + [System.Guid]::NewGuid().ToString("N"))
$venvDir = Join-Path $tempBase "venv"
$nuitkaBuildDir = Join-Path $tempBase "nuitka"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$launcherPath = Join-Path $repoRoot "src\utriage\__main__.py"

try {
    Write-Host "Creating clean build venv in $venvDir..."
    New-BuildVenv -VenvPath $venvDir

    Write-Host "Installing CPU torch, package, and build dependencies..."
    Install-PackageAndDeps -PythonExe $pythonExe -RepoRoot $repoRoot

    Write-Host "Running pytest validation..."
    Invoke-TestGate -PythonExe $pythonExe

    Write-Host "Compiling standalone launcher with Nuitka..."
    Invoke-NuitkaCompile -PythonExe $pythonExe -LauncherPath $launcherPath -OutputDir $nuitkaBuildDir

    Write-Host "Copying distribution assets to $resolvedOutDir..."
    Copy-ReleaseAssets -RepoRoot $repoRoot -NuitkaBuildDir $nuitkaBuildDir -TargetDir $resolvedOutDir

    Write-Host "Writing SHA256 checksums..."
    Write-Sha256Checksums -TargetDir $resolvedOutDir

    Write-Host "Packaging release archive..."
    $outParent = Split-Path -Path $resolvedOutDir -Parent
    if (-not $outParent) {
        $outParent = $repoRoot
    }
    Compress-ReleaseArchive -TargetDir $resolvedOutDir -Version $Version -OutDirParent $outParent

    Write-Host "Release build complete: $resolvedOutDir"
}
finally {
    if (Test-Path -Path $tempBase) {
        Remove-Item -Path $tempBase -Recurse -Force -ErrorAction SilentlyContinue
    }
}
