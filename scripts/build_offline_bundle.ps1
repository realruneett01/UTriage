# The INSTALL step makes no network calls; the BUILD step needs network.

<#
.SYNOPSIS
    Builds an air-gapped offline installer bundle for UTriage.

.PARAMETER OutDir
    Destination folder for unpacked offline bundle assets (default: 'dist/utriage-offline').

.PARAMETER Python
    Python executable command or path used during build step (default: 'python').
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$OutDir = "dist/utriage-offline",

    [Parameter(Mandatory = $false)]
    [string]$Python = "python"
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutDir = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir
} else {
    Join-Path $repoRoot $OutDir
}

$wheelsDir = Join-Path $resolvedOutDir "wheels"
Write-Host "Creating output and wheels directory at $wheelsDir..."
New-Item -ItemType Directory -Path $wheelsDir -Force | Out-Null

Write-Host "Building project wheel..."
$wheelArgs = @("-m", "pip", "wheel", ".", "--no-deps", "-w", $wheelsDir)
Invoke-CheckedCommand -Executable $Python -Arguments $wheelArgs

Write-Host "Downloading dependency wheels..."
$deps = @(
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pydantic>=2",
    "pyyaml",
    "typer",
    "matplotlib",
    "torch",
    "streamlit"
)
$downloadArgs = @(
    "-m", "pip", "download",
    "-d", $wheelsDir,
    "--only-binary=:all:"
) + $deps + @(
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--extra-index-url", "https://pypi.org/simple"
)
Invoke-CheckedCommand -Executable $Python -Arguments $downloadArgs

Write-Host "Copying assets..."
$configsSrc = Join-Path $repoRoot "configs"
if (Test-Path -Path $configsSrc) {
    Copy-Item -Path $configsSrc -Destination (Join-Path $resolvedOutDir "configs") -Recurse -Force
}

$modelsSrc = Join-Path $repoRoot "models"
if (Test-Path -Path $modelsSrc) {
    $targetModelsDir = Join-Path $resolvedOutDir "models"
    $modelFiles = Get-ChildItem -Path $modelsSrc -File | Where-Object {
        $_.Extension -eq ".pt" -or $_.Name -like "*.pt.sha256" -or $_.Extension -eq ".sha256"
    }
    if ($modelFiles) {
        New-Item -ItemType Directory -Path $targetModelsDir -Force | Out-Null
        foreach ($file in $modelFiles) {
            Copy-Item -Path $file.FullName -Destination (Join-Path $targetModelsDir $file.Name) -Force
        }
    }
}

Copy-Item -Path (Join-Path $repoRoot "scripts\verify_bundle.py") -Destination (Join-Path $resolvedOutDir "verify_bundle.py") -Force

Write-Host "Writing install script..."
$installPs1Content = @'
# The INSTALL step makes no network calls.
$ErrorActionPreference = "Stop"

Write-Host "Running bundle checksum check..."
& python verify_bundle.py verify .
if ($LASTEXITCODE -ne 0) {
    throw "Checksum check failed with exit code $LASTEXITCODE"
}

Write-Host "Creating local virtual environment (.venv)..."
& python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    throw "Virtual environment creation failed"
}

$venvPython = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { ".venv/bin/python" }

Write-Host "Installing utriage from local wheels..."
& $venvPython -m pip install --no-index --find-links wheels utriage
if ($LASTEXITCODE -ne 0) {
    throw "Package installation failed"
}

Write-Host "Running utriage version check..."
$utriageExe = if (Test-Path ".venv\Scripts\utriage.exe") { ".venv\Scripts\utriage.exe" } else { ".venv/bin/utriage" }
if (Test-Path $utriageExe) {
    & $utriageExe version
} else {
    & $venvPython -m utriage.cli version
}
'@

Set-Content -Path (Join-Path $resolvedOutDir "install.ps1") -Value $installPs1Content -Encoding UTF8

Write-Host "Generating bundle checksums..."
$verifyScript = Join-Path $resolvedOutDir "verify_bundle.py"
Invoke-CheckedCommand -Executable $Python -Arguments @($verifyScript, "write", $resolvedOutDir)

Write-Host "Compressing bundle archive..."
$outParent = Split-Path -Path $resolvedOutDir -Parent
if (-not $outParent) {
    $outParent = $repoRoot
}
$leafName = Split-Path -Path $resolvedOutDir -Leaf
$zipPath = Join-Path $outParent "$leafName.zip"
if (Test-Path -Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}
Compress-Archive -Path (Join-Path $resolvedOutDir "*") -DestinationPath $zipPath -Force

Write-Host "Offline bundle build complete: $resolvedOutDir"
