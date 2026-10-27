<#
.SYNOPSIS
    Phase 6 gate smoke test: verifies fresh clone, pip install -e .[app],
    and streamlit offline review app startup within 120 seconds.

.PARAMETER Python
    Python invocation command or executable path (default: 'py -3.14').

.PARAMETER KeepTemp
    Preserve temporary clone and venv directory for inspection.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Python = "py -3.14",

    [Parameter(Mandatory = $false)]
    [switch]$KeepTemp
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [string]$Command,
        [string[]]$Arguments = @()
    )
    $parts = $Command.Trim() -split '\s+'
    $exe = $parts[0]
    $prefixArgs = if ($parts.Length -gt 1) { $parts[1..($parts.Length - 1)] } else { @() }
    $allArgs = $prefixArgs + $Arguments
    & $exe @allArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

function Get-VenvPythonPath {
    param([string]$VenvDirectory)
    $candidate = Join-Path $VenvDirectory "Scripts\python.exe"
    if (-not (Test-Path -Path $candidate)) {
        $candidate = Join-Path $VenvDirectory "bin\python"
    }
    if (-not (Test-Path -Path $candidate)) {
        throw "Venv python executable not found in $VenvDirectory"
    }
    return $candidate
}

function Wait-ForStreamlitHealth {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$HealthUrl,
        [int]$TimeoutSeconds = 120
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($Process.HasExited) {
            throw "Streamlit process exited prematurely with code $($Process.ExitCode)"
        }
        try {
            $resp = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200 -and $resp.Content.Trim() -eq "ok") {
                return [math]::Round($sw.Elapsed.TotalSeconds, 2)
            }
        }
        catch {
            # Connection refused or server warming up
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Streamlit health check did not return ok within $TimeoutSeconds seconds"
}

function Test-RootEndpoint {
    param([string]$RootUrl)
    $resp = Invoke-WebRequest -Uri $RootUrl -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($resp.StatusCode -ne 200) {
        throw "Root endpoint $RootUrl returned status $($resp.StatusCode), expected 200"
    }
}

function Stop-StreamlitProcess {
    param([System.Diagnostics.Process]$Process)
    if ($Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        $Process.WaitForExit(3000)
    }
}

function Remove-TempDirectory {
    param(
        [string]$Path,
        [switch]$Keep
    )
    if (-not $Keep -and (Test-Path -Path $Path)) {
        Start-Sleep -Milliseconds 500
        Remove-Item -Path $Path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tempFolder = Join-Path ([System.IO.Path]::GetTempPath()) ("utriage-smoke-" + [System.Guid]::NewGuid().ToString("N"))
$initialLocation = Get-Location
$appProcess = $null
$healthOk = $false
$appSeconds = 0.0

try {
    # 1. start a stopwatch;
    $installSw = [System.Diagnostics.Stopwatch]::StartNew()

    # 2. git clone the local repo into a new temp folder (no network);
    Write-Host "Cloning local repo from $repoRoot to $tempFolder..."
    Invoke-CheckedCommand -Command "git" -Arguments @("clone", $repoRoot, $tempFolder)

    # 3. create a venv there;
    $venvDir = Join-Path $tempFolder ".venv"
    Write-Host "Creating venv using $Python..."
    Invoke-CheckedCommand -Command $Python -Arguments @("-m", "venv", $venvDir)
    $venvPython = Get-VenvPythonPath -VenvDirectory $venvDir

    # 4. pip install -e .[app] (print note that app makes no network calls at runtime);
    Write-Host "NOTE: Installing dependencies from index. UTriage app makes no network calls at runtime (fully offline)."
    Push-Location $tempFolder
    try {
        Invoke-CheckedCommand -Command $venvPython -Arguments @("-m", "pip", "install", "-e", ".[app]")
    }
    finally {
        Pop-Location
    }
    $installSw.Stop()
    $installSeconds = [math]::Round($installSw.Elapsed.TotalSeconds, 2)

    # 5. start streamlit run app/streamlit_app.py --server.headless true --server.port 8599 as background process;
    Write-Host "Starting Streamlit background process on port 8599..."
    $logFile = Join-Path $tempFolder "streamlit.log"
    $errFile = Join-Path $tempFolder "streamlit_err.log"
    $appArgs = @(
        "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.headless", "true",
        "--server.port", "8599"
    )
    $appProcess = Start-Process -FilePath $venvPython `
        -ArgumentList $appArgs `
        -WorkingDirectory $tempFolder `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $errFile `
        -PassThru

    # 6. poll http://localhost:8599/_stcore/health until it returns ok or 120 s pass;
    $healthUrl = "http://localhost:8599/_stcore/health"
    $appSeconds = Wait-ForStreamlitHealth -Process $appProcess -HealthUrl $healthUrl -TimeoutSeconds 120
    $healthOk = $true

    # 7. also fetch http://localhost:8599/ and check status 200;
    $rootUrl = "http://localhost:8599/"
    Test-RootEndpoint -RootUrl $rootUrl

    # 8. print elapsed seconds for install and for app start separately;
    Write-Host "Install elapsed: $installSeconds s"
    Write-Host "App start elapsed: $appSeconds s"

    # 9. stop the process;
    Write-Host "Stopping Streamlit process..."
    Stop-StreamlitProcess -Process $appProcess

    # 10. remove the temp folder unless -KeepTemp;
    Set-Location $initialLocation
    Remove-TempDirectory -Path $tempFolder -Keep:$KeepTemp

    # 11. exit 0 only if the app answered within 120 s of starting.
    if ($healthOk -and $appSeconds -le 120) {
        exit 0
    }
    else {
        exit 1
    }
}
finally {
    Set-Location $initialLocation
    Stop-StreamlitProcess -Process $appProcess
    Remove-TempDirectory -Path $tempFolder -Keep:$KeepTemp
}
