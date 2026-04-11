param(
    [string]$PythonExe = "",
    [string]$VenvDir = ".venv-win",
    [switch]$Recreate,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$lockFile = Join-Path $repoRoot "requirements.lock.txt"
$venvPath = Join-Path $repoRoot $VenvDir
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $lockFile)) {
    throw "requirements.lock.txt not found: $lockFile"
}

$launcher = $null
$launcherArgs = @()
if ($PythonExe) {
    $launcher = $PythonExe
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $launcher = (Get-Command py).Source
    $launcherArgs = @("-3.11")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $launcher = (Get-Command python).Source
} else {
    throw "Python launcher not found. Install Python 3.11 or pass -PythonExe."
}

if ($Recreate -and (Test-Path $venvPath)) {
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPython)) {
    & $launcher @launcherArgs -m venv $venvPath
}

& $venvPython -m pip install --no-cache-dir --disable-pip-version-check -r $lockFile

if (-not $SkipTests) {
    & $venvPython -m pytest tests/python_engine -q
    & $venvPython -m pytest tests/langgraph_sidecar -q
}

Write-Host "Environment ready: $venvPath"
Write-Host "Python: $venvPython"
