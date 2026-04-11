$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$localConfig = Join-Path $PSScriptRoot "langgraph.local.ps1"
$appshellDir = Join-Path $repoRoot "appshell"

if (-not (Test-Path $localConfig)) {
    throw "Local LangGraph config not found: $localConfig"
}

. $localConfig

Push-Location $appshellDir
try {
    & wails dev
} finally {
    Pop-Location
}
