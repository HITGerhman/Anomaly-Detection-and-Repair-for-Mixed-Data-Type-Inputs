$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv-win\Scripts\python.exe"

$env:APPSHELL_LANGGRAPH_ENABLED = "true"
$env:APPSHELL_LANGGRAPH_PYTHON_BIN = $venvPython
$env:APPSHELL_LANGGRAPH_LLM_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:APPSHELL_LANGGRAPH_LLM_API_KEY = "your-api-key"
$env:APPSHELL_LANGGRAPH_LLM_MODEL = "your-model-name"
$env:APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS = "8000"

Write-Host "LangGraph local environment loaded."
Write-Host "Base URL: $env:APPSHELL_LANGGRAPH_LLM_BASE_URL"
Write-Host "Model: $env:APPSHELL_LANGGRAPH_LLM_MODEL"
Write-Host "Python: $env:APPSHELL_LANGGRAPH_PYTHON_BIN"
