# Python Environment

## Purpose

This repository now uses a project-local Windows virtual environment for development and regression testing.
The goal is to keep project dependencies isolated from the Anaconda `base` environment and make the tested dependency set reproducible.

## Files

- `requirements.txt`: broad compatibility ranges for manual exploration
- `requirements.lock.txt`: exact dependency versions validated in `.venv-win`
- `scripts/setup_windows_env.ps1`: creates or refreshes the local Windows virtual environment

## Recommended workflow

1. Create or refresh the environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows_env.ps1
```

2. Activate it when working locally:

```powershell
.\.venv-win\Scripts\Activate.ps1
```

3. Run the Python regression suites:

```powershell
pytest tests/python_engine tests/langgraph_sidecar -q
```

## What this does guarantee

- On Windows with Python 3.11, developers can recreate the same pinned dependency set used by the latest passing regression run.
- The environment is isolated from unrelated system or Anaconda `base` packages.
- The project can be validated with the same `health` and `pytest` commands after recreation.

## What this does not fully guarantee

- Cross-OS parity is not automatic. A Windows lock file does not guarantee byte-for-byte parity on Linux or macOS.
- End-user distribution consistency is still a packaging problem, not just an environment problem.
- For shipping the desktop app to other machines, the final installer should bundle the Python runtime and the validated dependency payload instead of relying on the target machine's Python.

## Current validated baseline

- Python 3.11
- `numpy==1.26.4`
- `pandas==2.3.3`
- `lightgbm==4.6.0`
- `scikit-learn==1.8.0`
- `shap==0.45.1`
- `langgraph==1.1.2`
- `pytest==7.4.0`

## Phase B note

- `appshell/core/langgraph_sidecar/` is the local Python sidecar used by the LangGraph Phase B skeleton.
- The recommended `.venv-win` environment now includes `langgraph` and its pinned transitive dependencies.
- Startup checks treat the LangGraph sidecar as warning-only. If it cannot start, the Go planner falls back to the deterministic `MockPlanner`.

## Phase C note

- The LangGraph sidecar now supports `GET /health`, `POST /v1/plan`, and `POST /v1/explain`.
- Phase C keeps the deterministic execution boundary unchanged: Go still owns scan, preview, execute, validation, rescan, and rollback.
- To enable real cognition instead of deterministic fallback, configure an OpenAI-compatible endpoint before starting Wails or the demo CLI:

```powershell
$env:APPSHELL_LANGGRAPH_LLM_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:APPSHELL_LANGGRAPH_LLM_API_KEY = "your-api-key"
$env:APPSHELL_LANGGRAPH_LLM_MODEL = "your-model-name"
```

- Optional timeout override:

```powershell
$env:APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS = "4000"
```

- If the sidecar starts but these variables are missing or the endpoint is unavailable, the system stays enterable and automatically falls back to the deterministic `MockPlanner`.
