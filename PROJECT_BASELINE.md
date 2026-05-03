# M0 Project Baseline

- Baseline date: 2026-05-03 18:29:16 +08:00
- Scope: M0 project baseline confirmation only.
- Roadmap: `GRADUATION_PROJECT_ROADMAP.md`
- Rule: do not execute M1 or later milestones, do not refactor the main architecture, do not add dependencies.

## Current Entrypoints

- Streamlit demo: `app.py`
  - Intended command: `streamlit run app.py`
  - Current README still mainly describes this legacy/demo path.
- Python engine: `appshell/core/python_engine/engine_main.py`
  - Stable JSON protocol over stdin/stdout or `--input` / `--output`.
  - Supported actions observed from health check: `health`, `train`, `repair`, `scan_file`, `repair_batch`, `rollback_repair_batch`.
- Go backend demo: `appshell/backend/cmd/demo`
  - Intended command from `appshell/backend`: `go run ./cmd/demo -action health`
  - Used for task orchestration, timeout/cancel behavior, and Python engine subprocess calls.
- Wails shell backend: `appshell/backend/cmd/wails`
  - Intended command from `appshell/backend`: `go run ./cmd/wails -engine ../core/python_engine/engine_main.py`
  - Exposes task APIs for the frontend.
- Frontend static preview: `appshell/frontend/index.html`
  - Can be opened directly for browser/mock preview.
  - `appshell/frontend/src/main.js` contains real Wails bindings fallback and mock task logic.

## Environment Snapshot

- Project virtualenv Python:
  - Command: `.\.venv-win\Scripts\python.exe --version`
  - Result: `Python 3.11.7`
- Python engine dependencies:
  - Root: `requirements.txt`
  - Engine minimal runtime: `appshell/core/python_engine/requirements.txt`
  - `.\.venv-win\Scripts\python.exe` can import the tested engine dependencies.
- Go:
  - Observed earlier in this M0 pass: `go version go1.26.1 windows/amd64`
  - Backend module declares `go 1.25` in `appshell/backend/go.mod`.
- Node/npm:
  - `node --version` fails with `Access is denied`.
  - `npm --version` fails because `npm` is not recognized.
  - `appshell/frontend/package.json` is absent, so frontend build commands are not part of the current reliable baseline.
- Current untracked items at M0 start:
  - `GRADUATION_PROJECT_ROADMAP.md`
  - `out/figma-verify/`
  - `scripts/langgraph.local.ps1`
  - The latter two were pre-existing and are not part of this M0 work.

## Validation Results

| Command | Result |
|---|---|
| `git status --short --branch` | Passed. Current branch is `main...origin/main`; untracked items include `GRADUATION_PROJECT_ROADMAP.md`, `out/figma-verify/`, `scripts/langgraph.local.ps1`. |
| `.\.venv-win\Scripts\python.exe --version` | Passed: `Python 3.11.7`. |
| `.\.venv-win\Scripts\python.exe -m pytest --collect-only tests/python_engine -q` | Passed: 21 Python engine tests collected in 1.68s. |
| `.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q` | Passed: 21 passed in 33.86s. |
| `'{"task_id":"health-m0","action":"health","payload":{}}' \| .\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py` | Passed: returned `status=ok`, Python `3.11.7`, and actions `health/train/repair/scan_file/repair_batch/rollback_repair_batch`. |
| `Push-Location appshell\backend; go test ./internal/engine ./internal/task ./cmd/wails; Pop-Location` | Failed in default environment: `appshell/backend/internal/engine` failed with Python subprocess `exit status 9009`; `internal/task` and `cmd/wails` passed. |
| `$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH; Push-Location appshell\backend; go test ./internal/engine ./internal/task ./cmd/wails; Pop-Location` | Passed: all three Go packages passed when the project virtualenv Python was prepended to `PATH`. |
| `node --version` | Failed: `node.exe` access denied. |
| `npm --version` | Failed: `npm` command not found. |

## Known Issues

- Default shell Python entry is not reliable for Go subprocess tests. The Go engine runner tests need a usable `python` on `PATH`; prepending `.\.venv-win\Scripts` fixes the observed `exit status 9009`.
- Node/npm are not usable in the current environment, so frontend JavaScript build/lint checks are not currently part of the verified baseline.
- `appshell/frontend/package.json` and `appshell/wails.json` are absent; Wails/frontend packaging is not treated as a completed baseline capability.
- Root `README.md` still primarily describes the Streamlit application, while AppShell now has Python engine, Go backend, and Wails shell assets. Documentation drift remains.
- Go all-package discovery/testing may be slower than the focused package set; the M0 baseline uses the focused backend command above.
- Windows clean-machine packaging/installer validation is not proven in this baseline.

## High-Risk Areas

- `src/training_core.py`: model training, target-type inference, metric calculation, saved artifacts.
- `src/repair_core.py`: repair candidate generation, scoring, minimal edit search.
- `appshell/core/python_engine/*`: stable JSON action protocol and action routing.
- `appshell/backend/internal/engine/*`: Python subprocess boundary, timeout, stderr handling.
- `appshell/backend/internal/task/*`: task lifecycle, cancellation, history persistence.
- `appshell/frontend/src/main.js`: Wails bindings, scan/repair flow, mock fallback, task polling.

## Current Snapshot

The project currently has a working Python engine baseline under `.venv-win`, including scan, batch repair, rollback, train, repair, and health actions. The Go backend can pass focused tests when the same virtualenv Python is made visible through `PATH`. Streamlit remains the documented legacy/demo entrypoint. Wails/frontend assets exist, but Node/npm and packaging are not reliable in the current machine environment.

M0 is complete as a baseline documentation step. M1 and later milestone work has not been executed.
