# AppShell Template (Python core + Go backend + Wails frontend)

This folder is a practical template for migrating the current Streamlit project into a desktop app:

- Python: algorithm engine (`stdin/stdout` JSON protocol)
- Go: backend orchestrator (runner, timeout, cancellation, task status)
- Wails frontend: UI shell that calls Go methods
- Windows build scripts: package Python engine + Wails app into an installer

The template does not replace the existing `app.py`. It runs in parallel so migration can be incremental.

## Stage 0 Foundation

- The current Python engine actions remain the stable base for the future Tool Layer: `health`, `train`, `scan_file`, `repair`, `repair_batch`, and `rollback_repair_batch`.
- `../MULTI_AGENT_BLUEPRINT.md` defines the long-term intelligent upgrade direction, while `../TOOL_LAYER_FOUNDATION.md` records the Stage 0 mapping from actions to future tools and algorithm assets.
- Stage 0 only preserves and wraps existing assets. It does not add user-visible intelligent workflows or change the current request/response contract.

## Structure

```text
appshell/
  PHASES_ACCEPTANCE.md
  core/python_engine/
    engine_main.py
    engine_service.py
    engine_protocol.py
    sample_train_request.json
  backend/
    go.mod
    cmd/demo/main.go
    cmd/wails/main.go
    cmd/wails/app.go
    internal/engine/*.go
    internal/task/service.go
  frontend/
    index.html
    src/main.js
    src/style.css
    README.md
  build/windows/
    build.ps1
    installer.iss
```

## Quick Start

Run the following commands from the repository root.

Recommended first step on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows_env.ps1
```

This creates `.venv-win` from the validated `requirements.lock.txt` set.
See `ENVIRONMENT.md` in the repository root for consistency guarantees and limits.

1. Python engine health check:

```bash
echo '{"task_id":"health-1","action":"health","payload":{}}' | python3 appshell/core/python_engine/engine_main.py
```

2. Python train action:

```bash
.\.venv-win\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.lock.txt
.\.venv-win\Scripts\python.exe appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/sample_train_request.json
```

3. Go backend demo (polling task status):

```bash
cd appshell/backend
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke
```

4. Wails desktop MVP:

```bash
cd appshell/backend
go run ./cmd/wails -engine ../core/python_engine/engine_main.py
```

5. Frontend template preview (static):

Open `appshell/frontend/index.html` in a browser. In Wails runtime, JS calls `window.go.main.App.*` bindings.

## Notes

- Protocol uses JSON only, suitable for local process calls.
- For large data, pass file paths (CSV/output dir), not huge JSON blobs.
- Current template now includes phase 3 MVP UI path (`configure -> run -> inspect -> export`).
- Full milestones and acceptance checks are in `appshell/PHASES_ACCEPTANCE.md`.
