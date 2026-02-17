# AppShell Template (Python core + Go backend + Wails frontend)

This folder is a practical template for migrating the current Streamlit project into a desktop app:

- Python: algorithm engine (`stdin/stdout` JSON protocol)
- Go: backend orchestrator (runner, timeout, cancellation, task status)
- Wails frontend: UI shell that calls Go methods
- Windows build scripts: package Python engine + Wails app into an installer

The template does not replace the existing `app.py`. It runs in parallel so migration can be incremental.

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

1. Python engine health check:

```bash
echo '{"task_id":"health-1","action":"health","payload":{}}' | python3 appshell/core/python_engine/engine_main.py
```

2. Python train action:

```bash
pip install -r appshell/core/python_engine/requirements.txt
python3 appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/sample_train_request.json
```

3. Go backend demo (polling task status):

```bash
cd appshell/backend
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke
```

4. Frontend template preview (static):

Open `appshell/frontend/index.html` in a browser. In a real Wails app, bind JS to Go methods with the same payload shape.

## Notes

- Protocol uses JSON only, suitable for local process calls.
- For large data, pass file paths (CSV/output dir), not huge JSON blobs.
- Current template focuses on phase 0/1/2 execution path and includes a frontend contract for phase 3.
- Full milestones and acceptance checks are in `appshell/PHASES_ACCEPTANCE.md`.
