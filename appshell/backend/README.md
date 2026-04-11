# Go Backend Orchestration Layer

This backend runs Python engine requests as managed tasks and owns task
lifecycle. Frontend should never call Python directly.

## Main pieces

- `internal/engine`: request/response protocol + process runner
- `internal/task`: orchestration + sqlite task history (recent N records)
- `internal/observability`: structured JSON logs for Go runtime
- `cmd/demo`: CLI demo used for phase acceptance tests
- `cmd/wails`: desktop app entrypoint exposing Go bindings to frontend

## Task APIs

- `RunTask(req, timeout)` submits a task into queue.
- `CancelTask(taskID)` cancels pending/running task.
- `GetTaskStatus(taskID)` returns current task snapshot.
- `ListRecentTasks(limit)` returns persisted recent history.

Status values:

- `pending`
- `running`
- `succeeded`
- `failed`
- `canceled`
- `timed_out`

## Startup checks

`cmd/wails` now uses a blocking startup self-check before enabling task APIs.
Frontend should call `RunStartupChecks()` first, then enter the main wizard only when blocking checks pass.

Blocking checks:

- `engine_script`
- `engine_health`
- `runtime_dependencies`
- `task_history_sqlite`
- `results_output_root`

Warning-only check:

- `model_artifacts`: missing default model files means “not trained yet”, not a broken runtime

If any blocking check fails, `RunTask`, `GetTaskStatus`, `CancelTask`, and `ListTaskHistory` return a unified startup-blocked error and the backend keeps `task.Service` uninitialized.

## Python Engine Actions

The Python engine supports the following actions via JSON protocol:

- **health:** Health check and diagnostics
- **train:** Model training with task_type support (auto/classification/regression)
- **scan_file:** Full-table column-level scanning with outputs:
  - `column_thumbnails`: Column-level summaries
  - `hot_segments`: High-confidence anomaly segments
  - `issue_type_counts`: Aggregated issue statistics
  - `confidence`: Overall confidence score
  - `explain_features`: Feature importance explanations
- **repair:** Single-sample model-driven repair search
- **repair_batch:** Batch repair by selected issue_ids, generates rollback manifest
- **rollback_repair_batch:** Rollback a previous batch repair operation

Supported anomaly types:

- Missing values
- Outliers
- Rare categories
- Time series shift
- Cross-column consistency issues
- Duplicate records

## Demo commands

From this folder:

```bash
# health check
go run ./cmd/demo -action health

# train model
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke

# scan entire file for anomalies
go run ./cmd/demo -action scan_file -model-dir ../../outputs/results/wails_mvp

# single-sample repair
go run ./cmd/demo -action repair -model-dir ../../outputs/results/wails_mvp -sample-index 0 -max-changes 3

# batch repair by issue IDs
go run ./cmd/demo -action repair_batch -model-dir ../../outputs/results/wails_mvp -issue-ids "issue_1,issue_2,issue_3"

# rollback batch repair
go run ./cmd/demo -action rollback_repair_batch -model-dir ../../outputs/results/wails_mvp -rollback-manifest rollback_manifest.json

# run 3 tasks concurrently
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -parallel 3 -output ../../outputs/results/parallel

# cancel first task after 1s
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -parallel 3 -cancel-after 1s

# timeout control
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -timeout 2s
```

## Performance benchmark

From this folder, using the project-local Python environment:

```bash
go run ./cmd/bench \
  -scenario all \
  -python-bin ../../.venv-win/Scripts/python.exe \
  -plan-csv ../../data/raw/simple_obvious_anomaly.csv \
  -plan-model-dir ../../outputs/results/wails_mvp \
  -output ../../outputs/results/backend_benchmark_latest.json

# compare sequential vs parallel agent_retrieve preview
go run ./cmd/bench \
  -scenario agent-plan \
  -python-bin ../../.venv-win/Scripts/python.exe \
  -plan-csv ../../data/raw/simple_obvious_anomaly.csv \
  -plan-model-dir ../../outputs/results/wails_mvp \
  -plan-warmups 1 \
  -plan-iterations 5 \
  -agent-retrieve-mode sequential \
  -output ../../outputs/results/backend_benchmark_retrieve_sequential.json

go run ./cmd/bench \
  -scenario agent-plan \
  -python-bin ../../.venv-win/Scripts/python.exe \
  -plan-csv ../../data/raw/simple_obvious_anomaly.csv \
  -plan-model-dir ../../outputs/results/wails_mvp \
  -plan-warmups 1 \
  -plan-iterations 5 \
  -agent-retrieve-mode parallel \
  -output ../../outputs/results/backend_benchmark_retrieve_parallel.json
```

What this benchmark covers:

- synthetic scheduler scaling for the Go task queue
- synthetic approval pause/resume latency for the agent runtime
- end-to-end `agent.session.plan` latency and stage breakdown on a real sample CSV

The `agent-plan` report includes `retrieve_mode`, so sequential and parallel
`agent_retrieve` experiments can be compared with the same benchmark entrypoint.

The benchmark writes a JSON report to `outputs/results/` so the latest numbers can be reused in resume bullets, defense slides, and follow-up optimization work.

## Wails MVP run

From this folder:

```bash
# ensure frontend path can be resolved from cwd
go run ./cmd/wails -engine ../core/python_engine/engine_main.py
```

## Integration contract for Wails

Expose Go methods mirroring task service behavior:

- `RunStartupChecks() (StartupCheckReport, error)`
- `RunTask(payload map[string]any) (TaskSnapshot, error)`
- `GetTaskStatus(taskID string) (TaskSnapshot, error)`
- `CancelTask(taskID string) (bool, error)`
- `SelectCSV() (string, error)`
- `SelectOutputDir() (string, error)`

Frontend keeps the same JSON request shape used by Python engine.

Current supported actions:

- `health`: Diagnostics
- `train`: Model training
- `scan_file`: Full-table column-level scanning
- `repair`: Single-sample repair
- `repair_batch`: Batch repair by issue IDs
- `rollback_repair_batch`: Rollback batch repair

Additional method for history:

- `ListTaskHistory(limit int) ([]TaskSnapshot, error)`

Startup check report fields:

- `overall_status`: `ok | warning | failed`
- `can_enter`: whether frontend may enter the main workflow
- `checked_at`: RFC3339 timestamp
- `items[]`: per-check status, blocking flag, message, optional path/detail/auto_fixed
- `summary`: counts for passed/warning/failed items
- `raw`: optional aggregated engine health and resolved path details

## Observability and history persistence

- Go emits structured JSON logs to `stderr` (includes `task_id` for every task event).
- Go emits structured JSON logs to `stderr` and defaults to `outputs/appshell/go_backend.log`.
- Python engine logs are captured from `stderr` and re-emitted by Go with `task_id`.
- Frontend emits JSON lines to browser console with `task_id` for UI events.
- Task snapshots are persisted to sqlite and can be queried after restart.

Environment variables:

- `APPSHELL_TASK_DB`: sqlite file path for task history.
- `APPSHELL_TASK_HISTORY_KEEP`: keep only latest N tasks (default `100`).
- `APPSHELL_GO_LOG_FILE`: override Go JSON log file path (default is auto-set).
