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

## Demo commands

From this folder:

```bash
# single task
go run ./cmd/demo -action health
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke
go run ./cmd/demo -action repair -model-dir ../../outputs/results/wails_mvp -sample-index 0 -max-changes 3

# run 3 tasks concurrently
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -parallel 3 -output ../../outputs/results/parallel

# cancel first task after 1s
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -parallel 3 -cancel-after 1s

# timeout control
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -timeout 2s
```

## Wails MVP run

From this folder:

```bash
# ensure frontend path can be resolved from cwd
go run ./cmd/wails -engine ../core/python_engine/engine_main.py
```

## Integration contract for Wails

Expose Go methods mirroring task service behavior:

- `RunTask(payload map[string]any) (TaskSnapshot, error)`
- `GetTaskStatus(taskID string) (TaskSnapshot, error)`
- `CancelTask(taskID string) (bool, error)`
- `SelectCSV() (string, error)`
- `SelectOutputDir() (string, error)`

Frontend keeps the same JSON request shape used by Python engine.
Current MVP actions are `train` and `repair` (plus `health` for diagnostics).

Additional method for history:

- `ListTaskHistory(limit int) ([]TaskSnapshot, error)`

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
