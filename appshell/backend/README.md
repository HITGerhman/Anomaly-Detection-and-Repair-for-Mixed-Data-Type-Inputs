# Go Backend Orchestration Layer

This backend runs Python engine requests as managed tasks and owns task
lifecycle. Frontend should never call Python directly.

## Main pieces

- `internal/engine`: request/response protocol + process runner
- `internal/task`: in-memory orchestration with queue/timeout/cancel APIs
- `cmd/demo`: CLI demo used for phase acceptance tests
- `cmd/wails`: desktop app entrypoint exposing Go bindings to frontend

## Task APIs

- `RunTask(req, timeout)` submits a task into queue.
- `CancelTask(taskID)` cancels pending/running task.
- `GetTaskStatus(taskID)` returns current task snapshot.

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
