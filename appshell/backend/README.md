# Go Backend Template

This backend runs Python engine requests as managed tasks.

## Main pieces

- `internal/engine`: request/response protocol + process runner
- `internal/task`: in-memory task lifecycle (`Start`, `Get`, `Cancel`, `List`)
- `cmd/demo`: CLI demo used for phase acceptance tests

## Demo commands

From this folder:

```bash
go run ./cmd/demo -action health
go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke
```

## Integration contract for Wails

Expose Go methods mirroring task service behavior:

- `RunTrainTask(payload map[string]any) (TaskSnapshot, error)`
- `GetTask(taskID string) (TaskSnapshot, error)`
- `CancelTask(taskID string) (bool, error)`

Frontend keeps the same JSON request shape used by Python engine.
