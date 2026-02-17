# Frontend Template (Wails)

This folder is a UI template for phase 3.

## Expected Go binding

The JS calls `window.go.main.App.RunTrainTask(payload)` when available.

Recommended Go method signature:

```go
func (a *App) RunTrainTask(payload map[string]any) (map[string]any, error)
```

It should internally call backend task service and return a snapshot containing:

- task id
- status
- response payload (from Python engine)
- error (if failed)

## Local preview

You can open `index.html` directly to preview layout and mock flow.
