# Frontend MVP (Wails Phase 3)

This frontend provides the minimal usable flow:

1. Configure parameters (`csv_path`, `target_col`, `output_dir`, `timeout_ms`)
2. Select CSV/output directory
3. Start task
4. Poll and display progress
5. Show result and errors
6. Export task result (`JSON` and metrics `CSV`)

## Expected Go bindings

When running in Wails runtime, JS calls:

- `window.go.main.App.RunTask(payload)`
- `window.go.main.App.GetTaskStatus(taskID)`
- `window.go.main.App.CancelTask(taskID)`
- `window.go.main.App.SelectCSV()`
- `window.go.main.App.SelectOutputDir()`

## Fallback mode

Open `index.html` directly to preview UI. In browser mode, file/dir pickers and task
execution use local mock logic.

