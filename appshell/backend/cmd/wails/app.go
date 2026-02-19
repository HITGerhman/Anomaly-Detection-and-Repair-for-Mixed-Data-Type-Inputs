package main

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"appshell/backend/internal/engine"
	"appshell/backend/internal/task"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

type App struct {
	ctx context.Context

	service *task.Service
}

func NewApp(engineScript string) (*App, error) {
	absEngine, err := filepath.Abs(engineScript)
	if err != nil {
		return nil, fmt.Errorf("resolve engine path failed: %w", err)
	}

	runner := engine.NewRunner(absEngine)
	service := task.NewService(runner)
	return &App{
		service: service,
	}, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
}

func (a *App) shutdown(context.Context) {
	if a.service != nil {
		a.service.Close()
	}
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	case fmt.Stringer:
		return strings.TrimSpace(x.String())
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func timeoutFromPayload(payload map[string]any, fallback time.Duration) time.Duration {
	raw := payload["timeout_ms"]
	if raw == nil {
		return fallback
	}

	switch v := raw.(type) {
	case float64:
		if v <= 0 {
			return fallback
		}
		return time.Duration(v) * time.Millisecond
	case int:
		if v <= 0 {
			return fallback
		}
		return time.Duration(v) * time.Millisecond
	case int64:
		if v <= 0 {
			return fallback
		}
		return time.Duration(v) * time.Millisecond
	case string:
		v = strings.TrimSpace(v)
		if v == "" {
			return fallback
		}
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}

	return fallback
}

func normalizeRequest(payload map[string]any) (engine.Request, time.Duration, error) {
	if payload == nil {
		payload = map[string]any{}
	}

	action := asString(payload["action"])
	if action == "" {
		action = "train"
	}
	taskID := asString(payload["task_id"])
	timeout := timeoutFromPayload(payload, 90*time.Second)

	req := engine.Request{
		TaskID:  taskID,
		Action:  action,
		Payload: map[string]any{},
	}

	// If caller already provides a nested payload object, use it directly.
	if nested, ok := payload["payload"].(map[string]any); ok {
		req.Payload = nested
		return req, timeout, nil
	}

	// MVP path: flatten train fields.
	if action == "train" {
		csvPath := asString(payload["csv_path"])
		targetCol := asString(payload["target_col"])
		outputDir := asString(payload["output_dir"])

		if csvPath == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: csv_path")
		}
		if targetCol == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: target_col")
		}

		req.Payload["csv_path"] = csvPath
		req.Payload["target_col"] = targetCol
		if outputDir != "" {
			req.Payload["output_dir"] = outputDir
		}
	}

	return req, timeout, nil
}

func (a *App) RunTask(payload map[string]any) (task.Task, error) {
	if a.service == nil {
		return task.Task{}, fmt.Errorf("task service is not initialized")
	}

	req, timeout, err := normalizeRequest(payload)
	if err != nil {
		return task.Task{}, err
	}

	taskID, err := a.service.RunTask(req, timeout)
	if err != nil {
		return task.Task{}, err
	}

	snapshot, ok := a.service.GetTaskStatus(taskID)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found after submit: %s", taskID)
	}
	return *snapshot, nil
}

func (a *App) GetTaskStatus(taskID string) (task.Task, error) {
	if a.service == nil {
		return task.Task{}, fmt.Errorf("task service is not initialized")
	}
	id := strings.TrimSpace(taskID)
	if id == "" {
		return task.Task{}, fmt.Errorf("task id is required")
	}

	snapshot, ok := a.service.GetTaskStatus(id)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found: %s", id)
	}
	return *snapshot, nil
}

func (a *App) CancelTask(taskID string) (bool, error) {
	if a.service == nil {
		return false, fmt.Errorf("task service is not initialized")
	}
	id := strings.TrimSpace(taskID)
	if id == "" {
		return false, fmt.Errorf("task id is required")
	}
	return a.service.CancelTask(id), nil
}

// Backward-compatible alias used by previous frontend template.
func (a *App) RunTrainTask(payload map[string]any) (task.Task, error) {
	if payload == nil {
		payload = map[string]any{}
	}
	payload["action"] = "train"
	return a.RunTask(payload)
}

func (a *App) SelectCSV() (string, error) {
	if a.ctx == nil {
		return "", fmt.Errorf("runtime is not initialized")
	}

	path, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select CSV File",
		Filters: []runtime.FileFilter{
			{
				DisplayName: "CSV",
				Pattern:     "*.csv",
			},
		},
	})
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(path), nil
}

func (a *App) SelectOutputDir() (string, error) {
	if a.ctx == nil {
		return "", fmt.Errorf("runtime is not initialized")
	}

	path, err := runtime.OpenDirectoryDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select Output Directory",
	})
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(path), nil
}
