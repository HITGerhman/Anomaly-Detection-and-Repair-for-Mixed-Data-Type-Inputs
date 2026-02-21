package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"appshell/backend/internal/observability"
)

type Runner struct {
	PythonBin      string
	EngineScript   string
	DefaultTimeout time.Duration
}

func defaultPythonBin() string {
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func NewRunner(engineScript string) *Runner {
	return &Runner{
		PythonBin:      defaultPythonBin(),
		EngineScript:   engineScript,
		DefaultTimeout: 60 * time.Second,
	}
}

func (r *Runner) Run(ctx context.Context, req Request) (Response, error) {
	if r == nil {
		return Response{}, fmt.Errorf("runner is nil")
	}
	if strings.TrimSpace(r.EngineScript) == "" {
		return Response{}, fmt.Errorf("engine script path is empty")
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("marshal request: %w", err)
	}

	started := time.Now()
	observability.Info("engine_run_started", map[string]any{
		"task_id":       req.TaskID,
		"action":        req.Action,
		"engine_script": r.EngineScript,
	})

	timeout := r.DefaultTimeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, r.PythonBin, r.EngineScript)
	cmd.Stdin = bytes.NewReader(payload)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		emitPythonStderr(req.TaskID, stderr.String())
		observability.Error("engine_run_failed", map[string]any{
			"task_id":      req.TaskID,
			"action":       req.Action,
			"duration_ms":  time.Since(started).Milliseconds(),
			"runner_error": err.Error(),
		})
		return Response{}, fmt.Errorf("engine process failed: %w; stderr=%s", err, strings.TrimSpace(stderr.String()))
	}
	emitPythonStderr(req.TaskID, stderr.String())

	raw := strings.TrimSpace(stdout.String())
	if raw == "" {
		observability.Error("engine_run_failed", map[string]any{
			"task_id":     req.TaskID,
			"action":      req.Action,
			"duration_ms": time.Since(started).Milliseconds(),
			"reason":      "empty engine stdout",
		})
		return Response{}, fmt.Errorf("empty engine stdout; stderr=%s", strings.TrimSpace(stderr.String()))
	}

	var resp Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		observability.Error("engine_run_failed", map[string]any{
			"task_id":     req.TaskID,
			"action":      req.Action,
			"duration_ms": time.Since(started).Milliseconds(),
			"reason":      "invalid engine response json",
		})
		return Response{}, fmt.Errorf("invalid engine response json: %w; raw=%s", err, raw)
	}

	observability.Info("engine_run_succeeded", map[string]any{
		"task_id":     req.TaskID,
		"action":      req.Action,
		"duration_ms": time.Since(started).Milliseconds(),
		"status":      resp.Status,
	})

	return resp, nil
}

func emitPythonStderr(taskID string, rawStderr string) {
	lines := strings.Split(strings.TrimSpace(rawStderr), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		fields := map[string]any{
			"task_id": taskID,
			"line":    line,
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(line), &parsed); err == nil {
			fields["python_log"] = parsed
			if v, ok := parsed["event"]; ok {
				fields["python_event"] = v
			}
		}
		observability.Info("python_stderr", fields)
	}
}
