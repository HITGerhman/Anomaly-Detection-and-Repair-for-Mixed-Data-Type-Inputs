package engine

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"appshell/backend/internal/observability"
)

type StderrEvent struct {
	TaskID     string
	Line       string
	Parsed     map[string]any
	ObservedAt time.Time
}

type StderrObserver func(event StderrEvent)

type Runner struct {
	PythonBin      string
	EngineScript   string
	DefaultTimeout time.Duration

	observerMu sync.RWMutex
	observer   StderrObserver
}

func defaultPythonBin() string {
	if raw := strings.TrimSpace(os.Getenv("APPSHELL_PYTHON_BIN")); raw != "" {
		return raw
	}
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

func (r *Runner) SetStderrObserver(observer StderrObserver) {
	if r == nil {
		return
	}
	r.observerMu.Lock()
	r.observer = observer
	r.observerMu.Unlock()
}

func (r *Runner) stderrObserver() StderrObserver {
	if r == nil {
		return nil
	}
	r.observerMu.RLock()
	defer r.observerMu.RUnlock()
	return r.observer
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

	command, args := r.commandForEngine()
	cmd := exec.CommandContext(ctx, command, args...)
	cmd.Stdin = bytes.NewReader(payload)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return Response{}, fmt.Errorf("create stderr pipe failed: %w", err)
	}

	var stderrMu sync.Mutex
	stderrLines := make([]string, 0, 128)
	appendStderr := func(line string) {
		stderrMu.Lock()
		stderrLines = append(stderrLines, line)
		stderrMu.Unlock()
	}

	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		streamPythonStderr(req.TaskID, stderrPipe, r.stderrObserver(), appendStderr)
	}()

	if err := cmd.Start(); err != nil {
		return Response{}, fmt.Errorf("start engine process failed: %w", err)
	}
	if err := cmd.Wait(); err != nil {
		<-stderrDone
		stderrText := joinStderrLines(stderrLines)
		observability.Error("engine_run_failed", map[string]any{
			"task_id":      req.TaskID,
			"action":       req.Action,
			"duration_ms":  time.Since(started).Milliseconds(),
			"runner_error": err.Error(),
		})
		return Response{}, fmt.Errorf("engine process failed: %w; stderr=%s", err, strings.TrimSpace(stderrText))
	}
	<-stderrDone

	raw := strings.TrimSpace(stdout.String())
	stderrText := joinStderrLines(stderrLines)
	if raw == "" {
		observability.Error("engine_run_failed", map[string]any{
			"task_id":     req.TaskID,
			"action":      req.Action,
			"duration_ms": time.Since(started).Milliseconds(),
			"reason":      "empty engine stdout",
		})
		return Response{}, fmt.Errorf("empty engine stdout; stderr=%s", strings.TrimSpace(stderrText))
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

func (r *Runner) commandForEngine() (string, []string) {
	enginePath := strings.TrimSpace(r.EngineScript)
	if runEngineDirectly(enginePath) {
		return enginePath, nil
	}

	pythonBin := strings.TrimSpace(r.PythonBin)
	if pythonBin == "" {
		pythonBin = defaultPythonBin()
	}
	return pythonBin, []string{enginePath}
}

func runEngineDirectly(enginePath string) bool {
	if strings.TrimSpace(enginePath) == "" {
		return false
	}

	ext := strings.ToLower(filepath.Ext(enginePath))
	if runtime.GOOS == "windows" {
		return ext == ".exe" || ext == ".com" || ext == ".bat" || ext == ".cmd"
	}
	if ext == ".py" {
		return false
	}

	info, err := os.Stat(enginePath)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Mode()&0o111 != 0
}

func joinStderrLines(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func streamPythonStderr(taskID string, reader io.Reader, observer StderrObserver, sink func(line string)) {
	scanner := bufio.NewScanner(reader)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		sink(line)
		emitPythonStderrLine(taskID, line, observer)
	}
	if err := scanner.Err(); err != nil {
		observability.Warn("python_stderr_stream_failed", map[string]any{
			"task_id": taskID,
			"error":   err.Error(),
		})
	}
}

func emitPythonStderrLine(taskID string, line string, observer StderrObserver) {
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

	if observer != nil {
		observer(StderrEvent{
			TaskID:     taskID,
			Line:       line,
			Parsed:     parsed,
			ObservedAt: time.Now(),
		})
	}
}
