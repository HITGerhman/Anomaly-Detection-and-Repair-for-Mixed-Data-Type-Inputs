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
		return Response{}, fmt.Errorf("engine process failed: %w; stderr=%s", err, strings.TrimSpace(stderr.String()))
	}

	raw := strings.TrimSpace(stdout.String())
	if raw == "" {
		return Response{}, fmt.Errorf("empty engine stdout; stderr=%s", strings.TrimSpace(stderr.String()))
	}

	var resp Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		return Response{}, fmt.Errorf("invalid engine response json: %w; raw=%s", err, raw)
	}

	return resp, nil
}
