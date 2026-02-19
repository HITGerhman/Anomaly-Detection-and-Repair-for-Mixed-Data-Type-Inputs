package engine

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func writeTempEngineScript(t *testing.T, dir string) string {
	t.Helper()

	script := `import json, os, sys, time
req = json.load(sys.stdin)
payload = req.get("payload", {})
pid_file = payload.get("pid_file")
if pid_file:
    with open(pid_file, "w", encoding="utf-8") as fp:
        fp.write(str(os.getpid()))
sleep_ms = int(payload.get("sleep_ms", 0))
if sleep_ms > 0:
    time.sleep(sleep_ms / 1000.0)
resp = {
    "task_id": req.get("task_id", "unknown"),
    "status": "ok",
    "result": {"slept_ms": sleep_ms},
    "error": None,
    "timestamp": "2026-01-01T00:00:00+00:00",
    "duration_ms": 0
}
print(json.dumps(resp))
`

	path := filepath.Join(dir, "engine_script.py")
	if err := os.WriteFile(path, []byte(script), 0o644); err != nil {
		t.Fatalf("write script failed: %v", err)
	}
	return path
}

func waitUntil(timeout time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return cond()
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}

	if runtime.GOOS == "windows" {
		out, err := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid)).Output()
		if err != nil {
			return false
		}
		text := string(out)
		return strings.Contains(text, fmt.Sprintf(" %d ", pid)) ||
			strings.Contains(text, fmt.Sprintf(" %d\r", pid)) ||
			strings.Contains(text, fmt.Sprintf("\t%d ", pid))
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = proc.Signal(syscall.Signal(0))
	return err == nil
}

func TestRunnerRunSuccess(t *testing.T) {
	if _, err := exec.LookPath(defaultPythonBin()); err != nil {
		t.Skipf("python not found: %v", err)
	}

	dir := t.TempDir()
	engineScript := writeTempEngineScript(t, dir)
	runner := NewRunner(engineScript)
	runner.DefaultTimeout = 2 * time.Second

	resp, err := runner.Run(context.Background(), Request{
		TaskID: "run-success-1",
		Action: "health",
		Payload: map[string]any{
			"sleep_ms": 20,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("expected status ok, got %s", resp.Status)
	}
	if resp.TaskID != "run-success-1" {
		t.Fatalf("unexpected task id: %s", resp.TaskID)
	}
}

func TestRunnerTimeoutDoesNotLeaveZombieProcess(t *testing.T) {
	if _, err := exec.LookPath(defaultPythonBin()); err != nil {
		t.Skipf("python not found: %v", err)
	}

	dir := t.TempDir()
	engineScript := writeTempEngineScript(t, dir)
	pidFile := filepath.Join(dir, "pid.txt")

	runner := NewRunner(engineScript)
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	_, err := runner.Run(ctx, Request{
		TaskID: "timeout-1",
		Action: "train",
		Payload: map[string]any{
			"sleep_ms": 5000,
			"pid_file": pidFile,
		},
	})
	if err == nil {
		t.Fatalf("expected timeout error, got nil")
	}

	var pid int
	ok := waitUntil(2*time.Second, func() bool {
		raw, readErr := os.ReadFile(pidFile)
		if readErr != nil {
			return false
		}
		value, convErr := strconv.Atoi(strings.TrimSpace(string(raw)))
		if convErr != nil {
			return false
		}
		pid = value
		return true
	})
	if !ok {
		t.Fatalf("pid file was not created in time: %s", pidFile)
	}

	terminated := waitUntil(2*time.Second, func() bool {
		return !processAlive(pid)
	})
	if !terminated {
		t.Fatalf("python process still alive after timeout, pid=%d", pid)
	}
}
