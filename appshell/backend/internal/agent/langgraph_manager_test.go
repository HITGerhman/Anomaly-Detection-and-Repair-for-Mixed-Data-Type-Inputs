package agent

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func requirePythonBin(t *testing.T) string {
	t.Helper()
	pythonBin, err := exec.LookPath("python")
	if err != nil {
		t.Skipf("python not found: %v", err)
	}
	return pythonBin
}

func findFreeTCPPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}

func writeHealthOnlySidecarScript(t *testing.T) string {
	t.Helper()
	script := `import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=58331)
args = parser.parse_args()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return
    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"status":"ok","service":"langgraph-sidecar","planner_mode":"mock","ready":True,"graph_id":"phase_b_mock_plan_graph","version":"phase_b"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer((args.host, args.port), Handler)
server.serve_forever()
`
	path := filepath.Join(t.TempDir(), "health_sidecar.py")
	if err := os.WriteFile(path, []byte(script), 0o644); err != nil {
		t.Fatalf("write sidecar script failed: %v", err)
	}
	return path
}

func TestLangGraphSidecarManagerCanStartReuseAndRestart(t *testing.T) {
	port := findFreeTCPPort(t)
	pythonBin := requirePythonBin(t)
	script := writeHealthOnlySidecarScript(t)
	config := LangGraphConfig{
		Enabled:        true,
		Host:           "127.0.0.1",
		Port:           port,
		ScriptPath:     script,
		PythonBin:      pythonBin,
		StartupTimeout: 5 * time.Second,
		RequestTimeout: time.Second,
	}
	client := NewLangGraphClient(config.BaseURL(), config.RequestTimeout)
	manager := NewLangGraphSidecarManager(config, client)
	defer manager.Close()

	health, err := manager.EnsureHealthy(context.Background())
	if err != nil {
		t.Fatalf("EnsureHealthy failed: %v", err)
	}
	if health.GraphID != "phase_b_mock_plan_graph" {
		t.Fatalf("unexpected graph id: %s", health.GraphID)
	}
	if !manager.Managed() {
		t.Fatalf("expected manager to own the launched process")
	}
	firstPID := manager.cmd.Process.Pid

	if _, err := manager.EnsureHealthy(context.Background()); err != nil {
		t.Fatalf("EnsureHealthy reuse failed: %v", err)
	}
	if manager.cmd.Process.Pid != firstPID {
		t.Fatalf("expected sidecar process to be reused")
	}

	if err := manager.cmd.Process.Kill(); err != nil {
		t.Fatalf("kill sidecar failed: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	if _, err := manager.EnsureHealthy(context.Background()); err != nil {
		t.Fatalf("EnsureHealthy restart failed: %v", err)
	}
	if manager.cmd == nil || manager.cmd.Process == nil {
		t.Fatalf("expected restarted sidecar process")
	}
	if manager.cmd.Process.Pid == firstPID {
		t.Fatalf("expected restarted sidecar pid to change")
	}
}

func TestLangGraphSidecarManagerDetectsPortOccupiedByAnotherProcess(t *testing.T) {
	port := findFreeTCPPort(t)
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer listener.Close()

	pythonBin := requirePythonBin(t)
	script := writeHealthOnlySidecarScript(t)
	config := LangGraphConfig{
		Enabled:        true,
		Host:           "127.0.0.1",
		Port:           port,
		ScriptPath:     script,
		PythonBin:      pythonBin,
		StartupTimeout: time.Second,
		RequestTimeout: 300 * time.Millisecond,
	}
	manager := NewLangGraphSidecarManager(config, NewLangGraphClient(config.BaseURL(), config.RequestTimeout))
	defer manager.Close()

	if _, err := manager.EnsureHealthy(context.Background()); err == nil {
		t.Fatalf("expected port occupied error")
	}
	if manager.Managed() {
		t.Fatalf("manager should not claim ownership when another process holds the port")
	}
}
