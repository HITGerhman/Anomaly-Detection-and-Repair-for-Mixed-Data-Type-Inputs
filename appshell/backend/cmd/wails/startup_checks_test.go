package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"appshell/backend/internal/agent"
)

func writeStartupCheckEngineScript(t *testing.T, root string, mode string) string {
	t.Helper()

	script := "import json, sys\n" +
		"req = json.load(sys.stdin)\n" +
		"action = req.get('action', '')\n" +
		"if action == 'health':\n" +
		"    if '" + mode + "' == 'fail_health':\n" +
		"        resp = {'task_id': req.get('task_id', 'startup'), 'status': 'error', 'result': {}, 'error': {'code': 'MISSING_DEPENDENCY', 'message': 'numpy unavailable', 'details': {'dependency': 'numpy'}}, 'timestamp': '2026-03-12T00:00:00Z', 'duration_ms': 0}\n" +
		"    else:\n" +
		"        resp = {'task_id': req.get('task_id', 'startup'), 'status': 'ok', 'result': {'engine': 'python-anomaly-engine', 'project_root': '" + strings.ReplaceAll(root, "\\", "\\\\") + "', 'python': '3.11.0', 'platform': 'test-platform', 'actions': ['health', 'train'], 'dependencies': {'pandas': {'status': 'ok', 'module': 'pandas', 'version': '2.2.3'}, 'numpy': {'status': 'ok', 'module': 'numpy', 'version': '1.26.4'}, 'lightgbm': {'status': 'ok', 'module': 'lightgbm', 'version': '4.6.0'}, 'scikit-learn': {'status': 'ok', 'module': 'sklearn', 'version': '1.5.2'}, 'joblib': {'status': 'ok', 'module': 'joblib', 'version': '1.4.2'}}}, 'error': None, 'timestamp': '2026-03-12T00:00:00Z', 'duration_ms': 0}\n" +
		"else:\n" +
		"    resp = {'task_id': req.get('task_id', 'task'), 'status': 'ok', 'result': {'ok': True, 'action': action}, 'error': None, 'timestamp': '2026-03-12T00:00:00Z', 'duration_ms': 0}\n" +
		"print(json.dumps(resp))\n"

	path := filepath.Join(root, "engine_script.py")
	if err := os.WriteFile(path, []byte(script), 0o644); err != nil {
		t.Fatalf("write engine script failed: %v", err)
	}
	return path
}

func findFreePortForStartupTest(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}

func writeStartupCheckLangGraphScript(t *testing.T, root string, plannerMode string, llmMode string) string {
	t.Helper()
	if strings.TrimSpace(plannerMode) == "" {
		plannerMode = "llm"
	}
	if strings.TrimSpace(llmMode) == "" {
		llmMode = "configured"
	}
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
        body = json.dumps({"status":"ok","service":"langgraph-sidecar","planner_mode":"` + plannerMode + `","llm_mode":"` + llmMode + `","model":"gpt-test","ready":True,"graph_id":"phase_c_cognition_graph","version":"phase_c"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer((args.host, args.port), Handler)
server.serve_forever()
`
	path := filepath.Join(root, "langgraph_sidecar.py")
	if err := os.WriteFile(path, []byte(script), 0o644); err != nil {
		t.Fatalf("write langgraph sidecar script failed: %v", err)
	}
	return path
}

func withTempBackendWorkingDir(t *testing.T) string {
	t.Helper()

	if _, err := exec.LookPath("python"); err != nil {
		t.Skipf("python not found: %v", err)
	}

	root := t.TempDir()
	backendDir := filepath.Join(root, "appshell", "backend")
	if err := os.MkdirAll(backendDir, 0o755); err != nil {
		t.Fatalf("mkdir backend dir failed: %v", err)
	}

	prevWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd failed: %v", err)
	}
	if err := os.Chdir(backendDir); err != nil {
		t.Fatalf("chdir failed: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(prevWD)
	})
	return root
}

func findStartupItem(t *testing.T, report StartupCheckReport, key string) StartupCheckItem {
	t.Helper()
	for _, item := range report.Items {
		if item.Key == key {
			return item
		}
	}
	t.Fatalf("startup check item not found: %s", key)
	return StartupCheckItem{}
}

func waitForTerminalTask(t *testing.T, app *App, taskID string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		taskSnapshot, err := app.GetTaskStatus(taskID)
		if err == nil {
			status := strings.ToLower(strings.TrimSpace(taskSnapshot.Status))
			if status == "succeeded" || status == "failed" || status == "canceled" || status == "timed_out" {
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("task did not reach terminal status in time: %s", taskID)
}

func TestRunStartupChecksFailsWhenEngineScriptMissing(t *testing.T) {
	root := withTempBackendWorkingDir(t)
	t.Setenv("APPSHELL_TASK_DB", filepath.Join(root, "outputs", "appshell", "task_history.sqlite"))

	app, err := NewApp(filepath.Join(root, "missing_engine.py"))
	if err != nil {
		t.Fatalf("NewApp failed: %v", err)
	}
	t.Cleanup(func() {
		app.shutdown(context.Background())
	})

	report, err := app.RunStartupChecks()
	if err != nil {
		t.Fatalf("RunStartupChecks failed: %v", err)
	}
	if report.CanEnter {
		t.Fatalf("expected startup checks to block entry")
	}
	if report.OverallStatus != "failed" {
		t.Fatalf("unexpected overall status: %s", report.OverallStatus)
	}

	engineItem := findStartupItem(t, report, "engine_script")
	if engineItem.Status != "fail" {
		t.Fatalf("expected engine_script fail, got %s", engineItem.Status)
	}

	_, runErr := app.RunTask(map[string]any{"action": "health"})
	if runErr == nil || !strings.Contains(runErr.Error(), startupChecksBlockedMessage) {
		t.Fatalf("expected startup gate error, got %v", runErr)
	}
}

func TestRunStartupChecksAutoCreatesOutputAndSQLiteDirs(t *testing.T) {
	root := withTempBackendWorkingDir(t)
	t.Setenv("APPSHELL_TASK_DB", filepath.Join(root, "outputs", "custom", "task_history.sqlite"))

	engineScript := writeStartupCheckEngineScript(t, root, "success")
	app, err := NewApp(engineScript)
	if err != nil {
		t.Fatalf("NewApp failed: %v", err)
	}
	t.Cleanup(func() {
		app.shutdown(context.Background())
	})

	report, err := app.RunStartupChecks()
	if err != nil {
		t.Fatalf("RunStartupChecks failed: %v", err)
	}
	if !report.CanEnter {
		t.Fatalf("expected startup checks to pass, got %+v", report)
	}
	if report.OverallStatus != "warning" {
		t.Fatalf("expected warning overall status because model artifacts are absent, got %s", report.OverallStatus)
	}

	sqliteItem := findStartupItem(t, report, "task_history_sqlite")
	if sqliteItem.Status != "pass" || !sqliteItem.AutoFixed {
		t.Fatalf("expected sqlite auto-fixed pass, got %+v", sqliteItem)
	}

	resultsItem := findStartupItem(t, report, "results_output_root")
	if resultsItem.Status != "pass" || !resultsItem.AutoFixed {
		t.Fatalf("expected results dir auto-fixed pass, got %+v", resultsItem)
	}

	modelItem := findStartupItem(t, report, "model_artifacts")
	if modelItem.Status != "warning" || modelItem.Blocking {
		t.Fatalf("expected non-blocking model warning, got %+v", modelItem)
	}

	if _, err := os.Stat(filepath.Join(root, "outputs", "custom")); err != nil {
		t.Fatalf("expected sqlite directory to exist: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "outputs", "results")); err != nil {
		t.Fatalf("expected results directory to exist: %v", err)
	}

	taskSnapshot, err := app.RunTask(map[string]any{"action": "health", "payload": map[string]any{}})
	if err != nil {
		t.Fatalf("RunTask after startup checks failed: %v", err)
	}
	if strings.TrimSpace(taskSnapshot.ID) == "" {
		t.Fatalf("expected submitted task id")
	}
	waitForTerminalTask(t, app, taskSnapshot.ID, 5*time.Second)
}

func TestRunStartupChecksPrewarmsLangGraphSidecarAsWarningOnlyPass(t *testing.T) {
	root := withTempBackendWorkingDir(t)
	t.Setenv("APPSHELL_TASK_DB", filepath.Join(root, "outputs", "custom", "task_history.sqlite"))
	t.Setenv("APPSHELL_LANGGRAPH_SCRIPT", writeStartupCheckLangGraphScript(t, root, "llm", "configured"))
	t.Setenv("APPSHELL_LANGGRAPH_PORT", fmt.Sprintf("%d", findFreePortForStartupTest(t)))

	engineScript := writeStartupCheckEngineScript(t, root, "success")
	app, err := NewApp(engineScript)
	if err != nil {
		t.Fatalf("NewApp failed: %v", err)
	}
	t.Cleanup(func() {
		app.shutdown(context.Background())
	})

	report, err := app.RunStartupChecks()
	if err != nil {
		t.Fatalf("RunStartupChecks failed: %v", err)
	}
	if !report.CanEnter {
		t.Fatalf("expected startup checks to remain enterable")
	}
	item := findStartupItem(t, report, "langgraph_sidecar")
	if item.Status != "pass" {
		t.Fatalf("expected langgraph sidecar pass, got %+v", item)
	}
	if item.Detail["planner_mode"] != "llm" {
		t.Fatalf("unexpected planner mode detail: %+v", item.Detail)
	}
	if item.Detail["llm_mode"] != "configured" {
		t.Fatalf("unexpected llm mode detail: %+v", item.Detail)
	}
	if item.Detail["provider"] != agent.CognitionProviderLangGraph || item.Detail["cognition_status"] != agent.CognitionStatusEngaged {
		t.Fatalf("expected engaged langgraph cognition detail, got %+v", item.Detail)
	}
	if item.Detail["fallback_active"] != false {
		t.Fatalf("expected fallback inactive, got %+v", item.Detail)
	}
}

func TestRunStartupChecksWarnsWhenLangGraphSidecarCannotStart(t *testing.T) {
	root := withTempBackendWorkingDir(t)
	t.Setenv("APPSHELL_TASK_DB", filepath.Join(root, "outputs", "custom", "task_history.sqlite"))
	t.Setenv("APPSHELL_LANGGRAPH_SCRIPT", filepath.Join(root, "missing_sidecar.py"))
	t.Setenv("APPSHELL_LANGGRAPH_PORT", fmt.Sprintf("%d", findFreePortForStartupTest(t)))

	engineScript := writeStartupCheckEngineScript(t, root, "success")
	app, err := NewApp(engineScript)
	if err != nil {
		t.Fatalf("NewApp failed: %v", err)
	}
	t.Cleanup(func() {
		app.shutdown(context.Background())
	})

	report, err := app.RunStartupChecks()
	if err != nil {
		t.Fatalf("RunStartupChecks failed: %v", err)
	}
	if !report.CanEnter {
		t.Fatalf("expected sidecar failure to remain warning-only")
	}
	item := findStartupItem(t, report, "langgraph_sidecar")
	if item.Status != "warning" {
		t.Fatalf("expected langgraph sidecar warning, got %+v", item)
	}
	if item.Detail["fallback_reason_code"] != agent.CognitionFallbackScriptMissing {
		t.Fatalf("expected script missing fallback detail, got %+v", item.Detail)
	}
	if item.Detail["provider"] != agent.CognitionProviderDeterministic {
		t.Fatalf("expected deterministic provider during fallback, got %+v", item.Detail)
	}
}

func TestRunStartupChecksWarnsWhenLangGraphReportsFallbackPlannerMode(t *testing.T) {
	root := withTempBackendWorkingDir(t)
	t.Setenv("APPSHELL_TASK_DB", filepath.Join(root, "outputs", "custom", "task_history.sqlite"))
	t.Setenv("APPSHELL_LANGGRAPH_SCRIPT", writeStartupCheckLangGraphScript(t, root, "fallback", "unavailable"))
	t.Setenv("APPSHELL_LANGGRAPH_PORT", fmt.Sprintf("%d", findFreePortForStartupTest(t)))

	engineScript := writeStartupCheckEngineScript(t, root, "success")
	app, err := NewApp(engineScript)
	if err != nil {
		t.Fatalf("NewApp failed: %v", err)
	}
	t.Cleanup(func() {
		app.shutdown(context.Background())
	})

	report, err := app.RunStartupChecks()
	if err != nil {
		t.Fatalf("RunStartupChecks failed: %v", err)
	}
	if !report.CanEnter {
		t.Fatalf("expected planner fallback to remain warning-only")
	}
	item := findStartupItem(t, report, "langgraph_sidecar")
	if item.Status != "warning" {
		t.Fatalf("expected warning status, got %+v", item)
	}
	if item.Detail["fallback_reason_code"] != agent.CognitionFallbackPlannerMode {
		t.Fatalf("expected planner mode fallback detail, got %+v", item.Detail)
	}
	if item.Detail["provider"] != agent.CognitionProviderDeterministic || item.Detail["cognition_status"] != agent.CognitionStatusFallback {
		t.Fatalf("expected deterministic fallback detail, got %+v", item.Detail)
	}
}

func TestRunStartupChecksReportsSQLiteFailure(t *testing.T) {
	root := withTempBackendWorkingDir(t)
	blockedParent := filepath.Join(root, "blocked-parent")
	if err := os.WriteFile(blockedParent, []byte("file"), 0o644); err != nil {
		t.Fatalf("write blocked file failed: %v", err)
	}
	t.Setenv("APPSHELL_TASK_DB", filepath.Join(blockedParent, "task_history.sqlite"))

	engineScript := writeStartupCheckEngineScript(t, root, "success")
	app, err := NewApp(engineScript)
	if err != nil {
		t.Fatalf("NewApp failed: %v", err)
	}
	t.Cleanup(func() {
		app.shutdown(context.Background())
	})

	report, err := app.RunStartupChecks()
	if err != nil {
		t.Fatalf("RunStartupChecks failed: %v", err)
	}
	if report.CanEnter {
		t.Fatalf("expected startup checks to block entry due to sqlite failure")
	}

	sqliteItem := findStartupItem(t, report, "task_history_sqlite")
	if sqliteItem.Status != "fail" {
		t.Fatalf("expected sqlite fail, got %+v", sqliteItem)
	}
}
