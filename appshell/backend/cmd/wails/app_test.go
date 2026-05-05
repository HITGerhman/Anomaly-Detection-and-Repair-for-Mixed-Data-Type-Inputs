package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
	"appshell/backend/internal/task"
)

func TestNormalizeRequestForTrainPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":     "train",
		"csv_path":   "data/raw/demo.csv",
		"target_col": "stroke",
		"output_dir": "outputs/results/mvp",
		"task_type":  "classification",
		"timeout_ms": 12000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}

	if req.Action != "train" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["csv_path"] != "data/raw/demo.csv" {
		t.Fatalf("unexpected csv_path: %v", req.Payload["csv_path"])
	}
	if req.Payload["target_col"] != "stroke" {
		t.Fatalf("unexpected target_col: %v", req.Payload["target_col"])
	}
	if req.Payload["output_dir"] != "outputs/results/mvp" {
		t.Fatalf("unexpected output_dir: %v", req.Payload["output_dir"])
	}
	if req.Payload["task_type"] != "classification" {
		t.Fatalf("unexpected task_type: %v", req.Payload["task_type"])
	}
	if timeout != 12*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestMissingTrainFields(t *testing.T) {
	_, _, err := normalizeRequest(map[string]any{
		"action": "train",
	})
	if err == nil {
		t.Fatalf("expected error for missing fields")
	}
}

func TestNormalizeRequestForRepairPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":       "repair",
		"model_dir":    "outputs/results/wails_mvp",
		"sample_index": 7,
		"dry_run":      true,
		"max_changes":  4,
		"k_neighbors":  12,
		"timeout_ms":   15000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}

	if req.Action != "repair" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["model_dir"] != "outputs/results/wails_mvp" {
		t.Fatalf("unexpected model_dir: %v", req.Payload["model_dir"])
	}
	if req.Payload["sample_index"] != 7 {
		t.Fatalf("unexpected sample_index: %v", req.Payload["sample_index"])
	}
	if req.Payload["dry_run"] != true {
		t.Fatalf("unexpected dry_run: %v", req.Payload["dry_run"])
	}
	if req.Payload["max_changes"] != 4 {
		t.Fatalf("unexpected max_changes: %v", req.Payload["max_changes"])
	}
	if req.Payload["k_neighbors"] != 12 {
		t.Fatalf("unexpected k_neighbors: %v", req.Payload["k_neighbors"])
	}
	if timeout != 15*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestForScanFilePayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":                          "scan_file",
		"csv_path":                        "data/raw/healthcare-dataset-stroke-data.csv",
		"max_bins":                        96,
		"max_issues":                      500,
		"numeric_iqr_factor":              2.0,
		"enable_time_series_shift":        true,
		"time_series_z_threshold":         4.5,
		"enable_cross_column_consistency": true,
		"consistency_rules": []map[string]any{
			{
				"type":      "lte",
				"left_col":  "start_day",
				"right_col": "end_day",
			},
		},
		"enable_duplicate_record": true,
		"duplicate_subset":        []string{"id", "name"},
		"timeout_ms":              8000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}

	if req.Action != "scan_file" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["csv_path"] != "data/raw/healthcare-dataset-stroke-data.csv" {
		t.Fatalf("unexpected csv_path: %v", req.Payload["csv_path"])
	}
	if req.Payload["max_bins"] != 96 {
		t.Fatalf("unexpected max_bins: %v", req.Payload["max_bins"])
	}
	if req.Payload["max_issues"] != 500 {
		t.Fatalf("unexpected max_issues: %v", req.Payload["max_issues"])
	}
	if req.Payload["numeric_iqr_factor"] != 2.0 {
		t.Fatalf("unexpected numeric_iqr_factor: %v", req.Payload["numeric_iqr_factor"])
	}
	if req.Payload["time_series_z_threshold"] != 4.5 {
		t.Fatalf("unexpected time_series_z_threshold: %v", req.Payload["time_series_z_threshold"])
	}
	if req.Payload["consistency_rules"] == nil {
		t.Fatalf("consistency_rules should be forwarded")
	}
	if req.Payload["duplicate_subset"] == nil {
		t.Fatalf("duplicate_subset should be forwarded")
	}
	if timeout != 8*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestForRepairBatchPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":          "repair_batch",
		"csv_path":        "data/raw/healthcare-dataset-stroke-data.csv",
		"issue_ids":       []string{"age::numeric_outlier", "bmi::missing_values"},
		"write_output":    false,
		"plan_only":       true,
		"enable_rollback": true,
		"rollback_dir":    "outputs/rollback",
		"repair_strategy": map[string]any{
			"conflict_policy": "last_wins",
		},
		"column_dependencies": map[string]any{
			"bmi": []string{"age"},
		},
		"max_issues":              1200,
		"time_series_z_threshold": 5.2,
		"duplicate_subset":        []string{"id", "age"},
		"scan_config": map[string]any{
			"max_bins": 88,
		},
		"timeout_ms": 20000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}

	if req.Action != "repair_batch" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["csv_path"] != "data/raw/healthcare-dataset-stroke-data.csv" {
		t.Fatalf("unexpected csv_path: %v", req.Payload["csv_path"])
	}
	if req.Payload["write_output"] != false {
		t.Fatalf("unexpected write_output: %v", req.Payload["write_output"])
	}
	if req.Payload["plan_only"] != true {
		t.Fatalf("unexpected plan_only: %v", req.Payload["plan_only"])
	}
	if req.Payload["enable_rollback"] != true {
		t.Fatalf("unexpected enable_rollback: %v", req.Payload["enable_rollback"])
	}
	if req.Payload["rollback_dir"] != "outputs/rollback" {
		t.Fatalf("unexpected rollback_dir: %v", req.Payload["rollback_dir"])
	}
	if req.Payload["repair_strategy"] == nil {
		t.Fatalf("repair_strategy should be forwarded")
	}
	if req.Payload["column_dependencies"] == nil {
		t.Fatalf("column_dependencies should be forwarded")
	}
	if req.Payload["max_issues"] != 1200 {
		t.Fatalf("unexpected max_issues: %v", req.Payload["max_issues"])
	}
	if req.Payload["time_series_z_threshold"] != 5.2 {
		t.Fatalf("unexpected time_series_z_threshold: %v", req.Payload["time_series_z_threshold"])
	}
	if req.Payload["duplicate_subset"] == nil {
		t.Fatalf("duplicate_subset should be forwarded")
	}
	if req.Payload["scan_config"] == nil {
		t.Fatalf("scan_config should be forwarded")
	}
	if timeout != 20*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestForRepairWithGowerPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":          "repair_with_gower",
		"csv_path":        "data/raw/healthcare-dataset-stroke-data.csv",
		"issue_ids":       []string{"age::numeric_outlier"},
		"write_output":    false,
		"plan_only":       true,
		"enable_rollback": true,
		"column_dependencies": map[string]any{
			"bmi": []string{"age"},
		},
		"gower_strategy": map[string]any{
			"k_neighbors": 7,
			"weight_mode": "uniform",
		},
		"model_dir":   "outputs/results/model",
		"scan_config": map[string]any{"max_bins": 88},
		"timeout_ms":  17000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}
	if req.Action != "repair_with_gower" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["gower_strategy"] == nil {
		t.Fatalf("gower_strategy should be forwarded")
	}
	if req.Payload["model_dir"] != "outputs/results/model" {
		t.Fatalf("unexpected model_dir: %v", req.Payload["model_dir"])
	}
	if timeout != 17*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestForRollbackRepairBatchPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":         "rollback_repair_batch",
		"manifest_path":  "outputs/results/.rollback/rb-1.json",
		"restore_target": "output_csv",
		"target_csv":     "outputs/results/recovered.csv",
		"timeout_ms":     30000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}

	if req.Action != "rollback_repair_batch" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["manifest_path"] != "outputs/results/.rollback/rb-1.json" {
		t.Fatalf("unexpected manifest_path: %v", req.Payload["manifest_path"])
	}
	if req.Payload["restore_target"] != "output_csv" {
		t.Fatalf("unexpected restore_target: %v", req.Payload["restore_target"])
	}
	if req.Payload["target_csv"] != "outputs/results/recovered.csv" {
		t.Fatalf("unexpected target_csv: %v", req.Payload["target_csv"])
	}
	if timeout != 30*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestMissingRepairModelDir(t *testing.T) {
	_, _, err := normalizeRequest(map[string]any{
		"action": "repair",
	})
	if err == nil {
		t.Fatalf("expected error for missing model_dir")
	}
}

func TestNormalizeRequestUsesNestedPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action": "health",
		"payload": map[string]any{
			"x": 1,
		},
		"timeout_ms": "3s",
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}
	if req.Action != "health" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["x"] != 1 {
		t.Fatalf("unexpected nested payload: %v", req.Payload["x"])
	}
	if timeout != 3*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestDefaultsToStableTrainAction(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"csv_path":   "data/raw/demo.csv",
		"target_col": "stroke",
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}
	if req.Action != string(engine.ActionTrain) {
		t.Fatalf("unexpected default action: %s", req.Action)
	}
	if timeout != 90*time.Second {
		t.Fatalf("unexpected default timeout: %s", timeout)
	}
}

func TestEntrypointsReferenceEngineActionConstants(t *testing.T) {
	appSource, err := os.ReadFile("app.go")
	if err != nil {
		t.Fatalf("read app.go failed: %v", err)
	}
	appText := string(appSource)
	for _, token := range []string{
		"engine.ActionTrain",
		"engine.ActionRepairBatch",
		"engine.ActionRepairWithGower",
		"engine.ActionRollbackRepairBatch",
	} {
		if !strings.Contains(appText, token) {
			t.Fatalf("app.go should reference %s", token)
		}
	}

	demoSource, err := os.ReadFile(filepath.Join("..", "demo", "main.go"))
	if err != nil {
		t.Fatalf("read demo main failed: %v", err)
	}
	demoText := string(demoSource)
	for _, token := range []string{
		"engine.ActionHealth",
		"engine.ActionTrain",
		"engine.ActionRepair",
		"engine.ActionRepairWithGower",
	} {
		if !strings.Contains(demoText, token) {
			t.Fatalf("demo main.go should reference %s", token)
		}
	}
}

func TestHistoryKeepFromEnv(t *testing.T) {
	t.Setenv("APPSHELL_TASK_HISTORY_KEEP", "25")
	if got := historyKeepFromEnv(100); got != 25 {
		t.Fatalf("expected 25, got %d", got)
	}

	t.Setenv("APPSHELL_TASK_HISTORY_KEEP", "invalid")
	if got := historyKeepFromEnv(100); got != 100 {
		t.Fatalf("invalid value should fallback, got %d", got)
	}
}

func TestResolveTaskDBPathUsesEnv(t *testing.T) {
	t.Setenv("APPSHELL_TASK_DB", "outputs/custom/tasks.sqlite")
	got, err := resolveTaskDBPath()
	if err != nil {
		t.Fatalf("resolveTaskDBPath failed: %v", err)
	}
	if filepath.Base(got) != "tasks.sqlite" {
		t.Fatalf("unexpected db file: %s", got)
	}
}

func TestListCSVColumnsFromFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.csv")
	content := "id, name ,stroke,stroke,\n1,a,0,0,\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
	}

	cols, err := listCSVColumnsFromFile(path)
	if err != nil {
		t.Fatalf("listCSVColumnsFromFile failed: %v", err)
	}
	if len(cols) != 4 {
		t.Fatalf("expected 4 unique columns, got %d (%v)", len(cols), cols)
	}
	if cols[0] != "id" || cols[1] != "name" || cols[2] != "stroke" || cols[3] != "column_5" {
		t.Fatalf("unexpected columns: %v", cols)
	}
}

func TestListCSVColumnsFromFileEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.csv")
	if err := os.WriteFile(path, []byte(""), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
	}

	if _, err := listCSVColumnsFromFile(path); err == nil {
		t.Fatalf("expected error for empty csv")
	}
}

func TestResolveExistingFilePathSupportsProjectRelativeCSVPath(t *testing.T) {
	root := t.TempDir()
	backendDir := filepath.Join(root, "appshell", "backend")
	dataDir := filepath.Join(root, "data", "raw")
	if err := os.MkdirAll(backendDir, 0o755); err != nil {
		t.Fatalf("mkdir backend dir failed: %v", err)
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("mkdir data dir failed: %v", err)
	}

	csvPath := filepath.Join(dataDir, "demo.csv")
	if err := os.WriteFile(csvPath, []byte("a,b,c\n1,2,3\n"), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
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

	resolved, err := resolveExistingFilePath("data/raw/demo.csv")
	if err != nil {
		t.Fatalf("resolveExistingFilePath failed: %v", err)
	}
	if resolved != csvPath {
		t.Fatalf("unexpected resolved path: got=%s want=%s", resolved, csvPath)
	}
}

type fakeAgentAppRunner struct {
	mu               sync.Mutex
	observer         engine.StderrObserver
	validationCanRun bool
	tempDir          string
	sourceCSV        string
	postScanMode     string
	highRiskColumns  []string
}

func (r *fakeAgentAppRunner) SetStderrObserver(observer engine.StderrObserver) {
	r.observer = observer
}

func (r *fakeAgentAppRunner) Run(_ context.Context, req engine.Request) (engine.Response, error) {
	if r.observer != nil {
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     "load_csv",
				"phase":     "start",
				"progress":  10,
				"message":   "tool started",
				"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
			},
			ObservedAt: time.Now(),
		})
	}

	switch req.Action {
	case string(engine.ActionScanFile):
		csvPath := asString(req.Payload["csv_path"])
		if strings.TrimSpace(r.sourceCSV) != "" && filepath.Clean(csvPath) != filepath.Clean(r.sourceCSV) {
			return engine.Response{
				TaskID: req.TaskID,
				Status: "ok",
				Result: map[string]any{
					"issue_count": 1,
					"issues": []any{
						map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "column": "city", "risk_level": "low", "issue_score": 0.1},
					},
					"scan_summary": map[string]any{"total_issues": 1, "high_risk_columns": []string{}},
					"data_profile": map[string]any{"rows": 10},
				},
			}, nil
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"issue_count": 2,
				"issues": []any{
					map[string]any{"issue_id": "issue-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.7},
					map[string]any{"issue_id": "issue-2", "issue_type": "rare_category", "column": "city", "risk_level": "medium", "issue_score": 0.3},
				},
				"scan_summary": map[string]any{"total_issues": 2, "high_risk_columns": append([]string{}, r.highRiskColumns...)},
				"data_profile": map[string]any{"rows": 10},
			},
		}, nil
	case string(engine.ActionRepairBatch):
		planOnly, _ := req.Payload["plan_only"].(bool)
		if planOnly {
			if !r.validationCanRun {
				return engine.Response{
					TaskID: req.TaskID,
					Status: "ok",
					Result: map[string]any{
						"comparison": map[string]any{
							"before_issue_count":   2,
							"after_issue_count":    2,
							"resolved_issue_count": 0,
							"changed_cell_count":   0,
						},
						"skipped_issues": []any{
							map[string]any{"issue_id": "issue-1", "reason": "strategy_disabled"},
							map[string]any{"issue_id": "issue-2", "reason": "strategy_disabled"},
						},
					},
				}, nil
			}
			return engine.Response{
				TaskID: req.TaskID,
				Status: "ok",
				Result: map[string]any{
					"comparison": map[string]any{
						"before_issue_count":   2,
						"after_issue_count":    1,
						"resolved_issue_count": 1,
						"changed_cell_count":   1,
					},
					"applied_repairs": []any{
						map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.82},
					},
				},
			}, nil
		}
		outputCSV, rollback := r.writeExecutionArtifacts(req, "rule")
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"output_csv":          outputCSV,
				"applied_issue_count": 1,
				"rollback":            rollback,
				"comparison": map[string]any{
					"before_issue_count":   2,
					"after_issue_count":    1,
					"resolved_issue_count": 1,
					"changed_cell_count":   1,
				},
			},
		}, nil
	case string(engine.ActionRepairWithGower):
		planOnly, _ := req.Payload["plan_only"].(bool)
		if planOnly {
			if !r.validationCanRun {
				return engine.Response{
					TaskID: req.TaskID,
					Status: "ok",
					Result: map[string]any{
						"comparison": map[string]any{
							"before_issue_count":   2,
							"after_issue_count":    2,
							"resolved_issue_count": 0,
							"changed_cell_count":   0,
						},
						"skipped_issues": []any{
							map[string]any{"issue_id": "issue-1", "reason": "no_healthy_neighbors"},
							map[string]any{"issue_id": "issue-2", "reason": "no_healthy_neighbors"},
						},
					},
				}, nil
			}
			return engine.Response{
				TaskID: req.TaskID,
				Status: "ok",
				Result: map[string]any{
					"comparison": map[string]any{
						"before_issue_count":   2,
						"after_issue_count":    0,
						"resolved_issue_count": 2,
						"changed_cell_count":   2,
					},
					"applied_repairs": []any{
						map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.91},
						map[string]any{"issue_id": "issue-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.87},
					},
					"neighbor_evidence": []any{
						map[string]any{"issue_id": "issue-1", "candidate_confidence": 0.91},
						map[string]any{"issue_id": "issue-2", "candidate_confidence": 0.87},
					},
				},
			}, nil
		}
		outputCSV, rollback := r.writeExecutionArtifacts(req, "gower")
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"output_csv":          outputCSV,
				"applied_issue_count": 2,
				"rollback":            rollback,
				"comparison": map[string]any{
					"before_issue_count":   2,
					"after_issue_count":    0,
					"resolved_issue_count": 2,
					"changed_cell_count":   2,
				},
			},
		}, nil
	case string(engine.ActionRollbackRepairBatch):
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{"restore_target": asString(req.Payload["restore_target"])},
		}, nil
	default:
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{"action": req.Action},
		}, nil
	}
}

func (r *fakeAgentAppRunner) writeExecutionArtifacts(req engine.Request, source string) (string, map[string]any) {
	outputCSV := asString(req.Payload["output_csv"])
	if outputCSV == "" {
		baseDir := r.tempDir
		if baseDir == "" {
			baseDir = filepath.Join("outputs", "results")
		}
		outputCSV = filepath.Join(baseDir, source+".repaired.csv")
	}
	_ = os.MkdirAll(filepath.Dir(outputCSV), 0o755)
	content := []byte("age,city\n20,a\n")
	if csvPath := asString(req.Payload["csv_path"]); csvPath != "" {
		if data, err := os.ReadFile(csvPath); err == nil {
			content = data
		}
	}
	_ = os.WriteFile(outputCSV, content, 0o644)

	rollbackDir := asString(req.Payload["rollback_dir"])
	if rollbackDir == "" {
		rollbackDir = filepath.Join(filepath.Dir(outputCSV), ".rollback")
	}
	_ = os.MkdirAll(rollbackDir, 0o755)
	baseName := strings.TrimSuffix(filepath.Base(outputCSV), filepath.Ext(outputCSV))
	manifestPath := filepath.Join(rollbackDir, baseName+".json")
	backupCSV := filepath.Join(rollbackDir, baseName+".backup.csv")
	_ = os.WriteFile(manifestPath, []byte("{}"), 0o644)
	_ = os.WriteFile(backupCSV, content, 0o644)

	return outputCSV, map[string]any{
		"rollback_id":      baseName,
		"manifest_path":    manifestPath,
		"backup_csv":       backupCSV,
		"manifest_version": 2,
	}
}

func TestRunAgentSessionAndReadSessionArtifacts(t *testing.T) {
	store, err := agent.NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	runner := agent.NewRuntimeRunner(&fakeAgentAppRunner{}, store, agent.NewMockPlanner())
	svc := task.NewServiceWithConfig(runner, task.Config{MaxConcurrency: 1, QueueSize: 8})
	defer svc.Close()

	app := &App{
		service:    svc,
		agentStore: store,
	}
	csvPath := filepath.Join(t.TempDir(), "demo.csv")
	if err := os.WriteFile(csvPath, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
	}

	snapshot, err := app.RunAgentSession(map[string]any{
		"csv_path": csvPath,
	})
	if err != nil {
		t.Fatalf("RunAgentSession failed: %v", err)
	}
	waitForTerminalTask(t, app, snapshot.ID, 5*time.Second)

	finalSnapshot, err := app.GetTaskStatus(snapshot.ID)
	if err != nil {
		t.Fatalf("GetTaskStatus failed: %v", err)
	}
	agentBlock := finalSnapshot.Response.Result["agent"].(map[string]any)
	sessionID := agentBlock["session_id"].(string)

	session, err := app.GetAgentSession(sessionID)
	if err != nil {
		t.Fatalf("GetAgentSession failed: %v", err)
	}
	if session.LatestPlan.PlanID == "" {
		t.Fatalf("expected saved plan id")
	}
	if len(session.Presentation) == 0 {
		t.Fatalf("expected session presentation")
	}
	if session.PresentationArtifact != "" && filepath.Base(session.PresentationArtifact) != "presentation.json" {
		t.Fatalf("unexpected session presentation artifact: %s", session.PresentationArtifact)
	}
	if session.TraceSummary.Cognition.EventCount == 0 || session.TraceSummary.Cognition.Status == "" {
		t.Fatalf("expected cognition trace summary, got %+v", session.TraceSummary.Cognition)
	}

	trace, err := app.ListAgentTrace(sessionID)
	if err != nil {
		t.Fatalf("ListAgentTrace failed: %v", err)
	}
	if len(trace) == 0 {
		t.Fatalf("expected non-empty agent trace")
	}
}

func TestExecuteAgentPlanReturnsTaskSnapshot(t *testing.T) {
	store, err := agent.NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	runner := agent.NewRuntimeRunner(&fakeAgentAppRunner{validationCanRun: true}, store, agent.NewMockPlanner())
	svc := task.NewServiceWithConfig(runner, task.Config{MaxConcurrency: 1, QueueSize: 8})
	defer svc.Close()

	app := &App{
		service:    svc,
		agentStore: store,
	}
	csvPath := filepath.Join(t.TempDir(), "demo.csv")
	if err := os.WriteFile(csvPath, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
	}

	planTask, err := app.RunAgentSession(map[string]any{
		"csv_path": csvPath,
	})
	if err != nil {
		t.Fatalf("RunAgentSession failed: %v", err)
	}
	waitForTerminalTask(t, app, planTask.ID, 5*time.Second)

	planned, err := app.GetTaskStatus(planTask.ID)
	if err != nil {
		t.Fatalf("GetTaskStatus failed: %v", err)
	}
	agentBlock := planned.Response.Result["agent"].(map[string]any)
	sessionID := agentBlock["session_id"].(string)
	plan := agentBlock["plan"].(agent.AgentPlan)

	executeTask, err := app.ExecuteAgentPlan(map[string]any{
		"session_id": sessionID,
		"plan_id":    plan.PlanID,
	})
	if err != nil {
		t.Fatalf("ExecuteAgentPlan failed: %v", err)
	}
	waitForTerminalTask(t, app, executeTask.ID, 5*time.Second)

	executed, err := app.GetTaskStatus(executeTask.ID)
	if err != nil {
		t.Fatalf("GetTaskStatus failed: %v", err)
	}
	if executed.Status != task.StatusSucceeded {
		t.Fatalf("expected execute task succeeded, got %s", executed.Status)
	}
}

func TestRunAgentAutofixSessionReturnsTaskSnapshot(t *testing.T) {
	tempDir := t.TempDir()
	store, err := agent.NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	csvPath := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(csvPath, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
	}

	runner := agent.NewRuntimeRunner(&fakeAgentAppRunner{
		validationCanRun: true,
		tempDir:          tempDir,
		sourceCSV:        csvPath,
	}, store, agent.NewMockPlanner())
	svc := task.NewServiceWithConfig(runner, task.Config{MaxConcurrency: 1, QueueSize: 8})
	defer svc.Close()

	app := &App{
		service:    svc,
		agentStore: store,
	}

	snapshot, err := app.RunAgentAutofixSession(map[string]any{
		"csv_path": csvPath,
	})
	if err != nil {
		t.Fatalf("RunAgentAutofixSession failed: %v", err)
	}
	waitForTerminalTask(t, app, snapshot.ID, 5*time.Second)

	finalSnapshot, err := app.GetTaskStatus(snapshot.ID)
	if err != nil {
		t.Fatalf("GetTaskStatus failed: %v", err)
	}
	if finalSnapshot.Status != task.StatusSucceeded {
		t.Fatalf("expected autofix task succeeded, got %s", finalSnapshot.Status)
	}

	safety, _ := finalSnapshot.Response.Result["safety"].(map[string]any)
	if asString(safety["final_verdict"]) != "accepted" {
		t.Fatalf("expected accepted verdict, got %v", safety["final_verdict"])
	}

	agentBlock, ok := finalSnapshot.Response.Result["agent"].(map[string]any)
	if !ok {
		t.Fatalf("expected agent block in autofix response")
	}
	plan, ok := agentBlock["plan"].(agent.AgentPlan)
	if !ok {
		t.Fatalf("expected typed agent plan, got %T", agentBlock["plan"])
	}
	if len(plan.AutoRepairIssueIDs) == 0 {
		t.Fatalf("expected A2 auto repair issue bucket in Wails response plan")
	}
	if len(plan.SelectedIssueIDs) != len(plan.AutoRepairIssueIDs) {
		t.Fatalf("selected issues should match auto bucket for A6 autofix wiring: selected=%v auto=%v", plan.SelectedIssueIDs, plan.AutoRepairIssueIDs)
	}
	validation, ok := agentBlock["validation"].(map[string]any)
	if !ok {
		t.Fatalf("expected validation block in autofix response")
	}
	postValidation, ok := validation["post_execute"].(map[string]any)
	if !ok {
		t.Fatalf("expected post_execute validation block in autofix response")
	}
	if asString(postValidation["verdict"]) == "" {
		t.Fatalf("expected A4 validation gate verdict, got %#v", postValidation)
	}
	if _, ok := postValidation["total_cells_modified"]; !ok {
		t.Fatalf("expected A4 total_cells_modified in post_execute validation")
	}

	sessionID := asString(agentBlock["session_id"])
	session, err := app.GetAgentSession(sessionID)
	if err != nil {
		t.Fatalf("GetAgentSession failed for autofix session: %v", err)
	}
	if session.LatestPlan.PlanID != plan.PlanID || len(session.LatestPlan.AutoRepairIssueIDs) == 0 {
		t.Fatalf("expected session snapshot to preserve latest A2 plan buckets, got %+v", session.LatestPlan)
	}
	if len(session.Presentation) == 0 {
		t.Fatalf("expected session presentation for Wails autofix result")
	}
	trace, err := app.ListAgentTrace(sessionID)
	if err != nil {
		t.Fatalf("ListAgentTrace failed for autofix session: %v", err)
	}
	hasPostValidationTrace := false
	for _, event := range trace {
		if event.TraceType == agent.TraceValidation && asString(event.Payload["phase"]) == "post_execute" && asString(event.Payload["verdict"]) != "" {
			hasPostValidationTrace = true
			break
		}
	}
	if !hasPostValidationTrace {
		t.Fatalf("expected post_execute validation verdict in Wails trace")
	}
}

func TestGetAndSaveAgentPreferences(t *testing.T) {
	store, err := agent.NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	app := &App{
		service:    &task.Service{},
		agentStore: store,
	}
	csvPath := filepath.Join(t.TempDir(), "demo.csv")

	record, err := app.GetAgentPreferences("", csvPath)
	if err != nil {
		t.Fatalf("GetAgentPreferences failed: %v", err)
	}
	if record.WorkspaceID == "" {
		t.Fatalf("expected resolved workspace id")
	}
	if !record.Profile.AvoidTimeColumns || !record.Profile.RequireApprovalForHighRisk {
		t.Fatalf("expected default profile, got %#v", record.Profile)
	}

	saved, err := app.SaveAgentPreferences(map[string]any{
		"csv_path": csvPath,
		"profile": map[string]any{
			"conservative_mode":              true,
			"avoid_time_columns":             false,
			"protected_columns":              []string{"id", "created_at"},
			"require_approval_for_high_risk": false,
		},
	})
	if err != nil {
		t.Fatalf("SaveAgentPreferences failed: %v", err)
	}
	if !saved.Profile.ConservativeMode || saved.Profile.AvoidTimeColumns {
		t.Fatalf("unexpected saved profile: %#v", saved.Profile)
	}

	loaded, err := app.GetAgentPreferences("", csvPath)
	if err != nil {
		t.Fatalf("GetAgentPreferences after save failed: %v", err)
	}
	if !loaded.Profile.ConservativeMode || loaded.Profile.RequireApprovalForHighRisk {
		t.Fatalf("unexpected loaded profile: %#v", loaded.Profile)
	}
}

func TestApproveAgentSessionReturnsTaskSnapshot(t *testing.T) {
	tempDir := t.TempDir()
	store, err := agent.NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	csvPath := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(csvPath, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write csv failed: %v", err)
	}

	runner := agent.NewRuntimeRunner(&fakeAgentAppRunner{
		validationCanRun: true,
		tempDir:          tempDir,
		sourceCSV:        csvPath,
		highRiskColumns:  []string{"age"},
	}, store, agent.NewMockPlanner())
	svc := task.NewServiceWithConfig(runner, task.Config{MaxConcurrency: 1, QueueSize: 8})
	defer svc.Close()

	app := &App{
		service:    svc,
		agentStore: store,
	}

	paused, err := app.RunAgentAutofixSession(map[string]any{
		"csv_path": csvPath,
	})
	if err != nil {
		t.Fatalf("RunAgentAutofixSession failed: %v", err)
	}
	waitForTerminalTask(t, app, paused.ID, 5*time.Second)

	pausedTask, err := app.GetTaskStatus(paused.ID)
	if err != nil {
		t.Fatalf("GetTaskStatus failed: %v", err)
	}
	agentBlock := pausedTask.Response.Result["agent"].(map[string]any)
	sessionID := agentBlock["session_id"].(string)
	plan := agentBlock["plan"].(agent.AgentPlan)
	approval := agentBlock["approval"].(map[string]any)
	if asString(approval["status"]) != "required" {
		t.Fatalf("expected approval status required, got %#v", approval)
	}

	resumed, err := app.ApproveAgentSession(map[string]any{
		"session_id": sessionID,
		"plan_id":    plan.PlanID,
		"decision":   "approve",
	})
	if err != nil {
		t.Fatalf("ApproveAgentSession failed: %v", err)
	}
	waitForTerminalTask(t, app, resumed.ID, 5*time.Second)

	finalTask, err := app.GetTaskStatus(resumed.ID)
	if err != nil {
		t.Fatalf("GetTaskStatus resume failed: %v", err)
	}
	if finalTask.Status != task.StatusSucceeded {
		t.Fatalf("expected approve task succeeded, got %s", finalTask.Status)
	}
	safety := finalTask.Response.Result["safety"].(map[string]any)
	if asString(safety["final_verdict"]) != "accepted" {
		t.Fatalf("expected accepted verdict, got %v", safety["final_verdict"])
	}
}
