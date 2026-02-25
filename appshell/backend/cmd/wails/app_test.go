package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
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
