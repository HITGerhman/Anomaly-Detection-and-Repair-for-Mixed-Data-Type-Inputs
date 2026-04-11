package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"appshell/backend/internal/engine"
)

type fakeRuntimeBaseRunner struct {
	mu               sync.Mutex
	observer         engine.StderrObserver
	calls            []engine.Request
	validationCanRun bool
	baselineIssues   []map[string]any
	highRiskColumns  []string
	columnProfiles   []map[string]any
}

type spyPlanner struct {
	mu         sync.Mutex
	called     bool
	lastInput  PlanningInput
	buildCount int
}

func (p *spyPlanner) BuildPlan(_ context.Context, input PlanningInput) (AgentPlan, error) {
	p.mu.Lock()
	p.called = true
	p.buildCount++
	p.lastInput = PlanningInput{
		SessionID:               input.SessionID,
		Goal:                    input.Goal,
		CSVPath:                 input.CSVPath,
		ScanResult:              cloneMap(input.ScanResult),
		SelectedIssueIDs:        append([]string{}, input.SelectedIssueIDs...),
		SkippedIssues:           cloneSkippedIssues(input.SkippedIssues),
		RulePreview:             cloneMap(input.RulePreview),
		GowerPreview:            cloneMap(input.GowerPreview),
		ScanConfigOverrides:     cloneMap(input.ScanConfigOverrides),
		RepairStrategyOverrides: cloneMap(input.RepairStrategyOverrides),
		ColumnDependencies:      cloneMap(input.ColumnDependencies),
		GowerStrategyOverrides:  cloneMap(input.GowerStrategyOverrides),
		ModelDir:                input.ModelDir,
		OutputDir:               input.OutputDir,
	}
	p.mu.Unlock()

	candidate := RepairCandidate{
		CandidateID:      "candidate-spy",
		Source:           "rule",
		ToolSequence:     []string{"engine.repair_batch"},
		PlanPayloads:     []map[string]any{{"csv_path": input.CSVPath, "issue_ids": append([]string{}, input.SelectedIssueIDs...), "plan_only": true, "write_output": false, "enable_rollback": false}},
		ExecutePayloads:  []map[string]any{{"csv_path": input.CSVPath, "issue_ids": append([]string{}, input.SelectedIssueIDs...), "plan_only": false, "write_output": true, "enable_rollback": true}},
		SelectedIssueIDs: append([]string{}, input.SelectedIssueIDs...),
		IssueSourceMap:   map[string]any{},
		Comparison:       cloneMap(mapFromAny(input.RulePreview["comparison"])),
		Summary:          "spy planner candidate",
		Executable:       true,
	}

	return AgentPlan{
		PlanID:              "plan-spy",
		Status:              "planned",
		SelectedIssueIDs:    append([]string{}, input.SelectedIssueIDs...),
		SkippedIssues:       cloneSkippedIssues(input.SkippedIssues),
		Candidates:          []RepairCandidate{candidate},
		SelectedCandidateID: candidate.CandidateID,
		SelectedSource:      candidate.Source,
		IssueSourceMap:      map[string]any{},
		ProposedToolID:      "engine.repair_batch",
		ProposedPayload:     cloneMap(candidate.ExecutePayloads[0]),
		ReasoningSummary:    "spy planner",
		UserExplanation:     "spy planner",
	}, nil
}

func (r *fakeRuntimeBaseRunner) SetStderrObserver(observer engine.StderrObserver) {
	r.observer = observer
}

func (r *fakeRuntimeBaseRunner) Run(_ context.Context, req engine.Request) (engine.Response, error) {
	r.mu.Lock()
	r.calls = append(r.calls, req)
	r.mu.Unlock()

	if r.observer != nil {
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     "load_csv",
				"phase":     "start",
				"progress":  20,
				"message":   "tool started",
				"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
			},
			ObservedAt: time.Now(),
		})
	}

	switch req.Action {
	case string(engine.ActionHealth):
		return engine.Response{TaskID: req.TaskID, Status: "ok", Result: map[string]any{"ok": true}}, nil
	case string(engine.ActionScanFile):
		issues := r.baselineIssues
		if len(issues) == 0 {
			issues = []map[string]any{
				{"issue_id": "i-1", "issue_type": "missing_values", "column": "age"},
				{"issue_id": "i-2", "issue_type": "numeric_outlier", "column": "bmi"},
				{"issue_id": "i-3", "issue_type": "duplicate_record", "column": "id"},
			}
		}
		issueItems := make([]any, 0, len(issues))
		for _, issue := range issues {
			issueItems = append(issueItems, cloneMap(issue))
		}
		result := map[string]any{
			"issue_count": len(issues),
			"issues":      issueItems,
			"scan_summary": map[string]any{
				"total_issues":      len(issues),
				"high_risk_columns": append([]string{}, r.highRiskColumns...),
			},
			"data_profile": map[string]any{"rows": 10, "columns": 3},
		}
		if len(r.columnProfiles) > 0 {
			result["column_profiles"] = cloneValue(r.columnProfiles)
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: result,
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
							"before_issue_count":   3,
							"after_issue_count":    3,
							"resolved_issue_count": 0,
							"changed_cell_count":   0,
						},
						"skipped_issues": []any{
							map[string]any{"issue_id": "i-1", "reason": "strategy_disabled"},
							map[string]any{"issue_id": "i-2", "reason": "strategy_disabled"},
						},
					},
				}, nil
			}
			return engine.Response{
				TaskID: req.TaskID,
				Status: "ok",
				Result: map[string]any{
					"comparison": map[string]any{
						"before_issue_count":   3,
						"after_issue_count":    1,
						"resolved_issue_count": 2,
						"changed_cell_count":   2,
					},
					"applied_repairs": []any{
						map[string]any{"issue_id": "i-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.7},
						map[string]any{"issue_id": "i-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.75},
					},
				},
			}, nil
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"output_csv":          "outputs/results/demo.repaired.csv",
				"applied_issue_count": 2,
				"rollback": map[string]any{
					"manifest_path": "outputs/results/.rollback/demo.json",
				},
				"comparison": map[string]any{
					"before_issue_count":   3,
					"after_issue_count":    1,
					"resolved_issue_count": 2,
					"changed_cell_count":   2,
				},
			},
		}, nil
	case string(engine.ActionRepairWithGower):
		planOnly, _ := req.Payload["plan_only"].(bool)
		if planOnly {
			if r.validationCanRun {
				return engine.Response{
					TaskID: req.TaskID,
					Status: "ok",
					Result: map[string]any{
						"comparison": map[string]any{
							"before_issue_count":   3,
							"after_issue_count":    1,
							"resolved_issue_count": 2,
							"changed_cell_count":   2,
						},
						"applied_repairs": []any{
							map[string]any{"issue_id": "i-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.92},
							map[string]any{"issue_id": "i-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.88},
						},
						"neighbor_evidence": []any{
							map[string]any{"issue_id": "i-1", "candidate_confidence": 0.92},
							map[string]any{"issue_id": "i-2", "candidate_confidence": 0.88},
						},
					},
				}, nil
			}
			return engine.Response{
				TaskID: req.TaskID,
				Status: "ok",
				Result: map[string]any{
					"comparison": map[string]any{
						"before_issue_count":   3,
						"after_issue_count":    3,
						"resolved_issue_count": 0,
						"changed_cell_count":   0,
					},
					"skipped_issues": []any{
						map[string]any{"issue_id": "i-1", "reason": "no_healthy_neighbors"},
						map[string]any{"issue_id": "i-2", "reason": "no_healthy_neighbors"},
					},
				},
			}, nil
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"output_csv":          "outputs/results/demo.gower.repaired.csv",
				"applied_issue_count": 2,
				"rollback": map[string]any{
					"manifest_path": "outputs/results/.rollback/demo-gower.json",
				},
				"comparison": map[string]any{
					"before_issue_count":   3,
					"after_issue_count":    1,
					"resolved_issue_count": 2,
					"changed_cell_count":   2,
				},
			},
		}, nil
	default:
		return engine.Response{TaskID: req.TaskID, Status: "ok", Result: map[string]any{"action": req.Action}}, nil
	}
}

func (r *fakeRuntimeBaseRunner) callCount(action string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	total := 0
	for _, call := range r.calls {
		if call.Action == action {
			total++
		}
	}
	return total
}

func (r *fakeRuntimeBaseRunner) executionCallCount(action string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	total := 0
	for _, call := range r.calls {
		if call.Action != action {
			continue
		}
		planOnly, _ := call.Payload["plan_only"].(bool)
		if !planOnly {
			total++
		}
	}
	return total
}

type fakeAutoRuntimeBaseRunner struct {
	mu                   sync.Mutex
	observer             engine.StderrObserver
	tempDir              string
	sourceCSV            string
	previewCanExecute    bool
	postScanMode         string
	rollbackFails        bool
	omitRollbackMetadata bool
	calls                []engine.Request
	baselineIssues       []map[string]any
	highRiskColumns      []string
	columnProfiles       []map[string]any
}

func (r *fakeAutoRuntimeBaseRunner) SetStderrObserver(observer engine.StderrObserver) {
	r.observer = observer
}

func (r *fakeAutoRuntimeBaseRunner) Run(_ context.Context, req engine.Request) (engine.Response, error) {
	r.mu.Lock()
	r.calls = append(r.calls, req)
	r.mu.Unlock()

	if r.observer != nil {
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     "load_csv",
				"phase":     "start",
				"progress":  20,
				"message":   "tool started",
				"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
			},
			ObservedAt: time.Now(),
		})
	}

	switch req.Action {
	case string(engine.ActionScanFile):
		return r.scanResponse(req), nil
	case string(engine.ActionRepairBatch):
		return r.repairResponse(req, "rule"), nil
	case string(engine.ActionRepairWithGower):
		return r.repairResponse(req, "gower"), nil
	case string(engine.ActionRollbackRepairBatch):
		if r.rollbackFails {
			return engine.Response{
				TaskID: req.TaskID,
				Status: "error",
				Error: &engine.ErrorBody{
					Code:    "ROLLBACK_FAILED",
					Message: "rollback failed",
				},
			}, nil
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"restored_csv":   asString(req.Payload["target_csv"]),
				"restore_target": asString(req.Payload["restore_target"]),
			},
		}, nil
	default:
		return engine.Response{TaskID: req.TaskID, Status: "ok", Result: map[string]any{"action": req.Action}}, nil
	}
}

func (r *fakeAutoRuntimeBaseRunner) callCount(action string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	total := 0
	for _, call := range r.calls {
		if call.Action == action {
			total++
		}
	}
	return total
}

func (r *fakeAutoRuntimeBaseRunner) executionCallCount(action string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	total := 0
	for _, call := range r.calls {
		if call.Action != action {
			continue
		}
		planOnly, _ := call.Payload["plan_only"].(bool)
		if !planOnly {
			total++
		}
	}
	return total
}

func (r *fakeAutoRuntimeBaseRunner) baselineScanResult() map[string]any {
	issues := r.baselineIssues
	if len(issues) == 0 {
		issues = []map[string]any{
			{"issue_id": "issue-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.7},
			{"issue_id": "issue-2", "issue_type": "rare_category", "column": "city", "risk_level": "medium", "issue_score": 0.3},
		}
	}
	issueItems := make([]any, 0, len(issues))
	for _, issue := range issues {
		issueItems = append(issueItems, cloneMap(issue))
	}
	result := map[string]any{
		"issue_count": len(issues),
		"issues":      issueItems,
		"scan_summary": map[string]any{
			"total_issues":      len(issues),
			"high_risk_columns": append([]string{}, r.highRiskColumns...),
		},
		"data_profile": map[string]any{"rows": 10, "columns": 2},
	}
	if len(r.columnProfiles) > 0 {
		result["column_profiles"] = cloneValue(r.columnProfiles)
	}
	return result
}

func (r *fakeAutoRuntimeBaseRunner) postScanResult() engine.Response {
	switch r.postScanMode {
	case "reject":
		return engine.Response{
			Status: "ok",
			Result: map[string]any{
				"issue_count": 3,
				"issues": []any{
					map[string]any{"issue_id": "post-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.8},
					map[string]any{"issue_id": "post-2", "issue_type": "rare_category", "column": "city", "risk_level": "high", "issue_score": 0.5},
					map[string]any{"issue_id": "post-3", "issue_type": "duplicate_record", "column": "id", "risk_level": "medium", "issue_score": 0.2},
				},
				"scan_summary": map[string]any{"total_issues": 3},
				"data_profile": map[string]any{"rows": 10, "columns": 2},
			},
		}
	case "error":
		return engine.Response{
			Status: "error",
			Error: &engine.ErrorBody{
				Code:    "POST_SCAN_FAILED",
				Message: "post scan failed",
			},
		}
	default:
		return engine.Response{
			Status: "ok",
			Result: map[string]any{
				"issue_count": 1,
				"issues": []any{
					map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "column": "city", "risk_level": "low", "issue_score": 0.1},
				},
				"scan_summary": map[string]any{"total_issues": 1},
				"data_profile": map[string]any{"rows": 10, "columns": 2},
			},
		}
	}
}

func (r *fakeAutoRuntimeBaseRunner) scanResponse(req engine.Request) engine.Response {
	csvPath := asString(req.Payload["csv_path"])
	if csvPath != "" && filepath.Clean(csvPath) != filepath.Clean(r.sourceCSV) {
		resp := r.postScanResult()
		resp.TaskID = req.TaskID
		return resp
	}
	return engine.Response{
		TaskID: req.TaskID,
		Status: "ok",
		Result: r.baselineScanResult(),
	}
}

func (r *fakeAutoRuntimeBaseRunner) repairResponse(req engine.Request, source string) engine.Response {
	planOnly, _ := req.Payload["plan_only"].(bool)
	if planOnly {
		if !r.previewCanExecute {
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
			}
		}
		if source == "gower" {
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
						map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.92},
						map[string]any{"issue_id": "issue-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.88},
					},
					"neighbor_evidence": []any{
						map[string]any{"issue_id": "issue-1", "candidate_confidence": 0.92},
						map[string]any{"issue_id": "issue-2", "candidate_confidence": 0.88},
					},
				},
			}
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
					map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.75},
				},
			},
		}
	}

	outputCSV := asString(req.Payload["output_csv"])
	if outputCSV == "" {
		outputCSV = filepath.Join(r.tempDir, source+".repaired.csv")
	}
	if err := os.MkdirAll(filepath.Dir(outputCSV), 0o755); err != nil {
		return engine.Response{TaskID: req.TaskID, Status: "error", Error: &engine.ErrorBody{Code: "MKDIR_FAILED", Message: err.Error()}}
	}
	sourceCSV := asString(req.Payload["csv_path"])
	content := []byte("age,city\n20,a\n")
	if data, err := os.ReadFile(sourceCSV); err == nil {
		content = data
	}
	if err := os.WriteFile(outputCSV, content, 0o644); err != nil {
		return engine.Response{TaskID: req.TaskID, Status: "error", Error: &engine.ErrorBody{Code: "WRITE_FAILED", Message: err.Error()}}
	}

	rollbackDir := asString(req.Payload["rollback_dir"])
	if rollbackDir == "" {
		rollbackDir = filepath.Join(r.tempDir, ".rollback")
	}
	_ = os.MkdirAll(rollbackDir, 0o755)
	baseName := strings.TrimSuffix(filepath.Base(outputCSV), filepath.Ext(outputCSV))
	manifestPath := filepath.Join(rollbackDir, baseName+".json")
	backupCSV := filepath.Join(rollbackDir, baseName+".backup.csv")
	_ = os.WriteFile(manifestPath, []byte("{}"), 0o644)
	_ = os.WriteFile(backupCSV, content, 0o644)

	rollback := map[string]any{}
	if !r.omitRollbackMetadata {
		rollback = map[string]any{
			"rollback_id":      baseName,
			"manifest_path":    manifestPath,
			"backup_csv":       backupCSV,
			"manifest_version": 2,
			"source_tool_id": map[string]string{
				"rule":  "engine.repair_batch",
				"gower": "engine.repair_with_gower",
			}[source],
		}
	}

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
	}
}

func TestRuntimeRunnerPassthroughsStableActions(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeRuntimeBaseRunner{}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID:  "task-health",
		Action:  string(engine.ActionHealth),
		Payload: map[string]any{},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("unexpected response status: %s", resp.Status)
	}
	if base.callCount(string(engine.ActionHealth)) != 1 {
		t.Fatalf("expected passthrough health call")
	}
}

func TestRuntimeRunnerBuildsPlanAndPersistsTrace(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeRuntimeBaseRunner{}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	var observed []engine.StderrEvent
	runner.SetStderrObserver(func(event engine.StderrEvent) {
		observed = append(observed, event)
	})

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-plan",
		Action: ActionSessionPlan,
		Payload: map[string]any{
			"csv_path": "demo.csv",
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("unexpected response status: %s", resp.Status)
	}

	agentBlock := resp.Result["agent"].(map[string]any)
	plan := agentBlock["plan"].(AgentPlan)
	if len(plan.SelectedIssueIDs) != 2 {
		t.Fatalf("expected 2 selected issue ids, got %d", len(plan.SelectedIssueIDs))
	}
	if len(plan.Candidates) != 3 {
		t.Fatalf("expected 3 candidates, got %d", len(plan.Candidates))
	}
	if plan.Cognition.Provider != CognitionProviderDeterministic || plan.Cognition.Status != CognitionStatusFallback {
		t.Fatalf("expected deterministic cognition on mock planner plan, got %+v", plan.Cognition)
	}
	explanation := mapFromAny(agentBlock["explanation"])
	if asString(explanation["mode"]) != "langgraph_fallback" && asString(explanation["mode"]) != "deterministic" {
		t.Fatalf("expected structured explanation mode, got %#v", explanation)
	}
	sessionID := agentBlock["session_id"].(string)
	trace, err := store.ListTrace(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	if len(trace) == 0 {
		t.Fatalf("expected persisted trace events")
	}
	summary := SummarizeTraceEvents(trace)
	if summary.Cognition.EventCount == 0 || summary.Cognition.Status == "" {
		t.Fatalf("expected cognition trace summary, got %+v", summary.Cognition)
	}
	if len(observed) == 0 {
		t.Fatalf("expected forwarded stderr events")
	}
	for _, event := range observed {
		if event.TaskID != "task-plan" {
			t.Fatalf("expected observer task id to be remapped to parent, got %s", event.TaskID)
		}
	}
}

func TestRuntimeRunnerBuildsDeterministicPlanningSnapshotBeforeCallingPlanner(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeRuntimeBaseRunner{validationCanRun: true}
	planner := &spyPlanner{}
	runner := NewRuntimeRunner(base, store, planner)

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-plan-spy",
		Action: ActionSessionPlan,
		Payload: map[string]any{
			"csv_path":                  "demo.csv",
			"scan_config_overrides":     map[string]any{"preview_limit": 4},
			"repair_strategy_overrides": map[string]any{"strategy": "conservative"},
			"column_dependencies":       map[string]any{"city": []any{"country"}},
			"gower_strategy_overrides":  map[string]any{"weight_mode": "uniform"},
			"output_dir":                "outputs/results/demo",
			"model_dir":                 "outputs/models",
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("unexpected response status: %s", resp.Status)
	}
	if !planner.called || planner.buildCount != 1 {
		t.Fatalf("expected planner BuildPlan to be called exactly once")
	}
	if base.callCount(string(engine.ActionScanFile)) != 1 {
		t.Fatalf("expected exactly one scan call before planning")
	}
	if base.callCount(string(engine.ActionRepairBatch)) != 1 {
		t.Fatalf("expected exactly one rule preview call before planning")
	}
	if base.callCount(string(engine.ActionRepairWithGower)) != 1 {
		t.Fatalf("expected exactly one gower preview call before planning")
	}

	input := planner.lastInput
	if input.CSVPath != "demo.csv" {
		t.Fatalf("unexpected planner csv path: %s", input.CSVPath)
	}
	if input.SessionID == "" {
		t.Fatalf("expected runtime to populate session id before planning")
	}
	if len(input.SelectedIssueIDs) != 2 {
		t.Fatalf("expected 2 selected issue ids, got %d", len(input.SelectedIssueIDs))
	}
	if len(input.SkippedIssues) != 1 || input.SkippedIssues[0].IssueType != "duplicate_record" {
		t.Fatalf("expected duplicate_record to remain explain-only, got %#v", input.SkippedIssues)
	}
	if intFromAny(input.ScanResult["issue_count"]) != 3 {
		t.Fatalf("expected scan result to be provided to planner, got %#v", input.ScanResult)
	}
	if intFromAny(mapFromAny(input.RulePreview["comparison"])["resolved_issue_count"]) != 2 {
		t.Fatalf("expected rule preview comparison in planner input, got %#v", input.RulePreview)
	}
	if intFromAny(mapFromAny(input.GowerPreview["comparison"])["resolved_issue_count"]) != 2 {
		t.Fatalf("expected gower preview comparison in planner input, got %#v", input.GowerPreview)
	}
	if input.ScanConfigOverrides["preview_limit"] != 4 {
		t.Fatalf("expected scan overrides in planner input, got %#v", input.ScanConfigOverrides)
	}
	if input.RepairStrategyOverrides["strategy"] != "conservative" {
		t.Fatalf("expected repair overrides in planner input, got %#v", input.RepairStrategyOverrides)
	}
	if input.GowerStrategyOverrides["weight_mode"] != "uniform" {
		t.Fatalf("expected gower overrides in planner input, got %#v", input.GowerStrategyOverrides)
	}
	if input.OutputDir != "outputs/results/demo" {
		t.Fatalf("expected output dir in planner input, got %s", input.OutputDir)
	}
	if input.ModelDir != "outputs/models" {
		t.Fatalf("expected model dir in planner input, got %s", input.ModelDir)
	}
}

func TestRuntimeRunnerRejectsExecutionWhenValidationFails(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeRuntimeBaseRunner{validationCanRun: false}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	planResp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-plan",
		Action: ActionSessionPlan,
		Payload: map[string]any{
			"csv_path": "demo.csv",
		},
	})
	if err != nil {
		t.Fatalf("plan run failed: %v", err)
	}
	agentBlock := planResp.Result["agent"].(map[string]any)
	plan := agentBlock["plan"].(AgentPlan)
	sessionID := agentBlock["session_id"].(string)

	executeResp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-execute",
		Action: ActionSessionExecute,
		Payload: map[string]any{
			"session_id": sessionID,
			"plan_id":    plan.PlanID,
		},
	})
	if err != nil {
		t.Fatalf("execute run failed: %v", err)
	}
	if executeResp.Status != "error" {
		t.Fatalf("expected error status, got %s", executeResp.Status)
	}
	if executeResp.Error == nil || executeResp.Error.Code != ErrorValidationRejected {
		t.Fatalf("unexpected execute error: %+v", executeResp.Error)
	}
	if base.callCount(string(engine.ActionRepairWithGower)) < 1 {
		t.Fatalf("expected at least one gower preview call")
	}
}

func TestRuntimeRunnerExecutePausesForApprovalOnTimeColumns(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeRuntimeBaseRunner{
		validationCanRun: true,
		baselineIssues: []map[string]any{
			{"issue_id": "i-1", "issue_type": "missing_values", "column": "event_time", "risk_level": "high"},
		},
		columnProfiles: []map[string]any{
			{"column": "event_time", "dtype": "datetime64[ns]"},
		},
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	planResp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-plan-time",
		Action: ActionSessionPlan,
		Payload: map[string]any{
			"csv_path": "demo.csv",
			"user_preferences": map[string]any{
				"avoid_time_columns":             true,
				"require_approval_for_high_risk": false,
			},
		},
	})
	if err != nil {
		t.Fatalf("plan run failed: %v", err)
	}
	agentBlock := planResp.Result["agent"].(map[string]any)
	plan := agentBlock["plan"].(AgentPlan)
	sessionID := agentBlock["session_id"].(string)

	executeResp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-execute-time",
		Action: ActionSessionExecute,
		Payload: map[string]any{
			"session_id": sessionID,
			"plan_id":    plan.PlanID,
		},
	})
	if err != nil {
		t.Fatalf("execute run failed: %v", err)
	}
	if executeResp.Status != "ok" {
		t.Fatalf("expected ok response, got %s", executeResp.Status)
	}
	safety := mapFromAny(executeResp.Result["safety"])
	if asString(safety["final_verdict"]) != "approval_required" {
		t.Fatalf("expected approval_required verdict, got %v", safety["final_verdict"])
	}
	if base.executionCallCount(string(engine.ActionRepairBatch))+base.executionCallCount(string(engine.ActionRepairWithGower)) != 0 {
		t.Fatalf("expected no execution tool call before approval")
	}

	session, found, err := store.GetSession(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !found {
		t.Fatalf("expected saved session")
	}
	if session.Status != SessionStatusAwaitingApproval {
		t.Fatalf("expected session status %s, got %s", SessionStatusAwaitingApproval, session.Status)
	}
	if asString(mapFromAny(session.Context["cognition_state"])["status"]) == "" {
		t.Fatalf("expected persisted cognition_state in session context")
	}
}

func TestRuntimeRunnerAutoSessionAccepted(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: true,
		postScanMode:      "accept",
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-accepted",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("expected ok response, got %s", resp.Status)
	}

	safety := mapFromAny(resp.Result["safety"])
	if asString(safety["final_verdict"]) != "accepted" {
		t.Fatalf("expected accepted verdict, got %v", safety["final_verdict"])
	}
	if base.callCount(string(engine.ActionRollbackRepairBatch)) != 0 {
		t.Fatalf("rollback should not be called on accepted path")
	}

	agentBlock := mapFromAny(resp.Result["agent"])
	sessionID := asString(agentBlock["session_id"])
	trace, err := store.ListTrace(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	hasPreview := false
	hasPost := false
	for _, event := range trace {
		if event.TraceType != TraceValidation {
			continue
		}
		if asString(event.Payload["phase"]) == "preview" {
			hasPreview = true
		}
		if asString(event.Payload["phase"]) == "post_execute" {
			hasPost = true
		}
	}
	if !hasPreview || !hasPost {
		t.Fatalf("expected preview and post_execute validation trace events")
	}
}

func TestRuntimeRunnerAutoSessionPausesForApprovalOnHighRiskColumns(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: true,
		highRiskColumns:   []string{"age"},
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-approval",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("expected ok response, got %s", resp.Status)
	}

	safety := mapFromAny(resp.Result["safety"])
	if asString(safety["final_verdict"]) != "approval_required" {
		t.Fatalf("expected approval_required verdict, got %v", safety["final_verdict"])
	}
	agentBlock := mapFromAny(resp.Result["agent"])
	approval := mapFromAny(agentBlock["approval"])
	if asString(approval["status"]) != "required" {
		t.Fatalf("expected approval status required, got %#v", approval)
	}
	if base.executionCallCount(string(engine.ActionRepairBatch))+base.executionCallCount(string(engine.ActionRepairWithGower)) != 0 {
		t.Fatalf("expected no execution tool calls before approval")
	}

	sessionID := asString(agentBlock["session_id"])
	session, found, err := store.GetSession(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !found {
		t.Fatalf("expected saved session")
	}
	if session.Status != SessionStatusAwaitingApproval {
		t.Fatalf("expected session status %s, got %s", SessionStatusAwaitingApproval, session.Status)
	}
	cognitionState := mapFromAny(session.Context["cognition_state"])
	if asString(cognitionState["status"]) == "" {
		t.Fatalf("expected persisted cognition_state, got %#v", cognitionState)
	}

	trace, err := store.ListTrace(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	hasApprovalRequested := false
	hasCognitionTrace := false
	for _, event := range trace {
		if event.TraceType == TraceApprovalRequested {
			hasApprovalRequested = true
		}
		if event.TraceType == TraceCognitionTrace {
			hasCognitionTrace = true
		}
	}
	if !hasApprovalRequested || !hasCognitionTrace {
		t.Fatalf("expected approval_requested and cognition_trace events")
	}
}

func TestRuntimeRunnerApproveResumesAwaitingAutoSession(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: true,
		highRiskColumns:   []string{"age"},
		postScanMode:      "accept",
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	paused, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-pause",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("pause run failed: %v", err)
	}
	agentBlock := mapFromAny(paused.Result["agent"])
	plan := agentBlock["plan"].(AgentPlan)
	sessionID := asString(agentBlock["session_id"])

	resumed, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-approve",
		Action: ActionSessionApprove,
		Payload: map[string]any{
			"session_id": sessionID,
			"plan_id":    plan.PlanID,
			"decision":   "approve",
		},
	})
	if err != nil {
		t.Fatalf("approve run failed: %v", err)
	}
	if resumed.Status != "ok" {
		t.Fatalf("expected ok response, got %s", resumed.Status)
	}
	safety := mapFromAny(resumed.Result["safety"])
	if asString(safety["final_verdict"]) != "accepted" {
		t.Fatalf("expected accepted verdict, got %v", safety["final_verdict"])
	}
	if base.executionCallCount(string(engine.ActionRepairBatch))+base.executionCallCount(string(engine.ActionRepairWithGower)) == 0 {
		t.Fatalf("expected execution tool call after approval")
	}

	session, found, err := store.GetSession(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !found {
		t.Fatalf("expected saved session")
	}
	if session.Status != SessionStatusCompleted {
		t.Fatalf("expected session status %s, got %s", SessionStatusCompleted, session.Status)
	}
	if asString(mapFromAny(session.Context["cognition_state"])["status"]) == "" {
		t.Fatalf("expected cognition_state to survive approval resume")
	}

	trace, err := store.ListTrace(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	hasApprovalGranted := false
	for _, event := range trace {
		if event.TraceType == TraceApprovalGranted {
			hasApprovalGranted = true
		}
	}
	if !hasApprovalGranted {
		t.Fatalf("expected approval_granted trace event")
	}
	if SummarizeTraceEvents(trace).Cognition.EventCount == 0 {
		t.Fatalf("expected cognition summary to survive approval resume")
	}
}

func TestRuntimeRunnerRejectApprovalSkipsWriteAndRollback(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: true,
		highRiskColumns:   []string{"age"},
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	paused, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-pause-reject",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("pause run failed: %v", err)
	}
	agentBlock := mapFromAny(paused.Result["agent"])
	plan := agentBlock["plan"].(AgentPlan)
	sessionID := asString(agentBlock["session_id"])

	rejected, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-reject",
		Action: ActionSessionApprove,
		Payload: map[string]any{
			"session_id": sessionID,
			"plan_id":    plan.PlanID,
			"decision":   "reject",
		},
	})
	if err != nil {
		t.Fatalf("reject run failed: %v", err)
	}
	if rejected.Status != "ok" {
		t.Fatalf("expected ok response, got %s", rejected.Status)
	}
	safety := mapFromAny(rejected.Result["safety"])
	if asString(safety["final_verdict"]) != "approval_rejected" {
		t.Fatalf("expected approval_rejected verdict, got %v", safety["final_verdict"])
	}
	if base.executionCallCount(string(engine.ActionRepairBatch))+base.executionCallCount(string(engine.ActionRepairWithGower)) != 0 {
		t.Fatalf("expected no execution tool calls after rejection")
	}
	if base.callCount(string(engine.ActionRollbackRepairBatch)) != 0 {
		t.Fatalf("rollback should not run when approval is rejected")
	}

	session, found, err := store.GetSession(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !found {
		t.Fatalf("expected saved session")
	}
	if session.Status != SessionStatusApprovalRejected {
		t.Fatalf("expected session status %s, got %s", SessionStatusApprovalRejected, session.Status)
	}
	if asString(mapFromAny(session.Context["cognition_state"])["status"]) == "" {
		t.Fatalf("expected cognition_state to survive approval rejection")
	}
}

func TestRuntimeRunnerAutoSessionRejectsDuringPreview(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: false,
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-preview-rejected",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "error" {
		t.Fatalf("expected error response, got %s", resp.Status)
	}
	if resp.Error == nil || resp.Error.Code != ErrorValidationRejected {
		t.Fatalf("unexpected response error: %+v", resp.Error)
	}
	if base.callCount(string(engine.ActionRollbackRepairBatch)) != 0 {
		t.Fatalf("rollback should not run when preview rejects execution")
	}

	agentBlock := mapFromAny(resp.Result["agent"])
	sessionID := asString(agentBlock["session_id"])
	session, found, err := store.GetSession(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !found {
		t.Fatalf("expected saved session")
	}
	if session.Status != SessionStatusValidationRejected {
		t.Fatalf("expected session status %s, got %s", SessionStatusValidationRejected, session.Status)
	}
}

func TestRuntimeRunnerAutoSessionRollsBackRejectedOutput(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: true,
		postScanMode:      "reject",
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-rollback",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "error" {
		t.Fatalf("expected error response after rollback path, got %s", resp.Status)
	}

	safety := mapFromAny(resp.Result["safety"])
	if asString(safety["final_verdict"]) != "rolled_back" {
		t.Fatalf("expected rolled_back verdict, got %v", safety["final_verdict"])
	}
	if base.callCount(string(engine.ActionRollbackRepairBatch)) != 1 {
		t.Fatalf("expected one rollback call")
	}

	snapshotPath := asString(safety["rejected_output_snapshot"])
	if strings.TrimSpace(snapshotPath) == "" || !fileExists(snapshotPath) {
		t.Fatalf("expected rejected output snapshot to exist, got %s", snapshotPath)
	}

	agentBlock := mapFromAny(resp.Result["agent"])
	sessionID := asString(agentBlock["session_id"])
	trace, err := store.ListTrace(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	hasDecision := false
	hasExecuted := false
	for _, event := range trace {
		if event.TraceType == TraceRollbackDecision {
			hasDecision = true
		}
		if event.TraceType == TraceRollbackExecuted {
			hasExecuted = true
		}
	}
	if !hasDecision || !hasExecuted {
		t.Fatalf("expected rollback decision and executed trace events")
	}
}

func TestRuntimeRunnerAutoSessionReportsRollbackFailure(t *testing.T) {
	tempDir := t.TempDir()
	sourceCSV := filepath.Join(tempDir, "demo.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		t.Fatalf("write source csv failed: %v", err)
	}

	store, err := NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &fakeAutoRuntimeBaseRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		previewCanExecute: true,
		postScanMode:      "reject",
		rollbackFails:     true,
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-auto-rollback-failed",
		Action: ActionSessionAuto,
		Payload: map[string]any{
			"csv_path": sourceCSV,
		},
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "error" {
		t.Fatalf("expected error response, got %s", resp.Status)
	}

	safety := mapFromAny(resp.Result["safety"])
	if asString(safety["final_verdict"]) != "rollback_failed" {
		t.Fatalf("expected rollback_failed verdict, got %v", safety["final_verdict"])
	}

	agentBlock := mapFromAny(resp.Result["agent"])
	sessionID := asString(agentBlock["session_id"])
	session, found, err := store.GetSession(t.Context(), sessionID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !found {
		t.Fatalf("expected saved session")
	}
	if session.Status != SessionStatusRollbackFailed {
		t.Fatalf("expected session status %s, got %s", SessionStatusRollbackFailed, session.Status)
	}
}
