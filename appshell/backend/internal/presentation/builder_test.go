package presentation

import (
	"path/filepath"
	"testing"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
)

func TestEnrichTaskResultForScanAddsPresentation(t *testing.T) {
	result := map[string]any{
		"csv_path":    "demo.csv",
		"issue_count": 2,
		"issues": []any{
			map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.8},
			map[string]any{"issue_id": "i-2", "issue_type": "rare_category", "column": "city", "risk_level": "medium", "issue_score": 0.3},
		},
		"scan_summary": map[string]any{
			"high_risk_columns": []string{"age"},
		},
		"data_profile": map[string]any{
			"rows":    20,
			"columns": 3,
		},
		"column_thumbnails": []any{
			map[string]any{"column": "age", "risk_score": 78.4, "risk_level": "high", "issue_count": 1, "anomaly_points": 2},
			map[string]any{"column": "city", "risk_score": 34.2, "risk_level": "medium", "issue_count": 1, "anomaly_points": 1},
		},
	}

	artifact, err := EnrichTaskResult(engine.Request{Action: string(engine.ActionScanFile)}, result)
	if err != nil {
		t.Fatalf("EnrichTaskResult failed: %v", err)
	}
	if artifact != "" {
		t.Fatalf("scan without output dir should not write artifact")
	}
	presentation := objectMap(result["presentation"])
	if asString(presentation["kind"]) != "scan" {
		t.Fatalf("expected scan presentation, got %v", presentation["kind"])
	}
	if len(sliceOfMaps(presentation["charts"])) == 0 {
		t.Fatalf("expected charts in scan presentation")
	}
}

func TestEnrichTaskResultForRepairWritesArtifact(t *testing.T) {
	outputDir := t.TempDir()
	result := map[string]any{
		"output_csv":           filepath.Join(outputDir, "demo.repaired.csv"),
		"applied_issue_count":  2,
		"selected_issue_count": 2,
		"comparison": map[string]any{
			"before_issue_count":         2,
			"after_issue_count":          0,
			"resolved_issue_count":       2,
			"changed_cell_count":         2,
			"before_column_issue_counts": map[string]any{"age": 1, "city": 1},
			"after_column_issue_counts":  map[string]any{"age": 0, "city": 0},
		},
		"rollback":       map[string]any{"manifest_path": filepath.Join(outputDir, ".rollback", "demo.json")},
		"skipped_issues": []any{},
	}

	artifact, err := EnrichTaskResult(engine.Request{
		Action:  string(engine.ActionRepairBatch),
		Payload: map[string]any{"output_dir": outputDir},
	}, result)
	if err != nil {
		t.Fatalf("EnrichTaskResult failed: %v", err)
	}
	if artifact == "" {
		t.Fatalf("expected presentation artifact path")
	}
	if filepath.Base(artifact) != "presentation.json" {
		t.Fatalf("unexpected artifact path: %s", artifact)
	}
	presentation := objectMap(result["presentation"])
	if asString(presentation["kind"]) != "repair" {
		t.Fatalf("expected repair presentation")
	}
}

func TestEnrichTaskResultForAgentAttachesNestedPresentation(t *testing.T) {
	outputDir := t.TempDir()
	result := map[string]any{
		"agent": map[string]any{
			"session_id": "session-1",
			"plan_id":    "plan-1",
			"run_mode":   "auto",
			"plan": agent.AgentPlan{
				PlanID:              "plan-1",
				SelectedIssueIDs:    []string{"i-1", "i-2"},
				SelectedCandidateID: "c-1",
				SelectedSource:      "gower",
				Cognition: agent.AgentCognitionState{
					Provider:            agent.CognitionProviderLangGraph,
					Status:              agent.CognitionStatusEngaged,
					PlannerMode:         "llm",
					LLMMode:             "configured",
					SelectedCandidateID: "c-1",
					Summary:             "LangGraph selected the Gower candidate.",
				},
				Candidates: []agent.RepairCandidate{
					{CandidateID: "c-1", Source: "gower"},
					{CandidateID: "c-2", Source: "rule"},
				},
				UserExplanation: "Picked the gower candidate.",
			},
			"validation": map[string]any{
				"status":       "checked",
				"message":      "Validation passed.",
				"preview":      map[string]any{"status": "checked", "message": "Validation passed."},
				"post_execute": map[string]any{"status": "accepted", "message": "Post validation accepted."},
			},
			"execution": map[string]any{
				"status":     "executed",
				"output_csv": filepath.Join(outputDir, "demo.repaired.csv"),
			},
			"trace_summary": map[string]any{
				"event_count": 6,
				"agent_names": []string{"supervisor", "validator"},
				"cognition": map[string]any{
					"event_count":  1,
					"provider":     "langgraph",
					"status":       "engaged",
					"last_summary": "LangGraph selected the Gower candidate.",
				},
			},
		},
		"safety": map[string]any{
			"final_verdict":         "accepted",
			"baseline_scan_summary": map[string]any{"issue_count": 2, "high_risk_issue_count": 1, "total_issue_score": 1.0},
			"post_scan_summary":     map[string]any{"issue_count": 1, "high_risk_issue_count": 0, "total_issue_score": 0.2},
		},
	}

	artifact, err := EnrichTaskResult(engine.Request{Action: agent.ActionSessionAuto}, result)
	if err != nil {
		t.Fatalf("EnrichTaskResult failed: %v", err)
	}
	if artifact == "" {
		t.Fatalf("expected agent presentation artifact path")
	}
	agentBlock := objectMap(result["agent"])
	presentation := objectMap(agentBlock["presentation"])
	if asString(presentation["kind"]) != "agent" {
		t.Fatalf("expected agent presentation")
	}
	highlights := sliceOfMaps(presentation["highlights"])
	if len(highlights) == 0 || asString(highlights[0]["id"]) != "cognition_status" {
		t.Fatalf("expected cognition highlight, got %#v", highlights)
	}
}

func TestEnrichAgentSessionSnapshotAddsPresentation(t *testing.T) {
	outputDir := t.TempDir()
	snapshot := agent.AgentSessionSnapshot{
		SessionID: "session-1",
		Status:    agent.SessionStatusCompleted,
		Mode:      "auto",
		UserGoal:  "scan and repair",
		Context: map[string]any{
			"final_verdict": "accepted",
			"output_dir":    outputDir,
			"preview_validation": map[string]any{
				"status":  "checked",
				"message": "Preview accepted.",
			},
			"post_validation": map[string]any{
				"status":  "accepted",
				"message": "Post validation accepted.",
			},
			"execution_artifacts": map[string]any{
				"output_csv": filepath.Join(outputDir, "demo.repaired.csv"),
			},
			"baseline_scan": map[string]any{"issue_count": 2, "high_risk_issue_count": 1, "total_issue_score": 1.0},
			"post_scan":     map[string]any{"issue_count": 1, "high_risk_issue_count": 0, "total_issue_score": 0.2},
		},
		LatestPlan: agent.AgentPlan{
			PlanID:              "plan-1",
			SelectedIssueIDs:    []string{"i-1"},
			SelectedCandidateID: "c-1",
			SelectedSource:      "rule",
			Cognition: agent.AgentCognitionState{
				Provider:            agent.CognitionProviderDeterministic,
				Status:              agent.CognitionStatusFallback,
				PlannerMode:         "fallback",
				LLMMode:             "unavailable",
				SelectedCandidateID: "c-1",
				FallbackReasonCode:  agent.CognitionFallbackPlannerMode,
				Summary:             "Deterministic planning stayed active because LangGraph was unavailable.",
			},
			Candidates:      []agent.RepairCandidate{{CandidateID: "c-1", Source: "rule"}},
			UserExplanation: "Picked the safest candidate.",
		},
		TraceSummary: agent.TraceSummary{
			EventCount:    4,
			ToolCallCount: 2,
			AgentNames:    []string{"supervisor", "validator"},
			Cognition: agent.CognitionTraceSummary{
				EventCount:         1,
				Provider:           agent.CognitionProviderDeterministic,
				Status:             agent.CognitionStatusFallback,
				LastPhase:          "plan_complete",
				LastSummary:        "Deterministic planning stayed active because LangGraph was unavailable.",
				PlannerMode:        "fallback",
				LLMMode:            "unavailable",
				FallbackReasonCode: agent.CognitionFallbackPlannerMode,
			},
		},
	}

	if err := EnrichAgentSessionSnapshot(&snapshot); err != nil {
		t.Fatalf("EnrichAgentSessionSnapshot failed: %v", err)
	}
	if len(snapshot.Presentation) == 0 {
		t.Fatalf("expected session snapshot presentation")
	}
	if snapshot.PresentationArtifact == "" {
		t.Fatalf("expected session snapshot presentation artifact")
	}
}
