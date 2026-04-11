package agent

import "testing"

func TestMockPlannerBuildsThreeCandidatesFromDeterministicPreviews(t *testing.T) {
	planner := NewMockPlanner()

	plan, err := planner.BuildPlan(t.Context(), PlanningInput{
		CSVPath:          "demo.csv",
		SelectedIssueIDs: []string{"i-1", "i-2", "i-3"},
		SkippedIssues: []AgentSkippedIssue{
			{IssueID: "i-4", IssueType: "duplicate_record", Column: "id", Reason: "unsupported_issue_type"},
		},
		ScanResult: map[string]any{
			"issue_count": 4,
			"issues": []any{
				map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "column": "age"},
				map[string]any{"issue_id": "i-2", "issue_type": "numeric_outlier", "column": "bmi"},
				map[string]any{"issue_id": "i-3", "issue_type": "rare_category", "column": "city"},
				map[string]any{"issue_id": "i-4", "issue_type": "duplicate_record", "column": "id"},
			},
		},
		RulePreview: map[string]any{
			"comparison": map[string]any{
				"before_issue_count":   4,
				"after_issue_count":    2,
				"resolved_issue_count": 2,
				"changed_cell_count":   2,
			},
			"applied_repairs": []any{
				map[string]any{"issue_id": "i-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.8},
				map[string]any{"issue_id": "i-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.7},
			},
			"skipped_issues": []any{
				map[string]any{"issue_id": "i-3", "reason": "unsupported_issue_type"},
			},
		},
		GowerPreview: map[string]any{
			"comparison": map[string]any{
				"before_issue_count":   4,
				"after_issue_count":    1,
				"resolved_issue_count": 3,
				"changed_cell_count":   3,
			},
			"applied_repairs": []any{
				map[string]any{"issue_id": "i-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.9},
				map[string]any{"issue_id": "i-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.8},
				map[string]any{"issue_id": "i-3", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.95},
			},
			"neighbor_evidence": []any{
				map[string]any{"issue_id": "i-3", "candidate_confidence": 0.95},
			},
		},
	})
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	if plan.SelectedSource != "hybrid" {
		t.Fatalf("unexpected selected source: %s", plan.SelectedSource)
	}
	if len(plan.SelectedIssueIDs) != 3 {
		t.Fatalf("expected 3 selected issues, got %d", len(plan.SelectedIssueIDs))
	}
	if len(plan.SkippedIssues) != 1 {
		t.Fatalf("expected 1 skipped issue, got %d", len(plan.SkippedIssues))
	}
	if len(plan.Candidates) != 3 {
		t.Fatalf("expected 3 candidates, got %d", len(plan.Candidates))
	}
	if plan.ProposedToolID != "engine.repair_batch" {
		t.Fatalf("unexpected tool id: %s", plan.ProposedToolID)
	}
	if got := plan.ProposedPayload["plan_only"]; got != false {
		t.Fatalf("execute payload should set plan_only=false, got %v", got)
	}
}
