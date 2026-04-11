package agent

import "testing"

func TestSelectRepairableIssuesReturnsSupportedAndSkippedIssues(t *testing.T) {
	selectedIssueIDs, skipped := selectRepairableIssues(map[string]any{
		"issues": []any{
			map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "column": "age"},
			map[string]any{"issue_id": "i-2", "issue_type": "numeric_outlier", "column": "bmi"},
			map[string]any{"issue_id": "i-3", "issue_type": "rare_category", "column": "city"},
			map[string]any{"issue_id": "i-4", "issue_type": "duplicate_record", "column": "id"},
			map[string]any{"issue_type": "missing_values", "column": "weight"},
		},
	})

	if len(selectedIssueIDs) != 3 {
		t.Fatalf("expected 3 repairable issue ids, got %d", len(selectedIssueIDs))
	}
	if selectedIssueIDs[0] != "i-1" || selectedIssueIDs[1] != "i-2" || selectedIssueIDs[2] != "i-3" {
		t.Fatalf("unexpected selected issue ids: %#v", selectedIssueIDs)
	}
	if len(skipped) != 2 {
		t.Fatalf("expected 2 skipped issues, got %d", len(skipped))
	}
	if skipped[0].Reason != "unsupported_issue_type" {
		t.Fatalf("expected unsupported issue type skip, got %s", skipped[0].Reason)
	}
	if skipped[1].Reason != "missing_issue_id" {
		t.Fatalf("expected missing issue id skip, got %s", skipped[1].Reason)
	}
}

func TestBuildPlanningInputClonesDeterministicSnapshot(t *testing.T) {
	params := planningParams{
		CSVPath:            "demo.csv",
		Goal:               "scan and repair",
		OutputDir:          "outputs/results/demo",
		ModelDir:           "outputs/models",
		ScanOverrides:      map[string]any{"preview_limit": 5},
		RepairOverrides:    map[string]any{"strategy": "conservative"},
		ColumnDependencies: map[string]any{"city": []any{"country"}},
		GowerOverrides:     map[string]any{"weight_mode": "uniform"},
	}
	scanResult := map[string]any{
		"issue_count": 2,
		"scan_summary": map[string]any{
			"total_issues": 2,
		},
	}
	selectedIssueIDs := []string{"i-1", "i-2"}
	skipped := []AgentSkippedIssue{
		{IssueID: "i-3", IssueType: "duplicate_record", Details: map[string]any{"reason": "unsupported_issue_type"}},
	}
	rulePreview := map[string]any{"comparison": map[string]any{"resolved_issue_count": 1}}
	gowerPreview := map[string]any{"comparison": map[string]any{"resolved_issue_count": 2}}

	input := buildPlanningInput(
		"session-1",
		"scan and repair",
		params,
		scanResult,
		selectedIssueIDs,
		skipped,
		rulePreview,
		gowerPreview,
		map[string]any{"avoid_time_columns": true},
		map[string]any{"deterministic_required": false},
	)

	selectedIssueIDs[0] = "changed"
	skipped[0].Details["reason"] = "changed"
	params.ScanOverrides["preview_limit"] = 9
	params.RepairOverrides["strategy"] = "aggressive"
	params.ColumnDependencies["city"] = []any{"province"}
	params.GowerOverrides["weight_mode"] = "custom"
	mapFromAny(scanResult["scan_summary"])["total_issues"] = 9
	mapFromAny(rulePreview["comparison"])["resolved_issue_count"] = 9
	mapFromAny(gowerPreview["comparison"])["resolved_issue_count"] = 9

	if input.SessionID != "session-1" {
		t.Fatalf("unexpected session id: %s", input.SessionID)
	}
	if input.Goal != "scan and repair" {
		t.Fatalf("unexpected goal: %s", input.Goal)
	}
	if input.SelectedIssueIDs[0] != "i-1" {
		t.Fatalf("expected cloned issue ids, got %#v", input.SelectedIssueIDs)
	}
	if input.SkippedIssues[0].Details["reason"] != "unsupported_issue_type" {
		t.Fatalf("expected cloned skipped issue details, got %#v", input.SkippedIssues)
	}
	if mapFromAny(input.ScanResult["scan_summary"])["total_issues"] != 2 {
		t.Fatalf("expected cloned scan result, got %#v", input.ScanResult)
	}
	if mapFromAny(input.RulePreview["comparison"])["resolved_issue_count"] != 1 {
		t.Fatalf("expected cloned rule preview, got %#v", input.RulePreview)
	}
	if mapFromAny(input.GowerPreview["comparison"])["resolved_issue_count"] != 2 {
		t.Fatalf("expected cloned gower preview, got %#v", input.GowerPreview)
	}
	if input.ScanConfigOverrides["preview_limit"] != 5 {
		t.Fatalf("expected cloned scan overrides, got %#v", input.ScanConfigOverrides)
	}
	if input.RepairStrategyOverrides["strategy"] != "conservative" {
		t.Fatalf("expected cloned repair overrides, got %#v", input.RepairStrategyOverrides)
	}
	if input.GowerStrategyOverrides["weight_mode"] != "uniform" {
		t.Fatalf("expected cloned gower overrides, got %#v", input.GowerStrategyOverrides)
	}
}
