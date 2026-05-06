package agent

import (
	"reflect"
	"testing"
)

func TestSelectIssuePlanBucketsUsesConservativeA2Rules(t *testing.T) {
	buckets := selectIssuePlanBuckets(map[string]any{
		"issues": []any{
			map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "column": "age"},
			map[string]any{"issue_id": "i-2", "issue_type": "numeric_outlier", "column": "bmi"},
			map[string]any{"issue_id": "i-3", "issue_type": "rare_category", "column": "city"},
			map[string]any{"issue_id": "i-4", "issue_type": "duplicate_record", "column": "id"},
			map[string]any{"issue_id": "i-5", "issue_type": "cross_column_consistency", "column": "date"},
			map[string]any{"issue_id": "i-6", "issue_type": "new_issue_type", "column": "score"},
			map[string]any{"issue_type": "missing_values", "column": "weight"},
		},
	})

	if !reflect.DeepEqual(buckets.AutoRepairIssueIDs, []string{"i-1", "i-3"}) {
		t.Fatalf("unexpected auto issue ids: %#v", buckets.AutoRepairIssueIDs)
	}
	if !reflect.DeepEqual(buckets.CautiousIssueIDs, []string{"i-2"}) {
		t.Fatalf("unexpected cautious issue ids: %#v", buckets.CautiousIssueIDs)
	}
	if !reflect.DeepEqual(buckets.ManualReviewIssueIDs, []string{"i-4", "i-5"}) {
		t.Fatalf("unexpected manual issue ids: %#v", buckets.ManualReviewIssueIDs)
	}
	if !reflect.DeepEqual(buckets.BlockedIssueIDs, []string{"i-6"}) {
		t.Fatalf("unexpected blocked issue ids: %#v", buckets.BlockedIssueIDs)
	}

	reasons := map[string]string{}
	bucketsByIssue := map[string]string{}
	for _, item := range buckets.SkippedIssues {
		reasons[item.IssueID] = item.Reason
		bucketsByIssue[item.IssueID] = asString(item.Details["bucket"])
	}
	if reasons["i-2"] != "requires_human_review_before_auto_repair" || bucketsByIssue["i-2"] != issueBucketCautious {
		t.Fatalf("expected numeric_outlier to be cautious, got reasons=%#v buckets=%#v", reasons, bucketsByIssue)
	}
	if reasons["i-4"] != "manual_review_required" || reasons["i-5"] != "manual_review_required" {
		t.Fatalf("expected duplicate/consistency to be manual review, got %#v", reasons)
	}
	if reasons["i-6"] != "unsupported_issue_type" {
		t.Fatalf("expected unknown issue type to be blocked, got %#v", reasons)
	}
	if reasons[""] != "missing_issue_id" || bucketsByIssue[""] != issueBucketBlocked {
		t.Fatalf("expected missing issue id to be blocked skip, got reasons=%#v buckets=%#v", reasons, bucketsByIssue)
	}
}

func TestSelectRepairableIssuesReturnsOnlyAutoAndSkippedIssues(t *testing.T) {
	selectedIssueIDs, skipped := selectRepairableIssues(map[string]any{
		"issues": []any{
			map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "column": "age"},
			map[string]any{"issue_id": "i-2", "issue_type": "numeric_outlier", "column": "bmi"},
			map[string]any{"issue_id": "i-3", "issue_type": "rare_category", "column": "city"},
			map[string]any{"issue_id": "i-4", "issue_type": "duplicate_record", "column": "id"},
			map[string]any{"issue_id": "i-5", "issue_type": "cross_column_consistency", "column": "date"},
			map[string]any{"issue_id": "i-6", "issue_type": "new_issue_type", "column": "score"},
			map[string]any{"issue_type": "missing_values", "column": "weight"},
		},
	})

	if len(selectedIssueIDs) != 2 {
		t.Fatalf("expected 2 auto issue ids, got %d", len(selectedIssueIDs))
	}
	if selectedIssueIDs[0] != "i-1" || selectedIssueIDs[1] != "i-3" {
		t.Fatalf("unexpected selected issue ids: %#v", selectedIssueIDs)
	}
	if len(skipped) != 5 {
		t.Fatalf("expected 5 skipped issues, got %d", len(skipped))
	}
	if skipped[0].Reason != "requires_human_review_before_auto_repair" {
		t.Fatalf("expected cautious issue type skip, got %s", skipped[0].Reason)
	}
	if skipped[len(skipped)-1].Reason != "missing_issue_id" {
		t.Fatalf("expected missing issue id skip, got %s", skipped[len(skipped)-1].Reason)
	}
}

func TestIssueExplanationDetailsIncludeReasonsAndCounts(t *testing.T) {
	_, skipped := selectRepairableIssues(map[string]any{
		"issues": []any{
			map[string]any{"issue_id": "i-1", "issue_type": "numeric_outlier", "column": "bmi"},
			map[string]any{"issue_id": "i-2", "issue_type": "time_series_shift", "column": "date"},
			map[string]any{"issue_type": "missing_values", "column": "weight"},
		},
	})

	blocked, cautious, counts := issueExplanationDetails(skipped)

	if len(cautious) != 1 || cautious[0].IssueID != "i-1" || cautious[0].RiskReason != "requires_human_review_before_auto_repair" {
		t.Fatalf("unexpected cautious details: %#v", cautious)
	}
	if !cautious[0].ApprovalRequired || cautious[0].SuggestedAction == "" {
		t.Fatalf("expected cautious approval guidance, got %#v", cautious[0])
	}
	if len(blocked) != 2 {
		t.Fatalf("expected two blocked details, got %#v", blocked)
	}
	if counts["unsupported_issue_type"] != 1 || counts["missing_issue_id"] != 1 {
		t.Fatalf("unexpected blocked reason counts: %#v", counts)
	}
	if blocked[0].BlockedByRule == "" || blocked[0].SuggestedNextAction == "" {
		t.Fatalf("expected blocked rule explanation, got %#v", blocked[0])
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
