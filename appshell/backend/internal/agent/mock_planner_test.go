package agent

import (
	"reflect"
	"strings"
	"testing"
)

func TestMockPlannerBuildsThreeCandidatesFromDeterministicPreviews(t *testing.T) {
	planner := NewMockPlanner()

	plan, err := planner.BuildPlan(t.Context(), PlanningInput{
		CSVPath:          "demo.csv",
		SelectedIssueIDs: []string{"i-1", "i-2", "i-3", "i-4", "i-5", "i-6"},
		ScanResult: map[string]any{
			"issue_count": 7,
			"issues": []any{
				map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "column": "age"},
				map[string]any{"issue_id": "i-2", "issue_type": "numeric_outlier", "column": "bmi", "outlier_risk_level": "mild", "auto_repair_eligible": false},
				map[string]any{"issue_id": "i-3", "issue_type": "rare_category", "column": "city"},
				map[string]any{"issue_id": "i-4", "issue_type": "duplicate_record", "column": "id"},
				map[string]any{"issue_id": "i-5", "issue_type": "cross_column_consistency", "column": "event_date"},
				map[string]any{"issue_id": "i-6", "issue_type": "new_issue_type", "column": "score"},
				map[string]any{"issue_id": "i-7", "issue_type": "numeric_outlier", "column": "age", "outlier_risk_level": "extreme", "auto_repair_eligible": true},
			},
		},
		RulePreview: map[string]any{
			"comparison": map[string]any{
				"before_issue_count":   7,
				"after_issue_count":    5,
				"resolved_issue_count": 2,
				"changed_cell_count":   2,
			},
			"applied_repairs": []any{
				map[string]any{"issue_id": "i-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.8},
				map[string]any{"issue_id": "i-7", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.8},
			},
			"skipped_issues": []any{
				map[string]any{"issue_id": "i-3", "reason": "unsupported_issue_type"},
			},
		},
		GowerPreview: map[string]any{
			"comparison": map[string]any{
				"before_issue_count":   7,
				"after_issue_count":    4,
				"resolved_issue_count": 3,
				"changed_cell_count":   3,
			},
			"applied_repairs": []any{
				map[string]any{"issue_id": "i-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.9},
				map[string]any{"issue_id": "i-3", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.95},
				map[string]any{"issue_id": "i-7", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.95},
			},
			"neighbor_evidence": []any{
				map[string]any{"issue_id": "i-3", "candidate_confidence": 0.95},
				map[string]any{"issue_id": "i-7", "candidate_confidence": 0.95},
			},
		},
	})
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	if plan.SelectedSource != "hybrid" {
		t.Fatalf("unexpected selected source: %s", plan.SelectedSource)
	}
	if !reflect.DeepEqual(plan.SelectedIssueIDs, []string{"i-1", "i-3", "i-7"}) {
		t.Fatalf("unexpected selected issues: %#v", plan.SelectedIssueIDs)
	}
	if !reflect.DeepEqual(plan.AutoRepairIssueIDs, []string{"i-1", "i-3", "i-7"}) {
		t.Fatalf("unexpected auto issue ids: %#v", plan.AutoRepairIssueIDs)
	}
	if !reflect.DeepEqual(plan.CautiousIssueIDs, []string{"i-2"}) {
		t.Fatalf("unexpected cautious issue ids: %#v", plan.CautiousIssueIDs)
	}
	if !reflect.DeepEqual(plan.ManualReviewIssueIDs, []string{"i-4", "i-5"}) {
		t.Fatalf("unexpected manual issue ids: %#v", plan.ManualReviewIssueIDs)
	}
	if !reflect.DeepEqual(plan.BlockedIssueIDs, []string{"i-6"}) {
		t.Fatalf("unexpected blocked issue ids: %#v", plan.BlockedIssueIDs)
	}
	if len(plan.SkippedIssues) != 4 {
		t.Fatalf("expected 4 skipped issue explanations, got %d", len(plan.SkippedIssues))
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
	if !reflect.DeepEqual(stringsFromAny(plan.ProposedPayload["issue_ids"]), []string{"i-1", "i-3", "i-7"}) {
		t.Fatalf("expected proposed payload to contain only auto issues, got %#v", plan.ProposedPayload)
	}
	for _, candidate := range plan.Candidates {
		for _, payload := range candidate.PlanPayloads {
			issueIDs := stringsFromAny(payload["issue_ids"])
			for _, issueID := range issueIDs {
				if issueID == "i-2" || issueID == "i-4" || issueID == "i-5" || issueID == "i-6" {
					t.Fatalf("candidate %s preview payload included non-auto issue id %s: %#v", candidate.CandidateID, issueID, payload)
				}
			}
		}
	}
	reasonCodes := stringsFromAny(plan.ReasonCodes)
	for _, expected := range []string{"cautious_issues_pending_review", "manual_review_required", "blocked_issues_present"} {
		found := false
		for _, code := range reasonCodes {
			if code == expected {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected reason code %s in %#v", expected, reasonCodes)
		}
	}
	if !strings.Contains(plan.RiskNote, numericOutlierMildRiskNote) || !strings.Contains(plan.RiskNote, numericOutlierAutoRepairRestrictedRiskNote) {
		t.Fatalf("expected numeric_outlier policy risk note, got %q", plan.RiskNote)
	}
}
