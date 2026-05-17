package agent

import "testing"

func validationGateScan(issues ...map[string]any) map[string]any {
	items := make([]any, 0, len(issues))
	totalScore := 0.0
	highRiskCount := 0
	for _, issue := range issues {
		items = append(items, cloneMap(issue))
		totalScore += floatFromAny(issue["issue_score"])
		if asString(issue["risk_level"]) == "high" {
			highRiskCount++
		}
	}
	return map[string]any{
		"issue_count":           len(issues),
		"high_risk_issue_count": highRiskCount,
		"total_issue_score":     totalScore,
		"issues":                items,
	}
}

func validationGateRepair(changed int, rollback bool, issueIDs ...string) map[string]any {
	result := map[string]any{
		"status":             "executed",
		"output_csv":         "out/repaired.csv",
		"selected_issue_ids": append([]string{}, issueIDs...),
		"comparison": map[string]any{
			"before_issue_count":   len(issueIDs),
			"after_issue_count":    0,
			"resolved_issue_count": len(issueIDs),
			"changed_cell_count":   changed,
		},
	}
	if rollback {
		result["rollback"] = map[string]any{"manifest_path": "out/.rollback/repair.json"}
	}
	return result
}

func validationGateRepairWithApplied(changed int, rollback bool, appliedRepairs ...map[string]any) map[string]any {
	issueIDs := make([]string, 0, len(appliedRepairs))
	items := make([]any, 0, len(appliedRepairs))
	for _, item := range appliedRepairs {
		issueIDs = append(issueIDs, asString(item["issue_id"]))
		items = append(items, cloneMap(item))
	}
	result := validationGateRepair(changed, rollback, issueIDs...)
	result["applied_repairs"] = items
	return result
}

func validationGatePlan(autoIDs ...string) AgentPlan {
	return AgentPlan{
		SelectedIssueIDs:   append([]string{}, autoIDs...),
		AutoRepairIssueIDs: append([]string{}, autoIDs...),
	}
}

func TestValidationGateAcceptsIssueCountDecrease(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "medium", "issue_score": 0.7},
		map[string]any{"issue_id": "i-2", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.1},
	)

	result := buildPostValidation(baseline, validationGateRepair(2, true, "i-1", "i-2"), postScan, validationGatePlan("i-1", "i-2"))

	if result.Verdict != validationGateAccept || !result.Accepted {
		t.Fatalf("expected accept, got verdict=%s accepted=%v summary=%#v", result.Verdict, result.Accepted, result.Summary)
	}
	if intFromAny(result.Summary["before_issue_items"]) != 2 || intFromAny(result.Summary["after_issue_items"]) != 1 {
		t.Fatalf("expected before/after issue item counts, got %#v", result.Summary)
	}
	if intFromAny(result.Summary["resolved_issue_items"]) != 1 {
		t.Fatalf("expected resolved issue items from scan delta, got %#v", result.Summary)
	}
	if intFromAny(result.Summary["modified_cell_count"]) != 2 {
		t.Fatalf("expected modified cell count from repair result, got %#v", result.Summary)
	}
	if intFromAny(result.Summary["resolved_issue_count"]) != 1 {
		t.Fatalf("expected compatibility resolved_issue_count to mirror resolved_issue_items, got %#v", result.Summary)
	}
	if mapFromAny(result.Summary["metric_definitions"])["resolved_issue_items"] == nil {
		t.Fatalf("expected metric definitions, got %#v", result.Summary)
	}
	if asString(result.Summary["acceptance_reason"]) != "issue_count_improved_and_side_effects_controlled" {
		t.Fatalf("expected acceptance reason, got %#v", result.Summary)
	}
	if floatFromAny(result.Summary["modified_to_resolved_ratio"]) != 2.0 {
		t.Fatalf("expected modified/resolved ratio 2.0, got %#v", result.Summary)
	}
	numericSummary := mapFromAny(result.Summary["numeric_outlier_modification_summary"])
	if intFromAny(numericSummary["modified_cells"]) != 0 {
		t.Fatalf("expected no numeric_outlier modifications, got %#v", numericSummary)
	}
	rollbackAvailability := mapFromAny(result.Summary["rollback_availability"])
	if rollbackAvailability["available"] != true || asString(rollbackAvailability["manifest_path"]) == "" {
		t.Fatalf("expected rollback availability, got %#v", rollbackAvailability)
	}
	if len(stringsFromAny(result.Summary["side_effect_notes"])) != 0 {
		t.Fatalf("expected no side effect notes, got %#v", result.Summary["side_effect_notes"])
	}
}

func TestValidationGateSeparatesIssueItemsFromModifiedCells(t *testing.T) {
	baseline := validationGateScan()
	for i := 0; i < 17; i++ {
		baselineIssues := mapsFromAny(baseline["issues"])
		baselineIssues = append(baselineIssues, map[string]any{"issue_id": "i", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.1})
		baseline["issues"] = baselineIssues
	}
	baseline["issue_count"] = 17
	baseline["total_issue_score"] = 1.7
	postScan := validationGateScan()
	for i := 0; i < 10; i++ {
		postIssues := mapsFromAny(postScan["issues"])
		postIssues = append(postIssues, map[string]any{"issue_id": "post", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.05})
		postScan["issues"] = postIssues
	}
	postScan["issue_count"] = 10
	postScan["total_issue_score"] = 0.5

	result := buildPostValidation(baseline, validationGateRepair(48, true, "i-1"), postScan, validationGatePlan("i-1"))

	if intFromAny(result.Summary["before_issue_items"]) != 17 || intFromAny(result.Summary["after_issue_items"]) != 10 {
		t.Fatalf("unexpected issue item counts: %#v", result.Summary)
	}
	if intFromAny(result.Summary["resolved_issue_items"]) != 7 {
		t.Fatalf("expected issue item delta 7, got %#v", result.Summary)
	}
	if intFromAny(result.Summary["modified_cell_count"]) != 48 {
		t.Fatalf("expected modified cell count 48, got %#v", result.Summary)
	}
}

func TestValidationGateWarnsOnLargeCellChange(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "medium", "issue_score": 0.7},
		map[string]any{"issue_id": "i-2", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-3", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.2},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.1},
	)

	result := buildPostValidation(baseline, validationGateRepair(61, true, "i-1", "i-2", "i-3"), postScan, validationGatePlan("i-1", "i-2", "i-3"))

	if result.Verdict != validationGateWarn || !result.Accepted {
		t.Fatalf("expected warn but accepted, got verdict=%s accepted=%v summary=%#v", result.Verdict, result.Accepted, result.Summary)
	}
	if !hasString(stringsFromAny(result.Summary["risk_notes"]), "changed_cell_count_abnormally_high") {
		t.Fatalf("expected changed_cell_count_abnormally_high risk, got %#v", result.Summary["risk_notes"])
	}
	if !hasString(result.RiskFlags, validationRiskModifiedHighRelativeToResolved) {
		t.Fatalf("expected modified/resolved side-effect risk, got %#v", result.RiskFlags)
	}
}

func TestValidationGateWarnsWhenModifiedResolvedRatioIsHigh(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-2", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-3", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-4", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-5", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-6", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.1},
		map[string]any{"issue_id": "post-2", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.1},
	)

	result := buildPostValidation(baseline, validationGateRepair(55, true, "i-1", "i-2"), postScan, validationGatePlan("i-1", "i-2"))

	if result.Verdict != validationGateWarn || !result.Accepted {
		t.Fatalf("expected warn but accepted, got verdict=%s accepted=%v summary=%#v", result.Verdict, result.Accepted, result.Summary)
	}
	if !hasString(result.RiskFlags, validationRiskModifiedHighRelativeToResolved) {
		t.Fatalf("expected modified/resolved side-effect risk, got %#v", result.RiskFlags)
	}
	if hasString(result.RiskFlags, "changed_cell_count_abnormally_high") {
		t.Fatalf("expected ratio-specific warning without coarse changed-cell warning, got %#v", result.RiskFlags)
	}
}

func TestValidationGateRejectsMildNumericOutlierAutoRepair(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "n-1", "issue_type": "numeric_outlier", "outlier_risk_level": "mild", "risk_level": "low", "issue_score": 0.2},
	)
	repair := validationGateRepairWithApplied(
		1,
		true,
		map[string]any{"issue_id": "n-1", "issue_type": "numeric_outlier", "rows_touched": 1},
	)

	result := buildPostValidation(baseline, repair, validationGateScan(), validationGatePlan("n-1"))

	if result.Verdict != validationGateReject || result.Accepted {
		t.Fatalf("expected reject for mild numeric_outlier auto repair, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, validationRiskMildNumericOutlierAutoRepaired) {
		t.Fatalf("expected mild numeric_outlier risk, got %#v", result.RiskFlags)
	}
}

func TestValidationGateWarnsWhenNumericOutlierModificationShareIsHigh(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "n-1", "issue_type": "numeric_outlier", "outlier_risk_level": "extreme", "risk_level": "low", "issue_score": 0.3},
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "i-2", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.1},
	)
	repair := validationGateRepairWithApplied(
		40,
		true,
		map[string]any{"issue_id": "n-1", "issue_type": "numeric_outlier", "rows_touched": 30},
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "rows_touched": 10},
	)

	result := buildPostValidation(baseline, repair, postScan, validationGatePlan("n-1", "i-1"))

	if result.Verdict != validationGateWarn || !result.Accepted {
		t.Fatalf("expected warn for numeric_outlier modification share, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, validationRiskNumericOutlierModificationShareHigh) {
		t.Fatalf("expected numeric_outlier modification share risk, got %#v", result.RiskFlags)
	}
	numericSummary := mapFromAny(result.Summary["numeric_outlier_modification_summary"])
	if intFromAny(numericSummary["modified_cells"]) != 30 || numericSummary["high_share"] != true {
		t.Fatalf("unexpected numeric_outlier modification summary: %#v", numericSummary)
	}
	if !hasString(stringsFromAny(result.Summary["side_effect_notes"]), "numeric_outlier repairs changed many cells; manual review is recommended") {
		t.Fatalf("expected numeric_outlier side-effect note, got %#v", result.Summary["side_effect_notes"])
	}
}

func TestValidationGateWarnsWhenMissForestDoesNotConverge(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.5},
		map[string]any{"issue_id": "i-2", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.4},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
	)
	repair := validationGateRepair(2, true, "i-1")
	repair["model_evidence"] = []any{
		map[string]any{"issue_id": "i-1", "algorithm_mode": "iterative", "converged": false},
	}

	result := buildPostValidation(baseline, repair, postScan, validationGatePlan("i-1"))

	if result.Verdict != validationGateWarn || !result.Accepted {
		t.Fatalf("expected warn for non-converged MissForest with issue improvement, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, validationRiskMissForestNotConverged) {
		t.Fatalf("expected missforest_not_converged risk, got %#v", result.RiskFlags)
	}
}

func TestValidationGateRejectsNonConvergedMissForestWithoutImprovement(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.5},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "missing_values", "risk_level": "low", "issue_score": 0.5},
	)
	repair := validationGateRepair(1, true, "i-1")
	repair["model_evidence"] = []any{
		map[string]any{"issue_id": "i-1", "algorithm_mode": "iterative", "converged": false},
	}

	result := buildPostValidation(baseline, repair, postScan, validationGatePlan("i-1"))

	if result.Verdict != validationGateReject || result.Accepted {
		t.Fatalf("expected reject for non-converged MissForest without improvement, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, validationRiskMissForestNotConverged) || !hasString(result.RiskFlags, "issue_score_not_improved") {
		t.Fatalf("expected convergence and no-improvement risks, got %#v", result.RiskFlags)
	}
}

func TestValidationGateRejectsIssueCountIncrease(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "medium", "issue_score": 0.3},
	)
	postScan := validationGateScan(
		map[string]any{"issue_id": "post-1", "issue_type": "missing_values", "risk_level": "medium", "issue_score": 0.3},
		map[string]any{"issue_id": "post-2", "issue_type": "rare_category", "risk_level": "low", "issue_score": 0.2},
	)

	result := buildPostValidation(baseline, validationGateRepair(1, true, "i-1"), postScan, validationGatePlan("i-1"))

	if result.Verdict != validationGateReject || result.Accepted {
		t.Fatalf("expected reject, got verdict=%s accepted=%v summary=%#v", result.Verdict, result.Accepted, result.Summary)
	}
	if !hasString(result.RiskFlags, "issue_count_increased") {
		t.Fatalf("expected issue_count_increased risk, got %#v", result.RiskFlags)
	}
}

func TestValidationGateRejectsAffectedColumnIssueCountIncrease(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "age-1", "issue_type": "missing_values", "column": "age", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "city-1", "issue_type": "rare_category", "column": "city", "risk_level": "low", "issue_score": 0.2},
		map[string]any{"issue_id": "city-2", "issue_type": "rare_category", "column": "city", "risk_level": "low", "issue_score": 0.2},
	)
	baseline["column_issue_counts"] = map[string]any{"age": 1, "city": 2}
	postScan := validationGateScan(
		map[string]any{"issue_id": "age-post-1", "issue_type": "missing_values", "column": "age", "risk_level": "low", "issue_score": 0.05},
		map[string]any{"issue_id": "age-post-2", "issue_type": "numeric_outlier", "column": "age", "risk_level": "low", "issue_score": 0.05},
	)
	postScan["affected_columns"] = []string{"age"}
	postScan["column_issue_counts"] = map[string]any{"age": 2}

	result := buildPostValidation(baseline, validationGateRepair(1, true, "age-1"), postScan, validationGatePlan("age-1"))

	if result.Verdict != validationGateReject || result.Accepted {
		t.Fatalf("expected reject for affected column issue increase, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, validationRiskAffectedColumnIssueCountIncreased) {
		t.Fatalf("expected affected column issue-count risk, got %#v", result.RiskFlags)
	}
	deltas := mapsFromAny(result.Summary["affected_column_issue_deltas"])
	if len(deltas) != 1 || asString(deltas[0]["column"]) != "age" || intFromAny(deltas[0]["delta"]) != 1 {
		t.Fatalf("expected age issue-count delta, got %#v", result.Summary["affected_column_issue_deltas"])
	}
}

func TestValidationGateRejectsManualReviewIssueAutoRepair(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "manual-1", "issue_type": "duplicate_record", "risk_level": "medium", "issue_score": 0.5},
	)
	plan := validationGatePlan("manual-1")
	plan.ManualReviewIssueIDs = []string{"manual-1"}

	result := buildPostValidation(baseline, validationGateRepair(1, true, "manual-1"), validationGateScan(), plan)

	if result.Verdict != validationGateReject {
		t.Fatalf("expected reject, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, "manual_review_issue_auto_repaired") {
		t.Fatalf("expected manual_review_issue_auto_repaired risk, got %#v", result.RiskFlags)
	}
}

func TestValidationGateRejectsHighRiskIssueAutoRepair(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "high", "issue_score": 0.9},
	)

	result := buildPostValidation(baseline, validationGateRepair(1, true, "i-1"), validationGateScan(), validationGatePlan("i-1"))

	if result.Verdict != validationGateReject {
		t.Fatalf("expected reject, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, "high_risk_issue_auto_repaired") {
		t.Fatalf("expected high_risk_issue_auto_repaired risk, got %#v", result.RiskFlags)
	}
}

func TestValidationGateRecommendsRollbackWhenManifestMissing(t *testing.T) {
	baseline := validationGateScan(
		map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "medium", "issue_score": 0.5},
	)

	result := buildPostValidation(baseline, validationGateRepair(1, false, "i-1"), validationGateScan(), validationGatePlan("i-1"))

	if result.Verdict != validationGateRollbackRecommended || result.Accepted {
		t.Fatalf("expected rollback_recommended, got verdict=%s accepted=%v summary=%#v", result.Verdict, result.Accepted, result.Summary)
	}
	if result.Summary["rollback_recommended"] != true {
		t.Fatalf("expected rollback_recommended=true, got %#v", result.Summary)
	}
}

func TestValidationGateRejectsRepairError(t *testing.T) {
	result := evaluateValidationGate(validationGateInput{
		BaselineScan: validationGateScan(
			map[string]any{"issue_id": "i-1", "issue_type": "missing_values", "risk_level": "medium", "issue_score": 0.5},
		),
		RepairResult: validationGateRepair(0, false, "i-1"),
		PostScan:     validationGateScan(),
		Plan:         validationGatePlan("i-1"),
		RepairError:  "repair_batch failed",
	})

	if result.Verdict != validationGateReject || result.Summary["rollback_recommended"] != true {
		t.Fatalf("expected reject with rollback recommendation, got %#v", result.Summary)
	}
	if !hasString(result.RiskFlags, "repair_error") {
		t.Fatalf("expected repair_error risk, got %#v", result.RiskFlags)
	}
}
