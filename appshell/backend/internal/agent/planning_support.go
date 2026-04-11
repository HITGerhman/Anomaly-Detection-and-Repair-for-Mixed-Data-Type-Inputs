package agent

// supportedRepairIssueTypes returns the stable issue types that the current
// deterministic repair previews can execute end-to-end.
func supportedRepairIssueTypes() map[string]struct{} {
	return map[string]struct{}{
		"missing_values":  {},
		"numeric_outlier": {},
		"rare_category":   {},
	}
}

// selectRepairableIssues splits the full scan result into issue ids that can
// enter deterministic previews and issues that remain explain-only.
func selectRepairableIssues(scanResult map[string]any) ([]string, []AgentSkippedIssue) {
	issues := listOfMaps(scanResult["issues"])
	if issues == nil {
		issues = []map[string]any{}
	}

	supportedTypes := supportedRepairIssueTypes()
	selectedIssueIDs := make([]string, 0, len(issues))
	skipped := make([]AgentSkippedIssue, 0, len(issues))

	for _, issue := range issues {
		issueID := asString(issue["issue_id"])
		issueType := asString(issue["issue_type"])
		column := asString(issue["column"])
		if _, ok := supportedTypes[issueType]; ok && issueID != "" {
			selectedIssueIDs = append(selectedIssueIDs, issueID)
			continue
		}

		reason := "unsupported_issue_type"
		if issueID == "" {
			reason = "missing_issue_id"
		}
		skipped = append(skipped, AgentSkippedIssue{
			IssueID:   issueID,
			IssueType: issueType,
			Column:    column,
			Reason:    reason,
			Details: map[string]any{
				"issue_type": issueType,
				"column":     column,
			},
		})
	}
	return selectedIssueIDs, skipped
}

// buildPlanningInput snapshots the deterministic artifacts produced by the Go
// control plane before handing them to a Planner implementation.
func buildPlanningInput(sessionID string, goal string, params planningParams, scanResult map[string]any, selectedIssueIDs []string, skippedIssues []AgentSkippedIssue, rulePreview map[string]any, gowerPreview map[string]any, preferenceSnapshot map[string]any, approvalContext map[string]any) PlanningInput {
	return PlanningInput{
		SessionID:               sessionID,
		Goal:                    goal,
		CSVPath:                 params.CSVPath,
		ScanResult:              cloneMap(scanResult),
		SelectedIssueIDs:        append([]string{}, selectedIssueIDs...),
		SkippedIssues:           cloneSkippedIssues(skippedIssues),
		RulePreview:             cloneMap(rulePreview),
		GowerPreview:            cloneMap(gowerPreview),
		ScanConfigOverrides:     cloneMap(params.ScanOverrides),
		RepairStrategyOverrides: cloneMap(params.RepairOverrides),
		ColumnDependencies:      cloneMap(params.ColumnDependencies),
		GowerStrategyOverrides:  cloneMap(params.GowerOverrides),
		ModelDir:                params.ModelDir,
		OutputDir:               params.OutputDir,
		WorkspaceID:             params.WorkspaceID,
		PreferenceSnapshot:      cloneMap(preferenceSnapshot),
		ApprovalContext:         cloneMap(approvalContext),
	}
}
