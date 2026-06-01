package agent

import "strings"

const (
	issueBucketAutoRepair   = "auto_repair"
	issueBucketCautious     = "cautious"
	issueBucketManualReview = "manual_review"
	issueBucketBlocked      = "blocked"

	numericOutlierMildPromptOnlyReason            = "numeric_outlier_mild_prompt_only"
	numericOutlierStrongRequiresReviewReason      = "numeric_outlier_strong_requires_review"
	numericOutlierMissingRiskLevelCautiousReason  = "numeric_outlier_missing_risk_level_cautious"
	numericOutlierExtremeRequiresValidationReason = "numeric_outlier_extreme_requires_validation_or_review"
	numericOutlierRiskPolicyName                  = "numeric_outlier_eligible_extreme_only"
	numericOutlierMildRiskNote                    = "numeric_outlier mild values are statistical outliers but not necessarily data errors"
	numericOutlierAutoRepairRestrictedRiskNote    = "numeric_outlier automatic repair is restricted to eligible extreme cases"
)

type issuePlanBuckets struct {
	AutoRepairIssueIDs     []string
	CautiousIssueIDs       []string
	ManualReviewIssueIDs   []string
	BlockedIssueIDs        []string
	SkippedIssues          []AgentSkippedIssue
	NumericOutlierIssueIDs []string
}

func normalizeIssuePlanBuckets(buckets issuePlanBuckets) issuePlanBuckets {
	buckets.AutoRepairIssueIDs = uniqueStrings(buckets.AutoRepairIssueIDs)
	buckets.CautiousIssueIDs = uniqueStrings(buckets.CautiousIssueIDs)
	buckets.ManualReviewIssueIDs = uniqueStrings(buckets.ManualReviewIssueIDs)
	buckets.BlockedIssueIDs = uniqueStrings(buckets.BlockedIssueIDs)
	buckets.SkippedIssues = cloneSkippedIssues(buckets.SkippedIssues)
	buckets.NumericOutlierIssueIDs = uniqueStrings(buckets.NumericOutlierIssueIDs)
	return buckets
}

func issueBucketForType(issueType string) (string, string) {
	switch issueType {
	case "missing_values", "rare_category":
		return issueBucketAutoRepair, ""
	case "numeric_outlier":
		return issueBucketCautious, "requires_human_review_before_auto_repair"
	case "duplicate_record", "cross_column_consistency":
		return issueBucketManualReview, "manual_review_required"
	default:
		return issueBucketBlocked, "unsupported_issue_type"
	}
}

func numericOutlierBucketForIssue(issue map[string]any) (string, string) {
	riskLevel := strings.ToLower(asString(issue["outlier_risk_level"]))
	autoEligible, hasAutoEligible := boolFromAny(issue["auto_repair_eligible"])
	switch riskLevel {
	case "mild":
		return issueBucketCautious, numericOutlierMildPromptOnlyReason
	case "strong":
		return issueBucketCautious, numericOutlierStrongRequiresReviewReason
	case "extreme":
		if hasAutoEligible && autoEligible {
			return issueBucketAutoRepair, ""
		}
		return issueBucketCautious, numericOutlierExtremeRequiresValidationReason
	default:
		return issueBucketCautious, numericOutlierMissingRiskLevelCautiousReason
	}
}

func issueBucketForScanIssue(issue map[string]any) (string, string) {
	issueType := asString(issue["issue_type"])
	if issueType == "numeric_outlier" {
		return numericOutlierBucketForIssue(issue)
	}
	return issueBucketForType(issueType)
}

func recommendedActionForIssueBucket(bucket string) string {
	switch bucket {
	case issueBucketAutoRepair:
		return "auto_preview"
	case issueBucketCautious:
		return "review_before_auto_repair"
	case issueBucketManualReview:
		return "manual_review"
	default:
		return "block_until_supported"
	}
}

func skippedIssueFromScan(issue map[string]any, bucket string, reason string) AgentSkippedIssue {
	issueID := asString(issue["issue_id"])
	issueType := asString(issue["issue_type"])
	column := asString(issue["column"])
	if issueID == "" {
		bucket = issueBucketBlocked
		reason = "missing_issue_id"
	}
	skipped := AgentSkippedIssue{
		IssueID:   issueID,
		IssueType: issueType,
		Column:    column,
		Reason:    reason,
		Details: map[string]any{
			"issue_type":         issueType,
			"column":             column,
			"bucket":             bucket,
			"recommended_action": recommendedActionForIssueBucket(bucket),
			"reason":             reason,
		},
	}
	if issueType == "numeric_outlier" {
		skipped.Details["risk_policy"] = numericOutlierRiskPolicyName
		skipped.Details["outlier_risk_level"] = asString(issue["outlier_risk_level"])
		if eligible, ok := boolFromAny(issue["auto_repair_eligible"]); ok {
			skipped.Details["auto_repair_eligible"] = eligible
		}
		if policyReason := asString(issue["outlier_policy_reason"]); policyReason != "" {
			skipped.Details["outlier_policy_reason"] = policyReason
		}
	}
	return skipped
}

func selectIssuePlanBuckets(scanResult map[string]any) issuePlanBuckets {
	issues := mapsFromAny(scanResult["issues"])
	buckets := issuePlanBuckets{
		AutoRepairIssueIDs:   []string{},
		CautiousIssueIDs:     []string{},
		ManualReviewIssueIDs: []string{},
		BlockedIssueIDs:      []string{},
		SkippedIssues:        []AgentSkippedIssue{},
	}

	for _, issue := range issues {
		issueID := asString(issue["issue_id"])
		issueType := asString(issue["issue_type"])
		bucket, reason := issueBucketForScanIssue(issue)
		if issueID == "" {
			buckets.SkippedIssues = append(buckets.SkippedIssues, skippedIssueFromScan(issue, issueBucketBlocked, "missing_issue_id"))
			continue
		}
		if issueType == "numeric_outlier" {
			buckets.NumericOutlierIssueIDs = append(buckets.NumericOutlierIssueIDs, issueID)
		}

		switch bucket {
		case issueBucketAutoRepair:
			buckets.AutoRepairIssueIDs = append(buckets.AutoRepairIssueIDs, issueID)
		case issueBucketCautious:
			buckets.CautiousIssueIDs = append(buckets.CautiousIssueIDs, issueID)
			buckets.SkippedIssues = append(buckets.SkippedIssues, skippedIssueFromScan(issue, bucket, reason))
		case issueBucketManualReview:
			buckets.ManualReviewIssueIDs = append(buckets.ManualReviewIssueIDs, issueID)
			buckets.SkippedIssues = append(buckets.SkippedIssues, skippedIssueFromScan(issue, bucket, reason))
		default:
			buckets.BlockedIssueIDs = append(buckets.BlockedIssueIDs, issueID)
			buckets.SkippedIssues = append(buckets.SkippedIssues, skippedIssueFromScan(issue, issueBucketBlocked, reason))
		}
	}
	return normalizeIssuePlanBuckets(buckets)
}

func bucketFromSkippedIssue(item AgentSkippedIssue) string {
	if item.Details != nil {
		switch asString(item.Details["bucket"]) {
		case issueBucketAutoRepair:
			return issueBucketAutoRepair
		case issueBucketCautious:
			return issueBucketCautious
		case issueBucketManualReview:
			return issueBucketManualReview
		case issueBucketBlocked:
			return issueBucketBlocked
		}
	}
	switch item.Reason {
	case "cautious_issue_type", "requires_human_review_before_auto_repair":
		return issueBucketCautious
	case "manual_review_issue_type", "manual_review_required":
		return issueBucketManualReview
	case "missing_issue_id", "blocked_issue_type", "unsupported_issue_type":
		return issueBucketBlocked
	default:
		bucket, _ := issueBucketForType(item.IssueType)
		return bucket
	}
}

func blockedIssueDetail(item AgentSkippedIssue) AgentBlockedIssueDetail {
	reason := asString(item.Reason)
	if reason == "" {
		reason = "unsupported_issue_type"
	}
	suggested := asString(item.Details["recommended_action"])
	if suggested == "" {
		suggested = recommendedActionForIssueBucket(issueBucketBlocked)
	}
	return AgentBlockedIssueDetail{
		IssueID:             item.IssueID,
		IssueType:           item.IssueType,
		Column:              item.Column,
		BlockedReason:       reason,
		BlockedByRule:       "a2_deterministic_issue_bucket_policy",
		SuggestedNextAction: suggested,
	}
}

func cautiousIssueDetail(item AgentSkippedIssue) AgentCautiousIssueDetail {
	reason := asString(item.Reason)
	if reason == "" {
		reason = "requires_human_review_before_auto_repair"
	}
	suggested := asString(item.Details["recommended_action"])
	if suggested == "" {
		suggested = recommendedActionForIssueBucket(issueBucketCautious)
	}
	return AgentCautiousIssueDetail{
		IssueID:          item.IssueID,
		IssueType:        item.IssueType,
		Column:           item.Column,
		RiskReason:       reason,
		ApprovalRequired: true,
		SuggestedAction:  suggested,
	}
}

func issueExplanationDetails(skipped []AgentSkippedIssue) ([]AgentBlockedIssueDetail, []AgentCautiousIssueDetail, map[string]int) {
	blocked := []AgentBlockedIssueDetail{}
	cautious := []AgentCautiousIssueDetail{}
	reasonCounts := map[string]int{}
	for _, item := range skipped {
		switch bucketFromSkippedIssue(item) {
		case issueBucketBlocked:
			detail := blockedIssueDetail(item)
			blocked = append(blocked, detail)
			reasonCounts[detail.BlockedReason]++
		case issueBucketCautious:
			cautious = append(cautious, cautiousIssueDetail(item))
		}
	}
	return blocked, cautious, reasonCounts
}

func issuePlanBucketsFromPlanningInput(input PlanningInput) issuePlanBuckets {
	if len(mapsFromAny(input.ScanResult["issues"])) > 0 {
		return selectIssuePlanBuckets(input.ScanResult)
	}

	buckets := issuePlanBuckets{
		AutoRepairIssueIDs:   append([]string{}, input.SelectedIssueIDs...),
		CautiousIssueIDs:     []string{},
		ManualReviewIssueIDs: []string{},
		BlockedIssueIDs:      []string{},
		SkippedIssues:        cloneSkippedIssues(input.SkippedIssues),
	}
	for _, item := range input.SkippedIssues {
		issueID := asString(item.IssueID)
		if issueID == "" {
			continue
		}
		if item.IssueType == "numeric_outlier" {
			buckets.NumericOutlierIssueIDs = append(buckets.NumericOutlierIssueIDs, issueID)
		}
		switch bucketFromSkippedIssue(item) {
		case issueBucketCautious:
			buckets.CautiousIssueIDs = append(buckets.CautiousIssueIDs, issueID)
		case issueBucketManualReview:
			buckets.ManualReviewIssueIDs = append(buckets.ManualReviewIssueIDs, issueID)
		case issueBucketBlocked:
			buckets.BlockedIssueIDs = append(buckets.BlockedIssueIDs, issueID)
		}
	}
	return normalizeIssuePlanBuckets(buckets)
}

// selectRepairableIssues splits the full scan result into issue ids that can
// enter deterministic previews and issues that remain explain-only.
func selectRepairableIssues(scanResult map[string]any) ([]string, []AgentSkippedIssue) {
	buckets := selectIssuePlanBuckets(scanResult)
	return append([]string{}, buckets.AutoRepairIssueIDs...), cloneSkippedIssues(buckets.SkippedIssues)
}

// buildPlanningInput snapshots the deterministic artifacts produced by the Go
// control plane before handing them to a Planner implementation.
func buildPlanningInput(sessionID string, goal string, params planningParams, scanResult map[string]any, selectedIssueIDs []string, skippedIssues []AgentSkippedIssue, rulePreview map[string]any, gowerPreview map[string]any, missForestPreview map[string]any, preferenceSnapshot map[string]any, approvalContext map[string]any) PlanningInput {
	return PlanningInput{
		SessionID:                   sessionID,
		Goal:                        goal,
		CSVPath:                     params.CSVPath,
		ScanResult:                  cloneMap(scanResult),
		SelectedIssueIDs:            append([]string{}, selectedIssueIDs...),
		SkippedIssues:               cloneSkippedIssues(skippedIssues),
		RulePreview:                 cloneMap(rulePreview),
		GowerPreview:                cloneMap(gowerPreview),
		MissForestPreview:           cloneMap(missForestPreview),
		ScanConfigOverrides:         cloneMap(params.ScanOverrides),
		RepairStrategyOverrides:     cloneMap(params.RepairOverrides),
		ColumnDependencies:          cloneMap(params.ColumnDependencies),
		GowerStrategyOverrides:      cloneMap(params.GowerOverrides),
		MissForestStrategyOverrides: cloneMap(params.MissForestOverrides),
		ModelDir:                    params.ModelDir,
		LLMExplainMode:              params.LLMExplainMode,
		OutputDir:                   params.OutputDir,
		WorkspaceID:                 params.WorkspaceID,
		PreferenceSnapshot:          cloneMap(preferenceSnapshot),
		ApprovalContext:             cloneMap(approvalContext),
	}
}
