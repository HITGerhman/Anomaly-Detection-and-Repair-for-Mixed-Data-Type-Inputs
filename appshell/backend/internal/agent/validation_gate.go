package agent

import "fmt"

const (
	validationGateAccept              = "accept"
	validationGateWarn                = "warn"
	validationGateReject              = "reject"
	validationGateRollbackRecommended = "rollback_recommended"

	validationGateCellChangeFloor      = 50
	validationGateCellChangeMultiplier = 20
)

type validationGateInput struct {
	BaselineScan   map[string]any
	RepairResult   map[string]any
	PostScan       map[string]any
	Plan           AgentPlan
	ExtraRiskFlags []string
	RepairError    string
	Message        string
}

func buildPostValidation(baseline map[string]any, repairResult map[string]any, postScan map[string]any, plan AgentPlan) postValidationResult {
	return evaluateValidationGate(validationGateInput{
		BaselineScan: baseline,
		RepairResult: repairResult,
		PostScan:     postScan,
		Plan:         plan,
	})
}

func buildPostValidationFailure(verdict string, message string, riskFlags []string, baseline map[string]any, repairResult map[string]any, postScan map[string]any, plan AgentPlan) postValidationResult {
	if verdict == "" {
		verdict = validationGateReject
		if hasString(riskFlags, "missing_rollback_metadata") && !hasString(riskFlags, "output_csv_missing") && !hasString(riskFlags, "post_scan_failed") {
			verdict = validationGateRollbackRecommended
		}
	}
	result := validationGateSummary(validationGateSummaryInput{
		Verdict:               verdict,
		BaselineScan:          baseline,
		RepairResult:          repairResult,
		PostScan:              postScan,
		RiskNotes:             riskFlags,
		Message:               message,
		RollbackRecommended:   verdict == validationGateReject || verdict == validationGateRollbackRecommended,
		ResolvedIssueCount:    resolvedIssueCount(baseline, postScan, repairResult),
		TotalCellsModified:    repairChangedCellCount(repairResult),
		BeforeIssueCount:      scanIssueCount(baseline),
		AfterIssueCount:       scanIssueCount(postScan),
		BeforeHighRiskCount:   scanHighRiskIssueCount(baseline),
		AfterHighRiskCount:    scanHighRiskIssueCount(postScan),
		BeforeTotalIssueScore: scanTotalIssueScore(baseline),
		AfterTotalIssueScore:  scanTotalIssueScore(postScan),
	})
	return result
}

func evaluateValidationGate(input validationGateInput) postValidationResult {
	beforeIssueCount := scanIssueCount(input.BaselineScan)
	afterIssueCount := scanIssueCount(input.PostScan)
	beforeHighRiskCount := scanHighRiskIssueCount(input.BaselineScan)
	afterHighRiskCount := scanHighRiskIssueCount(input.PostScan)
	beforeTotalIssueScore := scanTotalIssueScore(input.BaselineScan)
	afterTotalIssueScore := scanTotalIssueScore(input.PostScan)
	resolvedCount := resolvedIssueCount(input.BaselineScan, input.PostScan, input.RepairResult)
	totalCellsModified := repairChangedCellCount(input.RepairResult)

	riskNotes := append([]string{}, input.ExtraRiskFlags...)
	if input.RepairError != "" {
		riskNotes = appendRiskFlag(riskNotes, "repair_error")
	}
	if afterIssueCount > beforeIssueCount {
		riskNotes = appendRiskFlag(riskNotes, "issue_count_increased")
	}
	if afterHighRiskCount > beforeHighRiskCount {
		riskNotes = appendRiskFlag(riskNotes, "high_risk_issue_count_increased")
	}
	if shouldWarnForCellChanges(beforeIssueCount, totalCellsModified) {
		riskNotes = appendRiskFlag(riskNotes, "changed_cell_count_abnormally_high")
	}
	appliedIDs := validationAppliedIssueIDs(input.RepairResult, input.Plan)
	if intersects(appliedIDs, input.Plan.ManualReviewIssueIDs) {
		riskNotes = appendRiskFlag(riskNotes, "manual_review_issue_auto_repaired")
	}
	if repairedHighRiskIssue(input.BaselineScan, appliedIDs) {
		riskNotes = appendRiskFlag(riskNotes, "high_risk_issue_auto_repaired")
	}
	if repairWroteOutput(input.RepairResult) && !repairHasRollbackManifest(input.RepairResult) {
		riskNotes = appendRiskFlag(riskNotes, "missing_rollback_metadata")
	}

	issueCountImproved := afterIssueCount < beforeIssueCount
	scoreImproved := afterTotalIssueScore < beforeTotalIssueScore
	verdict := validationGateAccept
	rollbackRecommended := false

	switch {
	case input.RepairError != "":
		verdict = validationGateReject
		rollbackRecommended = true
	case hasAnyString(riskNotes, []string{"issue_count_increased", "high_risk_issue_count_increased", "manual_review_issue_auto_repaired", "high_risk_issue_auto_repaired"}):
		verdict = validationGateReject
		rollbackRecommended = true
	case !issueCountImproved && !scoreImproved:
		verdict = validationGateReject
		rollbackRecommended = true
		riskNotes = appendRiskFlag(riskNotes, "issue_score_not_improved")
	case hasString(riskNotes, "missing_rollback_metadata"):
		verdict = validationGateRollbackRecommended
		rollbackRecommended = true
	case hasString(riskNotes, "changed_cell_count_abnormally_high") || (!issueCountImproved && scoreImproved):
		verdict = validationGateWarn
	default:
		verdict = validationGateAccept
	}

	return validationGateSummary(validationGateSummaryInput{
		Verdict:               verdict,
		BaselineScan:          input.BaselineScan,
		RepairResult:          input.RepairResult,
		PostScan:              input.PostScan,
		RiskNotes:             riskNotes,
		Message:               input.Message,
		RollbackRecommended:   rollbackRecommended,
		ResolvedIssueCount:    resolvedCount,
		TotalCellsModified:    totalCellsModified,
		BeforeIssueCount:      beforeIssueCount,
		AfterIssueCount:       afterIssueCount,
		BeforeHighRiskCount:   beforeHighRiskCount,
		AfterHighRiskCount:    afterHighRiskCount,
		BeforeTotalIssueScore: beforeTotalIssueScore,
		AfterTotalIssueScore:  afterTotalIssueScore,
	})
}

type validationGateSummaryInput struct {
	Verdict               string
	BaselineScan          map[string]any
	RepairResult          map[string]any
	PostScan              map[string]any
	RiskNotes             []string
	Message               string
	RollbackRecommended   bool
	ResolvedIssueCount    int
	TotalCellsModified    int
	BeforeIssueCount      int
	AfterIssueCount       int
	BeforeHighRiskCount   int
	AfterHighRiskCount    int
	BeforeTotalIssueScore float64
	AfterTotalIssueScore  float64
}

func validationGateSummary(input validationGateSummaryInput) postValidationResult {
	riskNotes := uniqueStrings(input.RiskNotes)
	accepted := input.Verdict == validationGateAccept || input.Verdict == validationGateWarn
	status := "rejected"
	if accepted {
		status = "accepted"
	}
	message := input.Message
	if message == "" {
		message = validationGateMessage(input.Verdict)
	}
	summary := map[string]any{
		"phase":                        "post_execute",
		"status":                       status,
		"accepted":                     accepted,
		"message":                      message,
		"verdict":                      input.Verdict,
		"before_issue_count":           input.BeforeIssueCount,
		"after_issue_count":            input.AfterIssueCount,
		"resolved_issue_count":         input.ResolvedIssueCount,
		"total_cells_modified":         input.TotalCellsModified,
		"changed_cell_count":           input.TotalCellsModified,
		"before_high_risk_issue_count": input.BeforeHighRiskCount,
		"after_high_risk_issue_count":  input.AfterHighRiskCount,
		"before_total_issue_score":     input.BeforeTotalIssueScore,
		"after_total_issue_score":      input.AfterTotalIssueScore,
		"risk_notes":                   append([]string{}, riskNotes...),
		"risk_flags":                   append([]string{}, riskNotes...),
		"rollback_recommended":         input.RollbackRecommended,
		"explanation":                  validationGateExplanation(input.Verdict, riskNotes, input.BeforeIssueCount, input.AfterIssueCount),
	}
	return postValidationResult{
		Summary:             summary,
		RiskFlags:           riskNotes,
		Accepted:            accepted,
		Verdict:             input.Verdict,
		RollbackRecommended: input.RollbackRecommended,
	}
}

func validationGateMessage(verdict string) string {
	switch verdict {
	case validationGateAccept:
		return "Validation gate accepted the repaired output."
	case validationGateWarn:
		return "Validation gate accepted the repaired output with warnings."
	case validationGateRollbackRecommended:
		return "Validation gate recommends rollback for the repaired output."
	default:
		return "Validation gate rejected the repaired output and requires rollback."
	}
}

func validationGateExplanation(verdict string, riskNotes []string, beforeCount int, afterCount int) string {
	if len(riskNotes) == 0 {
		return fmt.Sprintf("Issue count changed from %d to %d and no validation gate risks were detected.", beforeCount, afterCount)
	}
	return fmt.Sprintf("Validation gate verdict=%s because %s. Issue count changed from %d to %d.", verdict, joinStrings(riskNotes, ", "), beforeCount, afterCount)
}

func shouldWarnForCellChanges(beforeIssueCount int, totalCellsModified int) bool {
	threshold := validationGateCellChangeFloor
	if scaled := beforeIssueCount * validationGateCellChangeMultiplier; scaled > threshold {
		threshold = scaled
	}
	return totalCellsModified > threshold
}

func scanIssueCount(scan map[string]any) int {
	if scan == nil {
		return 0
	}
	if _, ok := scan["issue_count"]; ok {
		return intFromAny(scan["issue_count"])
	}
	if summary := mapFromAny(scan["scan_summary"]); summary != nil {
		if _, ok := summary["total_issues"]; ok {
			return intFromAny(summary["total_issues"])
		}
	}
	return len(mapsFromAny(scan["issues"]))
}

func scanHighRiskIssueCount(scan map[string]any) int {
	if scan == nil {
		return 0
	}
	if _, ok := scan["high_risk_issue_count"]; ok {
		return intFromAny(scan["high_risk_issue_count"])
	}
	total := 0
	for _, issue := range mapsFromAny(scan["issues"]) {
		if asString(issue["risk_level"]) == "high" {
			total++
		}
	}
	return total
}

func scanTotalIssueScore(scan map[string]any) float64 {
	if scan == nil {
		return 0
	}
	if _, ok := scan["total_issue_score"]; ok {
		return floatFromAny(scan["total_issue_score"])
	}
	total := 0.0
	for _, issue := range mapsFromAny(scan["issues"]) {
		total += floatFromAny(issue["issue_score"])
	}
	return total
}

func resolvedIssueCount(baseline map[string]any, postScan map[string]any, repairResult map[string]any) int {
	if comparison := mapFromAny(repairResult["comparison"]); comparison != nil {
		if _, ok := comparison["resolved_issue_count"]; ok {
			return intFromAny(comparison["resolved_issue_count"])
		}
	}
	before := scanIssueCount(baseline)
	after := scanIssueCount(postScan)
	if before > after {
		return before - after
	}
	return 0
}

func repairChangedCellCount(repairResult map[string]any) int {
	if repairResult == nil {
		return 0
	}
	if _, ok := repairResult["total_cells_modified"]; ok {
		return intFromAny(repairResult["total_cells_modified"])
	}
	if comparison := mapFromAny(repairResult["comparison"]); comparison != nil {
		if _, ok := comparison["changed_cell_count"]; ok {
			return intFromAny(comparison["changed_cell_count"])
		}
	}
	total := 0
	for _, step := range mapsFromAny(repairResult["execution_steps"]) {
		total += intFromAny(mapFromAny(step["comparison"])["changed_cell_count"])
	}
	return total
}

func validationAppliedIssueIDs(repairResult map[string]any, plan AgentPlan) []string {
	ids := []string{}
	ids = append(ids, stringsFromAny(repairResult["applied_issue_ids"])...)
	ids = append(ids, stringsFromAny(repairResult["issue_ids"])...)
	ids = append(ids, stringsFromAny(repairResult["selected_issue_ids"])...)
	for _, item := range mapsFromAny(repairResult["applied_repairs"]) {
		ids = append(ids, asString(item["issue_id"]))
	}
	for _, step := range mapsFromAny(repairResult["execution_steps"]) {
		ids = append(ids, stringsFromAny(step["selected_issue_ids"])...)
	}
	if len(uniqueStrings(ids)) == 0 {
		ids = append(ids, plan.SelectedIssueIDs...)
	}
	return uniqueStrings(ids)
}

func repairedHighRiskIssue(scan map[string]any, appliedIDs []string) bool {
	selected := stringSet(appliedIDs)
	for _, issue := range mapsFromAny(scan["issues"]) {
		issueID := asString(issue["issue_id"])
		if issueID == "" {
			continue
		}
		if _, ok := selected[issueID]; !ok {
			continue
		}
		if asString(issue["risk_level"]) == "high" {
			return true
		}
	}
	return false
}

func repairWroteOutput(repairResult map[string]any) bool {
	if repairResult == nil {
		return false
	}
	return asString(repairResult["output_csv"]) != "" || asString(repairResult["status"]) == "executed"
}

func repairHasRollbackManifest(repairResult map[string]any) bool {
	if repairResult == nil {
		return false
	}
	if asString(repairResult["rollback_manifest_path"]) != "" || asString(repairResult["manifest_path"]) != "" {
		return true
	}
	rollback := mapFromAny(repairResult["rollback"])
	return asString(rollback["manifest_path"]) != ""
}

func intersects(left []string, right []string) bool {
	if len(left) == 0 || len(right) == 0 {
		return false
	}
	rightSet := stringSet(right)
	for _, item := range left {
		if _, ok := rightSet[item]; ok {
			return true
		}
	}
	return false
}

func stringSet(items []string) map[string]struct{} {
	set := map[string]struct{}{}
	for _, item := range uniqueStrings(items) {
		set[item] = struct{}{}
	}
	return set
}

func hasString(items []string, needle string) bool {
	for _, item := range items {
		if item == needle {
			return true
		}
	}
	return false
}

func hasAnyString(items []string, needles []string) bool {
	for _, needle := range needles {
		if hasString(items, needle) {
			return true
		}
	}
	return false
}

func joinStrings(items []string, sep string) string {
	out := ""
	for _, item := range items {
		if item == "" {
			continue
		}
		if out != "" {
			out += sep
		}
		out += item
	}
	return out
}
