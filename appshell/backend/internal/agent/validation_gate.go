package agent

import "fmt"

const (
	validationGateAccept              = "accept"
	validationGateWarn                = "warn"
	validationGateReject              = "reject"
	validationGateRollbackRecommended = "rollback_recommended"

	validationGateCellChangeFloor      = 50
	validationGateCellChangeMultiplier = 20

	validationGateModifiedResolvedRatioWarn = 10.0
	validationGateModifiedResolvedMinCells  = 50
	validationGateNumericOutlierShareWarn   = 0.5
	validationGateNumericOutlierMinCells    = 20

	validationRiskMildNumericOutlierAutoRepaired      = "mild_numeric_outlier_auto_repaired"
	validationRiskModifiedHighRelativeToResolved      = "modified_cell_count_high_relative_to_resolved"
	validationRiskNumericOutlierModificationShareHigh = "numeric_outlier_modification_share_high"
	validationRiskMissForestNotConverged              = "missforest_not_converged"
	validationRiskAffectedColumnIssueCountIncreased   = "affected_column_issue_count_increased"
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
		ResolvedIssueItems:    resolvedIssueItems(scanIssueCount(baseline), scanIssueCount(postScan)),
		ModifiedCellCount:     repairChangedCellCount(repairResult),
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
	resolvedItems := resolvedIssueItems(beforeIssueCount, afterIssueCount)
	modifiedCellCount := repairChangedCellCount(input.RepairResult)
	affectedColumnDeltas := affectedColumnIssueDeltas(input.BaselineScan, input.PostScan)

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
	if shouldWarnForCellChanges(beforeIssueCount, modifiedCellCount) {
		riskNotes = appendRiskFlag(riskNotes, "changed_cell_count_abnormally_high")
	}
	appliedIDs := validationAppliedIssueIDs(input.RepairResult, input.Plan)
	if intersects(appliedIDs, input.Plan.ManualReviewIssueIDs) {
		riskNotes = appendRiskFlag(riskNotes, "manual_review_issue_auto_repaired")
	}
	if repairedMildNumericOutlier(input.BaselineScan, appliedIDs) {
		riskNotes = appendRiskFlag(riskNotes, validationRiskMildNumericOutlierAutoRepaired)
	}
	if repairedHighRiskIssue(input.BaselineScan, appliedIDs) {
		riskNotes = appendRiskFlag(riskNotes, "high_risk_issue_auto_repaired")
	}
	if repairWroteOutput(input.RepairResult) && !repairHasRollbackManifest(input.RepairResult) {
		riskNotes = appendRiskFlag(riskNotes, "missing_rollback_metadata")
	}
	if shouldWarnForModifiedResolvedRatio(modifiedCellCount, resolvedItems) {
		riskNotes = appendRiskFlag(riskNotes, validationRiskModifiedHighRelativeToResolved)
	}
	numericOutlierSummary := numericOutlierModificationSummary(input.RepairResult, input.BaselineScan)
	if boolValue(numericOutlierSummary["high_share"]) {
		riskNotes = appendRiskFlag(riskNotes, validationRiskNumericOutlierModificationShareHigh)
	}
	if missForestNotConverged(input.RepairResult) {
		riskNotes = appendRiskFlag(riskNotes, validationRiskMissForestNotConverged)
	}
	if len(affectedColumnDeltas) > 0 {
		riskNotes = appendRiskFlag(riskNotes, validationRiskAffectedColumnIssueCountIncreased)
	}

	issueCountImproved := afterIssueCount < beforeIssueCount
	scoreImproved := afterTotalIssueScore < beforeTotalIssueScore
	verdict := validationGateAccept
	rollbackRecommended := false

	switch {
	case input.RepairError != "":
		verdict = validationGateReject
		rollbackRecommended = true
	case hasAnyString(riskNotes, []string{"issue_count_increased", "high_risk_issue_count_increased", "manual_review_issue_auto_repaired", validationRiskMildNumericOutlierAutoRepaired, "high_risk_issue_auto_repaired", validationRiskAffectedColumnIssueCountIncreased}):
		verdict = validationGateReject
		rollbackRecommended = true
	case !issueCountImproved && !scoreImproved:
		verdict = validationGateReject
		rollbackRecommended = true
		riskNotes = appendRiskFlag(riskNotes, "issue_score_not_improved")
	case hasString(riskNotes, "missing_rollback_metadata"):
		verdict = validationGateRollbackRecommended
		rollbackRecommended = true
	case hasAnyString(riskNotes, []string{"changed_cell_count_abnormally_high", validationRiskModifiedHighRelativeToResolved, validationRiskNumericOutlierModificationShareHigh, validationRiskMissForestNotConverged, "post_scan_incremental_estimate"}) || (!issueCountImproved && scoreImproved):
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
		ResolvedIssueItems:    resolvedItems,
		ModifiedCellCount:     modifiedCellCount,
		BeforeIssueCount:      beforeIssueCount,
		AfterIssueCount:       afterIssueCount,
		BeforeHighRiskCount:   beforeHighRiskCount,
		AfterHighRiskCount:    afterHighRiskCount,
		BeforeTotalIssueScore: beforeTotalIssueScore,
		AfterTotalIssueScore:  afterTotalIssueScore,
		AffectedColumnDeltas:  affectedColumnDeltas,
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
	ResolvedIssueItems    int
	ModifiedCellCount     int
	BeforeIssueCount      int
	AfterIssueCount       int
	BeforeHighRiskCount   int
	AfterHighRiskCount    int
	BeforeTotalIssueScore float64
	AfterTotalIssueScore  float64
	AffectedColumnDeltas  []map[string]any
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
	modifiedResolvedRatio := modifiedToResolvedRatio(input.ModifiedCellCount, input.ResolvedIssueItems)
	numericOutlierSummary := numericOutlierModificationSummary(input.RepairResult, input.BaselineScan)
	rollbackAvailability := validationRollbackAvailability(input.RepairResult)
	sideEffectNotes := validationSideEffectNotes(riskNotes)
	summary := map[string]any{
		"phase":                                "post_execute",
		"status":                               status,
		"accepted":                             accepted,
		"message":                              message,
		"verdict":                              input.Verdict,
		"before_issue_items":                   input.BeforeIssueCount,
		"after_issue_items":                    input.AfterIssueCount,
		"resolved_issue_items":                 input.ResolvedIssueItems,
		"modified_cell_count":                  input.ModifiedCellCount,
		"before_issue_count":                   input.BeforeIssueCount,
		"after_issue_count":                    input.AfterIssueCount,
		"resolved_issue_count":                 input.ResolvedIssueItems,
		"total_cells_modified":                 input.ModifiedCellCount,
		"changed_cell_count":                   input.ModifiedCellCount,
		"before_high_risk_issue_count":         input.BeforeHighRiskCount,
		"after_high_risk_issue_count":          input.AfterHighRiskCount,
		"before_total_issue_score":             input.BeforeTotalIssueScore,
		"after_total_issue_score":              input.AfterTotalIssueScore,
		"affected_column_issue_deltas":         cloneValue(input.AffectedColumnDeltas),
		"risk_notes":                           append([]string{}, riskNotes...),
		"risk_flags":                           append([]string{}, riskNotes...),
		"rollback_recommended":                 input.RollbackRecommended,
		"acceptance_reason":                    validationGateAcceptanceReason(input.Verdict, riskNotes),
		"modified_to_resolved_ratio":           modifiedResolvedRatio,
		"numeric_outlier_modification_summary": numericOutlierSummary,
		"rollback_availability":                rollbackAvailability,
		"side_effect_notes":                    sideEffectNotes,
		"explanation":                          validationGateExplanation(input.Verdict, riskNotes, input.BeforeIssueCount, input.AfterIssueCount),
		"metric_definitions":                   validationMetricDefinitions(),
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

func validationMetricDefinitions() map[string]any {
	return map[string]any{
		"before_issue_items":                   "Number of issue records detected in the baseline scan before repair.",
		"after_issue_items":                    "Number of issue records detected in the post-repair scan.",
		"resolved_issue_items":                 "Issue-item delta computed as max(before_issue_items - after_issue_items, 0).",
		"modified_cell_count":                  "Number of individual CSV cells modified by the executed repair.",
		"modified_to_resolved_ratio":           "Modified cells divided by resolved issue items; high values indicate possible repair side effects.",
		"numeric_outlier_modification_summary": "Summary of how many repaired cells came from numeric_outlier issues and whether the share is high.",
		"rollback_availability":                "Whether a rollback manifest path is available for the written output.",
		"side_effect_notes":                    "Human-readable validation notes for side-effect risks.",
		"affected_column_issue_deltas":         "Columns touched by repair whose post-repair issue item count is higher than the baseline count.",
		"acceptance_reason":                    "Short machine-readable reason for the validation verdict.",
		"rollback_recommended":                 "Whether validation judged the repaired output unsafe enough to recommend or trigger rollback.",
		"rollback_manifest":                    "A recovery manifest created for written outputs; its existence does not imply rollback is recommended.",
		"resolved_issue_count":                 "Deprecated compatibility alias for resolved_issue_items in validation.post_execute; do not use for resume or defense metrics.",
		"total_cells_modified":                 "Deprecated compatibility alias for modified_cell_count.",
	}
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

func resolvedIssueItems(before int, after int) int {
	if before > after {
		return before - after
	}
	return 0
}

func columnIssueCountsFromScan(scan map[string]any) map[string]int {
	counts := map[string]int{}
	if scan == nil {
		return counts
	}
	switch raw := scan["column_issue_counts"].(type) {
	case map[string]any:
		for column, count := range raw {
			if text := asString(column); text != "" {
				counts[text] = intFromAny(count)
			}
		}
		if len(counts) > 0 {
			return counts
		}
	case map[string]int:
		for column, count := range raw {
			if text := asString(column); text != "" {
				counts[text] = count
			}
		}
		if len(counts) > 0 {
			return counts
		}
	}
	for _, profile := range mapsFromAny(scan["column_profiles"]) {
		column := asString(profile["column"])
		if column == "" {
			continue
		}
		counts[column] = intFromAny(profile["issue_count"])
	}
	if len(counts) > 0 {
		return counts
	}
	for _, issue := range mapsFromAny(scan["issues"]) {
		column := asString(issue["column"])
		if column == "" {
			continue
		}
		counts[column]++
	}
	return counts
}

func affectedColumnIssueDeltas(baseline map[string]any, postScan map[string]any) []map[string]any {
	affectedColumns := stringsFromAny(postScan["affected_columns"])
	if len(affectedColumns) == 0 {
		return []map[string]any{}
	}
	beforeCounts := columnIssueCountsFromScan(baseline)
	afterCounts := columnIssueCountsFromScan(postScan)
	deltas := []map[string]any{}
	for _, column := range affectedColumns {
		before := beforeCounts[column]
		after := afterCounts[column]
		if after <= before {
			continue
		}
		deltas = append(deltas, map[string]any{
			"column":       column,
			"before_count": before,
			"after_count":  after,
			"delta":        after - before,
		})
	}
	return deltas
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

func scanIssuesByID(scan map[string]any) map[string]map[string]any {
	out := map[string]map[string]any{}
	for _, issue := range mapsFromAny(scan["issues"]) {
		issueID := asString(issue["issue_id"])
		if issueID == "" {
			continue
		}
		out[issueID] = issue
	}
	return out
}

func repairedMildNumericOutlier(scan map[string]any, appliedIDs []string) bool {
	issuesByID := scanIssuesByID(scan)
	for _, issueID := range uniqueStrings(appliedIDs) {
		issue := issuesByID[issueID]
		if asString(issue["issue_type"]) != "numeric_outlier" {
			continue
		}
		if asString(issue["outlier_risk_level"]) == "mild" {
			return true
		}
	}
	return false
}

func repairedHighRiskIssue(scan map[string]any, appliedIDs []string) bool {
	issuesByID := scanIssuesByID(scan)
	for _, issueID := range uniqueStrings(appliedIDs) {
		issue := issuesByID[issueID]
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
	return validationRollbackManifestPath(repairResult) != ""
}

func validationRollbackManifestPath(repairResult map[string]any) string {
	if repairResult == nil {
		return ""
	}
	if asString(repairResult["rollback_manifest_path"]) != "" || asString(repairResult["manifest_path"]) != "" {
		if path := asString(repairResult["rollback_manifest_path"]); path != "" {
			return path
		}
		return asString(repairResult["manifest_path"])
	}
	rollback := mapFromAny(repairResult["rollback"])
	return asString(rollback["manifest_path"])
}

func validationRollbackAvailability(repairResult map[string]any) map[string]any {
	path := validationRollbackManifestPath(repairResult)
	return map[string]any{
		"available":     path != "",
		"manifest_path": path,
	}
}

func modifiedToResolvedRatio(modifiedCellCount int, resolvedItems int) float64 {
	if resolvedItems <= 0 {
		if modifiedCellCount > 0 {
			return float64(modifiedCellCount)
		}
		return 0
	}
	return float64(modifiedCellCount) / float64(resolvedItems)
}

func shouldWarnForModifiedResolvedRatio(modifiedCellCount int, resolvedItems int) bool {
	if modifiedCellCount < validationGateModifiedResolvedMinCells || resolvedItems <= 0 {
		return false
	}
	return modifiedToResolvedRatio(modifiedCellCount, resolvedItems) > validationGateModifiedResolvedRatioWarn
}

func numericOutlierModificationSummary(repairResult map[string]any, baseline map[string]any) map[string]any {
	totalModified := repairChangedCellCount(repairResult)
	numericModified := numericOutlierModifiedCells(repairResult, baseline)
	share := 0.0
	if totalModified > 0 {
		share = float64(numericModified) / float64(totalModified)
	}
	appliedIDs := validationAppliedIssueIDs(repairResult, AgentPlan{})
	numericIDs := []string{}
	mildIDs := []string{}
	issuesByID := scanIssuesByID(baseline)
	for _, issueID := range appliedIDs {
		issue := issuesByID[issueID]
		if asString(issue["issue_type"]) != "numeric_outlier" {
			continue
		}
		numericIDs = append(numericIDs, issueID)
		if asString(issue["outlier_risk_level"]) == "mild" {
			mildIDs = append(mildIDs, issueID)
		}
	}
	highShare := numericModified >= validationGateNumericOutlierMinCells && share >= validationGateNumericOutlierShareWarn
	return map[string]any{
		"modified_cells":            numericModified,
		"total_modified_cells":      totalModified,
		"share_of_modified_cells":   share,
		"applied_issue_count":       len(uniqueStrings(numericIDs)),
		"applied_issue_ids":         uniqueStrings(numericIDs),
		"mild_issue_ids":            uniqueStrings(mildIDs),
		"high_share":                highShare,
		"manual_review_recommended": highShare,
	}
}

func numericOutlierModifiedCells(repairResult map[string]any, baseline map[string]any) int {
	issuesByID := scanIssuesByID(baseline)
	appliedRepairs := mapsFromAny(repairResult["applied_repairs"])
	if len(appliedRepairs) > 0 {
		total := 0
		for _, item := range appliedRepairs {
			issueID := asString(item["issue_id"])
			issueType := asString(item["issue_type"])
			if issueType == "" {
				issueType = asString(issuesByID[issueID]["issue_type"])
			}
			if issueType != "numeric_outlier" {
				continue
			}
			total += repairItemChangedCellCount(item)
		}
		return total
	}

	executionSteps := mapsFromAny(repairResult["execution_steps"])
	if len(executionSteps) > 0 {
		total := 0
		for _, step := range executionSteps {
			selectedIDs := stringsFromAny(step["selected_issue_ids"])
			if !containsNumericOutlierIssue(selectedIDs, issuesByID) {
				continue
			}
			total += intFromAny(mapFromAny(step["comparison"])["changed_cell_count"])
		}
		return total
	}

	selectedIDs := validationAppliedIssueIDs(repairResult, AgentPlan{})
	if containsNumericOutlierIssue(selectedIDs, issuesByID) {
		return repairChangedCellCount(repairResult)
	}
	return 0
}

func repairItemChangedCellCount(item map[string]any) int {
	for _, key := range []string{"rows_touched", "resolved_count", "before_count", "changed_cell_count"} {
		if count := intFromAny(item[key]); count > 0 {
			return count
		}
	}
	return len(mapsFromAny(item["cells_preview"]))
}

func containsNumericOutlierIssue(issueIDs []string, issuesByID map[string]map[string]any) bool {
	for _, issueID := range uniqueStrings(issueIDs) {
		if asString(issuesByID[issueID]["issue_type"]) == "numeric_outlier" {
			return true
		}
	}
	return false
}

func missForestNotConverged(repairResult map[string]any) bool {
	if repairResult == nil {
		return false
	}
	for _, item := range mapsFromAny(repairResult["model_evidence"]) {
		if asString(item["algorithm_mode"]) == "iterative" && !boolValue(item["converged"]) {
			return true
		}
	}
	for _, step := range mapsFromAny(repairResult["execution_steps"]) {
		for _, item := range mapsFromAny(step["model_evidence"]) {
			if asString(item["algorithm_mode"]) == "iterative" && !boolValue(item["converged"]) {
				return true
			}
		}
	}
	return false
}

func validationSideEffectNotes(riskNotes []string) []string {
	notes := []string{}
	if hasString(riskNotes, validationRiskNumericOutlierModificationShareHigh) {
		notes = append(notes, "numeric_outlier repairs changed many cells; manual review is recommended")
	}
	if hasString(riskNotes, validationRiskMissForestNotConverged) {
		notes = append(notes, "MissForest iterative repair did not converge within max_iter; manual review is recommended")
	}
	if hasString(riskNotes, "post_scan_incremental_estimate") {
		notes = append(notes, "post-execute validation used an affected-column incremental estimate instead of a full scan")
	}
	if hasString(riskNotes, validationRiskModifiedHighRelativeToResolved) {
		notes = append(notes, "modified cell count is high relative to resolved issue count; manual review is recommended")
	}
	if hasString(riskNotes, validationRiskMildNumericOutlierAutoRepaired) {
		notes = append(notes, "mild numeric_outlier was repaired automatically; rollback is recommended")
	}
	if hasString(riskNotes, validationRiskAffectedColumnIssueCountIncreased) {
		notes = append(notes, "one or more affected columns have more issue items after repair than in the baseline; rollback is recommended")
	}
	if hasString(riskNotes, "missing_rollback_metadata") {
		notes = append(notes, "rollback manifest is unavailable for the repaired output")
	}
	return uniqueStrings(notes)
}

func validationGateAcceptanceReason(verdict string, riskNotes []string) string {
	switch verdict {
	case validationGateAccept:
		return "issue_count_improved_and_side_effects_controlled"
	case validationGateWarn:
		return "accepted_with_side_effect_warnings"
	case validationGateRollbackRecommended:
		return "rollback_manifest_missing"
	default:
		if len(riskNotes) > 0 {
			return "rejected_due_to_validation_risks"
		}
		return "rejected_without_measurable_improvement"
	}
}

func boolValue(value any) bool {
	typed, ok := value.(bool)
	return ok && typed
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
