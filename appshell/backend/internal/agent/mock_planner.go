package agent

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

type DeterministicPlanner struct{}

type MockPlanner = DeterministicPlanner

var _ Planner = (*DeterministicPlanner)(nil)

func NewDeterministicPlanner() *DeterministicPlanner {
	return &DeterministicPlanner{}
}

func NewMockPlanner() *MockPlanner {
	return NewDeterministicPlanner()
}

func toolSummaryForSource(source string) string {
	switch source {
	case "rule":
		return "Deterministic rule-based repair preview."
	case "gower":
		return "Gower neighbor retrieval repair preview."
	case "missforest":
		return "Iterative MissForest random forest imputation repair preview."
	case "hybrid":
		return "Issue-level merged plan from rule, Gower, and MissForest previews."
	default:
		return source
	}
}

func comparisonFromPreview(preview map[string]any, fallbackBefore int) map[string]any {
	comparison := mapFromAny(preview["comparison"])
	if comparison == nil {
		comparison = map[string]any{}
	}
	if intFromAny(comparison["before_issue_count"]) <= 0 {
		comparison["before_issue_count"] = fallbackBefore
	}
	if _, ok := comparison["after_issue_count"]; !ok {
		comparison["after_issue_count"] = fallbackBefore
	}
	if _, ok := comparison["resolved_issue_count"]; !ok {
		comparison["resolved_issue_count"] = 0
	}
	if _, ok := comparison["changed_cell_count"]; !ok {
		comparison["changed_cell_count"] = 0
	}
	return comparison
}

func issueMetricsFromPreview(preview map[string]any) (map[string]map[string]any, map[string]struct{}) {
	metrics := map[string]map[string]any{}
	skipped := map[string]struct{}{}

	for _, item := range listOfMaps(preview["applied_repairs"]) {
		issueID := asString(item["issue_id"])
		if issueID == "" {
			continue
		}
		metrics[issueID] = cloneMap(item)
	}
	for _, item := range listOfMaps(preview["neighbor_evidence"]) {
		issueID := asString(item["issue_id"])
		if issueID == "" {
			continue
		}
		existing := metrics[issueID]
		if existing == nil {
			existing = map[string]any{}
		}
		for key, value := range item {
			existing[key] = value
		}
		metrics[issueID] = existing
	}
	for _, item := range listOfMaps(preview["model_evidence"]) {
		issueID := asString(item["issue_id"])
		if issueID == "" {
			continue
		}
		existing := metrics[issueID]
		if existing == nil {
			existing = map[string]any{}
		}
		for key, value := range item {
			existing[key] = value
		}
		metrics[issueID] = existing
	}
	for _, item := range listOfMaps(preview["skipped_issues"]) {
		issueID := asString(item["issue_id"])
		if issueID == "" {
			continue
		}
		skipped[issueID] = struct{}{}
	}
	return metrics, skipped
}

func issueRowsTouched(issue map[string]any) int {
	rows := intFromAny(issue["rows_touched"])
	if rows > 0 {
		return rows
	}
	return 0
}

func issueConfidence(issue map[string]any) float64 {
	if issue == nil {
		return 0.0
	}
	switch value := issue["candidate_confidence"].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	default:
		return 0.0
	}
}

func chooseIssueSource(ruleIssue map[string]any, hasRule bool, gowerIssue map[string]any, hasGower bool) string {
	if hasRule && !hasGower {
		return "rule"
	}
	if hasGower && !hasRule {
		return "gower"
	}
	if !hasRule && !hasGower {
		return ""
	}

	ruleResolved := intFromAny(ruleIssue["resolved_count"])
	gowerResolved := intFromAny(gowerIssue["resolved_count"])
	if ruleResolved > gowerResolved {
		return "rule"
	}
	if gowerResolved > ruleResolved {
		return "gower"
	}

	ruleConfidence := issueConfidence(ruleIssue)
	gowerConfidence := issueConfidence(gowerIssue)
	if ruleConfidence > gowerConfidence {
		return "rule"
	}
	if gowerConfidence > ruleConfidence {
		return "gower"
	}

	ruleRows := issueRowsTouched(ruleIssue)
	gowerRows := issueRowsTouched(gowerIssue)
	if ruleRows < gowerRows {
		return "rule"
	}
	if gowerRows < ruleRows {
		return "gower"
	}
	return "rule"
}

type issueSourceOption struct {
	Source    string
	Issue     map[string]any
	Available bool
}

func chooseIssueSourceFromOptions(options []issueSourceOption) string {
	best := issueSourceOption{}
	for _, option := range options {
		if !option.Available {
			continue
		}
		if best.Source == "" {
			best = option
			continue
		}
		if issueSourceOptionBetter(option, best) {
			best = option
		}
	}
	return best.Source
}

func issueSourceOptionBetter(left issueSourceOption, right issueSourceOption) bool {
	leftResolved := intFromAny(left.Issue["resolved_count"])
	rightResolved := intFromAny(right.Issue["resolved_count"])
	if leftResolved != rightResolved {
		return leftResolved > rightResolved
	}
	leftConfidence := issueConfidence(left.Issue)
	rightConfidence := issueConfidence(right.Issue)
	if leftConfidence != rightConfidence {
		return leftConfidence > rightConfidence
	}
	leftRows := issueRowsTouched(left.Issue)
	rightRows := issueRowsTouched(right.Issue)
	if leftRows != rightRows {
		return leftRows < rightRows
	}
	priority := map[string]int{
		"rule":       0,
		"gower":      1,
		"missforest": 2,
	}
	return priority[left.Source] < priority[right.Source]
}

func buildCandidateScore(candidate RepairCandidate) map[string]any {
	before := intFromAny(candidate.Comparison["before_issue_count"])
	after := intFromAny(candidate.Comparison["after_issue_count"])
	resolved := intFromAny(candidate.Comparison["resolved_issue_count"])
	changed := intFromAny(candidate.Comparison["changed_cell_count"])
	if before <= 0 {
		before = after + resolved
	}
	improvementRatio := 0.0
	if before > 0 {
		improvementRatio = float64(resolved) / float64(before)
	}
	overall := float64(resolved)*100.0 + improvementRatio*25.0 - float64(after)*12.0 - float64(changed)*0.5
	if !candidate.Executable {
		overall -= 10_000.0
	}
	return map[string]any{
		"score_version":        "candidate_score_v1",
		"overall":              overall,
		"resolved_weight":      resolved,
		"remaining_penalty":    after,
		"side_effect_penalty":  changed,
		"improvement_ratio":    improvementRatio,
		"planner_write_policy": "plan_only_no_data_write",
	}
}

func attachCandidateScore(candidate RepairCandidate) RepairCandidate {
	candidate.Score = buildCandidateScore(candidate)
	comparison := cloneMap(candidate.Comparison)
	comparison["candidate_score"] = floatFromAny(candidate.Score["overall"])
	comparison["score_version"] = asString(candidate.Score["score_version"])
	candidate.Comparison = comparison
	return candidate
}

func candidateBetter(left RepairCandidate, right RepairCandidate) bool {
	leftScore := floatFromAny(left.Score["overall"])
	rightScore := floatFromAny(right.Score["overall"])
	if leftScore != rightScore {
		return leftScore > rightScore
	}

	leftAfter := intFromAny(left.Comparison["after_issue_count"])
	rightAfter := intFromAny(right.Comparison["after_issue_count"])
	if leftAfter != rightAfter {
		return leftAfter < rightAfter
	}

	leftResolved := intFromAny(left.Comparison["resolved_issue_count"])
	rightResolved := intFromAny(right.Comparison["resolved_issue_count"])
	if leftResolved != rightResolved {
		return leftResolved > rightResolved
	}

	leftChanged := intFromAny(left.Comparison["changed_cell_count"])
	rightChanged := intFromAny(right.Comparison["changed_cell_count"])
	if leftChanged != rightChanged {
		return leftChanged < rightChanged
	}

	priority := map[string]int{
		"hybrid":     0,
		"rule":       1,
		"gower":      2,
		"missforest": 3,
	}
	return priority[left.Source] < priority[right.Source]
}

func buildRulePayloads(input PlanningInput) ([]map[string]any, []map[string]any) {
	planPayload := map[string]any{
		"csv_path":        input.CSVPath,
		"issue_ids":       append([]string{}, input.SelectedIssueIDs...),
		"plan_only":       true,
		"write_output":    false,
		"enable_rollback": false,
	}
	if len(input.ScanConfigOverrides) > 0 {
		planPayload["scan_config"] = cloneMap(input.ScanConfigOverrides)
	}
	if len(input.RepairStrategyOverrides) > 0 {
		planPayload["repair_strategy"] = cloneMap(input.RepairStrategyOverrides)
	}
	if len(input.ColumnDependencies) > 0 {
		planPayload["column_dependencies"] = cloneMap(input.ColumnDependencies)
	}
	if strings.TrimSpace(input.OutputDir) != "" {
		planPayload["output_dir"] = strings.TrimSpace(input.OutputDir)
	}

	executePayload := cloneMap(planPayload)
	executePayload["plan_only"] = false
	executePayload["write_output"] = true
	executePayload["enable_rollback"] = true
	return []map[string]any{planPayload}, []map[string]any{executePayload}
}

func buildGowerPayloads(input PlanningInput) ([]map[string]any, []map[string]any) {
	planPayload := map[string]any{
		"csv_path":        input.CSVPath,
		"issue_ids":       append([]string{}, input.SelectedIssueIDs...),
		"plan_only":       true,
		"write_output":    false,
		"enable_rollback": false,
	}
	if len(input.ScanConfigOverrides) > 0 {
		planPayload["scan_config"] = cloneMap(input.ScanConfigOverrides)
	}
	if len(input.ColumnDependencies) > 0 {
		planPayload["column_dependencies"] = cloneMap(input.ColumnDependencies)
	}
	if len(input.GowerStrategyOverrides) > 0 {
		planPayload["gower_strategy"] = withDefaultGowerCandidateLimit(input.GowerStrategyOverrides)
	} else {
		planPayload["gower_strategy"] = withDefaultGowerCandidateLimit(nil)
	}
	if strings.TrimSpace(input.ModelDir) != "" {
		planPayload["model_dir"] = strings.TrimSpace(input.ModelDir)
	}
	if strings.TrimSpace(input.OutputDir) != "" {
		planPayload["output_dir"] = strings.TrimSpace(input.OutputDir)
	}

	executePayload := cloneMap(planPayload)
	executePayload["plan_only"] = false
	executePayload["write_output"] = true
	executePayload["enable_rollback"] = true
	return []map[string]any{planPayload}, []map[string]any{executePayload}
}

func buildMissForestPayloads(input PlanningInput) ([]map[string]any, []map[string]any) {
	planPayload := map[string]any{
		"csv_path":        input.CSVPath,
		"issue_ids":       append([]string{}, input.SelectedIssueIDs...),
		"plan_only":       true,
		"write_output":    false,
		"enable_rollback": false,
	}
	if len(input.ScanConfigOverrides) > 0 {
		planPayload["scan_config"] = cloneMap(input.ScanConfigOverrides)
	}
	planPayload["missforest_strategy"] = withDefaultMissForestStrategy(input.MissForestStrategyOverrides)
	if strings.TrimSpace(input.OutputDir) != "" {
		planPayload["output_dir"] = strings.TrimSpace(input.OutputDir)
	}

	executePayload := cloneMap(planPayload)
	executePayload["plan_only"] = false
	executePayload["write_output"] = true
	executePayload["enable_rollback"] = true
	return []map[string]any{planPayload}, []map[string]any{executePayload}
}

func buildHybridComparison(beforeCount int, issueSourceMap map[string]any, ruleIssues map[string]map[string]any, gowerIssues map[string]map[string]any, missForestIssues map[string]map[string]any) map[string]any {
	resolved := 0
	changed := 0
	for issueID, rawSource := range issueSourceMap {
		source := asString(rawSource)
		var item map[string]any
		switch source {
		case "gower":
			item = gowerIssues[issueID]
		case "missforest":
			item = missForestIssues[issueID]
		default:
			item = ruleIssues[issueID]
		}
		resolved += intFromAny(item["resolved_count"])
		changed += issueRowsTouched(item)
	}
	after := beforeCount - resolved
	if after < 0 {
		after = 0
	}
	return map[string]any{
		"before_issue_count":   beforeCount,
		"after_issue_count":    after,
		"resolved_issue_count": resolved,
		"changed_cell_count":   changed,
	}
}

func deterministicReasonCodes(buckets issuePlanBuckets, selectedSource string) []string {
	codes := []string{"deterministic_planner"}
	if len(buckets.AutoRepairIssueIDs) > 0 {
		codes = append(codes, "auto_repair_candidates_selected")
	} else {
		codes = append(codes, "no_auto_repair_candidates")
	}
	if len(buckets.CautiousIssueIDs) > 0 {
		codes = append(codes, "cautious_issues_pending_review")
	}
	if len(buckets.ManualReviewIssueIDs) > 0 {
		codes = append(codes, "manual_review_required")
	}
	if len(buckets.BlockedIssueIDs) > 0 {
		codes = append(codes, "blocked_issues_present")
	}
	if source := strings.TrimSpace(selectedSource); source != "" {
		codes = append(codes, "selected_"+source+"_candidate")
	}
	return uniqueStrings(codes)
}

func deterministicRiskNote(buckets issuePlanBuckets) string {
	notes := []string{}
	if len(buckets.CautiousIssueIDs)+len(buckets.ManualReviewIssueIDs)+len(buckets.BlockedIssueIDs) == 0 {
		if len(buckets.NumericOutlierIssueIDs) > 0 {
			notes = append(notes, "Only deterministic policy-approved issue types are selected for automatic execution.")
		} else {
			notes = append(notes, "Only deterministic low-risk issue types are selected for automatic execution.")
		}
	} else {
		notes = append(notes, fmt.Sprintf(
			"Automatic execution is limited to %d low-risk issues; %d cautious, %d manual-review, and %d blocked issues remain outside the write payload.",
			len(buckets.AutoRepairIssueIDs),
			len(buckets.CautiousIssueIDs),
			len(buckets.ManualReviewIssueIDs),
			len(buckets.BlockedIssueIDs),
		))
	}
	if len(buckets.NumericOutlierIssueIDs) > 0 {
		notes = append(notes, numericOutlierMildRiskNote+".")
		notes = append(notes, numericOutlierAutoRepairRestrictedRiskNote+".")
	}
	return strings.Join(notes, " ")
}

func deterministicExplanationBullets(buckets issuePlanBuckets, selectedCandidate RepairCandidate) []string {
	bullets := []string{
		fmt.Sprintf("Selected %d auto-repair issue ids for deterministic preview and execution.", len(buckets.AutoRepairIssueIDs)),
		fmt.Sprintf("Candidate source selected by deterministic score: %s.", selectedCandidate.Source),
	}
	if len(buckets.CautiousIssueIDs) > 0 {
		bullets = append(bullets, fmt.Sprintf("%d numeric_outlier issues are marked cautious and excluded from automatic write payloads.", len(buckets.CautiousIssueIDs)))
	}
	if len(buckets.NumericOutlierIssueIDs) > 0 {
		bullets = append(bullets, numericOutlierAutoRepairRestrictedRiskNote+".")
	}
	if len(buckets.ManualReviewIssueIDs) > 0 {
		bullets = append(bullets, fmt.Sprintf("%d duplicate or cross-column issues require manual review.", len(buckets.ManualReviewIssueIDs)))
	}
	if len(buckets.BlockedIssueIDs) > 0 {
		bullets = append(bullets, fmt.Sprintf("%d unsupported issues are blocked until a deterministic tool policy exists.", len(buckets.BlockedIssueIDs)))
	}
	return bullets
}

func (p *DeterministicPlanner) BuildPlan(_ context.Context, input PlanningInput) (AgentPlan, error) {
	buckets := issuePlanBucketsFromPlanningInput(input)
	selectedIssueIDs := append([]string{}, buckets.AutoRepairIssueIDs...)
	skipped := cloneSkippedIssues(buckets.SkippedIssues)
	previewInput := input
	previewInput.SelectedIssueIDs = append([]string{}, selectedIssueIDs...)
	previewInput.SkippedIssues = cloneSkippedIssues(skipped)
	beforeIssueCount := intFromAny(input.ScanResult["issue_count"])

	rulePlanPayloads, ruleExecutePayloads := buildRulePayloads(previewInput)
	gowerPlanPayloads, gowerExecutePayloads := buildGowerPayloads(previewInput)
	missForestPlanPayloads, missForestExecutePayloads := buildMissForestPayloads(previewInput)
	ruleComparison := comparisonFromPreview(input.RulePreview, beforeIssueCount)
	gowerComparison := comparisonFromPreview(input.GowerPreview, beforeIssueCount)
	missForestComparison := comparisonFromPreview(input.MissForestPreview, beforeIssueCount)
	ruleIssues, ruleSkipped := issueMetricsFromPreview(input.RulePreview)
	gowerIssues, gowerSkipped := issueMetricsFromPreview(input.GowerPreview)
	missForestIssues, missForestSkipped := issueMetricsFromPreview(input.MissForestPreview)

	ruleCandidate := RepairCandidate{
		CandidateID:      "candidate-rule",
		Source:           "rule",
		ToolSequence:     []string{"engine.repair_batch"},
		PlanPayloads:     rulePlanPayloads,
		ExecutePayloads:  ruleExecutePayloads,
		SelectedIssueIDs: append([]string{}, selectedIssueIDs...),
		IssueSourceMap:   map[string]any{},
		Comparison:       cloneMap(ruleComparison),
		Summary:          toolSummaryForSource("rule"),
		Executable:       intFromAny(ruleComparison["resolved_issue_count"]) > 0,
	}

	gowerCandidate := RepairCandidate{
		CandidateID:      "candidate-gower",
		Source:           "gower",
		ToolSequence:     []string{"engine.repair_with_gower"},
		PlanPayloads:     gowerPlanPayloads,
		ExecutePayloads:  gowerExecutePayloads,
		SelectedIssueIDs: append([]string{}, selectedIssueIDs...),
		IssueSourceMap:   map[string]any{},
		Comparison:       cloneMap(gowerComparison),
		Summary:          toolSummaryForSource("gower"),
		Executable:       intFromAny(gowerComparison["resolved_issue_count"]) > 0,
	}
	missForestCandidate := RepairCandidate{
		CandidateID:      "candidate-missforest",
		Source:           "missforest",
		ToolSequence:     []string{"engine.repair_with_missforest"},
		PlanPayloads:     missForestPlanPayloads,
		ExecutePayloads:  missForestExecutePayloads,
		SelectedIssueIDs: append([]string{}, selectedIssueIDs...),
		IssueSourceMap:   map[string]any{},
		Comparison:       cloneMap(missForestComparison),
		Summary:          toolSummaryForSource("missforest"),
		Executable:       intFromAny(missForestComparison["resolved_issue_count"]) > 0,
	}

	issueSourceMap := map[string]any{}
	hybridSelectedIssueIDs := make([]string, 0, len(selectedIssueIDs))
	for _, issueID := range selectedIssueIDs {
		_, ruleWasSkipped := ruleSkipped[issueID]
		_, gowerWasSkipped := gowerSkipped[issueID]
		_, missForestWasSkipped := missForestSkipped[issueID]
		ruleIssue, hasRule := ruleIssues[issueID]
		gowerIssue, hasGower := gowerIssues[issueID]
		missForestIssue, hasMissForest := missForestIssues[issueID]
		if ruleWasSkipped {
			hasRule = false
		}
		if gowerWasSkipped {
			hasGower = false
		}
		if missForestWasSkipped {
			hasMissForest = false
		}
		source := chooseIssueSourceFromOptions([]issueSourceOption{
			{Source: "rule", Issue: ruleIssue, Available: hasRule},
			{Source: "gower", Issue: gowerIssue, Available: hasGower},
			{Source: "missforest", Issue: missForestIssue, Available: hasMissForest},
		})
		if source == "" {
			continue
		}
		issueSourceMap[issueID] = source
		hybridSelectedIssueIDs = append(hybridSelectedIssueIDs, issueID)
	}

	hybridRuleIDs := make([]string, 0, len(issueSourceMap))
	hybridGowerIDs := make([]string, 0, len(issueSourceMap))
	hybridMissForestIDs := make([]string, 0, len(issueSourceMap))
	for _, issueID := range hybridSelectedIssueIDs {
		switch asString(issueSourceMap[issueID]) {
		case "gower":
			hybridGowerIDs = append(hybridGowerIDs, issueID)
		case "missforest":
			hybridMissForestIDs = append(hybridMissForestIDs, issueID)
		default:
			hybridRuleIDs = append(hybridRuleIDs, issueID)
		}
	}

	hybridPlanPayloads := make([]map[string]any, 0, 2)
	hybridExecutePayloads := make([]map[string]any, 0, 2)
	if len(hybridRuleIDs) > 0 {
		payload := cloneMap(rulePlanPayloads[0])
		payload["issue_ids"] = append([]string{}, hybridRuleIDs...)
		hybridPlanPayloads = append(hybridPlanPayloads, payload)

		execPayload := cloneMap(ruleExecutePayloads[0])
		execPayload["issue_ids"] = append([]string{}, hybridRuleIDs...)
		hybridExecutePayloads = append(hybridExecutePayloads, execPayload)
	}
	if len(hybridGowerIDs) > 0 {
		payload := cloneMap(gowerPlanPayloads[0])
		payload["issue_ids"] = append([]string{}, hybridGowerIDs...)
		hybridPlanPayloads = append(hybridPlanPayloads, payload)

		execPayload := cloneMap(gowerExecutePayloads[0])
		execPayload["issue_ids"] = append([]string{}, hybridGowerIDs...)
		hybridExecutePayloads = append(hybridExecutePayloads, execPayload)
	}
	if len(hybridMissForestIDs) > 0 {
		payload := cloneMap(missForestPlanPayloads[0])
		payload["issue_ids"] = append([]string{}, hybridMissForestIDs...)
		hybridPlanPayloads = append(hybridPlanPayloads, payload)

		execPayload := cloneMap(missForestExecutePayloads[0])
		execPayload["issue_ids"] = append([]string{}, hybridMissForestIDs...)
		hybridExecutePayloads = append(hybridExecutePayloads, execPayload)
	}

	hybridComparison := buildHybridComparison(beforeIssueCount, issueSourceMap, ruleIssues, gowerIssues, missForestIssues)
	hybridCandidate := RepairCandidate{
		CandidateID:      "candidate-hybrid",
		Source:           "hybrid",
		ToolSequence:     []string{},
		PlanPayloads:     hybridPlanPayloads,
		ExecutePayloads:  hybridExecutePayloads,
		SelectedIssueIDs: hybridSelectedIssueIDs,
		IssueSourceMap:   cloneMap(issueSourceMap),
		Comparison:       cloneMap(hybridComparison),
		Summary:          toolSummaryForSource("hybrid"),
		Executable:       intFromAny(hybridComparison["resolved_issue_count"]) > 0,
	}
	if len(hybridRuleIDs) > 0 {
		hybridCandidate.ToolSequence = append(hybridCandidate.ToolSequence, "engine.repair_batch")
	}
	if len(hybridGowerIDs) > 0 {
		hybridCandidate.ToolSequence = append(hybridCandidate.ToolSequence, "engine.repair_with_gower")
	}
	if len(hybridMissForestIDs) > 0 {
		hybridCandidate.ToolSequence = append(hybridCandidate.ToolSequence, "engine.repair_with_missforest")
	}

	candidates := []RepairCandidate{
		attachCandidateScore(ruleCandidate),
		attachCandidateScore(gowerCandidate),
		attachCandidateScore(missForestCandidate),
		attachCandidateScore(hybridCandidate),
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidateBetter(candidates[i], candidates[j])
	})
	selectedCandidate := candidates[0]

	skippedTypeSet := map[string]struct{}{}
	skippedTypes := make([]string, 0, len(skipped))
	for _, item := range skipped {
		if item.IssueType == "" {
			continue
		}
		if _, exists := skippedTypeSet[item.IssueType]; exists {
			continue
		}
		skippedTypeSet[item.IssueType] = struct{}{}
		skippedTypes = append(skippedTypes, item.IssueType)
	}
	sort.Strings(skippedTypes)

	reasoningSummary := fmt.Sprintf(
		"Compared rule, Gower, MissForest, and hybrid candidates for %d supported issues. Selected %s because it produced the best scored preview under deterministic tie-break rules.",
		len(selectedIssueIDs),
		selectedCandidate.Source,
	)
	userExplanation := fmt.Sprintf(
		"The agent compared rule-based repair, Gower neighbor repair, MissForest repair, and a hybrid issue-level merge. It selected the %s candidate with candidate_score=%.2f, after_issue_count=%d, and changed_cell_count=%d.",
		selectedCandidate.Source,
		floatFromAny(selectedCandidate.Score["overall"]),
		intFromAny(selectedCandidate.Comparison["after_issue_count"]),
		intFromAny(selectedCandidate.Comparison["changed_cell_count"]),
	)
	if len(skippedTypes) > 0 {
		userExplanation += fmt.Sprintf(" Non-auto issue types remain outside automatic execution: %s.", strings.Join(skippedTypes, ", "))
	}

	proposedToolID := ""
	proposedPayload := map[string]any{}
	if len(selectedCandidate.ToolSequence) > 0 {
		proposedToolID = selectedCandidate.ToolSequence[0]
	}
	if len(selectedCandidate.ExecutePayloads) > 0 {
		proposedPayload = cloneMap(selectedCandidate.ExecutePayloads[0])
	}
	blockedDetails, cautiousDetails, blockedReasonCounts := issueExplanationDetails(skipped)

	plan := AgentPlan{
		PlanID:               newPlanID(),
		Status:               "planned",
		SelectedIssueIDs:     selectedIssueIDs,
		AutoRepairIssueIDs:   append([]string{}, buckets.AutoRepairIssueIDs...),
		CautiousIssueIDs:     append([]string{}, buckets.CautiousIssueIDs...),
		ManualReviewIssueIDs: append([]string{}, buckets.ManualReviewIssueIDs...),
		BlockedIssueIDs:      append([]string{}, buckets.BlockedIssueIDs...),
		CautiousIssueDetails: cautiousDetails,
		BlockedIssueDetails:  blockedDetails,
		BlockedReasonCounts:  blockedReasonCounts,
		SkippedIssues:        skipped,
		Candidates:           candidates,
		SelectedCandidateID:  selectedCandidate.CandidateID,
		SelectedSource:       selectedCandidate.Source,
		IssueSourceMap:       cloneMap(selectedCandidate.IssueSourceMap),
		ProposedToolID:       proposedToolID,
		ProposedPayload:      proposedPayload,
		IntentLabel:          "auto_repair",
		StrategyLabel:        "deterministic_planner",
		ReasonCodes:          deterministicReasonCodes(buckets, selectedCandidate.Source),
		RiskNote:             deterministicRiskNote(buckets),
		ExplanationBullets:   deterministicExplanationBullets(buckets, selectedCandidate),
		ReasoningSummary:     reasoningSummary,
		UserExplanation:      userExplanation,
	}
	plan.Cognition = buildDeterministicCognitionState(plan)
	return plan, nil
}
