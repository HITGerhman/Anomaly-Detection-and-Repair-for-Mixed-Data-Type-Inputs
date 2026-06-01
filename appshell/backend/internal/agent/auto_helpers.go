package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"appshell/backend/internal/engine"
)

type planningProgress struct {
	IntentStart      int
	IntentComplete   int
	ProfileStart     int
	ScanStart        int
	ScanComplete     int
	ProfileComplete  int
	StrategyStart    int
	StrategyComplete int
	RetrieveStart    int
	RetrieveComplete int
	CompareStart     int
	CompareComplete  int
	PlanComplete     int
}

type planningParams struct {
	SessionID           string
	WorkspaceID         string
	CSVPath             string
	Goal                string
	OutputDir           string
	ModelDir            string
	LLMExplainMode      string
	ScanOverrides       map[string]any
	RepairOverrides     map[string]any
	ColumnDependencies  map[string]any
	GowerOverrides      map[string]any
	MissForestOverrides map[string]any
	UserPreferences     map[string]any
}

type planningResult struct {
	Session   AgentSession
	Goal      string
	Plan      AgentPlan
	Baseline  map[string]any
	ScanInput planningParams
}

type postValidationResult struct {
	Summary             map[string]any
	RiskFlags           []string
	Accepted            bool
	Verdict             string
	RollbackRecommended bool
}

type rollbackMetadata struct {
	RollbackID   string
	ManifestPath string
	Directory    string
}

type autoFinalizeInput struct {
	Session        *AgentSession
	Goal           string
	Plan           AgentPlan
	Validation     map[string]any
	Execution      map[string]any
	BaselineScan   map[string]any
	PostScan       map[string]any
	PostValidation map[string]any
	RiskFlags      []string
	Reason         string
	ErrorCode      string
	ErrorMessage   string
}

func defaultValidationResult() map[string]any {
	return map[string]any{
		"status":      "not_run",
		"message":     "Validation has not run.",
		"can_execute": false,
		"preview": map[string]any{
			"status": "not_run",
		},
		"post_execute": map[string]any{
			"status": "not_run",
		},
	}
}

func defaultExecutionResult(autoMode bool) map[string]any {
	return map[string]any{
		"status":               "not_run",
		"auto_mode":            autoMode,
		"post_scan_output_csv": "",
		"rollback_applied":     false,
	}
}

func buildRollbackRecommendation(recommended bool, reason string) map[string]any {
	item := map[string]any{
		"recommended": recommended,
		"reason":      reason,
	}
	if recommended {
		item["restore_target"] = "output_csv"
	}
	return item
}

func defaultSafetyResult() map[string]any {
	return map[string]any{
		"final_verdict":           "not_run",
		"risk_flags":              []string{},
		"baseline_scan_summary":   map[string]any{},
		"post_scan_summary":       map[string]any{},
		"rollback_recommendation": buildRollbackRecommendation(false, "Safety checks have not run."),
		"rollback_execution": map[string]any{
			"status": "not_run",
		},
		"rejected_output_snapshot": "",
	}
}

func buildSafetyResult(verdict string, riskFlags []string, baseline map[string]any, postScan map[string]any, rollbackRecommendation map[string]any, rollbackExecution map[string]any, rejectedOutputSnapshot string) map[string]any {
	safety := defaultSafetyResult()
	safety["final_verdict"] = strings.TrimSpace(verdict)
	safety["risk_flags"] = append([]string{}, uniqueStrings(riskFlags)...)
	safety["baseline_scan_summary"] = cloneMap(baseline)
	safety["post_scan_summary"] = cloneMap(postScan)
	if len(rollbackRecommendation) > 0 {
		safety["rollback_recommendation"] = cloneMap(rollbackRecommendation)
	}
	if len(rollbackExecution) > 0 {
		safety["rollback_execution"] = cloneMap(rollbackExecution)
	}
	safety["rejected_output_snapshot"] = strings.TrimSpace(rejectedOutputSnapshot)
	return safety
}

func buildValidationEnvelope(preview map[string]any) map[string]any {
	envelope := defaultValidationResult()
	if len(preview) == 0 {
		return envelope
	}
	envelope["status"] = asString(preview["status"])
	envelope["message"] = asString(preview["message"])
	envelope["can_execute"] = preview["can_execute"] == true
	envelope["preview"] = cloneMap(preview)
	return envelope
}

func attachPostValidation(validation map[string]any, post map[string]any) map[string]any {
	envelope := cloneMap(validation)
	if len(envelope) == 0 {
		envelope = defaultValidationResult()
	}
	envelope["post_execute"] = cloneMap(post)
	if len(post) == 0 {
		return envelope
	}
	envelope["status"] = asString(post["status"])
	envelope["message"] = asString(post["message"])
	if accepted, ok := post["accepted"].(bool); ok {
		envelope["accepted"] = accepted
	}
	return envelope
}

func uniqueStrings(items []string) []string {
	if len(items) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(items))
	for _, item := range items {
		text := strings.TrimSpace(item)
		if text == "" {
			continue
		}
		if _, exists := seen[text]; exists {
			continue
		}
		seen[text] = struct{}{}
		out = append(out, text)
	}
	return out
}

func appendRiskFlag(flags []string, flag string) []string {
	text := strings.TrimSpace(flag)
	if text == "" {
		return uniqueStrings(flags)
	}
	return uniqueStrings(append(flags, text))
}

func planningProgressForMode(mode string) planningProgress {
	switch strings.TrimSpace(mode) {
	case "auto":
		return planningProgress{
			IntentStart:      6,
			IntentComplete:   10,
			ProfileStart:     12,
			ScanStart:        18,
			ScanComplete:     30,
			ProfileComplete:  34,
			StrategyStart:    38,
			StrategyComplete: 42,
			RetrieveStart:    46,
			RetrieveComplete: 54,
			CompareStart:     58,
			CompareComplete:  64,
			PlanComplete:     68,
		}
	default:
		return planningProgress{
			IntentStart:      8,
			IntentComplete:   14,
			ProfileStart:     18,
			ScanStart:        24,
			ScanComplete:     42,
			ProfileComplete:  48,
			StrategyStart:    54,
			StrategyComplete: 58,
			RetrieveStart:    62,
			RetrieveComplete: 72,
			CompareStart:     76,
			CompareComplete:  82,
			PlanComplete:     90,
		}
	}
}

func buildScanPayload(csvPath string, scanOverrides map[string]any) map[string]any {
	payload := map[string]any{
		"csv_path": strings.TrimSpace(csvPath),
	}
	if len(scanOverrides) > 0 {
		payload["scan_config"] = cloneMap(scanOverrides)
	}
	return payload
}

const postValidationIncrementalThresholdBytes int64 = 512 * 1024 * 1024

func buildScopedScanPayloadFields(affectedColumns []string) map[string]any {
	return map[string]any{
		"scan_scope":       "affected_columns",
		"affected_columns": append([]string{}, uniqueStrings(affectedColumns)...),
	}
}

func outputFileSizeBytes(path string) (int64, bool) {
	info, err := os.Stat(strings.TrimSpace(path))
	if err != nil || info.IsDir() {
		return 0, false
	}
	return info.Size(), true
}

func outputSizeBytesForValidation(execution map[string]any, outputCSV string) (int64, bool) {
	if size := intFromAny(execution["output_size_bytes"]); size > 0 {
		return int64(size), true
	}
	return outputFileSizeBytes(outputCSV)
}

func affectedColumnsFromExecution(execution map[string]any) []string {
	columns := []string{}
	columns = appendColumnsFromRepairs(columns, mapsFromAny(execution["applied_repairs"]))
	columns = appendColumnsFromComparison(columns, mapFromAny(execution["comparison"]))
	for _, step := range mapsFromAny(execution["execution_steps"]) {
		columns = appendColumnsFromRepairs(columns, mapsFromAny(step["applied_repairs"]))
		columns = appendColumnsFromComparison(columns, mapFromAny(step["comparison"]))
	}
	return uniqueStrings(columns)
}

func appendColumnsFromRepairs(columns []string, repairs []map[string]any) []string {
	for _, repair := range repairs {
		column := strings.TrimSpace(asString(repair["column"]))
		if column != "" {
			columns = append(columns, column)
		}
	}
	return columns
}

func appendColumnsFromComparison(columns []string, comparison map[string]any) []string {
	for _, change := range mapsFromAny(comparison["changed_cells_preview"]) {
		column := strings.TrimSpace(asString(change["column"]))
		if column != "" {
			columns = append(columns, column)
		}
	}
	return columns
}

func markIncrementalPostScanEstimate(summary map[string]any, affectedColumns []string, outputSizeBytes int64) map[string]any {
	out := cloneMap(summary)
	out["post_scan_incremental_estimate"] = true
	out["scan_scope"] = "affected_columns"
	out["affected_columns"] = append([]string{}, uniqueStrings(affectedColumns)...)
	if outputSizeBytes >= 0 {
		out["output_size_bytes"] = outputSizeBytes
	}
	return out
}

func scanSummaryFromResult(csvPath string, scanResult map[string]any) map[string]any {
	issues := listOfMaps(scanResult["issues"])
	highRiskCount := 0
	totalIssueScore := 0.0
	for _, issue := range issues {
		if asString(issue["risk_level"]) == "high" {
			highRiskCount++
		}
		totalIssueScore += floatFromAny(issue["issue_score"])
	}
	scanSummary := cloneMap(mapFromAny(scanResult["scan_summary"]))
	columnIssueCounts := map[string]any{}
	for column, count := range columnIssueCountsFromScan(scanResult) {
		columnIssueCounts[column] = count
	}
	return map[string]any{
		"csv_path":               strings.TrimSpace(csvPath),
		"issue_count":            intFromAny(scanResult["issue_count"]),
		"high_risk_issue_count":  highRiskCount,
		"total_issue_score":      totalIssueScore,
		"scan_scope":             firstNonEmpty(asString(scanResult["scan_scope"]), "full"),
		"affected_columns":       stringsFromAny(scanResult["affected_columns"]),
		"column_issue_counts":    columnIssueCounts,
		"scan_summary":           scanSummary,
		"data_profile":           cloneMap(mapFromAny(scanResult["data_profile"])),
		"anomaly_column_count":   intFromAny(scanSummary["anomaly_column_count"]),
		"issue_type_counts":      cloneMap(mapFromAny(scanSummary["issue_type_counts"])),
		"high_risk_columns":      cloneValue(scanSummary["high_risk_columns"]),
		"medium_risk_columns":    cloneValue(scanSummary["medium_risk_columns"]),
		"baseline_source_exists": strings.TrimSpace(csvPath) != "",
	}
}

func newPlanningSession(taskID string, mode string, params planningParams) AgentSession {
	now := time.Now().UTC()
	return AgentSession{
		SessionID:     params.SessionID,
		RootTaskID:    taskID,
		CurrentTaskID: taskID,
		Status:        SessionStatusPlanning,
		Mode:          mode,
		UserGoal:      params.Goal,
		Context: map[string]any{
			"csv_path":                      params.CSVPath,
			"workspace_id":                  params.WorkspaceID,
			"user_goal":                     params.Goal,
			"scan_summary":                  map[string]any{},
			"selected_issue_ids":            []string{},
			"selected_issue_catalog":        []map[string]any{},
			"skipped_issue_types":           []string{},
			"latest_plan_id":                "",
			"validation_preview":            map[string]any{},
			"preview_validation":            map[string]any{},
			"post_scan":                     map[string]any{},
			"post_validation":               map[string]any{},
			"rollback_summary":              map[string]any{},
			"execution_artifacts":           map[string]any{},
			"scan_config_overrides":         cloneMap(params.ScanOverrides),
			"repair_strategy_overrides":     cloneMap(params.RepairOverrides),
			"column_dependencies":           cloneMap(params.ColumnDependencies),
			"gower_strategy_overrides":      cloneMap(params.GowerOverrides),
			"missforest_strategy_overrides": cloneMap(params.MissForestOverrides),
			"llm_explain_mode":              params.LLMExplainMode,
			"user_preferences":              cloneMap(params.UserPreferences),
			"preference_snapshot":           preferenceProfileToMap(defaultPreferenceProfile()),
			"approval_state":                defaultApprovalResult(),
			"risk_assessment":               map[string]any{},
			"candidate_columns":             []string{},
			"time_like_columns":             []string{},
			"model_dir":                     params.ModelDir,
			"baseline_scan":                 map[string]any{},
			"final_verdict":                 "",
			"rejected_output_snapshot":      "",
		},
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func (r *RuntimeRunner) successResponseWithSafety(taskID string, started time.Time, sessionID string, planID string, runMode string, goal string, plan AgentPlan, explanation string, validation map[string]any, execution map[string]any, safety map[string]any, traceSummary TraceSummary) engine.Response {
	resp := r.successResponse(taskID, started, sessionID, planID, runMode, goal, plan, explanation, validation, execution, traceSummary)
	resp.Result["safety"] = cloneMap(safety)
	return resp
}

func (r *RuntimeRunner) errorResponseWithSafety(taskID string, started time.Time, code string, message string, details map[string]any, sessionID string, planID string, runMode string, goal string, plan AgentPlan, validation map[string]any, execution map[string]any, safety map[string]any, explanation string) engine.Response {
	resp := r.errorResponse(taskID, started, code, message, details, sessionID, planID, runMode, goal, plan, validation, execution, explanation)
	resp.Result["safety"] = cloneMap(safety)
	return resp
}

func (r *RuntimeRunner) toolFailureResponseWithSafety(taskID string, started time.Time, sessionID string, planID string, runMode string, goal string, plan AgentPlan, validation map[string]any, execution map[string]any, safety map[string]any, toolResp engine.Response, toolID string) engine.Response {
	resp := r.toolFailureResponse(taskID, started, sessionID, planID, runMode, goal, plan, validation, execution, toolResp, toolID)
	resp.Result["safety"] = cloneMap(safety)
	return resp
}

func parsePlanningParams(payload map[string]any, action string) (planningParams, error) {
	params := planningParams{}
	params.SessionID = strings.TrimSpace(asString(payload["session_id"]))
	if params.SessionID == "" {
		params.SessionID = newSessionID()
	}
	params.CSVPath = strings.TrimSpace(asString(payload["csv_path"]))
	params.Goal = strings.TrimSpace(asString(payload["user_goal"]))
	if params.Goal == "" {
		params.Goal = DefaultUserGoal
	}
	if params.CSVPath == "" {
		return planningParams{}, fmt.Errorf("Field csv_path is required for %s", action)
	}

	var err error
	params.ScanOverrides, err = validateObjectField(payload, "scan_config_overrides")
	if err != nil {
		return planningParams{}, err
	}
	params.RepairOverrides, err = validateObjectField(payload, "repair_strategy_overrides")
	if err != nil {
		return planningParams{}, err
	}
	params.ColumnDependencies, err = validateObjectField(payload, "column_dependencies")
	if err != nil {
		return planningParams{}, err
	}
	params.GowerOverrides, err = validateObjectField(payload, "gower_strategy_overrides")
	if err != nil {
		return planningParams{}, err
	}
	params.MissForestOverrides, err = validateObjectField(payload, "missforest_strategy_overrides")
	if err != nil {
		return planningParams{}, err
	}
	params.OutputDir = strings.TrimSpace(asString(payload["output_dir"]))
	params.ModelDir = strings.TrimSpace(asString(payload["model_dir"]))
	params.LLMExplainMode = strings.TrimSpace(asString(payload["llm_explain_mode"]))
	params.WorkspaceID = strings.TrimSpace(asString(payload["workspace_id"]))
	params.UserPreferences, err = validateObjectField(payload, "user_preferences")
	if err != nil {
		return planningParams{}, err
	}
	return params, nil
}

func fileExists(path string) bool {
	info, err := os.Stat(strings.TrimSpace(path))
	return err == nil && !info.IsDir()
}

func rollbackMetaFromExecution(execution map[string]any) rollbackMetadata {
	rollback := mapFromAny(execution["rollback"])
	meta := rollbackMetadata{}
	meta.RollbackID = asString(rollback["rollback_id"])
	meta.ManifestPath = asString(rollback["manifest_path"])
	if meta.ManifestPath != "" {
		meta.Directory = filepath.Dir(meta.ManifestPath)
		if meta.RollbackID == "" {
			meta.RollbackID = strings.TrimSuffix(filepath.Base(meta.ManifestPath), filepath.Ext(meta.ManifestPath))
		}
	}
	if meta.Directory == "" {
		outputCSV := asString(execution["output_csv"])
		if outputCSV != "" {
			meta.Directory = filepath.Join(filepath.Dir(outputCSV), ".rollback")
		}
	}
	if meta.RollbackID == "" {
		meta.RollbackID = fmt.Sprintf("rb-%d-auto", time.Now().UnixMilli())
	}
	return meta
}

func rejectedSnapshotPath(execution map[string]any) string {
	meta := rollbackMetaFromExecution(execution)
	if strings.TrimSpace(meta.Directory) == "" {
		return ""
	}
	return filepath.Join(meta.Directory, meta.RollbackID+".rejected.csv")
}

func writeRejectedSnapshot(execution map[string]any) (string, error) {
	outputCSV := asString(execution["output_csv"])
	if outputCSV == "" || !fileExists(outputCSV) {
		return "", fmt.Errorf("output csv does not exist")
	}
	target := rejectedSnapshotPath(execution)
	if target == "" {
		return "", fmt.Errorf("rollback directory is not available")
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return "", err
	}
	if err := copyFile(outputCSV, target); err != nil {
		return "", err
	}
	return target, nil
}

func buildVerdictExplanation(plan AgentPlan, validation map[string]any, execution map[string]any, safety map[string]any) string {
	verdict := asString(safety["final_verdict"])
	switch verdict {
	case "accepted":
		if execution["auto_mode"] == true {
			post := mapFromAny(validation["post_execute"])
			return fmt.Sprintf(
				"The agent planned, executed, rescanned, and accepted the %s candidate for plan %s. Remaining issues=%d.",
				plan.SelectedSource,
				plan.PlanID,
				intFromAny(post["after_issue_count"]),
			)
		}
		return fmt.Sprintf(
			"The agent validated and executed the %s candidate for plan %s. Output=%s.",
			plan.SelectedSource,
			plan.PlanID,
			asString(execution["output_csv"]),
		)
	case "validation_rejected":
		return fmt.Sprintf(
			"The agent rejected execution for plan %s during preview validation. %s",
			plan.PlanID,
			asString(validation["message"]),
		)
	case "rolled_back":
		return fmt.Sprintf(
			"The agent executed plan %s, post-execute validation rejected the output, and rollback restored the repaired artifact.",
			plan.PlanID,
		)
	case "rollback_failed":
		return fmt.Sprintf(
			"The agent executed plan %s, but the safety rollback failed. Immediate inspection is required.",
			plan.PlanID,
		)
	default:
		if len(execution) == 0 || asString(execution["status"]) == "not_run" {
			return fmt.Sprintf("The saved plan %s is ready for execution.", plan.PlanID)
		}
		return fmt.Sprintf("The agent completed plan %s with source %s.", plan.PlanID, plan.SelectedSource)
	}
}
