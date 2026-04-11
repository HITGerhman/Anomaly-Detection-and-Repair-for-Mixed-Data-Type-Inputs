package agent

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	approvalStatusNotRequired = "not_required"
	approvalStatusRequired    = "required"
	approvalStatusApproved    = "approved"
	approvalStatusRejected    = "rejected"
)

type preferenceOverrides struct {
	ConservativeMode           *bool
	AvoidTimeColumns           *bool
	ProtectedColumns           []string
	HasProtectedColumns        bool
	RequireApprovalForHighRisk *bool
}

func defaultPreferenceProfile() AgentPreferenceProfile {
	return AgentPreferenceProfile{
		ConservativeMode:           false,
		AvoidTimeColumns:           true,
		ProtectedColumns:           []string{},
		RequireApprovalForHighRisk: true,
	}
}

func normalizePreferenceProfile(profile AgentPreferenceProfile) AgentPreferenceProfile {
	profile.ProtectedColumns = uniqueStrings(profile.ProtectedColumns)
	return profile
}

func preferenceProfileToMap(profile AgentPreferenceProfile) map[string]any {
	normalized := normalizePreferenceProfile(profile)
	return map[string]any{
		"conservative_mode":            normalized.ConservativeMode,
		"avoid_time_columns":           normalized.AvoidTimeColumns,
		"protected_columns":            append([]string{}, normalized.ProtectedColumns...),
		"require_approval_for_high_risk": normalized.RequireApprovalForHighRisk,
	}
}

func boolPointer(value bool) *bool {
	return &value
}

func boolFromAny(value any) (bool, bool) {
	switch typed := value.(type) {
	case bool:
		return typed, true
	case string:
		switch strings.ToLower(strings.TrimSpace(typed)) {
		case "true", "1", "yes", "on":
			return true, true
		case "false", "0", "no", "off":
			return false, true
		}
	}
	return false, false
}

func stringsFromAny(value any) []string {
	switch typed := value.(type) {
	case []string:
		return uniqueStrings(typed)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			text := asString(item)
			if text != "" {
				out = append(out, text)
			}
		}
		return uniqueStrings(out)
	case string:
		parts := strings.Split(typed, ",")
		out := make([]string, 0, len(parts))
		for _, part := range parts {
			if text := strings.TrimSpace(part); text != "" {
				out = append(out, text)
			}
		}
		return uniqueStrings(out)
	default:
		return []string{}
	}
}

func mapsFromAny(value any) []map[string]any {
	switch typed := value.(type) {
	case []map[string]any:
		out := make([]map[string]any, len(typed))
		for idx, item := range typed {
			out[idx] = cloneMap(item)
		}
		return out
	case []any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			mapped := mapFromAny(item)
			if mapped != nil {
				out = append(out, mapped)
			}
		}
		return out
	default:
		return []map[string]any{}
	}
}

func parsePreferenceOverrides(raw map[string]any) preferenceOverrides {
	if len(raw) == 0 {
		return preferenceOverrides{}
	}
	overrides := preferenceOverrides{}
	if value, ok := boolFromAny(raw["conservative_mode"]); ok {
		overrides.ConservativeMode = boolPointer(value)
	}
	if value, ok := boolFromAny(raw["avoid_time_columns"]); ok {
		overrides.AvoidTimeColumns = boolPointer(value)
	}
	if value, exists := raw["protected_columns"]; exists {
		overrides.ProtectedColumns = stringsFromAny(value)
		overrides.HasProtectedColumns = true
	}
	if value, ok := boolFromAny(raw["require_approval_for_high_risk"]); ok {
		overrides.RequireApprovalForHighRisk = boolPointer(value)
	}
	return overrides
}

func mergePreferenceProfile(saved AgentPreferenceProfile, overrides preferenceOverrides) AgentPreferenceProfile {
	profile := normalizePreferenceProfile(saved)
	if overrides.ConservativeMode != nil {
		profile.ConservativeMode = *overrides.ConservativeMode
	}
	if overrides.AvoidTimeColumns != nil {
		profile.AvoidTimeColumns = *overrides.AvoidTimeColumns
	}
	if overrides.HasProtectedColumns {
		profile.ProtectedColumns = uniqueStrings(overrides.ProtectedColumns)
	}
	if overrides.RequireApprovalForHighRisk != nil {
		profile.RequireApprovalForHighRisk = *overrides.RequireApprovalForHighRisk
	}
	return normalizePreferenceProfile(profile)
}

func resolvePreferenceSnapshot(store SessionStore, workspaceID string, raw map[string]any) (AgentPreferenceRecord, bool, AgentPreferenceProfile, error) {
	defaults := defaultPreferenceProfile()
	saved := defaults
	record := AgentPreferenceRecord{
		WorkspaceID: strings.TrimSpace(workspaceID),
		Profile:     defaults,
	}
	found := false
	if store != nil && strings.TrimSpace(workspaceID) != "" {
		ctx, cancel := persistenceContext()
		defer cancel()
		stored, ok, err := store.GetPreferences(ctx, workspaceID)
		if err != nil {
			return AgentPreferenceRecord{}, false, AgentPreferenceProfile{}, err
		}
		if ok {
			record = stored
			saved = normalizePreferenceProfile(stored.Profile)
			found = true
		}
	}
	profile := mergePreferenceProfile(saved, parsePreferenceOverrides(raw))
	record.WorkspaceID = strings.TrimSpace(workspaceID)
	record.Profile = profile
	return record, found, profile, nil
}

func applyPreferenceRepairDefaults(repairOverrides map[string]any, profile AgentPreferenceProfile) map[string]any {
	resolved := cloneMap(repairOverrides)
	if !profile.ConservativeMode {
		return resolved
	}
	if len(resolved) == 0 {
		resolved = map[string]any{}
	}
	if _, ok := resolved["conflict_policy"]; !ok {
		resolved["conflict_policy"] = "skip_conflict"
	}
	if _, ok := resolved["outlier"]; !ok {
		resolved["outlier"] = "skip"
	}
	if _, ok := resolved["missing_numeric"]; !ok {
		resolved["missing_numeric"] = "median"
	}
	if _, ok := resolved["missing_categorical"]; !ok {
		resolved["missing_categorical"] = "mode"
	}
	if _, ok := resolved["rare_category"]; !ok {
		resolved["rare_category"] = "mode"
	}
	return resolved
}

func resolveWorkspaceID(explicitWorkspaceID string, csvPath string) string {
	if workspaceID := strings.TrimSpace(explicitWorkspaceID); workspaceID != "" {
		return workspaceID
	}
	cleanCSV := strings.TrimSpace(csvPath)
	if cleanCSV == "" {
		return ""
	}
	if absPath, err := filepath.Abs(cleanCSV); err == nil {
		return filepath.Clean(filepath.Dir(absPath))
	}
	return filepath.Clean(filepath.Dir(cleanCSV))
}

func buildSelectedIssueCatalog(scanResult map[string]any, selectedIssueIDs []string) []map[string]any {
	selected := map[string]struct{}{}
	for _, issueID := range selectedIssueIDs {
		if text := strings.TrimSpace(issueID); text != "" {
			selected[text] = struct{}{}
		}
	}
	out := make([]map[string]any, 0, len(selected))
	for _, issue := range mapsFromAny(scanResult["issues"]) {
		issueID := asString(issue["issue_id"])
		if issueID == "" {
			continue
		}
		if _, ok := selected[issueID]; !ok {
			continue
		}
		out = append(out, map[string]any{
			"issue_id":    issueID,
			"column":      asString(issue["column"]),
			"issue_type":  asString(issue["issue_type"]),
			"risk_level":  asString(issue["risk_level"]),
			"severity":    asString(issue["severity"]),
			"issue_score": floatFromAny(issue["issue_score"]),
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		leftColumn := asString(out[i]["column"])
		rightColumn := asString(out[j]["column"])
		if leftColumn == rightColumn {
			return asString(out[i]["issue_id"]) < asString(out[j]["issue_id"])
		}
		return leftColumn < rightColumn
	})
	return out
}

func buildCandidateColumns(selectedIssueCatalog []map[string]any) []string {
	columns := make([]string, 0, len(selectedIssueCatalog))
	for _, item := range selectedIssueCatalog {
		if column := asString(item["column"]); column != "" {
			columns = append(columns, column)
		}
	}
	sort.Strings(columns)
	return uniqueStrings(columns)
}

func detectTimeLikeColumns(scanResult map[string]any, candidateColumns []string) []string {
	if len(candidateColumns) == 0 {
		return []string{}
	}
	columnSet := map[string]struct{}{}
	for _, column := range candidateColumns {
		if text := strings.TrimSpace(column); text != "" {
			columnSet[text] = struct{}{}
		}
	}

	timeLike := map[string]struct{}{}
	for _, item := range mapsFromAny(scanResult["column_profiles"]) {
		column := asString(item["column"])
		if _, ok := columnSet[column]; !ok {
			continue
		}
		dtype := strings.ToLower(asString(item["dtype"]))
		if strings.Contains(dtype, "datetime") ||
			strings.Contains(dtype, "timestamp") ||
			strings.Contains(dtype, "timedelta") ||
			strings.Contains(dtype, "date") ||
			strings.Contains(dtype, "time") {
			timeLike[column] = struct{}{}
		}
	}
	for column := range columnSet {
		lower := strings.ToLower(column)
		if strings.Contains(lower, "datetime") ||
			strings.Contains(lower, "timestamp") ||
			strings.Contains(lower, "date") ||
			strings.Contains(lower, "time") {
			timeLike[column] = struct{}{}
		}
	}
	out := make([]string, 0, len(timeLike))
	for column := range timeLike {
		out = append(out, column)
	}
	sort.Strings(out)
	return out
}

func buildPlanningApprovalContext(baseline map[string]any, candidateColumns []string, timeLikeColumns []string, profile AgentPreferenceProfile) map[string]any {
	highRiskColumns := intersectionStrings(stringsFromAny(baseline["high_risk_columns"]), candidateColumns)
	protectedColumns := intersectionStrings(profile.ProtectedColumns, candidateColumns)
	reasonCodes := make([]string, 0, 4)
	if profile.RequireApprovalForHighRisk && len(highRiskColumns) > 0 {
		reasonCodes = append(reasonCodes, "high_risk_columns_selected")
	}
	if profile.AvoidTimeColumns && len(timeLikeColumns) > 0 {
		reasonCodes = append(reasonCodes, "time_like_columns_selected")
	}
	if len(protectedColumns) > 0 {
		reasonCodes = append(reasonCodes, "protected_columns_selected")
	}
	return map[string]any{
		"candidate_columns":      append([]string{}, candidateColumns...),
		"high_risk_columns":      highRiskColumns,
		"time_like_columns":      append([]string{}, timeLikeColumns...),
		"protected_columns":      protectedColumns,
		"deterministic_reasons":  uniqueStrings(reasonCodes),
		"deterministic_required": len(reasonCodes) > 0,
	}
}

func buildRiskAssessment(plan AgentPlan, baseline map[string]any, selectedIssueCatalog []map[string]any, candidateColumns []string, timeLikeColumns []string, profile AgentPreferenceProfile) map[string]any {
	highRiskColumns := intersectionStrings(stringsFromAny(baseline["high_risk_columns"]), candidateColumns)
	protectedColumns := intersectionStrings(profile.ProtectedColumns, candidateColumns)
	reasonCodes := make([]string, 0, 6)
	if profile.RequireApprovalForHighRisk && len(highRiskColumns) > 0 {
		reasonCodes = append(reasonCodes, "high_risk_columns_selected")
	}
	if profile.AvoidTimeColumns && len(timeLikeColumns) > 0 {
		reasonCodes = append(reasonCodes, "time_like_columns_selected")
	}
	if len(protectedColumns) > 0 {
		reasonCodes = append(reasonCodes, "protected_columns_selected")
	}
	if plan.ApprovalNeeded {
		reasonCodes = append(reasonCodes, "planner_requested_approval")
	}
	reasonCodes = uniqueStrings(reasonCodes)

	return map[string]any{
		"required":               len(reasonCodes) > 0,
		"reason_codes":           reasonCodes,
		"candidate_columns":      append([]string{}, candidateColumns...),
		"risk_columns":           highRiskColumns,
		"protected_columns":      protectedColumns,
		"time_like_columns":      append([]string{}, timeLikeColumns...),
		"selected_issue_count":   len(selectedIssueCatalog),
		"selected_issue_catalog": cloneValue(selectedIssueCatalog),
		"selected_source":        strings.TrimSpace(plan.SelectedSource),
		"planner_requested":      plan.ApprovalNeeded,
		"message":                buildRiskAssessmentMessage(reasonCodes),
	}
}

func buildRiskAssessmentMessage(reasonCodes []string) string {
	if len(reasonCodes) == 0 {
		return "Deterministic preview passed and no approval gate is required."
	}
	parts := make([]string, 0, len(reasonCodes))
	for _, code := range uniqueStrings(reasonCodes) {
		switch code {
		case "high_risk_columns_selected":
			parts = append(parts, "selected issues touch high-risk columns")
		case "time_like_columns_selected":
			parts = append(parts, "selected issues touch time-like columns")
		case "protected_columns_selected":
			parts = append(parts, "selected issues touch protected columns")
		case "planner_requested_approval":
			parts = append(parts, "LangGraph requested approval")
		default:
			parts = append(parts, code)
		}
	}
	return "Approval is required before writing output because " + strings.Join(parts, "; ") + "."
}

func buildApprovalState(status string, required bool, riskAssessment map[string]any) map[string]any {
	return map[string]any{
		"status":       strings.TrimSpace(status),
		"required":     required,
		"decision":     strings.TrimSpace(status),
		"reason_codes": stringsFromAny(riskAssessment["reason_codes"]),
		"message":      asString(riskAssessment["message"]),
	}
}

func enrichApprovalState(state map[string]any, taskID string, decision string) map[string]any {
	updated := cloneMap(state)
	now := time.Now().UTC().Format(time.RFC3339Nano)
	if strings.TrimSpace(taskID) != "" {
		updated["task_id"] = strings.TrimSpace(taskID)
	}
	if text := strings.TrimSpace(decision); text != "" {
		updated["decision"] = text
		updated["decided_at"] = now
	} else if updated["requested_at"] == nil {
		updated["requested_at"] = now
	}
	return updated
}

func defaultApprovalResult() map[string]any {
	return map[string]any{
		"status":            approvalStatusNotRequired,
		"required":          false,
		"reason_codes":      []string{},
		"risk_columns":      []string{},
		"protected_columns": []string{},
		"time_like_columns": []string{},
		"message":           "No approval gate is currently active.",
	}
}

func approvalResultFromContext(context map[string]any) map[string]any {
	riskAssessment := mapFromAny(context["risk_assessment"])
	approvalState := mapFromAny(context["approval_state"])
	if len(riskAssessment) == 0 && len(approvalState) == 0 {
		return defaultApprovalResult()
	}
	result := defaultApprovalResult()
	status := approvalStatusNotRequired
	if len(approvalState) > 0 {
		status = firstNonEmpty(asString(approvalState["status"]), status)
	}
	required, _ := approvalState["required"].(bool)
	if !required {
		required, _ = riskAssessment["required"].(bool)
	}
	result["status"] = status
	result["required"] = required
	result["reason_codes"] = stringsFromAny(firstNonNil(approvalState["reason_codes"], riskAssessment["reason_codes"]))
	result["risk_columns"] = stringsFromAny(riskAssessment["risk_columns"])
	result["protected_columns"] = stringsFromAny(riskAssessment["protected_columns"])
	result["time_like_columns"] = stringsFromAny(riskAssessment["time_like_columns"])
	result["message"] = firstNonEmpty(asString(approvalState["message"]), asString(riskAssessment["message"]), asString(result["message"]))
	return result
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
}

func firstNonNil(values ...any) any {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func intersectionStrings(left []string, right []string) []string {
	if len(left) == 0 || len(right) == 0 {
		return []string{}
	}
	set := map[string]struct{}{}
	for _, item := range right {
		if text := strings.TrimSpace(item); text != "" {
			set[text] = struct{}{}
		}
	}
	out := make([]string, 0, len(left))
	for _, item := range left {
		text := strings.TrimSpace(item)
		if text == "" {
			continue
		}
		if _, ok := set[text]; ok {
			out = append(out, text)
		}
	}
	sort.Strings(out)
	return uniqueStrings(out)
}

func updateSessionApprovalContext(session *AgentSession, riskAssessment map[string]any, approvalState map[string]any) {
	if session == nil {
		return
	}
	if session.Context == nil {
		session.Context = map[string]any{}
	}
	session.Context["risk_assessment"] = cloneMap(riskAssessment)
	session.Context["approval_state"] = cloneMap(approvalState)
}

func preferenceProfileFromContext(value any) AgentPreferenceProfile {
	raw := mapFromAny(value)
	if len(raw) == 0 {
		return defaultPreferenceProfile()
	}
	profile := defaultPreferenceProfile()
	if parsed, ok := boolFromAny(raw["conservative_mode"]); ok {
		profile.ConservativeMode = parsed
	}
	if parsed, ok := boolFromAny(raw["avoid_time_columns"]); ok {
		profile.AvoidTimeColumns = parsed
	}
	if parsed, ok := boolFromAny(raw["require_approval_for_high_risk"]); ok {
		profile.RequireApprovalForHighRisk = parsed
	}
	profile.ProtectedColumns = stringsFromAny(raw["protected_columns"])
	return normalizePreferenceProfile(profile)
}

func refreshRiskAssessmentForSession(session *AgentSession, overrideProfile *AgentPreferenceProfile) map[string]any {
	if session == nil {
		return map[string]any{}
	}
	profile := preferenceProfileFromContext(session.Context["preference_snapshot"])
	if overrideProfile != nil {
		profile = normalizePreferenceProfile(*overrideProfile)
	}
	return buildRiskAssessment(
		session.LatestPlan,
		mapFromAny(session.Context["baseline_scan"]),
		mapsFromAny(session.Context["selected_issue_catalog"]),
		stringsFromAny(session.Context["candidate_columns"]),
		stringsFromAny(session.Context["time_like_columns"]),
		profile,
	)
}

func fileParentExists(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	info, err := os.Stat(filepath.Dir(path))
	return err == nil && info.IsDir()
}
