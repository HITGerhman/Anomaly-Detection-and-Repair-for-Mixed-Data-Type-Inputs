package agent

import "strings"

const (
	CognitionProviderLangGraph     = "langgraph"
	CognitionProviderDeterministic = "deterministic"
)

const (
	CognitionStatusEngaged     = "engaged"
	CognitionStatusDegraded    = "degraded"
	CognitionStatusFallback    = "fallback"
	CognitionStatusDisabled    = "disabled"
	CognitionStatusUnavailable = "unavailable"
)

const (
	CognitionFallbackDisabled          = "disabled"
	CognitionFallbackScriptMissing     = "script_missing"
	CognitionFallbackStartupFailed     = "startup_failed"
	CognitionFallbackPortOccupied      = "port_occupied"
	CognitionFallbackHealthcheckFailed = "healthcheck_failed"
	CognitionFallbackPlannerMode       = "planner_mode_fallback"
	CognitionFallbackPlanRequest       = "plan_request_failed"
	CognitionFallbackInvalidCandidate  = "invalid_candidate"
	CognitionFallbackExplainRequest    = "explain_request_failed"
)

type AgentCognitionState struct {
	Provider            string   `json:"provider"`
	Status              string   `json:"status"`
	PlannerMode         string   `json:"planner_mode,omitempty"`
	LLMMode             string   `json:"llm_mode,omitempty"`
	GraphID             string   `json:"graph_id,omitempty"`
	Version             string   `json:"version,omitempty"`
	SelectedCandidateID string   `json:"selected_candidate_id,omitempty"`
	ReasonCodes         []string `json:"reason_codes,omitempty"`
	RiskNote            string   `json:"risk_note,omitempty"`
	Summary             string   `json:"summary,omitempty"`
	FallbackReasonCode  string   `json:"fallback_reason_code,omitempty"`
	FallbackMessage     string   `json:"fallback_message,omitempty"`
}

func (s AgentCognitionState) IsZero() bool {
	return strings.TrimSpace(s.Provider) == "" &&
		strings.TrimSpace(s.Status) == "" &&
		strings.TrimSpace(s.PlannerMode) == "" &&
		strings.TrimSpace(s.LLMMode) == "" &&
		strings.TrimSpace(s.GraphID) == "" &&
		strings.TrimSpace(s.Version) == "" &&
		strings.TrimSpace(s.SelectedCandidateID) == "" &&
		len(s.ReasonCodes) == 0 &&
		strings.TrimSpace(s.RiskNote) == "" &&
		strings.TrimSpace(s.Summary) == "" &&
		strings.TrimSpace(s.FallbackReasonCode) == "" &&
		strings.TrimSpace(s.FallbackMessage) == ""
}

type CognitionTraceSummary struct {
	EventCount          int      `json:"event_count"`
	Provider            string   `json:"provider,omitempty"`
	Status              string   `json:"status,omitempty"`
	LastPhase           string   `json:"last_phase,omitempty"`
	LastSummary         string   `json:"last_summary,omitempty"`
	PlannerMode         string   `json:"planner_mode,omitempty"`
	LLMMode             string   `json:"llm_mode,omitempty"`
	FallbackReasonCode  string   `json:"fallback_reason_code,omitempty"`
	ReasonCodes         []string `json:"reason_codes,omitempty"`
	SelectedCandidateID string   `json:"selected_candidate_id,omitempty"`
}

func (s CognitionTraceSummary) IsZero() bool {
	return s.EventCount == 0 &&
		strings.TrimSpace(s.Provider) == "" &&
		strings.TrimSpace(s.Status) == "" &&
		strings.TrimSpace(s.LastPhase) == "" &&
		strings.TrimSpace(s.LastSummary) == "" &&
		strings.TrimSpace(s.PlannerMode) == "" &&
		strings.TrimSpace(s.LLMMode) == "" &&
		strings.TrimSpace(s.FallbackReasonCode) == "" &&
		len(s.ReasonCodes) == 0 &&
		strings.TrimSpace(s.SelectedCandidateID) == ""
}

func cloneCognitionState(state AgentCognitionState) AgentCognitionState {
	return AgentCognitionState{
		Provider:            strings.TrimSpace(state.Provider),
		Status:              strings.TrimSpace(state.Status),
		PlannerMode:         strings.TrimSpace(state.PlannerMode),
		LLMMode:             strings.TrimSpace(state.LLMMode),
		GraphID:             strings.TrimSpace(state.GraphID),
		Version:             strings.TrimSpace(state.Version),
		SelectedCandidateID: strings.TrimSpace(state.SelectedCandidateID),
		ReasonCodes:         append([]string{}, state.ReasonCodes...),
		RiskNote:            strings.TrimSpace(state.RiskNote),
		Summary:             strings.TrimSpace(state.Summary),
		FallbackReasonCode:  strings.TrimSpace(state.FallbackReasonCode),
		FallbackMessage:     strings.TrimSpace(state.FallbackMessage),
	}
}

func cloneCognitionTraceSummary(summary CognitionTraceSummary) CognitionTraceSummary {
	return CognitionTraceSummary{
		EventCount:          summary.EventCount,
		Provider:            strings.TrimSpace(summary.Provider),
		Status:              strings.TrimSpace(summary.Status),
		LastPhase:           strings.TrimSpace(summary.LastPhase),
		LastSummary:         strings.TrimSpace(summary.LastSummary),
		PlannerMode:         strings.TrimSpace(summary.PlannerMode),
		LLMMode:             strings.TrimSpace(summary.LLMMode),
		FallbackReasonCode:  strings.TrimSpace(summary.FallbackReasonCode),
		ReasonCodes:         append([]string{}, summary.ReasonCodes...),
		SelectedCandidateID: strings.TrimSpace(summary.SelectedCandidateID),
	}
}

func normalizeCognitionState(state AgentCognitionState) AgentCognitionState {
	state = cloneCognitionState(state)
	state.ReasonCodes = uniqueStrings(stringsFromAny(state.ReasonCodes))
	if strings.TrimSpace(state.Provider) == "" {
		state.Provider = CognitionProviderDeterministic
	}
	if strings.TrimSpace(state.Status) == "" {
		state.Status = CognitionStatusFallback
	}
	if strings.TrimSpace(state.PlannerMode) == "" && state.Provider == CognitionProviderDeterministic {
		state.PlannerMode = "fallback"
	}
	if strings.TrimSpace(state.Summary) == "" {
		state.Summary = firstNonEmpty(state.FallbackMessage, "Deterministic planning remained active.")
	}
	return state
}

func cognitionStateToMap(state AgentCognitionState) map[string]any {
	normalized := normalizeCognitionState(state)
	return map[string]any{
		"provider":              normalized.Provider,
		"status":                normalized.Status,
		"planner_mode":          normalized.PlannerMode,
		"llm_mode":              normalized.LLMMode,
		"graph_id":              normalized.GraphID,
		"version":               normalized.Version,
		"selected_candidate_id": normalized.SelectedCandidateID,
		"reason_codes":          append([]string{}, normalized.ReasonCodes...),
		"risk_note":             normalized.RiskNote,
		"summary":               normalized.Summary,
		"fallback_reason_code":  normalized.FallbackReasonCode,
		"fallback_message":      normalized.FallbackMessage,
	}
}

func cognitionSummaryToMap(summary CognitionTraceSummary) map[string]any {
	cloned := cloneCognitionTraceSummary(summary)
	return map[string]any{
		"event_count":           cloned.EventCount,
		"provider":              cloned.Provider,
		"status":                cloned.Status,
		"last_phase":            cloned.LastPhase,
		"last_summary":          cloned.LastSummary,
		"planner_mode":          cloned.PlannerMode,
		"llm_mode":              cloned.LLMMode,
		"fallback_reason_code":  cloned.FallbackReasonCode,
		"reason_codes":          append([]string{}, cloned.ReasonCodes...),
		"selected_candidate_id": cloned.SelectedCandidateID,
	}
}

func buildDeterministicCognitionState(plan AgentPlan) AgentCognitionState {
	summary := firstNonEmpty(plan.ReasoningSummary, plan.UserExplanation, "Deterministic planning selected the current repair path.")
	return normalizeCognitionState(AgentCognitionState{
		Provider:            CognitionProviderDeterministic,
		Status:              CognitionStatusFallback,
		PlannerMode:         "fallback",
		SelectedCandidateID: strings.TrimSpace(plan.SelectedCandidateID),
		ReasonCodes:         append([]string{}, plan.ReasonCodes...),
		RiskNote:            strings.TrimSpace(plan.RiskNote),
		Summary:             summary,
	})
}

func ensurePlanCognition(plan AgentPlan) AgentPlan {
	if strings.TrimSpace(plan.Cognition.Provider) == "" &&
		strings.TrimSpace(plan.Cognition.Status) == "" &&
		strings.TrimSpace(plan.Cognition.Summary) == "" &&
		strings.TrimSpace(plan.Cognition.FallbackReasonCode) == "" {
		plan.Cognition = buildDeterministicCognitionState(plan)
		return plan
	}
	plan.Cognition = normalizeCognitionState(plan.Cognition)
	return plan
}

func langGraphFallbackMessage(reasonCode string) string {
	switch strings.TrimSpace(reasonCode) {
	case CognitionFallbackDisabled:
		return "LangGraph sidecar is disabled, so Go kept deterministic planning active."
	case CognitionFallbackScriptMissing:
		return "LangGraph sidecar script is unavailable, so Go kept deterministic planning active."
	case CognitionFallbackStartupFailed:
		return "LangGraph sidecar could not start, so Go kept deterministic planning active."
	case CognitionFallbackPortOccupied:
		return "LangGraph sidecar port is occupied, so Go kept deterministic planning active."
	case CognitionFallbackHealthcheckFailed:
		return "LangGraph sidecar health checks failed, so Go kept deterministic planning active."
	case CognitionFallbackPlannerMode:
		return "LangGraph reported fallback planner mode, so Go kept deterministic planning active."
	case CognitionFallbackPlanRequest:
		return "LangGraph planning failed, so Go kept deterministic planning active."
	case CognitionFallbackInvalidCandidate:
		return "LangGraph returned an invalid candidate, so Go kept deterministic planning active."
	case CognitionFallbackExplainRequest:
		return "LangGraph explanation degraded, but the selected candidate remained under Go safety controls."
	default:
		return "Deterministic planning remained active."
	}
}

func ClassifyLangGraphAvailabilityError(err error) (string, string, string) {
	if err == nil {
		return CognitionStatusFallback, "", langGraphFallbackMessage("")
	}
	text := strings.ToLower(strings.TrimSpace(err.Error()))
	switch {
	case strings.Contains(text, "disabled"):
		return CognitionStatusDisabled, CognitionFallbackDisabled, langGraphFallbackMessage(CognitionFallbackDisabled)
	case strings.Contains(text, "script unavailable"), strings.Contains(text, "script path is empty"), strings.Contains(text, "file does not exist"), strings.Contains(text, "cannot find the path"):
		return CognitionStatusUnavailable, CognitionFallbackScriptMissing, langGraphFallbackMessage(CognitionFallbackScriptMissing)
	case strings.Contains(text, "port is occupied"):
		return CognitionStatusUnavailable, CognitionFallbackPortOccupied, langGraphFallbackMessage(CognitionFallbackPortOccupied)
	case strings.Contains(text, "start langgraph sidecar failed"):
		return CognitionStatusUnavailable, CognitionFallbackStartupFailed, langGraphFallbackMessage(CognitionFallbackStartupFailed)
	default:
		return CognitionStatusUnavailable, CognitionFallbackHealthcheckFailed, langGraphFallbackMessage(CognitionFallbackHealthcheckFailed)
	}
}

func buildLangGraphFallbackState(plan AgentPlan, health LangGraphHealth, status string, reasonCode string) AgentCognitionState {
	message := langGraphFallbackMessage(reasonCode)
	if strings.TrimSpace(status) == "" {
		status = CognitionStatusFallback
	}
	return normalizeCognitionState(AgentCognitionState{
		Provider:            CognitionProviderDeterministic,
		Status:              status,
		PlannerMode:         firstNonEmpty(strings.TrimSpace(health.PlannerMode), "fallback"),
		LLMMode:             strings.TrimSpace(health.LLMMode),
		GraphID:             strings.TrimSpace(health.GraphID),
		Version:             strings.TrimSpace(health.Version),
		SelectedCandidateID: strings.TrimSpace(plan.SelectedCandidateID),
		ReasonCodes:         append([]string{}, plan.ReasonCodes...),
		RiskNote:            strings.TrimSpace(plan.RiskNote),
		Summary:             firstNonEmpty(plan.ReasoningSummary, plan.UserExplanation, message),
		FallbackReasonCode:  strings.TrimSpace(reasonCode),
		FallbackMessage:     message,
	})
}

func buildLangGraphDegradedState(health LangGraphHealth, plan AgentPlan) AgentCognitionState {
	message := langGraphFallbackMessage(CognitionFallbackExplainRequest)
	return normalizeCognitionState(AgentCognitionState{
		Provider:            CognitionProviderLangGraph,
		Status:              CognitionStatusDegraded,
		PlannerMode:         strings.TrimSpace(health.PlannerMode),
		LLMMode:             strings.TrimSpace(health.LLMMode),
		GraphID:             strings.TrimSpace(health.GraphID),
		Version:             strings.TrimSpace(health.Version),
		SelectedCandidateID: strings.TrimSpace(plan.SelectedCandidateID),
		ReasonCodes:         append([]string{}, plan.ReasonCodes...),
		RiskNote:            strings.TrimSpace(plan.RiskNote),
		Summary:             firstNonEmpty(plan.ReasoningSummary, plan.UserExplanation, message),
		FallbackReasonCode:  CognitionFallbackExplainRequest,
		FallbackMessage:     message,
	})
}

func buildLangGraphEngagedState(health LangGraphHealth, plan AgentPlan) AgentCognitionState {
	return normalizeCognitionState(AgentCognitionState{
		Provider:            CognitionProviderLangGraph,
		Status:              CognitionStatusEngaged,
		PlannerMode:         strings.TrimSpace(health.PlannerMode),
		LLMMode:             strings.TrimSpace(health.LLMMode),
		GraphID:             strings.TrimSpace(health.GraphID),
		Version:             strings.TrimSpace(health.Version),
		SelectedCandidateID: strings.TrimSpace(plan.SelectedCandidateID),
		ReasonCodes:         append([]string{}, plan.ReasonCodes...),
		RiskNote:            strings.TrimSpace(plan.RiskNote),
		Summary:             firstNonEmpty(plan.ReasoningSummary, plan.UserExplanation, strings.TrimSpace(plan.Cognition.Summary)),
	})
}

func cognitionTracePayload(state AgentCognitionState, phase string) map[string]any {
	payload := cognitionStateToMap(state)
	payload["phase"] = strings.TrimSpace(phase)
	payload["summary"] = strings.TrimSpace(state.Summary)
	return payload
}

func cognitionTraceSummary(state AgentCognitionState) string {
	if summary := strings.TrimSpace(state.Summary); summary != "" {
		return summary
	}
	if message := strings.TrimSpace(state.FallbackMessage); message != "" {
		return message
	}
	return langGraphFallbackMessage(strings.TrimSpace(state.FallbackReasonCode))
}
