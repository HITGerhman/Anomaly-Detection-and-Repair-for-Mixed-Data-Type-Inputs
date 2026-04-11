package agent

import "testing"

func TestSummarizeTraceEventsIncludesCognitionSummary(t *testing.T) {
	events := []AgentTraceEvent{
		{
			AgentName: "repair_planner",
			TraceType: TraceCognitionTrace,
			Summary:   "LangGraph selected the primary candidate.",
			Payload: map[string]any{
				"phase":                 "plan_complete",
				"provider":              CognitionProviderLangGraph,
				"status":                CognitionStatusEngaged,
				"planner_mode":          "llm",
				"llm_mode":              "configured",
				"selected_candidate_id": "candidate-gower",
				"reason_codes":          []string{"phase_e"},
				"summary":               "LangGraph selected the primary candidate.",
			},
		},
		{
			AgentName: "validator",
			TraceType: TraceValidation,
			Summary:   "Preview validation passed.",
			Payload:   map[string]any{"phase": "preview"},
		},
	}

	summary := SummarizeTraceEvents(events)
	if summary.EventCount != 2 || summary.TraceTypeCounts[TraceCognitionTrace] != 1 {
		t.Fatalf("unexpected trace summary counts: %+v", summary)
	}
	if summary.Cognition.EventCount != 1 {
		t.Fatalf("expected cognition event count, got %+v", summary.Cognition)
	}
	if summary.Cognition.Provider != CognitionProviderLangGraph || summary.Cognition.Status != CognitionStatusEngaged {
		t.Fatalf("unexpected cognition provider/status: %+v", summary.Cognition)
	}
	if summary.Cognition.LastPhase != "plan_complete" || summary.Cognition.LastSummary == "" {
		t.Fatalf("expected cognition phase/summary, got %+v", summary.Cognition)
	}
}

func TestBuildAgentExplanationPayloadUsesCognitionMode(t *testing.T) {
	plan := AgentPlan{
		PlanID:           "plan-1",
		ReasoningSummary: "LangGraph selected the safest candidate.",
		UserExplanation:  "The run stayed inside the deterministic safety boundary.",
		ReasonCodes:      []string{"phase_e"},
		RiskNote:         "validation first",
		Cognition: AgentCognitionState{
			Provider:            CognitionProviderLangGraph,
			Status:              CognitionStatusDegraded,
			PlannerMode:         "llm",
			LLMMode:             "configured",
			SelectedCandidateID: "candidate-gower",
			ReasonCodes:         []string{"phase_e"},
			RiskNote:            "validation first",
			Summary:             "LangGraph selected the candidate, but explanation degraded.",
			FallbackReasonCode:  CognitionFallbackExplainRequest,
			FallbackMessage:     "Go kept the deterministic fallback explanation.",
		},
	}

	payload := buildAgentExplanationPayload(plan, "")
	if asString(payload["mode"]) != "langgraph_degraded" {
		t.Fatalf("expected langgraph_degraded mode, got %#v", payload)
	}
	cognition := mapFromAny(payload["cognition"])
	if asString(cognition["status"]) != CognitionStatusDegraded {
		t.Fatalf("expected degraded cognition payload, got %#v", cognition)
	}
	if asString(cognition["fallback_reason_code"]) != CognitionFallbackExplainRequest {
		t.Fatalf("expected explain fallback reason, got %#v", cognition)
	}
}
