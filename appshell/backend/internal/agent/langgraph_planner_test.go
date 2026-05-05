package agent

import (
	"context"
	"testing"
)

type stubFallbackPlanner struct {
	plan  AgentPlan
	err   error
	calls int
}

func (p *stubFallbackPlanner) BuildPlan(_ context.Context, _ PlanningInput) (AgentPlan, error) {
	p.calls++
	if p.err != nil {
		return AgentPlan{}, p.err
	}
	return clonePlan(p.plan), nil
}

type stubSidecarManager struct {
	healthErr error
	health    LangGraphHealth
}

func (m stubSidecarManager) EnsureHealthy(_ context.Context) (LangGraphHealth, error) {
	if m.healthErr != nil {
		return LangGraphHealth{}, m.healthErr
	}
	if m.health.Status != "" {
		return m.health, nil
	}
	return LangGraphHealth{
		Status:      "ok",
		Service:     "langgraph-sidecar",
		PlannerMode: "llm",
		LLMMode:     "configured",
		Model:       "gpt-test",
		Ready:       true,
	}, nil
}

type stubCognitionCaller struct {
	planResp       LangGraphPlanResponse
	planErr        error
	planCalls      int
	lastPlanReq    LangGraphPlanRequest
	explainResp    LangGraphExplainResponse
	explainErr     error
	explainCalls   int
	lastExplainReq LangGraphExplainRequest
}

func (c *stubCognitionCaller) Plan(_ context.Context, req LangGraphPlanRequest) (LangGraphPlanResponse, error) {
	c.planCalls++
	c.lastPlanReq = req
	if c.planErr != nil {
		return LangGraphPlanResponse{}, c.planErr
	}
	return c.planResp, nil
}

func (c *stubCognitionCaller) Explain(_ context.Context, req LangGraphExplainRequest) (LangGraphExplainResponse, error) {
	c.explainCalls++
	c.lastExplainReq = req
	if c.explainErr != nil {
		return LangGraphExplainResponse{}, c.explainErr
	}
	return c.explainResp, nil
}

func TestLangGraphPlannerOverlaysCognitiveFieldsWhenSidecarIsHealthy(t *testing.T) {
	basePlan := AgentPlan{
		PlanID:               "plan-1",
		Status:               "planned",
		SelectedIssueIDs:     []string{"i-1"},
		AutoRepairIssueIDs:   []string{"i-1"},
		CautiousIssueIDs:     []string{"i-2"},
		ManualReviewIssueIDs: []string{"i-3"},
		BlockedIssueIDs:      []string{"i-4"},
		Candidates: []RepairCandidate{
			{
				CandidateID:      "candidate-rule",
				Source:           "rule",
				ToolSequence:     []string{"engine.repair_batch"},
				ExecutePayloads:  []map[string]any{{"csv_path": "demo.csv", "plan_only": false}},
				SelectedIssueIDs: []string{"i-1"},
				IssueSourceMap:   map[string]any{},
				Comparison:       map[string]any{"after_issue_count": 1},
				Summary:          "rule",
				Executable:       true,
			},
			{
				CandidateID:      "candidate-gower",
				Source:           "gower",
				ToolSequence:     []string{"engine.repair_with_gower"},
				ExecutePayloads:  []map[string]any{{"csv_path": "demo.csv", "plan_only": false}},
				SelectedIssueIDs: []string{"i-1"},
				IssueSourceMap:   map[string]any{"i-1": "gower"},
				Comparison:       map[string]any{"after_issue_count": 0},
				Summary:          "gower",
				Executable:       true,
			},
		},
		SelectedCandidateID: "candidate-rule",
		SelectedSource:      "rule",
		IssueSourceMap:      map[string]any{},
		ProposedToolID:      "engine.repair_batch",
		ProposedPayload:     map[string]any{"csv_path": "demo.csv"},
		ReasoningSummary:    "base",
		UserExplanation:     "base",
	}
	fallback := &stubFallbackPlanner{plan: basePlan}
	client := &stubCognitionCaller{
		planResp: LangGraphPlanResponse{
			StrategyLabel:       "neighbor_similarity",
			SelectedCandidateID: "candidate-gower",
			ReasonCodes:         []string{"phase_c_llm"},
			RiskNote:            "validation first",
			IntentLabel:         "auto_repair",
			OneSentenceSummary:  "LangGraph prefers the Gower candidate.",
			ShortBullets:        []string{"short note"},
		},
		explainResp: LangGraphExplainResponse{
			Summary:      "LLM summary",
			FinalMessage: "LLM final message",
			ShortBullets: []string{"bullet-1", "bullet-2"},
			ReasonCodes:  []string{"phase_c_llm", "selected_gower"},
			RiskNote:     "validation first",
		},
	}

	planner := NewLangGraphPlanner(fallback, stubSidecarManager{}, client)
	plan, err := planner.BuildPlan(t.Context(), PlanningInput{
		SessionID: "session-1",
		Goal:      "scan and repair",
		ScanResult: map[string]any{
			"scan_summary": map[string]any{"total_issues": 1},
		},
	})
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}
	if fallback.calls != 1 || client.planCalls != 1 || client.explainCalls != 1 {
		t.Fatalf("expected fallback, plan, and explain calls exactly once")
	}
	safety := client.lastPlanReq.SafetyContext
	if len(stringsFromAny(safety["auto_repair_issue_ids"])) != 1 || stringsFromAny(safety["auto_repair_issue_ids"])[0] != "i-1" {
		t.Fatalf("expected A2 auto issue ids in plan request safety context, got %#v", safety)
	}
	if len(stringsFromAny(safety["cautious_issue_ids"])) != 1 || stringsFromAny(safety["cautious_issue_ids"])[0] != "i-2" {
		t.Fatalf("expected cautious issue ids in plan request safety context, got %#v", safety)
	}
	if len(stringsFromAny(safety["manual_review_issue_ids"])) != 1 || stringsFromAny(safety["manual_review_issue_ids"])[0] != "i-3" {
		t.Fatalf("expected manual issue ids in plan request safety context, got %#v", safety)
	}
	if len(stringsFromAny(safety["blocked_issue_ids"])) != 1 || stringsFromAny(safety["blocked_issue_ids"])[0] != "i-4" {
		t.Fatalf("expected blocked issue ids in plan request safety context, got %#v", safety)
	}
	if plan.SelectedCandidateID != "candidate-gower" || plan.SelectedSource != "gower" {
		t.Fatalf("expected sidecar-selected candidate, got %s / %s", plan.SelectedCandidateID, plan.SelectedSource)
	}
	if plan.ProposedToolID != "engine.repair_with_gower" {
		t.Fatalf("unexpected proposed tool id: %s", plan.ProposedToolID)
	}
	if plan.ReasoningSummary != "LLM summary" || plan.UserExplanation != "LLM final message" {
		t.Fatalf("expected explain response to overwrite display fields: %+v", plan)
	}
	if plan.IntentLabel != "auto_repair" || plan.StrategyLabel != "neighbor_similarity" {
		t.Fatalf("expected structured cognition fields: %+v", plan)
	}
	if plan.RiskNote != "validation first" || len(plan.ExplanationBullets) != 2 {
		t.Fatalf("expected structured explanation fields: %+v", plan)
	}
	if plan.Cognition.Provider != CognitionProviderLangGraph || plan.Cognition.Status != CognitionStatusEngaged {
		t.Fatalf("expected engaged langgraph cognition, got %+v", plan.Cognition)
	}
	if plan.Cognition.PlannerMode != "llm" || plan.Cognition.LLMMode != "configured" {
		t.Fatalf("expected llm planner metadata, got %+v", plan.Cognition)
	}
	if plan.Cognition.SelectedCandidateID != "candidate-gower" || plan.Cognition.FallbackReasonCode != "" {
		t.Fatalf("expected engaged cognition without fallback, got %+v", plan.Cognition)
	}
}

func TestLangGraphPlannerFallsBackWhenSidecarUnavailableOrInvalid(t *testing.T) {
	basePlan := AgentPlan{
		PlanID:              "plan-1",
		Status:              "planned",
		SelectedCandidateID: "candidate-rule",
		SelectedSource:      "rule",
		ProposedToolID:      "engine.repair_batch",
		ProposedPayload:     map[string]any{"csv_path": "demo.csv"},
		ReasoningSummary:    "base",
		UserExplanation:     "base",
		Candidates: []RepairCandidate{
			{CandidateID: "candidate-rule", Source: "rule", ToolSequence: []string{"engine.repair_batch"}, ExecutePayloads: []map[string]any{{"csv_path": "demo.csv"}}, Executable: true},
		},
	}

	t.Run("manager_error", func(t *testing.T) {
		fallback := &stubFallbackPlanner{plan: basePlan}
		client := &stubCognitionCaller{planResp: LangGraphPlanResponse{StrategyLabel: "deterministic_rule", SelectedCandidateID: "candidate-rule"}}
		planner := NewLangGraphPlanner(fallback, stubSidecarManager{healthErr: context.DeadlineExceeded}, client)

		plan, err := planner.BuildPlan(t.Context(), PlanningInput{})
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if plan.SelectedCandidateID != "candidate-rule" || client.planCalls != 0 || client.explainCalls != 0 {
			t.Fatalf("expected pure fallback behavior")
		}
		if plan.Cognition.Status != CognitionStatusUnavailable || plan.Cognition.FallbackReasonCode != CognitionFallbackHealthcheckFailed {
			t.Fatalf("expected healthcheck fallback cognition, got %+v", plan.Cognition)
		}
	})

	t.Run("invalid_candidate", func(t *testing.T) {
		fallback := &stubFallbackPlanner{plan: basePlan}
		client := &stubCognitionCaller{planResp: LangGraphPlanResponse{StrategyLabel: "deterministic_rule", SelectedCandidateID: "candidate-missing", OneSentenceSummary: "ignored"}}
		planner := NewLangGraphPlanner(fallback, stubSidecarManager{}, client)

		plan, err := planner.BuildPlan(t.Context(), PlanningInput{})
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if plan.SelectedCandidateID != "candidate-rule" || plan.ReasoningSummary != "base" {
			t.Fatalf("expected invalid sidecar candidate to preserve fallback plan")
		}
		if plan.Cognition.Status != CognitionStatusFallback || plan.Cognition.FallbackReasonCode != CognitionFallbackInvalidCandidate {
			t.Fatalf("expected invalid candidate fallback cognition, got %+v", plan.Cognition)
		}
	})

	t.Run("planner_mode_fallback", func(t *testing.T) {
		fallback := &stubFallbackPlanner{plan: basePlan}
		client := &stubCognitionCaller{planResp: LangGraphPlanResponse{StrategyLabel: "deterministic_rule", SelectedCandidateID: "candidate-rule"}}
		planner := NewLangGraphPlanner(
			fallback,
			stubSidecarManager{health: LangGraphHealth{Status: "ok", Service: "langgraph-sidecar", PlannerMode: "fallback", LLMMode: "unavailable", Ready: true}},
			client,
		)

		plan, err := planner.BuildPlan(t.Context(), PlanningInput{})
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if plan.SelectedCandidateID != "candidate-rule" || client.planCalls != 0 || client.explainCalls != 0 {
			t.Fatalf("expected planner to return deterministic fallback when llm is unavailable")
		}
		if plan.Cognition.Status != CognitionStatusFallback || plan.Cognition.FallbackReasonCode != CognitionFallbackPlannerMode {
			t.Fatalf("expected planner_mode fallback cognition, got %+v", plan.Cognition)
		}
	})

	t.Run("explain_error_keeps_plan_overlay", func(t *testing.T) {
		fallback := &stubFallbackPlanner{plan: basePlan}
		client := &stubCognitionCaller{
			planResp: LangGraphPlanResponse{
				StrategyLabel:       "deterministic_rule",
				SelectedCandidateID: "candidate-rule",
				ReasonCodes:         []string{"phase_c_llm"},
				RiskNote:            "still validated",
				IntentLabel:         "auto_repair",
				OneSentenceSummary:  "plan overlay",
				ShortBullets:        []string{"plan bullet"},
			},
			explainErr: context.Canceled,
		}
		planner := NewLangGraphPlanner(fallback, stubSidecarManager{}, client)

		plan, err := planner.BuildPlan(t.Context(), PlanningInput{})
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if plan.ReasoningSummary != "plan overlay" || plan.UserExplanation == "base" {
			t.Fatalf("expected plan overlay to survive explain error: %+v", plan)
		}
		if plan.Cognition.Status != CognitionStatusDegraded || plan.Cognition.FallbackReasonCode != CognitionFallbackExplainRequest {
			t.Fatalf("expected degraded cognition on explain failure, got %+v", plan.Cognition)
		}
	})

	t.Run("plan_request_failed", func(t *testing.T) {
		fallback := &stubFallbackPlanner{plan: basePlan}
		client := &stubCognitionCaller{planErr: context.DeadlineExceeded}
		planner := NewLangGraphPlanner(fallback, stubSidecarManager{}, client)

		plan, err := planner.BuildPlan(t.Context(), PlanningInput{})
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if plan.SelectedCandidateID != "candidate-rule" || client.explainCalls != 0 {
			t.Fatalf("expected fallback plan after plan request failure")
		}
		if plan.Cognition.Status != CognitionStatusFallback || plan.Cognition.FallbackReasonCode != CognitionFallbackPlanRequest {
			t.Fatalf("expected plan request fallback cognition, got %+v", plan.Cognition)
		}
	})

	t.Run("approval_context_forces_overlay_approval", func(t *testing.T) {
		fallback := &stubFallbackPlanner{plan: basePlan}
		client := &stubCognitionCaller{
			planResp: LangGraphPlanResponse{
				StrategyLabel:       "deterministic_rule",
				SelectedCandidateID: "candidate-rule",
				ReasonCodes:         []string{"phase_c_llm"},
				IntentLabel:         "auto_repair",
				OneSentenceSummary:  "LLM tried to avoid approval.",
				ApprovalNeeded:      false,
			},
			explainResp: LangGraphExplainResponse{
				Summary:      "LLM summary",
				FinalMessage: "LLM final message",
				ReasonCodes:  []string{"phase_c_llm"},
			},
		}
		planner := NewLangGraphPlanner(fallback, stubSidecarManager{}, client)

		plan, err := planner.BuildPlan(t.Context(), PlanningInput{
			ApprovalContext: map[string]any{"deterministic_required": true},
		})
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if !plan.ApprovalNeeded {
			t.Fatalf("expected approval context to force approval")
		}
		found := false
		for _, code := range plan.ReasonCodes {
			if code == "approval_context_enforced" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected approval_context_enforced reason code, got %#v", plan.ReasonCodes)
		}
	})
}
