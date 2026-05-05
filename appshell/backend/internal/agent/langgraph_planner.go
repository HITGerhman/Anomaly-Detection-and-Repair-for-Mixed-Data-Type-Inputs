package agent

import (
	"context"
	"sort"
	"strings"

	"appshell/backend/internal/observability"
)

type sidecarHealthManager interface {
	EnsureHealthy(ctx context.Context) (LangGraphHealth, error)
}

type cognitionCaller interface {
	planCaller
	explainCaller
}

type LangGraphPlanner struct {
	fallback Planner
	manager  sidecarHealthManager
	client   cognitionCaller
}

func NewLangGraphPlanner(fallback Planner, manager sidecarHealthManager, client cognitionCaller) *LangGraphPlanner {
	if fallback == nil {
		fallback = NewDeterministicPlanner()
	}
	return &LangGraphPlanner{
		fallback: fallback,
		manager:  manager,
		client:   client,
	}
}

var _ Planner = (*LangGraphPlanner)(nil)

func (p *LangGraphPlanner) BuildPlan(ctx context.Context, input PlanningInput) (AgentPlan, error) {
	basePlan, err := p.fallback.BuildPlan(ctx, input)
	if err != nil {
		return AgentPlan{}, err
	}
	basePlan = ensurePlanCognition(basePlan)
	if p == nil || p.manager == nil || p.client == nil {
		return basePlan, nil
	}

	health, err := p.manager.EnsureHealthy(ctx)
	if err != nil {
		status, reasonCode, _ := ClassifyLangGraphAvailabilityError(err)
		observability.Warn("langgraph_sidecar_fallback", map[string]any{
			"session_id":  input.SessionID,
			"reason":      err.Error(),
			"status":      status,
			"reason_code": reasonCode,
		})
		basePlan.Cognition = buildLangGraphFallbackState(basePlan, LangGraphHealth{}, status, reasonCode)
		return basePlan, nil
	}
	if !strings.EqualFold(strings.TrimSpace(health.PlannerMode), "llm") {
		observability.Info("langgraph_planner_fallback_mode", map[string]any{
			"session_id":   input.SessionID,
			"planner_mode": health.PlannerMode,
			"llm_mode":     health.LLMMode,
			"model":        health.Model,
		})
		basePlan.Cognition = buildLangGraphFallbackState(basePlan, health, CognitionStatusFallback, CognitionFallbackPlannerMode)
		return basePlan, nil
	}

	planResp, err := p.client.Plan(ctx, buildLangGraphPlanRequest(input, basePlan))
	if err != nil {
		observability.Warn("langgraph_plan_fallback", map[string]any{
			"session_id": input.SessionID,
			"reason":     err.Error(),
		})
		basePlan.Cognition = buildLangGraphFallbackState(basePlan, health, CognitionStatusFallback, CognitionFallbackPlanRequest)
		return basePlan, nil
	}

	candidate, ok := candidateByID(basePlan, planResp.SelectedCandidateID)
	if !ok || !candidate.Executable {
		observability.Warn("langgraph_plan_invalid_candidate", map[string]any{
			"session_id":            input.SessionID,
			"selected_candidate_id": planResp.SelectedCandidateID,
		})
		basePlan.Cognition = buildLangGraphFallbackState(basePlan, health, CognitionStatusFallback, CognitionFallbackInvalidCandidate)
		return basePlan, nil
	}

	updated := clonePlan(basePlan)
	updated.SelectedCandidateID = candidate.CandidateID
	updated.SelectedSource = candidate.Source
	updated.IssueSourceMap = cloneMap(candidate.IssueSourceMap)
	if len(candidate.ToolSequence) > 0 {
		updated.ProposedToolID = candidate.ToolSequence[0]
	}
	if len(candidate.ExecutePayloads) > 0 {
		updated.ProposedPayload = cloneMap(candidate.ExecutePayloads[0])
	} else {
		updated.ProposedPayload = map[string]any{}
	}
	updated.IntentLabel = strings.TrimSpace(planResp.IntentLabel)
	updated.StrategyLabel = strings.TrimSpace(planResp.StrategyLabel)
	updated.ReasonCodes = append([]string{}, planResp.ReasonCodes...)
	updated.RiskNote = strings.TrimSpace(planResp.RiskNote)
	updated.ReasoningSummary = strings.TrimSpace(planResp.OneSentenceSummary)
	updated.ExplanationBullets = append([]string{}, planResp.ShortBullets...)
	updated.ApprovalNeeded = planResp.ApprovalNeeded
	updated = enforceLangGraphApprovalContext(updated, input.ApprovalContext)

	explainResp, err := p.client.Explain(ctx, buildLangGraphExplainRequest(input, candidate, updated, planResp))
	if err != nil {
		observability.Warn("langgraph_explain_fallback", map[string]any{
			"session_id": input.SessionID,
			"reason":     err.Error(),
		})
		updated.UserExplanation = buildLangGraphExplanation(planResp)
		updated = enforceLangGraphApprovalContext(updated, input.ApprovalContext)
		updated.Cognition = buildLangGraphDegradedState(health, updated)
		return updated, nil
	}

	if summary := strings.TrimSpace(explainResp.Summary); summary != "" {
		updated.ReasoningSummary = summary
	}
	if finalMessage := strings.TrimSpace(explainResp.FinalMessage); finalMessage != "" {
		updated.UserExplanation = finalMessage
	} else {
		updated.UserExplanation = buildLangGraphExplanation(planResp)
	}
	if len(explainResp.ShortBullets) > 0 {
		updated.ExplanationBullets = append([]string{}, explainResp.ShortBullets...)
	}
	if len(explainResp.ReasonCodes) > 0 {
		updated.ReasonCodes = append([]string{}, explainResp.ReasonCodes...)
	}
	if riskNote := strings.TrimSpace(explainResp.RiskNote); riskNote != "" {
		updated.RiskNote = riskNote
	}
	updated = enforceLangGraphApprovalContext(updated, input.ApprovalContext)
	updated.ReasonCodes = uniqueStrings(updated.ReasonCodes)
	updated.Cognition = buildLangGraphEngagedState(health, updated)
	return updated, nil
}

func enforceLangGraphApprovalContext(plan AgentPlan, approvalContext map[string]any) AgentPlan {
	required, _ := boolFromAny(approvalContext["deterministic_required"])
	if !required {
		return plan
	}
	plan.ApprovalNeeded = true
	plan.ReasonCodes = uniqueStrings(append(plan.ReasonCodes, "approval_context_enforced"))
	return plan
}

func buildLangGraphPlanRequest(input PlanningInput, basePlan AgentPlan) LangGraphPlanRequest {
	scanSummary := cloneMap(mapFromAny(input.ScanResult["scan_summary"]))
	if len(scanSummary) == 0 {
		scanSummary = map[string]any{"issue_count": intFromAny(input.ScanResult["issue_count"])}
	}

	candidatePreviews := make([]LangGraphCandidatePreview, 0, len(basePlan.Candidates))
	for _, candidate := range basePlan.Candidates {
		candidatePreviews = append(candidatePreviews, LangGraphCandidatePreview{
			CandidateID:      candidate.CandidateID,
			Source:           candidate.Source,
			Comparison:       cloneMap(candidate.Comparison),
			SelectedIssueIDs: append([]string{}, candidate.SelectedIssueIDs...),
			ToolSequence:     append([]string{}, candidate.ToolSequence...),
			Summary:          candidate.Summary,
		})
	}
	sort.SliceStable(candidatePreviews, func(i, j int) bool {
		leftSelected := candidatePreviews[i].CandidateID == basePlan.SelectedCandidateID
		rightSelected := candidatePreviews[j].CandidateID == basePlan.SelectedCandidateID
		if leftSelected == rightSelected {
			return i < j
		}
		return leftSelected
	})

	skippedTypes := make([]string, 0, len(basePlan.SkippedIssues))
	seenSkipped := map[string]struct{}{}
	for _, skipped := range basePlan.SkippedIssues {
		if skipped.IssueType == "" {
			continue
		}
		if _, exists := seenSkipped[skipped.IssueType]; exists {
			continue
		}
		seenSkipped[skipped.IssueType] = struct{}{}
		skippedTypes = append(skippedTypes, skipped.IssueType)
	}

	return LangGraphPlanRequest{
		SessionID:         input.SessionID,
		Goal:              input.Goal,
		ScanSummary:       scanSummary,
		CandidatePreviews: candidatePreviews,
		SafetyContext: map[string]any{
			"selected_candidate_id":     basePlan.SelectedCandidateID,
			"selected_source":           basePlan.SelectedSource,
			"skipped_issue_types":       skippedTypes,
			"selected_issue_count":      len(basePlan.SelectedIssueIDs),
			"auto_repair_issue_ids":     append([]string{}, basePlan.AutoRepairIssueIDs...),
			"cautious_issue_ids":        append([]string{}, basePlan.CautiousIssueIDs...),
			"manual_review_issue_ids":   append([]string{}, basePlan.ManualReviewIssueIDs...),
			"blocked_issue_ids":         append([]string{}, basePlan.BlockedIssueIDs...),
			"auto_repair_issue_count":   len(basePlan.AutoRepairIssueIDs),
			"cautious_issue_count":      len(basePlan.CautiousIssueIDs),
			"manual_review_issue_count": len(basePlan.ManualReviewIssueIDs),
			"blocked_issue_count":       len(basePlan.BlockedIssueIDs),
		},
		ApprovalContext: cloneMap(input.ApprovalContext),
		UserPreferences: cloneMap(input.PreferenceSnapshot),
		OutputConstraints: map[string]any{
			"strategy_label":       true,
			"risk_note":            true,
			"one_sentence_summary": true,
			"max_bullets":          3,
			"json_only":            true,
		},
	}
}

func buildLangGraphExplainRequest(input PlanningInput, candidate RepairCandidate, plan AgentPlan, planResp LangGraphPlanResponse) LangGraphExplainRequest {
	return LangGraphExplainRequest{
		SessionID: input.SessionID,
		Goal:      input.Goal,
		SelectedCandidate: LangGraphCandidatePreview{
			CandidateID:      candidate.CandidateID,
			Source:           candidate.Source,
			Comparison:       cloneMap(candidate.Comparison),
			SelectedIssueIDs: append([]string{}, candidate.SelectedIssueIDs...),
			ToolSequence:     append([]string{}, candidate.ToolSequence...),
			Summary:          candidate.Summary,
		},
		StrategyLabel:     strings.TrimSpace(planResp.StrategyLabel),
		ReasonCodes:       append([]string{}, planResp.ReasonCodes...),
		RiskNote:          strings.TrimSpace(planResp.RiskNote),
		ValidationPreview: cloneMap(candidate.Comparison),
		SafetyContext: map[string]any{
			"selected_source":      candidate.Source,
			"selected_issue_count": len(plan.SelectedIssueIDs),
			"skipped_issue_types":  skippedIssueTypes(plan.SkippedIssues),
		},
		ApprovalContext: cloneMap(input.ApprovalContext),
		OutputConstraints: map[string]any{
			"summary":       true,
			"final_message": true,
			"risk_note":     true,
			"max_bullets":   3,
			"json_only":     true,
			"max_sentences": 2,
		},
	}
}

func buildLangGraphExplanation(resp LangGraphPlanResponse) string {
	parts := []string{strings.TrimSpace(resp.OneSentenceSummary)}
	if risk := strings.TrimSpace(resp.RiskNote); risk != "" {
		parts = append(parts, risk)
	}
	if len(resp.ShortBullets) > 0 {
		parts = append(parts, strings.Join(resp.ShortBullets, " "))
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

func skippedIssueTypes(items []AgentSkippedIssue) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if item.IssueType == "" {
			continue
		}
		if _, exists := seen[item.IssueType]; exists {
			continue
		}
		seen[item.IssueType] = struct{}{}
		out = append(out, item.IssueType)
	}
	sort.Strings(out)
	return out
}
