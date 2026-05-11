package agent

import (
	"fmt"
	"strings"
	"time"
)

func cloneMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = cloneValue(value)
	}
	return out
}

func cloneSlice(input []any) []any {
	if len(input) == 0 {
		return []any{}
	}
	out := make([]any, len(input))
	for idx, value := range input {
		out[idx] = cloneValue(value)
	}
	return out
}

func cloneValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		return cloneSlice(typed)
	case []string:
		out := make([]string, len(typed))
		copy(out, typed)
		return out
	case []map[string]any:
		out := make([]map[string]any, len(typed))
		for idx, item := range typed {
			out[idx] = cloneMap(item)
		}
		return out
	default:
		return typed
	}
}

func clonePlan(plan AgentPlan) AgentPlan {
	selected := make([]string, len(plan.SelectedIssueIDs))
	copy(selected, plan.SelectedIssueIDs)
	autoRepair := make([]string, len(plan.AutoRepairIssueIDs))
	copy(autoRepair, plan.AutoRepairIssueIDs)
	cautious := make([]string, len(plan.CautiousIssueIDs))
	copy(cautious, plan.CautiousIssueIDs)
	manualReview := make([]string, len(plan.ManualReviewIssueIDs))
	copy(manualReview, plan.ManualReviewIssueIDs)
	blocked := make([]string, len(plan.BlockedIssueIDs))
	copy(blocked, plan.BlockedIssueIDs)
	cautiousDetails := cloneCautiousIssueDetails(plan.CautiousIssueDetails)
	blockedDetails := cloneBlockedIssueDetails(plan.BlockedIssueDetails)
	blockedReasonCounts := cloneIntMap(plan.BlockedReasonCounts)

	skipped := make([]AgentSkippedIssue, len(plan.SkippedIssues))
	for idx, item := range plan.SkippedIssues {
		skipped[idx] = AgentSkippedIssue{
			IssueID:   item.IssueID,
			IssueType: item.IssueType,
			Column:    item.Column,
			Reason:    item.Reason,
			Details:   cloneMap(item.Details),
		}
	}

	candidates := make([]RepairCandidate, len(plan.Candidates))
	for idx, candidate := range plan.Candidates {
		planPayloads := make([]map[string]any, len(candidate.PlanPayloads))
		for payloadIdx, payload := range candidate.PlanPayloads {
			planPayloads[payloadIdx] = cloneMap(payload)
		}
		executePayloads := make([]map[string]any, len(candidate.ExecutePayloads))
		for payloadIdx, payload := range candidate.ExecutePayloads {
			executePayloads[payloadIdx] = cloneMap(payload)
		}
		selectedIssueIDs := make([]string, len(candidate.SelectedIssueIDs))
		copy(selectedIssueIDs, candidate.SelectedIssueIDs)
		toolSequence := make([]string, len(candidate.ToolSequence))
		copy(toolSequence, candidate.ToolSequence)
		candidates[idx] = RepairCandidate{
			CandidateID:      candidate.CandidateID,
			Source:           candidate.Source,
			ToolSequence:     toolSequence,
			PlanPayloads:     planPayloads,
			ExecutePayloads:  executePayloads,
			SelectedIssueIDs: selectedIssueIDs,
			IssueSourceMap:   cloneMap(candidate.IssueSourceMap),
			Comparison:       cloneMap(candidate.Comparison),
			Summary:          candidate.Summary,
			Executable:       candidate.Executable,
		}
	}

	return AgentPlan{
		PlanID:               plan.PlanID,
		Status:               plan.Status,
		SelectedIssueIDs:     selected,
		AutoRepairIssueIDs:   autoRepair,
		CautiousIssueIDs:     cautious,
		ManualReviewIssueIDs: manualReview,
		BlockedIssueIDs:      blocked,
		CautiousIssueDetails: cautiousDetails,
		BlockedIssueDetails:  blockedDetails,
		BlockedReasonCounts:  blockedReasonCounts,
		SkippedIssues:        skipped,
		Candidates:           candidates,
		SelectedCandidateID:  plan.SelectedCandidateID,
		SelectedSource:       plan.SelectedSource,
		IssueSourceMap:       cloneMap(plan.IssueSourceMap),
		ProposedToolID:       plan.ProposedToolID,
		ProposedPayload:      cloneMap(plan.ProposedPayload),
		IntentLabel:          plan.IntentLabel,
		StrategyLabel:        plan.StrategyLabel,
		ReasonCodes:          append([]string{}, plan.ReasonCodes...),
		RiskNote:             plan.RiskNote,
		ExplanationBullets:   append([]string{}, plan.ExplanationBullets...),
		ApprovalNeeded:       plan.ApprovalNeeded,
		Cognition:            cloneCognitionState(plan.Cognition),
		TimingsMS:            cloneMap(plan.TimingsMS),
		ReasoningSummary:     plan.ReasoningSummary,
		UserExplanation:      plan.UserExplanation,
	}
}

func cloneBlockedIssueDetails(items []AgentBlockedIssueDetail) []AgentBlockedIssueDetail {
	out := make([]AgentBlockedIssueDetail, len(items))
	copy(out, items)
	return out
}

func cloneCautiousIssueDetails(items []AgentCautiousIssueDetail) []AgentCautiousIssueDetail {
	out := make([]AgentCautiousIssueDetail, len(items))
	copy(out, items)
	return out
}

func cloneIntMap(input map[string]int) map[string]int {
	if len(input) == 0 {
		return map[string]int{}
	}
	out := make(map[string]int, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func mergeTimingMS(existing map[string]any, updates map[string]any) map[string]any {
	out := cloneMap(existing)
	for key, value := range updates {
		if strings.TrimSpace(key) == "" {
			continue
		}
		out[key] = value
	}
	return out
}

func ensureTimingKeys(timings map[string]any, keys ...string) map[string]any {
	out := cloneMap(timings)
	for _, key := range keys {
		if strings.TrimSpace(key) == "" {
			continue
		}
		if _, exists := out[key]; !exists {
			out[key] = nil
		}
	}
	return out
}

func cloneSkippedIssues(items []AgentSkippedIssue) []AgentSkippedIssue {
	out := make([]AgentSkippedIssue, len(items))
	for idx, item := range items {
		out[idx] = AgentSkippedIssue{
			IssueID:   item.IssueID,
			IssueType: item.IssueType,
			Column:    item.Column,
			Reason:    item.Reason,
			Details:   cloneMap(item.Details),
		}
	}
	return out
}

func asString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	case fmt.Stringer:
		return strings.TrimSpace(typed.String())
	default:
		return strings.TrimSpace(fmt.Sprint(value))
	}
}

func mapFromAny(value any) map[string]any {
	if value == nil {
		return nil
	}
	typed, ok := value.(map[string]any)
	if !ok {
		return nil
	}
	return cloneMap(typed)
}

func listOfMaps(value any) []map[string]any {
	items, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		mapped, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, mapped)
	}
	return out
}

func intFromAny(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float32:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func floatFromAny(value any) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int32:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return 0
	}
}

func newSessionID() string {
	return fmt.Sprintf("session-%d", time.Now().UnixNano())
}

func newPlanID() string {
	return fmt.Sprintf("plan-%d", time.Now().UnixNano())
}

func SummarizeTraceEvents(events []AgentTraceEvent) TraceSummary {
	summary := TraceSummary{
		EventCount:      len(events),
		ToolCallCount:   0,
		AgentNames:      make([]string, 0, 8),
		TraceTypeCounts: map[string]int{},
		Cognition: CognitionTraceSummary{
			ReasonCodes: []string{},
		},
	}
	seenAgents := map[string]struct{}{}
	for _, event := range events {
		summary.TraceTypeCounts[event.TraceType]++
		if event.TraceType == TraceToolCall {
			summary.ToolCallCount++
		}
		if agentName := strings.TrimSpace(event.AgentName); agentName != "" {
			if _, exists := seenAgents[agentName]; !exists {
				seenAgents[agentName] = struct{}{}
				summary.AgentNames = append(summary.AgentNames, agentName)
			}
		}
		if event.TraceType == TraceCognitionTrace {
			summary.Cognition.EventCount++
			summary.Cognition.Provider = asString(event.Payload["provider"])
			summary.Cognition.Status = asString(event.Payload["status"])
			summary.Cognition.LastPhase = asString(event.Payload["phase"])
			summary.Cognition.LastSummary = firstNonEmpty(asString(event.Payload["summary"]), event.Summary)
			summary.Cognition.PlannerMode = asString(event.Payload["planner_mode"])
			summary.Cognition.LLMMode = asString(event.Payload["llm_mode"])
			summary.Cognition.FallbackReasonCode = asString(event.Payload["fallback_reason_code"])
			summary.Cognition.ReasonCodes = uniqueStrings(stringsFromAny(event.Payload["reason_codes"]))
			summary.Cognition.SelectedCandidateID = asString(event.Payload["selected_candidate_id"])
		}
		summary.LastTraceType = event.TraceType
		summary.LastTraceSummary = event.Summary
	}
	return summary
}

func summarizeTrace(events []AgentTraceEvent) TraceSummary {
	return SummarizeTraceEvents(events)
}
