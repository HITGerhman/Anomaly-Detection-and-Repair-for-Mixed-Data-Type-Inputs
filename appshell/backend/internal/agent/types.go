package agent

import "time"

type AgentSkippedIssue struct {
	IssueID   string         `json:"issue_id,omitempty"`
	IssueType string         `json:"issue_type,omitempty"`
	Column    string         `json:"column,omitempty"`
	Reason    string         `json:"reason"`
	Details   map[string]any `json:"details,omitempty"`
}

type AgentBlockedIssueDetail struct {
	IssueID             string `json:"issue_id,omitempty"`
	IssueType           string `json:"issue_type,omitempty"`
	Column              string `json:"column,omitempty"`
	BlockedReason       string `json:"blocked_reason"`
	BlockedByRule       string `json:"blocked_by_rule"`
	SuggestedNextAction string `json:"suggested_next_action"`
}

type AgentCautiousIssueDetail struct {
	IssueID          string `json:"issue_id,omitempty"`
	IssueType        string `json:"issue_type,omitempty"`
	Column           string `json:"column,omitempty"`
	RiskReason       string `json:"risk_reason"`
	ApprovalRequired bool   `json:"approval_required"`
	SuggestedAction  string `json:"suggested_action"`
}

type RepairCandidate struct {
	CandidateID      string           `json:"candidate_id"`
	Source           string           `json:"source"`
	ToolSequence     []string         `json:"tool_sequence"`
	PlanPayloads     []map[string]any `json:"plan_payloads"`
	ExecutePayloads  []map[string]any `json:"execute_payloads"`
	SelectedIssueIDs []string         `json:"selected_issue_ids"`
	IssueSourceMap   map[string]any   `json:"issue_source_map"`
	Comparison       map[string]any   `json:"comparison"`
	Score            map[string]any   `json:"score,omitempty"`
	Summary          string           `json:"summary"`
	Executable       bool             `json:"executable"`
}

type AgentPlan struct {
	PlanID               string                     `json:"plan_id"`
	Status               string                     `json:"status"`
	SelectedIssueIDs     []string                   `json:"selected_issue_ids"`
	AutoRepairIssueIDs   []string                   `json:"auto_repair_issue_ids,omitempty"`
	CautiousIssueIDs     []string                   `json:"cautious_issue_ids,omitempty"`
	ManualReviewIssueIDs []string                   `json:"manual_review_issue_ids,omitempty"`
	BlockedIssueIDs      []string                   `json:"blocked_issue_ids,omitempty"`
	CautiousIssueDetails []AgentCautiousIssueDetail `json:"cautious_issue_details,omitempty"`
	BlockedIssueDetails  []AgentBlockedIssueDetail  `json:"blocked_issue_details,omitempty"`
	BlockedReasonCounts  map[string]int             `json:"blocked_reason_counts,omitempty"`
	SkippedIssues        []AgentSkippedIssue        `json:"skipped_issues"`
	Candidates           []RepairCandidate          `json:"candidates"`
	SelectedCandidateID  string                     `json:"selected_candidate_id"`
	SelectedSource       string                     `json:"selected_source"`
	IssueSourceMap       map[string]any             `json:"issue_source_map"`
	ProposedToolID       string                     `json:"proposed_tool_id"`
	ProposedPayload      map[string]any             `json:"proposed_payload"`
	IntentLabel          string                     `json:"intent_label,omitempty"`
	StrategyLabel        string                     `json:"strategy_label,omitempty"`
	ReasonCodes          []string                   `json:"reason_codes,omitempty"`
	RiskNote             string                     `json:"risk_note,omitempty"`
	ExplanationBullets   []string                   `json:"explanation_bullets,omitempty"`
	ApprovalNeeded       bool                       `json:"approval_needed,omitempty"`
	Cognition            AgentCognitionState        `json:"cognition,omitempty"`
	TimingsMS            map[string]any             `json:"timings_ms,omitempty"`
	ReasoningSummary     string                     `json:"reasoning_summary"`
	UserExplanation      string                     `json:"user_explanation"`
}

type AgentSession struct {
	SessionID     string         `json:"session_id"`
	RootTaskID    string         `json:"root_task_id"`
	CurrentTaskID string         `json:"current_task_id"`
	Status        string         `json:"status"`
	Mode          string         `json:"mode"`
	UserGoal      string         `json:"user_goal"`
	Context       map[string]any `json:"context"`
	LatestPlan    AgentPlan      `json:"latest_plan"`
	CreatedAt     time.Time      `json:"created_at"`
	UpdatedAt     time.Time      `json:"updated_at"`
}

type AgentTraceEvent struct {
	ID        int64          `json:"id"`
	SessionID string         `json:"session_id"`
	TaskID    string         `json:"task_id"`
	Seq       int            `json:"seq"`
	AgentName string         `json:"agent_name"`
	TraceType string         `json:"trace_type"`
	Summary   string         `json:"summary"`
	Payload   map[string]any `json:"payload,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
}

type TraceSummary struct {
	EventCount       int                   `json:"event_count"`
	ToolCallCount    int                   `json:"tool_call_count"`
	AgentNames       []string              `json:"agent_names"`
	TraceTypeCounts  map[string]int        `json:"trace_type_counts"`
	Cognition        CognitionTraceSummary `json:"cognition"`
	LastTraceType    string                `json:"last_trace_type,omitempty"`
	LastTraceSummary string                `json:"last_trace_summary,omitempty"`
}

type AgentSessionSnapshot struct {
	SessionID            string         `json:"session_id"`
	RootTaskID           string         `json:"root_task_id"`
	CurrentTaskID        string         `json:"current_task_id"`
	Status               string         `json:"status"`
	Mode                 string         `json:"mode"`
	UserGoal             string         `json:"user_goal"`
	Context              map[string]any `json:"context"`
	LatestPlan           AgentPlan      `json:"latest_plan"`
	Presentation         map[string]any `json:"presentation,omitempty"`
	PresentationArtifact string         `json:"presentation_artifact,omitempty"`
	CreatedAt            time.Time      `json:"created_at"`
	UpdatedAt            time.Time      `json:"updated_at"`
	TraceSummary         TraceSummary   `json:"trace_summary"`
}

func (s AgentSession) Snapshot(summary TraceSummary) AgentSessionSnapshot {
	return AgentSessionSnapshot{
		SessionID:            s.SessionID,
		RootTaskID:           s.RootTaskID,
		CurrentTaskID:        s.CurrentTaskID,
		Status:               s.Status,
		Mode:                 s.Mode,
		UserGoal:             s.UserGoal,
		Context:              cloneMap(s.Context),
		LatestPlan:           clonePlan(s.LatestPlan),
		Presentation:         map[string]any{},
		PresentationArtifact: "",
		CreatedAt:            s.CreatedAt,
		UpdatedAt:            s.UpdatedAt,
		TraceSummary:         summary,
	}
}
