package agent

import "context"

// PlanningInput is a deterministic snapshot assembled by the Go runtime after
// scan and preview tool calls complete. Planner implementations must treat it
// as read-only planning context rather than fetching additional data on their
// own.
type PlanningInput struct {
	SessionID               string
	Goal                    string
	CSVPath                 string
	ScanResult              map[string]any
	SelectedIssueIDs        []string
	SkippedIssues           []AgentSkippedIssue
	RulePreview             map[string]any
	GowerPreview            map[string]any
	ScanConfigOverrides     map[string]any
	RepairStrategyOverrides map[string]any
	ColumnDependencies      map[string]any
	GowerStrategyOverrides  map[string]any
	ModelDir                string
	OutputDir               string
	WorkspaceID             string
	PreferenceSnapshot      map[string]any
	ApprovalContext         map[string]any
}

// Planner is the cognitive planning boundary for agent sessions.
//
// Implementations must only consume PlanningInput and return an AgentPlan. They
// must not call tools directly, persist state, validate or roll back execution,
// or write files. Those responsibilities remain in the Go runtime and the
// deterministic tool layer.
type Planner interface {
	BuildPlan(ctx context.Context, input PlanningInput) (AgentPlan, error)
}
