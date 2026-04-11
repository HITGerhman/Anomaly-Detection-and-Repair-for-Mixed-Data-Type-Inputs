package agent

const (
	ActionSessionPlan    = "agent.session.plan"
	ActionSessionExecute = "agent.session.execute"
	ActionSessionAuto    = "agent.session.auto"
	ActionSessionApprove = "agent.session.approve"
)

const (
	DefaultUserGoal = "扫描并给出修复计划"
)

const (
	SessionStatusPlanning           = "planning"
	SessionStatusPlanned            = "planned"
	SessionStatusExecuting          = "executing"
	SessionStatusAwaitingApproval   = "awaiting_approval"
	SessionStatusApprovalRejected   = "approval_rejected"
	SessionStatusCompleted          = "completed"
	SessionStatusFailed             = "failed"
	SessionStatusValidationRejected = "validation_rejected"
	SessionStatusRolledBack         = "rolled_back"
	SessionStatusRollbackFailed     = "rollback_failed"
)

const (
	TraceSessionStarted    = "session_started"
	TraceAgentStarted      = "agent_started"
	TraceAgentDecision     = "agent_decision"
	TraceToolCall          = "tool_call"
	TraceToolResult        = "tool_result"
	TraceValidation        = "validation"
	TraceRollbackDecision  = "rollback_decision"
	TraceRollbackExecuted  = "rollback_executed"
	TraceCognitionTrace    = "cognition_trace"
	TraceApprovalRequested = "approval_requested"
	TraceApprovalGranted   = "approval_granted"
	TraceApprovalRejected  = "approval_rejected"
	TraceMemoryUpdated     = "memory_updated"
	TraceSessionCompleted  = "session_completed"
	TraceSessionFailed     = "session_failed"
)

const (
	AgentSupervisor    = "supervisor"
	AgentIntent        = "intent"
	AgentProfile       = "profile"
	AgentStrategy      = "strategy"
	AgentRepairPlanner = "repair_planner"
	AgentExplainer     = "explainer"
	AgentValidator     = "validator"
)

const (
	ErrorInvalidInput       = "AGENT_INVALID_INPUT"
	ErrorSessionNotFound    = "AGENT_SESSION_NOT_FOUND"
	ErrorPlanNotFound       = "AGENT_PLAN_NOT_FOUND"
	ErrorApprovalRejected   = "AGENT_APPROVAL_REJECTED"
	ErrorValidationRejected = "AGENT_VALIDATION_REJECTED"
	ErrorToolFailed         = "AGENT_TOOL_FAILED"
	ErrorPlannerFailed      = "AGENT_PLANNER_FAILED"
)
