package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"appshell/backend/internal/engine"
	"appshell/backend/internal/observability"
)

type baseRunner interface {
	Run(ctx context.Context, req engine.Request) (engine.Response, error)
}

type observerSetter interface {
	SetStderrObserver(engine.StderrObserver)
}

type RuntimeRunner struct {
	base     baseRunner
	registry *ToolRegistry
	planner  Planner
	store    SessionStore

	observerMu sync.RWMutex
	observer   engine.StderrObserver

	childTaskMu      sync.RWMutex
	childTaskParents map[string]string
}

func NewRuntimeRunner(base baseRunner, store SessionStore, planner Planner) *RuntimeRunner {
	if planner == nil {
		planner = NewDeterministicPlanner()
	}
	runner := &RuntimeRunner{
		base:             base,
		registry:         NewToolRegistry(),
		planner:          planner,
		store:            store,
		childTaskParents: map[string]string{},
	}
	if setter, ok := base.(observerSetter); ok {
		setter.SetStderrObserver(runner.onBaseRunnerStderr)
	}
	return runner
}

func (r *RuntimeRunner) SetStderrObserver(observer engine.StderrObserver) {
	if r == nil {
		return
	}
	r.observerMu.Lock()
	r.observer = observer
	r.observerMu.Unlock()
}

func (r *RuntimeRunner) stderrObserver() engine.StderrObserver {
	if r == nil {
		return nil
	}
	r.observerMu.RLock()
	defer r.observerMu.RUnlock()
	return r.observer
}

func (r *RuntimeRunner) Run(ctx context.Context, req engine.Request) (engine.Response, error) {
	if r == nil {
		return engine.Response{}, fmt.Errorf("runtime runner is nil")
	}
	switch strings.TrimSpace(req.Action) {
	case ActionSessionPlan:
		return r.runPlanSession(ctx, req)
	case ActionSessionExecute:
		return r.runExecutePlan(ctx, req)
	case ActionSessionAuto:
		return r.runAutoSession(ctx, req)
	case ActionSessionApprove:
		return r.runApproveSession(ctx, req)
	default:
		return r.base.Run(ctx, req)
	}
}

func (r *RuntimeRunner) onBaseRunnerStderr(event engine.StderrEvent) {
	parentTaskID := event.TaskID
	r.childTaskMu.RLock()
	if mapped, ok := r.childTaskParents[event.TaskID]; ok {
		parentTaskID = mapped
	}
	r.childTaskMu.RUnlock()

	observer := r.stderrObserver()
	if observer == nil {
		return
	}
	event.TaskID = parentTaskID
	observer(event)
}

func (r *RuntimeRunner) registerChildTask(parentTaskID string, childTaskID string) {
	r.childTaskMu.Lock()
	r.childTaskParents[childTaskID] = parentTaskID
	r.childTaskMu.Unlock()
}

func (r *RuntimeRunner) unregisterChildTask(childTaskID string) {
	r.childTaskMu.Lock()
	delete(r.childTaskParents, childTaskID)
	r.childTaskMu.Unlock()
}

func (r *RuntimeRunner) callTool(ctx context.Context, parentTaskID string, toolID string, payload map[string]any) (engine.Response, error) {
	spec, err := r.registry.MustGet(toolID)
	if err != nil {
		return engine.Response{}, err
	}
	childTaskID := fmt.Sprintf("%s/%s/%d", parentTaskID, strings.ReplaceAll(toolID, ".", "_"), time.Now().UnixNano())
	r.registerChildTask(parentTaskID, childTaskID)
	defer r.unregisterChildTask(childTaskID)
	return r.base.Run(ctx, engine.Request{
		TaskID:  childTaskID,
		Action:  spec.Action,
		Payload: cloneMap(payload),
	})
}

func (r *RuntimeRunner) emitStage(taskID string, stage string, phase string, progress int, message string, extra map[string]any) {
	observer := r.stderrObserver()
	if observer == nil {
		return
	}
	parsed := map[string]any{
		"event":     "stage_progress",
		"stage":     stage,
		"phase":     phase,
		"progress":  progress,
		"message":   message,
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
	}
	for key, value := range extra {
		parsed[key] = value
	}
	observer(engine.StderrEvent{
		TaskID:     taskID,
		Parsed:     parsed,
		ObservedAt: time.Now(),
	})
}

func persistenceContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 5*time.Second)
}

func (r *RuntimeRunner) saveSession(session AgentSession) error {
	ctx, cancel := persistenceContext()
	defer cancel()
	return r.store.SaveSession(ctx, session)
}

func (r *RuntimeRunner) loadSession(sessionID string) (*AgentSession, bool, error) {
	ctx, cancel := persistenceContext()
	defer cancel()
	return r.store.GetSession(ctx, sessionID)
}

func (r *RuntimeRunner) saveTrace(event AgentTraceEvent) error {
	ctx, cancel := persistenceContext()
	defer cancel()
	_, err := r.store.SaveTraceEvent(ctx, event)
	return err
}

func (r *RuntimeRunner) loadTraceSummary(sessionID string) TraceSummary {
	if r == nil || r.store == nil || strings.TrimSpace(sessionID) == "" {
		return TraceSummary{TraceTypeCounts: map[string]int{}}
	}
	ctx, cancel := persistenceContext()
	defer cancel()
	events, err := r.store.ListTrace(ctx, sessionID)
	if err != nil {
		observability.Warn("agent_trace_summary_failed", map[string]any{"session_id": sessionID, "error": err.Error()})
		return TraceSummary{TraceTypeCounts: map[string]int{}}
	}
	return SummarizeTraceEvents(events)
}

func (r *RuntimeRunner) failSession(session AgentSession, summary string) {
	session.Status = SessionStatusFailed
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(session); err != nil {
		observability.Warn("agent_session_save_failed", map[string]any{"session_id": session.SessionID, "error": err.Error()})
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    session.CurrentTaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceSessionFailed,
		Summary:   summary,
		Payload:   map[string]any{"status": session.Status},
	})
}

func (r *RuntimeRunner) successResponse(taskID string, started time.Time, sessionID string, planID string, runMode string, goal string, plan AgentPlan, explanation string, validation map[string]any, execution map[string]any, traceSummary TraceSummary) engine.Response {
	return engine.Response{
		TaskID:     taskID,
		Status:     "ok",
		Result:     r.agentResult(sessionID, planID, runMode, goal, plan, explanation, validation, execution, traceSummary),
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		DurationMS: int(time.Since(started).Milliseconds()),
	}
}

func (r *RuntimeRunner) errorResponse(taskID string, started time.Time, code string, message string, details map[string]any, sessionID string, planID string, runMode string, goal string, plan AgentPlan, validation map[string]any, execution map[string]any, explanation string) engine.Response {
	traceSummary := r.loadTraceSummary(sessionID)
	return engine.Response{
		TaskID: taskID,
		Status: "error",
		Result: r.agentResult(sessionID, planID, runMode, goal, plan, explanation, validation, execution, traceSummary),
		Error: &engine.ErrorBody{
			Code:    code,
			Message: message,
			Details: cloneMap(details),
		},
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		DurationMS: int(time.Since(started).Milliseconds()),
	}
}

func (r *RuntimeRunner) agentResult(sessionID string, planID string, runMode string, goal string, plan AgentPlan, explanation string, validation map[string]any, execution map[string]any, traceSummary TraceSummary) map[string]any {
	return map[string]any{
		"agent": map[string]any{
			"session_id":     sessionID,
			"plan_id":        planID,
			"run_mode":       runMode,
			"goal":           goal,
			"plan":           clonePlan(plan),
			"explanation":    buildAgentExplanationPayload(plan, explanation),
			"strategy_label": strings.TrimSpace(plan.StrategyLabel),
			"intent_label":   strings.TrimSpace(plan.IntentLabel),
			"validation":     cloneMap(validation),
			"execution":      cloneMap(execution),
			"approval":       defaultApprovalResult(),
			"trace_summary":  traceSummary,
		},
		"safety": defaultSafetyResult(),
	}
}

func attachApprovalToResponse(resp engine.Response, approval map[string]any) engine.Response {
	result := cloneMap(resp.Result)
	agentBlock := mapFromAny(result["agent"])
	if agentBlock == nil {
		agentBlock = map[string]any{}
	}
	agentBlock["approval"] = cloneMap(approval)
	result["agent"] = agentBlock
	resp.Result = result
	return resp
}

func (r *RuntimeRunner) toolFailureResponse(taskID string, started time.Time, sessionID string, planID string, runMode string, goal string, plan AgentPlan, validation map[string]any, execution map[string]any, toolResp engine.Response, toolID string) engine.Response {
	message := "Agent tool execution failed"
	details := map[string]any{"tool_id": toolID, "tool_status": toolResp.Status}
	if toolResp.Error != nil {
		message = fmt.Sprintf("Tool %s failed: %s", toolID, toolResp.Error.Message)
		details["tool_error_code"] = toolResp.Error.Code
		details["tool_error_message"] = toolResp.Error.Message
		details["tool_error_details"] = cloneMap(toolResp.Error.Details)
	}
	return r.errorResponse(taskID, started, ErrorToolFailed, message, details, sessionID, planID, runMode, goal, plan, validation, execution, "")
}

func candidateByID(plan AgentPlan, candidateID string) (RepairCandidate, bool) {
	for _, candidate := range plan.Candidates {
		if strings.TrimSpace(candidate.CandidateID) == strings.TrimSpace(candidateID) {
			return candidate, true
		}
	}
	return RepairCandidate{}, false
}

func selectedCandidate(plan AgentPlan) (RepairCandidate, bool) {
	if candidate, ok := candidateByID(plan, plan.SelectedCandidateID); ok {
		return candidate, true
	}
	if len(plan.Candidates) == 0 {
		return RepairCandidate{}, false
	}
	return plan.Candidates[0], true
}

func validateObjectField(payload map[string]any, field string) (map[string]any, error) {
	if payload[field] == nil {
		return nil, nil
	}
	value := mapFromAny(payload[field])
	if value == nil {
		return nil, fmt.Errorf("Field %s must be an object", field)
	}
	return value, nil
}

func buildRulePreviewPayload(csvPath string, issueIDs []string, scanOverrides map[string]any, repairOverrides map[string]any, columnDependencies map[string]any, outputDir string) map[string]any {
	payload := map[string]any{
		"csv_path":        csvPath,
		"issue_ids":       append([]string{}, issueIDs...),
		"plan_only":       true,
		"write_output":    false,
		"enable_rollback": false,
	}
	if len(scanOverrides) > 0 {
		payload["scan_config"] = cloneMap(scanOverrides)
	}
	if len(repairOverrides) > 0 {
		payload["repair_strategy"] = cloneMap(repairOverrides)
	}
	if len(columnDependencies) > 0 {
		payload["column_dependencies"] = cloneMap(columnDependencies)
	}
	if strings.TrimSpace(outputDir) != "" {
		payload["output_dir"] = strings.TrimSpace(outputDir)
	}
	return payload
}

func buildGowerPreviewPayload(csvPath string, issueIDs []string, scanOverrides map[string]any, columnDependencies map[string]any, gowerOverrides map[string]any, outputDir string, modelDir string) map[string]any {
	payload := map[string]any{
		"csv_path":        csvPath,
		"issue_ids":       append([]string{}, issueIDs...),
		"plan_only":       true,
		"write_output":    false,
		"enable_rollback": false,
	}
	if len(scanOverrides) > 0 {
		payload["scan_config"] = cloneMap(scanOverrides)
	}
	if len(columnDependencies) > 0 {
		payload["column_dependencies"] = cloneMap(columnDependencies)
	}
	if len(gowerOverrides) > 0 {
		payload["gower_strategy"] = cloneMap(gowerOverrides)
	}
	if strings.TrimSpace(outputDir) != "" {
		payload["output_dir"] = strings.TrimSpace(outputDir)
	}
	if strings.TrimSpace(modelDir) != "" {
		payload["model_dir"] = strings.TrimSpace(modelDir)
	}
	return payload
}

func buildAgentExplanationPayload(plan AgentPlan, explanation string) map[string]any {
	plan = ensurePlanCognition(plan)
	finalMessage := strings.TrimSpace(explanation)
	if finalMessage == "" {
		finalMessage = strings.TrimSpace(plan.UserExplanation)
	}
	summary := strings.TrimSpace(plan.ReasoningSummary)
	if summary == "" {
		summary = finalMessage
	}
	mode := "deterministic"
	switch plan.Cognition.Status {
	case CognitionStatusEngaged:
		mode = "langgraph_llm"
	case CognitionStatusDegraded:
		mode = "langgraph_degraded"
	case CognitionStatusFallback, CognitionStatusDisabled, CognitionStatusUnavailable:
		if plan.Cognition.Provider == CognitionProviderLangGraph || strings.TrimSpace(plan.Cognition.FallbackReasonCode) != "" {
			mode = "langgraph_fallback"
		}
	}
	return map[string]any{
		"mode":          mode,
		"summary":       summary,
		"final_message": finalMessage,
		"short_bullets": append([]string{}, plan.ExplanationBullets...),
		"reason_codes":  append([]string{}, plan.ReasonCodes...),
		"risk_note":     strings.TrimSpace(plan.RiskNote),
		"cognition":     cognitionStateToMap(plan.Cognition),
	}
}

func (r *RuntimeRunner) buildExecutionExplanation(plan AgentPlan, validation map[string]any, execution map[string]any) string {
	preview := mapFromAny(validation["preview"])
	if len(preview) == 0 {
		preview = cloneMap(validation)
	}
	if len(execution) == 0 {
		return fmt.Sprintf("The saved plan %s was validated but not executed. %s", plan.PlanID, asString(validation["message"]))
	}
	return fmt.Sprintf(
		"The agent validated and executed the %s candidate for plan %s. Resolved=%d, output=%s.",
		plan.SelectedSource,
		plan.PlanID,
		intFromAny(preview["resolved_issue_count"]),
		asString(execution["output_csv"]),
	)
}

func (r *RuntimeRunner) runPlanSession(ctx context.Context, req engine.Request) (engine.Response, error) {
	started := time.Now()
	result, handled, err := r.runPlanningFlow(ctx, req, started, "plan")
	if err != nil {
		return engine.Response{}, err
	}
	if handled != nil {
		return *handled, nil
	}

	explanation := result.Plan.UserExplanation
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: result.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentExplainer,
		TraceType: TraceAgentDecision,
		Summary:   "Prepared plan explanation",
		Payload:   map[string]any{"explanation": explanation},
	})
	r.emitStage(req.TaskID, "agent_explain", "complete", 96, "Plan explanation is ready", nil)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: result.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceSessionCompleted,
		Summary:   "Agent planning session completed",
		Payload:   map[string]any{"plan_id": result.Plan.PlanID, "selected_issue_count": len(result.Plan.SelectedIssueIDs)},
	})
	r.emitStage(req.TaskID, "complete", "complete", 100, "Agent planning session completed", nil)

	traceSummary := r.loadTraceSummary(result.Session.SessionID)
	resp := r.successResponseWithSafety(
		req.TaskID,
		started,
		result.Session.SessionID,
		result.Plan.PlanID,
		"plan",
		result.Goal,
		result.Plan,
		explanation,
		defaultValidationResult(),
		defaultExecutionResult(false),
		defaultSafetyResult(),
		traceSummary,
	)
	return attachApprovalToResponse(resp, approvalResultFromContext(result.Session.Context)), nil
}

func hybridOutputPath(sourceCSV string, requestedOutputDir string, requestedOutputCSV string) string {
	if strings.TrimSpace(requestedOutputCSV) != "" {
		return strings.TrimSpace(requestedOutputCSV)
	}
	baseName := strings.TrimSuffix(filepath.Base(sourceCSV), filepath.Ext(sourceCSV))
	if strings.TrimSpace(requestedOutputDir) != "" {
		return filepath.Join(strings.TrimSpace(requestedOutputDir), baseName+".repaired.hybrid.csv")
	}
	return filepath.Join(filepath.Dir(sourceCSV), baseName+".repaired.hybrid.csv")
}

func copyFile(src string, dst string) error {
	input, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, input, 0o644)
}

func (r *RuntimeRunner) runValidationPreview(ctx context.Context, parentTaskID string, sessionID string, taskID string, candidate RepairCandidate) (map[string]any, engine.Response, string, error) {
	totalResolved := 0
	totalChanged := 0
	beforeIssueCount := 0
	lastResp := engine.Response{}
	lastToolID := ""

	for idx, payload := range candidate.PlanPayloads {
		if idx >= len(candidate.ToolSequence) {
			break
		}
		toolID := candidate.ToolSequence[idx]
		_ = r.saveTrace(AgentTraceEvent{SessionID: sessionID, TaskID: taskID, AgentName: AgentValidator, TraceType: TraceToolCall, Summary: "Calling preview tool for validation", Payload: map[string]any{"tool_id": toolID, "payload": cloneMap(payload)}})
		resp, err := r.callTool(ctx, parentTaskID, toolID, payload)
		if err != nil {
			return nil, engine.Response{}, toolID, err
		}
		lastResp = resp
		lastToolID = toolID
		if strings.ToLower(strings.TrimSpace(resp.Status)) != "ok" {
			return nil, resp, toolID, nil
		}
		_ = r.saveTrace(AgentTraceEvent{SessionID: sessionID, TaskID: taskID, AgentName: AgentValidator, TraceType: TraceToolResult, Summary: "Preview validation tool completed", Payload: map[string]any{"tool_id": toolID, "result": cloneMap(resp.Result)}})
		comparison := mapFromAny(resp.Result["comparison"])
		if comparison == nil {
			comparison = map[string]any{}
		}
		if idx == 0 {
			beforeIssueCount = intFromAny(comparison["before_issue_count"])
		}
		totalResolved += intFromAny(comparison["resolved_issue_count"])
		totalChanged += intFromAny(comparison["changed_cell_count"])
	}

	afterIssueCount := beforeIssueCount - totalResolved
	if afterIssueCount < 0 {
		afterIssueCount = 0
	}
	if candidate.Source != "hybrid" && lastResp.Result != nil {
		if comparison := mapFromAny(lastResp.Result["comparison"]); comparison != nil {
			if beforeIssueCount <= 0 {
				beforeIssueCount = intFromAny(comparison["before_issue_count"])
			}
			afterIssueCount = intFromAny(comparison["after_issue_count"])
			totalResolved = intFromAny(comparison["resolved_issue_count"])
			totalChanged = intFromAny(comparison["changed_cell_count"])
		}
	}

	validation := map[string]any{
		"status":               "checked",
		"candidate_id":         candidate.CandidateID,
		"selected_source":      candidate.Source,
		"before_issue_count":   beforeIssueCount,
		"after_issue_count":    afterIssueCount,
		"resolved_issue_count": totalResolved,
		"changed_cell_count":   totalChanged,
	}
	validation["can_execute"] = totalResolved > 0 && afterIssueCount <= beforeIssueCount
	if canExecute, _ := validation["can_execute"].(bool); canExecute {
		validation["message"] = "Validation passed. The selected candidate can be executed."
	} else {
		validation["status"] = "rejected"
		validation["message"] = "Validation rejected the selected candidate because it did not improve the issue count."
	}
	return validation, lastResp, lastToolID, nil
}

func (r *RuntimeRunner) executeHybridCandidate(ctx context.Context, parentTaskID string, sessionID string, taskID string, candidate RepairCandidate, outputDir string) (map[string]any, error) {
	if len(candidate.ExecutePayloads) == 0 {
		return map[string]any{"status": "not_run"}, nil
	}
	sourceCSV := asString(candidate.ExecutePayloads[0]["csv_path"])
	if sourceCSV == "" {
		return nil, fmt.Errorf("hybrid candidate is missing source csv")
	}

	tempDir, err := os.MkdirTemp("", "agent-hybrid-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	lastPayload := candidate.ExecutePayloads[len(candidate.ExecutePayloads)-1]
	finalOutput := hybridOutputPath(sourceCSV, outputDir, asString(lastPayload["output_csv"]))
	if outputDir == "" {
		if embedded := asString(lastPayload["output_dir"]); embedded != "" {
			finalOutput = hybridOutputPath(sourceCSV, embedded, asString(lastPayload["output_csv"]))
		}
	}

	currentCSV := sourceCSV
	executionSteps := make([]map[string]any, 0, len(candidate.ExecutePayloads))
	for idx, payload := range candidate.ExecutePayloads {
		if idx >= len(candidate.ToolSequence) {
			break
		}
		toolID := candidate.ToolSequence[idx]
		callPayload := cloneMap(payload)
		callPayload["csv_path"] = currentCSV
		callPayload["plan_only"] = false
		callPayload["write_output"] = true
		callPayload["enable_rollback"] = false
		if idx < len(candidate.ExecutePayloads)-1 {
			callPayload["output_csv"] = filepath.Join(tempDir, fmt.Sprintf("hybrid_step_%d.csv", idx+1))
		} else {
			callPayload["output_csv"] = finalOutput
			if strings.TrimSpace(outputDir) != "" {
				callPayload["output_dir"] = strings.TrimSpace(outputDir)
			}
		}

		_ = r.saveTrace(AgentTraceEvent{SessionID: sessionID, TaskID: taskID, AgentName: AgentSupervisor, TraceType: TraceToolCall, Summary: "Calling hybrid execution step", Payload: map[string]any{"tool_id": toolID, "payload": cloneMap(callPayload)}})
		resp, err := r.callTool(ctx, parentTaskID, toolID, callPayload)
		if err != nil {
			return nil, err
		}
		if strings.ToLower(strings.TrimSpace(resp.Status)) != "ok" {
			return nil, fmt.Errorf("hybrid step failed for tool %s", toolID)
		}
		_ = r.saveTrace(AgentTraceEvent{SessionID: sessionID, TaskID: taskID, AgentName: AgentSupervisor, TraceType: TraceToolResult, Summary: "Hybrid execution step completed", Payload: map[string]any{"tool_id": toolID, "result": cloneMap(resp.Result)}})
		if stepOutput := asString(resp.Result["output_csv"]); stepOutput != "" {
			currentCSV = stepOutput
		}
		executionSteps = append(executionSteps, map[string]any{
			"step":               idx + 1,
			"tool_id":            toolID,
			"selected_issue_ids": cloneValue(callPayload["issue_ids"]),
			"output_csv":         currentCSV,
			"comparison":         cloneMap(mapFromAny(resp.Result["comparison"])),
		})
	}

	rollbackDir := filepath.Join(filepath.Dir(finalOutput), ".rollback")
	if raw := asString(lastPayload["rollback_dir"]); raw != "" {
		rollbackDir = raw
	}
	if err := os.MkdirAll(rollbackDir, 0o755); err != nil {
		return nil, err
	}
	rollbackID := fmt.Sprintf("rb-%d-hybrid", time.Now().UnixMilli())
	backupCSV := filepath.Join(rollbackDir, rollbackID+"."+filepath.Base(sourceCSV)+".bak.csv")
	if err := copyFile(sourceCSV, backupCSV); err != nil {
		return nil, err
	}
	manifestPath := filepath.Join(rollbackDir, rollbackID+".json")
	manifest := map[string]any{
		"manifest_version":   2,
		"source_tool_id":     "engine.hybrid_repair",
		"rollback_id":        rollbackID,
		"created_at":         time.Now().Unix(),
		"source_csv":         sourceCSV,
		"output_csv":         finalOutput,
		"backup_csv":         backupCSV,
		"execution_steps":    executionSteps,
		"selected_issue_ids": append([]string{}, candidate.SelectedIssueIDs...),
		"issue_source_map":   cloneMap(candidate.IssueSourceMap),
	}
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(manifestPath, append(manifestBytes, '\n'), 0o644); err != nil {
		return nil, err
	}

	return map[string]any{
		"status":              "executed",
		"selected_source":     "hybrid",
		"output_csv":          finalOutput,
		"applied_issue_count": len(candidate.SelectedIssueIDs),
		"rollback": map[string]any{
			"rollback_id":      rollbackID,
			"manifest_path":    manifestPath,
			"backup_csv":       backupCSV,
			"restore_action":   "rollback_repair_batch",
			"manifest_version": 2,
			"source_tool_id":   "engine.hybrid_repair",
		},
		"comparison":      cloneMap(candidate.Comparison),
		"execution_steps": executionSteps,
	}, nil
}

func (r *RuntimeRunner) runExecutePlan(ctx context.Context, req engine.Request) (engine.Response, error) {
	started := time.Now()
	payload := cloneMap(req.Payload)
	sessionID := strings.TrimSpace(asString(payload["session_id"]))
	planID := strings.TrimSpace(asString(payload["plan_id"]))
	outputDir := strings.TrimSpace(asString(payload["output_dir"]))

	if sessionID == "" {
		return r.errorResponse(req.TaskID, started, ErrorInvalidInput, "Field session_id is required for agent.session.execute", map[string]any{"field": "session_id"}, "", planID, "execute", "", AgentPlan{}, map[string]any{"status": "not_run"}, map[string]any{"status": "not_run"}, ""), nil
	}
	if planID == "" {
		return r.errorResponse(req.TaskID, started, ErrorInvalidInput, "Field plan_id is required for agent.session.execute", map[string]any{"field": "plan_id"}, sessionID, "", "execute", "", AgentPlan{}, map[string]any{"status": "not_run"}, map[string]any{"status": "not_run"}, ""), nil
	}

	session, found, err := r.loadSession(sessionID)
	if err != nil {
		return engine.Response{}, err
	}
	if !found {
		return r.errorResponse(req.TaskID, started, ErrorSessionNotFound, "Agent session was not found", map[string]any{"session_id": sessionID}, sessionID, planID, "execute", "", AgentPlan{}, map[string]any{"status": "not_run"}, map[string]any{"status": "not_run"}, ""), nil
	}
	if strings.TrimSpace(session.LatestPlan.PlanID) != planID {
		return r.errorResponse(req.TaskID, started, ErrorPlanNotFound, "Agent plan was not found for the requested session", map[string]any{"session_id": sessionID, "requested_plan_id": planID, "latest_plan_id": session.LatestPlan.PlanID}, sessionID, planID, "execute", session.UserGoal, session.LatestPlan, map[string]any{"status": "not_run"}, map[string]any{"status": "not_run"}, session.LatestPlan.UserExplanation), nil
	}
	candidate, ok := selectedCandidate(session.LatestPlan)
	if !ok {
		return r.errorResponse(req.TaskID, started, ErrorPlanNotFound, "Selected candidate was not found for the requested plan", map[string]any{"session_id": sessionID, "plan_id": planID}, sessionID, planID, "execute", session.UserGoal, session.LatestPlan, map[string]any{"status": "not_run"}, map[string]any{"status": "not_run"}, session.LatestPlan.UserExplanation), nil
	}

	if err := r.applyExecutionPreferenceOverrides(session, payload, req.TaskID); err != nil {
		return r.errorResponse(req.TaskID, started, ErrorInvalidInput, err.Error(), map[string]any{"field": "user_preferences"}, sessionID, planID, "execute", session.UserGoal, session.LatestPlan, defaultValidationResult(), defaultExecutionResult(false), session.LatestPlan.UserExplanation), nil
	}
	session.CurrentTaskID = req.TaskID
	session.Status = SessionStatusExecuting
	session.Mode = "execute"
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}

	r.emitStage(req.TaskID, "agent_validate", "start", 28, "Agent is validating the selected candidate", map[string]any{"selected_source": candidate.Source})
	previewValidation, previewResp, previewToolID, err := r.runValidationPreview(ctx, req.TaskID, sessionID, req.TaskID, candidate)
	if err != nil {
		r.failSession(*session, "Agent execute failed while validating the selected candidate")
		return engine.Response{}, err
	}
	if strings.TrimSpace(previewToolID) != "" && strings.ToLower(strings.TrimSpace(previewResp.Status)) != "ok" {
		r.failSession(*session, "Validation preview tool returned an error")
		return r.toolFailureResponse(req.TaskID, started, sessionID, planID, "execute", session.UserGoal, session.LatestPlan, defaultValidationResult(), defaultExecutionResult(false), previewResp, previewToolID), nil
	}
	validation := buildValidationEnvelope(previewValidation)
	session.Context["validation_preview"] = cloneMap(previewValidation)
	session.Context["preview_validation"] = cloneMap(previewValidation)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: sessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceValidation,
		Summary:   asString(previewValidation["message"]),
		Payload: func() map[string]any {
			payload := cloneMap(previewValidation)
			payload["phase"] = "preview"
			return payload
		}(),
	})
	r.emitStage(req.TaskID, "agent_validate", "complete", 52, "Plan validation finished", cloneMap(previewValidation))

	canExecute, _ := validation["can_execute"].(bool)
	if !canExecute {
		session.Status = SessionStatusValidationRejected
		session.UpdatedAt = time.Now().UTC()
		if err := r.saveSession(*session); err != nil {
			return engine.Response{}, err
		}
		return r.errorResponse(req.TaskID, started, ErrorValidationRejected, "Plan validation rejected execution", map[string]any{"session_id": sessionID, "plan_id": planID, "validation": cloneMap(validation)}, sessionID, planID, "execute", session.UserGoal, session.LatestPlan, validation, map[string]any{"status": "not_run"}, r.buildExecutionExplanation(session.LatestPlan, validation, nil)), nil
	}
	approvalResp, err := r.maybePauseForApproval(req, started, session, session.UserGoal, session.LatestPlan, validation, defaultExecutionResult(false), defaultSafetyResult())
	if err != nil {
		return engine.Response{}, err
	}
	if approvalResp != nil {
		return *approvalResp, nil
	}
	return r.continueExecuteApprovedSession(ctx, started, req, session, validation, outputDir)
}
