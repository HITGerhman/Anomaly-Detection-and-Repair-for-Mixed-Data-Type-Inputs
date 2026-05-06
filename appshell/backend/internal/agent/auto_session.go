package agent

import (
	"context"
	"strings"
	"time"

	"appshell/backend/internal/engine"
)

func (r *RuntimeRunner) executeCandidate(ctx context.Context, parentTaskID string, sessionID string, taskID string, candidate RepairCandidate, outputDir string) (map[string]any, *engine.Response, string, error) {
	switch candidate.Source {
	case "rule", "gower":
		if len(candidate.ExecutePayloads) == 0 || len(candidate.ToolSequence) == 0 {
			return defaultExecutionResult(false), nil, "", nil
		}
		toolID := candidate.ToolSequence[0]
		executePayload := cloneMap(candidate.ExecutePayloads[0])
		executePayload["plan_only"] = false
		executePayload["write_output"] = true
		executePayload["enable_rollback"] = true
		if strings.TrimSpace(outputDir) != "" {
			executePayload["output_dir"] = strings.TrimSpace(outputDir)
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: sessionID,
			TaskID:    taskID,
			AgentName: AgentSupervisor,
			TraceType: TraceToolCall,
			Summary:   "Calling execution tool",
			Payload:   map[string]any{"tool_id": toolID, "payload": cloneMap(executePayload)},
		})
		resp, err := r.callTool(ctx, parentTaskID, toolID, executePayload)
		if err != nil {
			return nil, nil, toolID, err
		}
		if strings.ToLower(strings.TrimSpace(resp.Status)) != "ok" {
			return nil, &resp, toolID, nil
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: sessionID,
			TaskID:    taskID,
			AgentName: AgentSupervisor,
			TraceType: TraceToolResult,
			Summary:   "Execution tool completed",
			Payload:   map[string]any{"tool_id": toolID, "result": cloneMap(resp.Result)},
		})
		execution := defaultExecutionResult(false)
		execution["status"] = "executed"
		execution["selected_source"] = candidate.Source
		execution["output_csv"] = asString(resp.Result["output_csv"])
		execution["applied_issue_count"] = intFromAny(resp.Result["applied_issue_count"])
		execution["rollback"] = cloneMap(mapFromAny(resp.Result["rollback"]))
		execution["comparison"] = cloneMap(mapFromAny(resp.Result["comparison"]))
		return execution, nil, toolID, nil
	case "hybrid":
		execution, err := r.executeHybridCandidate(ctx, parentTaskID, sessionID, taskID, candidate, outputDir)
		if err != nil {
			return nil, nil, "engine.hybrid_repair", err
		}
		return execution, nil, "engine.hybrid_repair", nil
	default:
		return defaultExecutionResult(false), nil, "", nil
	}
}

func (r *RuntimeRunner) rescanOutput(ctx context.Context, parentTaskID string, sessionID string, taskID string, outputCSV string, scanOverrides map[string]any) (map[string]any, map[string]any, *engine.Response, string, error) {
	scanPayload := buildScanPayload(outputCSV, scanOverrides)
	toolID := "engine.scan_table"
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: sessionID,
		TaskID:    taskID,
		AgentName: AgentValidator,
		TraceType: TraceToolCall,
		Summary:   "Calling engine.scan_table for post-execute validation",
		Payload:   map[string]any{"tool_id": toolID, "payload": cloneMap(scanPayload)},
	})
	resp, err := r.callTool(ctx, parentTaskID, toolID, scanPayload)
	if err != nil {
		return nil, nil, nil, toolID, err
	}
	if strings.ToLower(strings.TrimSpace(resp.Status)) != "ok" {
		return nil, nil, &resp, toolID, nil
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: sessionID,
		TaskID:    taskID,
		AgentName: AgentValidator,
		TraceType: TraceToolResult,
		Summary:   "Post-execute scan completed",
		Payload:   map[string]any{"tool_id": toolID, "result": cloneMap(resp.Result)},
	})
	result := cloneMap(resp.Result)
	return result, scanSummaryFromResult(outputCSV, result), nil, toolID, nil
}

func (r *RuntimeRunner) finishAutoRollback(ctx context.Context, started time.Time, req engine.Request, input autoFinalizeInput) (engine.Response, error) {
	recommendation := buildRollbackRecommendation(true, input.Reason)
	r.emitStage(req.TaskID, "agent_rollback", "start", 95, "Agent is rolling back the unsafe output", nil)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: input.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceRollbackDecision,
		Summary:   input.Reason,
		Payload: map[string]any{
			"reason":     input.Reason,
			"risk_flags": append([]string{}, uniqueStrings(input.RiskFlags)...),
			"execution":  cloneMap(input.Execution),
		},
	})

	snapshotPath := ""
	if candidateSnapshot, err := writeRejectedSnapshot(input.Execution); err == nil {
		snapshotPath = candidateSnapshot
	} else if asString(input.Execution["output_csv"]) != "" {
		input.RiskFlags = appendRiskFlag(input.RiskFlags, "rejected_output_snapshot_failed")
	}

	meta := rollbackMetaFromExecution(input.Execution)
	if strings.TrimSpace(meta.ManifestPath) == "" {
		input.Session.Status = SessionStatusRollbackFailed
		input.Session.Context["rollback_summary"] = map[string]any{
			"status":  "failed",
			"reason":  "missing_rollback_metadata",
			"message": "Automatic rollback metadata is missing from the execution result.",
		}
		input.Session.Context["final_verdict"] = "rollback_failed"
		input.Session.Context["rejected_output_snapshot"] = snapshotPath
		input.Session.Context["post_scan"] = cloneMap(input.PostScan)
		input.Session.Context["post_validation"] = cloneMap(input.PostValidation)
		input.Session.UpdatedAt = time.Now().UTC()
		if err := r.saveSession(*input.Session); err != nil {
			return engine.Response{}, err
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: input.Session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentSupervisor,
			TraceType: TraceSessionFailed,
			Summary:   "Automatic rollback failed because rollback metadata is missing",
			Payload:   map[string]any{"reason": "missing_rollback_metadata"},
		})
		r.emitStage(req.TaskID, "agent_explain", "complete", 99, "Rollback metadata is missing", nil)
		r.emitStage(req.TaskID, "complete", "complete", 100, "Agent automatic recovery failed", nil)
		rollbackExecution := map[string]any{
			"status":  "failed",
			"reason":  "missing_rollback_metadata",
			"message": "Automatic rollback metadata is missing from the execution result.",
		}
		safety := buildSafetyResult("rollback_failed", input.RiskFlags, input.BaselineScan, input.PostScan, recommendation, rollbackExecution, snapshotPath)
		explanation := buildVerdictExplanation(input.Plan, input.Validation, input.Execution, safety)
		return r.errorResponseWithSafety(req.TaskID, started, ErrorToolFailed, "Automatic rollback metadata is missing from the execution result", map[string]any{
			"session_id": input.Session.SessionID,
			"plan_id":    input.Plan.PlanID,
			"execution":  cloneMap(input.Execution),
		}, input.Session.SessionID, input.Plan.PlanID, "auto", input.Goal, input.Plan, input.Validation, input.Execution, safety, explanation), nil
	}

	rollbackPayload := map[string]any{
		"manifest_path":  meta.ManifestPath,
		"restore_target": "output_csv",
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: input.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceToolCall,
		Summary:   "Calling engine.rollback_batch",
		Payload:   map[string]any{"tool_id": "engine.rollback_batch", "payload": cloneMap(rollbackPayload)},
	})
	rollbackResp, err := r.callTool(ctx, req.TaskID, "engine.rollback_batch", rollbackPayload)
	if err != nil {
		input.Session.Status = SessionStatusRollbackFailed
		input.Session.Context["rollback_summary"] = map[string]any{"status": "failed", "reason": err.Error()}
		input.Session.Context["final_verdict"] = "rollback_failed"
		input.Session.Context["rejected_output_snapshot"] = snapshotPath
		input.Session.UpdatedAt = time.Now().UTC()
		if saveErr := r.saveSession(*input.Session); saveErr != nil {
			return engine.Response{}, saveErr
		}
		return engine.Response{}, err
	}
	if strings.ToLower(strings.TrimSpace(rollbackResp.Status)) != "ok" {
		rollbackExecution := map[string]any{
			"status":  "failed",
			"tool_id": "engine.rollback_batch",
		}
		if rollbackResp.Error != nil {
			rollbackExecution["error"] = map[string]any{
				"code":    rollbackResp.Error.Code,
				"message": rollbackResp.Error.Message,
				"details": cloneMap(rollbackResp.Error.Details),
			}
		}
		input.Session.Status = SessionStatusRollbackFailed
		input.Session.Context["rollback_summary"] = cloneMap(rollbackExecution)
		input.Session.Context["final_verdict"] = "rollback_failed"
		input.Session.Context["rejected_output_snapshot"] = snapshotPath
		input.Session.UpdatedAt = time.Now().UTC()
		if err := r.saveSession(*input.Session); err != nil {
			return engine.Response{}, err
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: input.Session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentSupervisor,
			TraceType: TraceSessionFailed,
			Summary:   "Automatic rollback tool returned an error",
			Payload:   map[string]any{"tool_id": "engine.rollback_batch"},
		})
		r.emitStage(req.TaskID, "agent_explain", "complete", 99, "Automatic rollback failed", nil)
		r.emitStage(req.TaskID, "complete", "complete", 100, "Agent automatic recovery failed", nil)
		safety := buildSafetyResult("rollback_failed", input.RiskFlags, input.BaselineScan, input.PostScan, recommendation, rollbackExecution, snapshotPath)
		explanation := buildVerdictExplanation(input.Plan, input.Validation, input.Execution, safety)
		return r.errorResponseWithSafety(req.TaskID, started, ErrorToolFailed, "Automatic rollback failed", map[string]any{
			"session_id": input.Session.SessionID,
			"plan_id":    input.Plan.PlanID,
			"rollback":   cloneMap(rollbackExecution),
		}, input.Session.SessionID, input.Plan.PlanID, "auto", input.Goal, input.Plan, input.Validation, input.Execution, safety, explanation), nil
	}

	_ = r.saveTrace(AgentTraceEvent{
		SessionID: input.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceToolResult,
		Summary:   "Rollback tool completed",
		Payload:   map[string]any{"tool_id": "engine.rollback_batch", "result": cloneMap(rollbackResp.Result)},
	})
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: input.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceRollbackExecuted,
		Summary:   "Automatic rollback restored the repaired artifact",
		Payload:   map[string]any{"rollback": cloneMap(rollbackResp.Result)},
	})

	input.Execution["rollback_applied"] = true
	input.Session.Status = SessionStatusRolledBack
	input.Session.Context["rollback_summary"] = map[string]any{
		"status": "executed",
		"result": cloneMap(rollbackResp.Result),
		"reason": input.Reason,
	}
	input.Session.Context["final_verdict"] = "rolled_back"
	input.Session.Context["rejected_output_snapshot"] = snapshotPath
	input.Session.Context["post_scan"] = cloneMap(input.PostScan)
	input.Session.Context["post_validation"] = cloneMap(input.PostValidation)
	input.Session.Context["execution_artifacts"] = map[string]any{
		"output_csv":               asString(input.Execution["output_csv"]),
		"rollback":                 cloneMap(mapFromAny(input.Execution["rollback"])),
		"selected_source":          asString(input.Execution["selected_source"]),
		"issue_source_map":         cloneMap(input.Plan.IssueSourceMap),
		"rollback_applied":         true,
		"rejected_output_snapshot": snapshotPath,
	}
	input.Session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*input.Session); err != nil {
		return engine.Response{}, err
	}
	r.emitStage(req.TaskID, "agent_rollback", "complete", 97, "Automatic rollback completed", nil)

	rollbackExecution := map[string]any{
		"status": "executed",
		"result": cloneMap(rollbackResp.Result),
	}
	safety := buildSafetyResult("rolled_back", input.RiskFlags, input.BaselineScan, input.PostScan, recommendation, rollbackExecution, snapshotPath)
	explanation := buildVerdictExplanation(input.Plan, input.Validation, input.Execution, safety)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: input.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentExplainer,
		TraceType: TraceAgentDecision,
		Summary:   "Prepared auto-recovery explanation",
		Payload:   map[string]any{"explanation": explanation},
	})
	r.emitStage(req.TaskID, "agent_explain", "complete", 99, "Explanation is ready", nil)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: input.Session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceSessionCompleted,
		Summary:   "Agent auto session completed with rollback",
		Payload:   map[string]any{"plan_id": input.Plan.PlanID, "final_verdict": "rolled_back"},
	})
	r.emitStage(req.TaskID, "complete", "complete", 100, "Agent automatic recovery completed", nil)

	return r.errorResponseWithSafety(req.TaskID, started, input.ErrorCode, input.ErrorMessage, map[string]any{
		"session_id":      input.Session.SessionID,
		"plan_id":         input.Plan.PlanID,
		"post_validation": cloneMap(input.PostValidation),
		"rollback":        cloneMap(rollbackResp.Result),
	}, input.Session.SessionID, input.Plan.PlanID, "auto", input.Goal, input.Plan, input.Validation, input.Execution, safety, explanation), nil
}

func (r *RuntimeRunner) runAutoSession(ctx context.Context, req engine.Request) (engine.Response, error) {
	started := time.Now()
	result, handled, err := r.runPlanningFlow(ctx, req, started, "auto")
	if err != nil {
		return engine.Response{}, err
	}
	if handled != nil {
		return *handled, nil
	}

	session := result.Session
	plan := result.Plan
	goal := result.Goal
	baseline := cloneMap(result.Baseline)
	validation := defaultValidationResult()
	execution := defaultExecutionResult(true)
	safety := defaultSafetyResult()

	candidate, ok := selectedCandidate(plan)
	if !ok {
		safety = buildSafetyResult("rollback_failed", []string{"selected_candidate_missing"}, cloneMap(baseline), map[string]any{}, buildRollbackRecommendation(false, "No candidate is available for automatic execution"), map[string]any{"status": "not_run"}, "")
		resp := r.errorResponseWithSafety(req.TaskID, started, ErrorPlanNotFound, "Selected candidate was not found for automatic execution", map[string]any{"session_id": session.SessionID, "plan_id": plan.PlanID}, session.SessionID, plan.PlanID, "auto", goal, plan, validation, execution, safety, plan.UserExplanation)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	session.CurrentTaskID = req.TaskID
	session.Mode = "auto"
	session.Status = SessionStatusExecuting
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(session); err != nil {
		return engine.Response{}, err
	}

	r.emitStage(req.TaskID, "agent_validate", "start", 72, "Agent is validating the selected candidate", map[string]any{"selected_source": candidate.Source})
	preview := r.runCachedValidationPreview(session.SessionID, req.TaskID, candidate)

	validation = buildValidationEnvelope(preview)
	session.Context["validation_preview"] = cloneMap(preview)
	session.Context["preview_validation"] = cloneMap(preview)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(session); err != nil {
		return engine.Response{}, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceValidation,
		Summary:   asString(preview["message"]),
		Payload: func() map[string]any {
			payload := cloneMap(preview)
			payload["phase"] = "preview"
			return payload
		}(),
	})
	r.emitStage(req.TaskID, "agent_validate", "complete", 76, "Preview validation finished", cloneMap(preview))

	canExecute, _ := preview["can_execute"].(bool)
	if !canExecute {
		session.Status = SessionStatusValidationRejected
		session.Context["post_validation"] = map[string]any{}
		session.Context["final_verdict"] = "validation_rejected"
		session.UpdatedAt = time.Now().UTC()
		if err := r.saveSession(session); err != nil {
			return engine.Response{}, err
		}
		safety = buildSafetyResult("validation_rejected", []string{"preview_validation_rejected"}, cloneMap(baseline), map[string]any{}, buildRollbackRecommendation(false, "Preview validation rejected automatic execution"), map[string]any{"status": "not_run"}, "")
		explanation := buildVerdictExplanation(plan, validation, execution, safety)
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentExplainer,
			TraceType: TraceAgentDecision,
			Summary:   "Prepared preview rejection explanation",
			Payload:   map[string]any{"explanation": explanation},
		})
		r.emitStage(req.TaskID, "agent_explain", "complete", 99, "Explanation is ready", nil)
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentSupervisor,
			TraceType: TraceSessionCompleted,
			Summary:   "Agent auto session rejected execution during preview validation",
			Payload:   map[string]any{"plan_id": plan.PlanID, "final_verdict": "validation_rejected"},
		})
		r.emitStage(req.TaskID, "complete", "complete", 100, "Agent auto session completed", nil)
		resp := r.errorResponseWithSafety(req.TaskID, started, ErrorValidationRejected, "Preview validation rejected automatic execution", map[string]any{"session_id": session.SessionID, "plan_id": plan.PlanID, "validation": cloneMap(validation)}, session.SessionID, plan.PlanID, "auto", goal, plan, validation, execution, safety, explanation)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}
	approvalResp, err := r.maybePauseForApproval(req, started, &session, goal, plan, validation, execution, safety)
	if err != nil {
		return engine.Response{}, err
	}
	if approvalResp != nil {
		return *approvalResp, nil
	}
	return r.continueAutoApprovedSession(ctx, started, req, &session, validation)
}
