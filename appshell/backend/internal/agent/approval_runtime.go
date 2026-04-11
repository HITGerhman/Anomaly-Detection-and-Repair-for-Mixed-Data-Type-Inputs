package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"appshell/backend/internal/engine"
)

func buildApprovalSafetyResult(verdict string, baseline map[string]any, riskAssessment map[string]any, message string) map[string]any {
	reasons := stringsFromAny(riskAssessment["reason_codes"])
	return buildSafetyResult(
		verdict,
		reasons,
		cloneMap(baseline),
		map[string]any{},
		buildRollbackRecommendation(false, message),
		map[string]any{"status": "not_run"},
		"",
	)
}

func buildApprovalExplanation(plan AgentPlan, riskAssessment map[string]any, decision string) string {
	switch strings.TrimSpace(decision) {
	case approvalStatusRejected:
		return fmt.Sprintf("Execution for plan %s was canceled before any files were written.", plan.PlanID)
	case approvalStatusApproved:
		return fmt.Sprintf("Approval was granted for plan %s, so the session resumed from the stored preview validation checkpoint.", plan.PlanID)
	default:
		return firstNonEmpty(
			asString(riskAssessment["message"]),
			fmt.Sprintf("Execution for plan %s is paused until approval is granted.", plan.PlanID),
		)
	}
}

func (r *RuntimeRunner) maybePauseForApproval(req engine.Request, started time.Time, session *AgentSession, goal string, plan AgentPlan, validation map[string]any, execution map[string]any, safety map[string]any) (*engine.Response, error) {
	if session == nil {
		return nil, fmt.Errorf("session is nil")
	}
	riskAssessment := refreshRiskAssessmentForSession(session, nil)
	required, _ := riskAssessment["required"].(bool)
	status := approvalStatusNotRequired
	if required {
		status = approvalStatusRequired
	}
	approvalState := buildApprovalState(status, required, riskAssessment)
	updateSessionApprovalContext(session, riskAssessment, approvalState)
	session.UpdatedAt = time.Now().UTC()
	if !required {
		if err := r.saveSession(*session); err != nil {
			return nil, err
		}
		return nil, nil
	}

	approvalState = enrichApprovalState(approvalState, req.TaskID, "")
	updateSessionApprovalContext(session, riskAssessment, approvalState)
	session.CurrentTaskID = req.TaskID
	session.Status = SessionStatusAwaitingApproval
	session.Context["final_verdict"] = "approval_required"
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return nil, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceApprovalRequested,
		Summary:   "Execution paused because approval is required before writing output",
		Payload: map[string]any{
			"approval_state":  cloneMap(approvalState),
			"risk_assessment": cloneMap(riskAssessment),
		},
	})
	r.emitStage(req.TaskID, "agent_approval", "complete", 78, "Execution is paused until approval is granted", map[string]any{
		"reason_codes": stringsFromAny(riskAssessment["reason_codes"]),
		"columns":      stringsFromAny(riskAssessment["candidate_columns"]),
	})
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentExplainer,
		TraceType: TraceAgentDecision,
		Summary:   "Prepared approval request summary",
		Payload: map[string]any{
			"message": buildApprovalExplanation(plan, riskAssessment, approvalStatusRequired),
		},
	})
	r.emitStage(req.TaskID, "complete", "complete", 100, "Approval is required before writing output", nil)

	traceSummary := r.loadTraceSummary(session.SessionID)
	resp := r.successResponseWithSafety(
		req.TaskID,
		started,
		session.SessionID,
		plan.PlanID,
		session.Mode,
		goal,
		plan,
		buildApprovalExplanation(plan, riskAssessment, approvalStatusRequired),
		validation,
		execution,
		buildApprovalSafetyResult("approval_required", mapFromAny(session.Context["baseline_scan"]), riskAssessment, asString(riskAssessment["message"])),
		traceSummary,
	)
	resp = attachApprovalToResponse(resp, approvalResultFromContext(session.Context))
	return &resp, nil
}

func (r *RuntimeRunner) applyExecutionPreferenceOverrides(session *AgentSession, payload map[string]any, taskID string) error {
	if session == nil {
		return fmt.Errorf("session is nil")
	}
	rawPreferences, err := validateObjectField(payload, "user_preferences")
	if err != nil {
		return err
	}
	workspaceID := resolveWorkspaceID(asString(payload["workspace_id"]), asString(session.Context["csv_path"]))
	if workspaceID == "" {
		workspaceID = asString(session.Context["workspace_id"])
	}
	if len(rawPreferences) == 0 && workspaceID == asString(session.Context["workspace_id"]) {
		return nil
	}
	_, _, snapshot, err := resolvePreferenceSnapshot(r.store, workspaceID, rawPreferences)
	if err != nil {
		return err
	}
	session.Context["workspace_id"] = workspaceID
	session.Context["preference_snapshot"] = preferenceProfileToMap(snapshot)
	session.Context["user_preferences"] = cloneMap(rawPreferences)
	riskAssessment := refreshRiskAssessmentForSession(session, &snapshot)
	updateSessionApprovalContext(session, riskAssessment, buildApprovalState(func() string {
		if riskAssessment["required"] == true {
			return approvalStatusRequired
		}
		return approvalStatusNotRequired
	}(), riskAssessment["required"] == true, riskAssessment))
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    taskID,
		AgentName: AgentProfile,
		TraceType: TraceMemoryUpdated,
		Summary:   "Execution request refreshed the preference snapshot for this session",
		Payload: map[string]any{
			"workspace_id":        workspaceID,
			"preference_snapshot": preferenceProfileToMap(snapshot),
		},
	})
	return nil
}

func (r *RuntimeRunner) continueExecuteApprovedSession(ctx context.Context, started time.Time, req engine.Request, session *AgentSession, validation map[string]any, outputDir string) (engine.Response, error) {
	if session == nil {
		return engine.Response{}, fmt.Errorf("session is nil")
	}
	sessionID := session.SessionID
	planID := session.LatestPlan.PlanID
	goal := session.UserGoal
	candidate, ok := selectedCandidate(session.LatestPlan)
	if !ok {
		resp := r.errorResponse(req.TaskID, started, ErrorPlanNotFound, "Selected candidate was not found for the requested plan", map[string]any{"session_id": sessionID, "plan_id": planID}, sessionID, planID, "execute", goal, session.LatestPlan, validation, map[string]any{"status": "not_run"}, session.LatestPlan.UserExplanation)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	session.CurrentTaskID = req.TaskID
	session.Status = SessionStatusExecuting
	session.Mode = "execute"
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}

	r.emitStage(req.TaskID, "agent_execute", "start", 64, "Agent is executing the selected candidate", map[string]any{"selected_source": candidate.Source})
	execution, toolResp, toolID, err := r.executeCandidate(ctx, req.TaskID, sessionID, req.TaskID, candidate, outputDir)
	if err != nil {
		r.failSession(*session, "Agent execute failed while applying the selected candidate")
		return engine.Response{}, err
	}
	if toolResp != nil {
		r.failSession(*session, "Execution tool returned an error")
		resp := r.toolFailureResponse(req.TaskID, started, sessionID, planID, "execute", goal, session.LatestPlan, validation, defaultExecutionResult(false), *toolResp, toolID)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	session.Context["execution_artifacts"] = map[string]any{
		"output_csv":       asString(execution["output_csv"]),
		"rollback":         cloneMap(mapFromAny(execution["rollback"])),
		"selected_source":  candidate.Source,
		"issue_source_map": cloneMap(candidate.IssueSourceMap),
	}
	session.Status = SessionStatusCompleted
	session.Context["final_verdict"] = ""
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	r.emitStage(req.TaskID, "agent_execute", "complete", 84, "Repair execution completed", nil)

	explanation := r.buildExecutionExplanation(session.LatestPlan, validation, execution)
	_ = r.saveTrace(AgentTraceEvent{SessionID: sessionID, TaskID: req.TaskID, AgentName: AgentExplainer, TraceType: TraceAgentDecision, Summary: "Prepared execution explanation", Payload: map[string]any{"explanation": explanation}})
	r.emitStage(req.TaskID, "agent_explain", "complete", 96, "Execution explanation is ready", nil)
	_ = r.saveTrace(AgentTraceEvent{SessionID: sessionID, TaskID: req.TaskID, AgentName: AgentSupervisor, TraceType: TraceSessionCompleted, Summary: "Agent execute flow completed", Payload: map[string]any{"plan_id": planID, "output": asString(execution["output_csv"]), "selected_source": candidate.Source}})
	r.emitStage(req.TaskID, "complete", "complete", 100, "Agent execution completed", nil)

	traceSummary := r.loadTraceSummary(sessionID)
	resp := r.successResponse(req.TaskID, started, sessionID, planID, "execute", goal, session.LatestPlan, explanation, validation, execution, traceSummary)
	return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
}

func (r *RuntimeRunner) continueAutoApprovedSession(ctx context.Context, started time.Time, req engine.Request, session *AgentSession, validation map[string]any) (engine.Response, error) {
	if session == nil {
		return engine.Response{}, fmt.Errorf("session is nil")
	}
	plan := session.LatestPlan
	goal := session.UserGoal
	baseline := cloneMap(mapFromAny(session.Context["baseline_scan"]))
	outputDir := asString(session.Context["output_dir"])
	scanOverrides := cloneMap(mapFromAny(session.Context["scan_config_overrides"]))
	safety := defaultSafetyResult()

	candidate, ok := selectedCandidate(plan)
	if !ok {
		safety = buildSafetyResult("rollback_failed", []string{"selected_candidate_missing"}, cloneMap(baseline), map[string]any{}, buildRollbackRecommendation(false, "No candidate is available for automatic execution"), map[string]any{"status": "not_run"}, "")
		resp := r.errorResponseWithSafety(req.TaskID, started, ErrorPlanNotFound, "Selected candidate was not found for automatic execution", map[string]any{"session_id": session.SessionID, "plan_id": plan.PlanID}, session.SessionID, plan.PlanID, "auto", goal, plan, validation, defaultExecutionResult(true), safety, plan.UserExplanation)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	session.CurrentTaskID = req.TaskID
	session.Mode = "auto"
	session.Status = SessionStatusExecuting
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}

	r.emitStage(req.TaskID, "agent_execute", "start", 80, "Agent is executing the selected candidate", map[string]any{"selected_source": candidate.Source})
	execution, toolResp, toolID, err := r.executeCandidate(ctx, req.TaskID, session.SessionID, req.TaskID, candidate, outputDir)
	if err != nil {
		r.failSession(*session, "Agent auto flow failed during execution")
		return engine.Response{}, err
	}
	if toolResp != nil {
		r.failSession(*session, "Automatic execution tool returned an error")
		resp := r.toolFailureResponseWithSafety(req.TaskID, started, session.SessionID, plan.PlanID, "auto", goal, plan, validation, execution, safety, *toolResp, toolID)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}
	execution["auto_mode"] = true
	execution["rollback_applied"] = false
	execution["post_scan_output_csv"] = ""
	session.Context["execution_artifacts"] = map[string]any{
		"output_csv":       asString(execution["output_csv"]),
		"rollback":         cloneMap(mapFromAny(execution["rollback"])),
		"selected_source":  asString(execution["selected_source"]),
		"issue_source_map": cloneMap(plan.IssueSourceMap),
		"rollback_applied": false,
	}
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	r.emitStage(req.TaskID, "agent_execute", "complete", 84, "Automatic execution completed", nil)

	riskFlags := []string{}
	outputCSV := asString(execution["output_csv"])
	if strings.TrimSpace(outputCSV) == "" || !fileExists(outputCSV) {
		riskFlags = appendRiskFlag(riskFlags, "output_csv_missing")
	}
	if strings.TrimSpace(rollbackMetaFromExecution(execution).ManifestPath) == "" {
		riskFlags = appendRiskFlag(riskFlags, "missing_rollback_metadata")
	}
	if len(riskFlags) > 0 {
		postValidation := map[string]any{
			"phase":    "post_execute",
			"status":   "rejected",
			"accepted": false,
			"message":  "Automatic safety checks rejected the execution result before post-scan completed.",
		}
		validation = attachPostValidation(validation, postValidation)
		session.Context["post_validation"] = cloneMap(postValidation)
		session.UpdatedAt = time.Now().UTC()
		if err := r.saveSession(*session); err != nil {
			return engine.Response{}, err
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentValidator,
			TraceType: TraceValidation,
			Summary:   asString(postValidation["message"]),
			Payload: func() map[string]any {
				payload := cloneMap(postValidation)
				payload["phase"] = "post_execute"
				payload["risk_flags"] = append([]string{}, riskFlags...)
				return payload
			}(),
		})
		r.emitStage(req.TaskID, "agent_post_validate", "complete", 94, "Post-execute safety checks rejected the output", map[string]any{"risk_flags": append([]string{}, riskFlags...)})
		resp, err := r.finishAutoRollback(ctx, started, req, autoFinalizeInput{
			Session:        session,
			Goal:           goal,
			Plan:           plan,
			Validation:     validation,
			Execution:      execution,
			BaselineScan:   baseline,
			PostScan:       map[string]any{},
			PostValidation: postValidation,
			RiskFlags:      riskFlags,
			Reason:         "Automatic safety checks rejected the execution result before post-scan completed.",
			ErrorCode:      ErrorValidationRejected,
			ErrorMessage:   "Automatic safety checks rejected the execution result",
		})
		if err != nil {
			return engine.Response{}, err
		}
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	r.emitStage(req.TaskID, "agent_rescan", "start", 86, "Agent is rescanning the repaired output", map[string]any{"output_csv": outputCSV})
	_, postScanSummary, postScanResp, postScanToolID, err := r.rescanOutput(ctx, req.TaskID, session.SessionID, req.TaskID, outputCSV, scanOverrides)
	if err != nil {
		riskFlags = appendRiskFlag(riskFlags, "post_scan_failed")
		postValidation := map[string]any{
			"phase":    "post_execute",
			"status":   "rejected",
			"accepted": false,
			"message":  "Post-execute scan failed and requires rollback.",
		}
		validation = attachPostValidation(validation, postValidation)
		session.Context["post_validation"] = cloneMap(postValidation)
		session.UpdatedAt = time.Now().UTC()
		if saveErr := r.saveSession(*session); saveErr != nil {
			return engine.Response{}, saveErr
		}
		resp, err := r.finishAutoRollback(ctx, started, req, autoFinalizeInput{
			Session:        session,
			Goal:           goal,
			Plan:           plan,
			Validation:     validation,
			Execution:      execution,
			BaselineScan:   baseline,
			PostScan:       map[string]any{},
			PostValidation: postValidation,
			RiskFlags:      riskFlags,
			Reason:         "Post-execute scan failed and requires rollback.",
			ErrorCode:      ErrorValidationRejected,
			ErrorMessage:   "Post-execute scan failed",
		})
		if err != nil {
			return engine.Response{}, err
		}
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}
	if postScanResp != nil {
		riskFlags = appendRiskFlag(riskFlags, "post_scan_failed")
		postValidation := map[string]any{
			"phase":    "post_execute",
			"status":   "rejected",
			"accepted": false,
			"message":  "Post-execute scan returned an error and requires rollback.",
		}
		validation = attachPostValidation(validation, postValidation)
		session.Context["post_validation"] = cloneMap(postValidation)
		session.UpdatedAt = time.Now().UTC()
		if saveErr := r.saveSession(*session); saveErr != nil {
			return engine.Response{}, saveErr
		}
		_ = postScanToolID
		resp, err := r.finishAutoRollback(ctx, started, req, autoFinalizeInput{
			Session:        session,
			Goal:           goal,
			Plan:           plan,
			Validation:     validation,
			Execution:      execution,
			BaselineScan:   baseline,
			PostScan:       map[string]any{},
			PostValidation: postValidation,
			RiskFlags:      riskFlags,
			Reason:         "Post-execute scan returned an error and requires rollback.",
			ErrorCode:      ErrorValidationRejected,
			ErrorMessage:   "Post-execute scan returned an error",
		})
		if err != nil {
			return engine.Response{}, err
		}
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}
	execution["post_scan_output_csv"] = outputCSV
	session.Context["post_scan"] = cloneMap(postScanSummary)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	r.emitStage(req.TaskID, "agent_rescan", "complete", 90, "Repaired output rescanned", map[string]any{"issue_count": intFromAny(postScanSummary["issue_count"])})

	r.emitStage(req.TaskID, "agent_post_validate", "start", 92, "Agent is evaluating post-execute safety checks", nil)
	postValidation := buildPostValidation(baseline, postScanSummary)
	validation = attachPostValidation(validation, postValidation.Summary)
	session.Context["post_validation"] = cloneMap(postValidation.Summary)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentValidator,
		TraceType: TraceValidation,
		Summary:   asString(postValidation.Summary["message"]),
		Payload: func() map[string]any {
			payload := cloneMap(postValidation.Summary)
			payload["phase"] = "post_execute"
			return payload
		}(),
	})
	r.emitStage(req.TaskID, "agent_post_validate", "complete", 94, "Post-execute safety validation completed", map[string]any{"accepted": postValidation.Accepted})

	riskFlags = uniqueStrings(append(riskFlags, postValidation.RiskFlags...))
	if !postValidation.Accepted {
		resp, err := r.finishAutoRollback(ctx, started, req, autoFinalizeInput{
			Session:        session,
			Goal:           goal,
			Plan:           plan,
			Validation:     validation,
			Execution:      execution,
			BaselineScan:   baseline,
			PostScan:       postScanSummary,
			PostValidation: postValidation.Summary,
			RiskFlags:      riskFlags,
			Reason:         asString(postValidation.Summary["message"]),
			ErrorCode:      ErrorValidationRejected,
			ErrorMessage:   "Post-execute validation rejected the repaired output",
		})
		if err != nil {
			return engine.Response{}, err
		}
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	session.Status = SessionStatusCompleted
	session.Context["final_verdict"] = "accepted"
	session.Context["execution_artifacts"] = map[string]any{
		"output_csv":               outputCSV,
		"rollback":                 cloneMap(mapFromAny(execution["rollback"])),
		"selected_source":          asString(execution["selected_source"]),
		"issue_source_map":         cloneMap(plan.IssueSourceMap),
		"rollback_applied":         false,
		"rejected_output_snapshot": "",
	}
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	safety = buildSafetyResult("accepted", riskFlags, baseline, postScanSummary, buildRollbackRecommendation(false, "Post-execute validation accepted the repaired output"), map[string]any{"status": "not_run"}, "")
	explanation := buildVerdictExplanation(plan, validation, execution, safety)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentExplainer,
		TraceType: TraceAgentDecision,
		Summary:   "Prepared auto execution explanation",
		Payload:   map[string]any{"explanation": explanation},
	})
	r.emitStage(req.TaskID, "agent_explain", "complete", 99, "Explanation is ready", nil)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceSessionCompleted,
		Summary:   "Agent auto session completed",
		Payload:   map[string]any{"plan_id": plan.PlanID, "final_verdict": "accepted"},
	})
	r.emitStage(req.TaskID, "complete", "complete", 100, "Agent auto session completed", nil)

	traceSummary := r.loadTraceSummary(session.SessionID)
	resp := r.successResponseWithSafety(req.TaskID, started, session.SessionID, plan.PlanID, "auto", goal, plan, explanation, validation, execution, safety, traceSummary)
	return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
}

func (r *RuntimeRunner) runApproveSession(ctx context.Context, req engine.Request) (engine.Response, error) {
	started := time.Now()
	payload := cloneMap(req.Payload)
	sessionID := strings.TrimSpace(asString(payload["session_id"]))
	planID := strings.TrimSpace(asString(payload["plan_id"]))
	decision := strings.ToLower(strings.TrimSpace(asString(payload["decision"])))
	if sessionID == "" {
		return r.errorResponse(req.TaskID, started, ErrorInvalidInput, "Field session_id is required for agent.session.approve", map[string]any{"field": "session_id"}, "", planID, "approve", "", AgentPlan{}, defaultValidationResult(), defaultExecutionResult(false), ""), nil
	}
	if planID == "" {
		return r.errorResponse(req.TaskID, started, ErrorInvalidInput, "Field plan_id is required for agent.session.approve", map[string]any{"field": "plan_id"}, sessionID, "", "approve", "", AgentPlan{}, defaultValidationResult(), defaultExecutionResult(false), ""), nil
	}
	if decision != "approve" && decision != "reject" {
		return r.errorResponse(req.TaskID, started, ErrorInvalidInput, "Field decision must be approve or reject", map[string]any{"field": "decision", "value": decision}, sessionID, planID, "approve", "", AgentPlan{}, defaultValidationResult(), defaultExecutionResult(false), ""), nil
	}

	session, found, err := r.loadSession(sessionID)
	if err != nil {
		return engine.Response{}, err
	}
	if !found {
		return r.errorResponse(req.TaskID, started, ErrorSessionNotFound, "Agent session was not found", map[string]any{"session_id": sessionID}, sessionID, planID, "approve", "", AgentPlan{}, defaultValidationResult(), defaultExecutionResult(false), ""), nil
	}
	if strings.TrimSpace(session.LatestPlan.PlanID) != planID {
		return r.errorResponse(req.TaskID, started, ErrorPlanNotFound, "Agent plan was not found for the requested session", map[string]any{"session_id": sessionID, "requested_plan_id": planID, "latest_plan_id": session.LatestPlan.PlanID}, sessionID, planID, "approve", session.UserGoal, session.LatestPlan, defaultValidationResult(), defaultExecutionResult(session.Mode == "auto"), session.LatestPlan.UserExplanation), nil
	}
	if strings.TrimSpace(session.Status) != SessionStatusAwaitingApproval {
		resp := r.errorResponse(req.TaskID, started, ErrorInvalidInput, "Session is not waiting for approval", map[string]any{"session_id": sessionID, "status": session.Status}, sessionID, planID, session.Mode, session.UserGoal, session.LatestPlan, buildValidationEnvelope(mapFromAny(session.Context["preview_validation"])), defaultExecutionResult(session.Mode == "auto"), session.LatestPlan.UserExplanation)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	riskAssessment := refreshRiskAssessmentForSession(session, nil)
	if decision == "reject" {
		approvalState := enrichApprovalState(buildApprovalState(approvalStatusRejected, true, riskAssessment), req.TaskID, approvalStatusRejected)
		updateSessionApprovalContext(session, riskAssessment, approvalState)
		session.CurrentTaskID = req.TaskID
		session.Status = SessionStatusApprovalRejected
		session.Context["final_verdict"] = "approval_rejected"
		session.UpdatedAt = time.Now().UTC()
		if err := r.saveSession(*session); err != nil {
			return engine.Response{}, err
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentSupervisor,
			TraceType: TraceApprovalRejected,
			Summary:   "Execution was canceled during the approval gate",
			Payload:   map[string]any{"approval_state": cloneMap(approvalState)},
		})
		r.emitStage(req.TaskID, "agent_approval", "complete", 100, "This execution was canceled before any files were written", nil)

		traceSummary := r.loadTraceSummary(session.SessionID)
		resp := r.successResponseWithSafety(
			req.TaskID,
			started,
			session.SessionID,
			planID,
			session.Mode,
			session.UserGoal,
			session.LatestPlan,
			buildApprovalExplanation(session.LatestPlan, riskAssessment, approvalStatusRejected),
			buildValidationEnvelope(mapFromAny(session.Context["preview_validation"])),
			defaultExecutionResult(session.Mode == "auto"),
			buildApprovalSafetyResult("approval_rejected", mapFromAny(session.Context["baseline_scan"]), riskAssessment, "Execution was canceled before any files were written."),
			traceSummary,
		)
		return attachApprovalToResponse(resp, approvalResultFromContext(session.Context)), nil
	}

	approvalState := enrichApprovalState(buildApprovalState(approvalStatusApproved, true, riskAssessment), req.TaskID, approvalStatusApproved)
	updateSessionApprovalContext(session, riskAssessment, approvalState)
	session.CurrentTaskID = req.TaskID
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(*session); err != nil {
		return engine.Response{}, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceApprovalGranted,
		Summary:   "Approval was granted and the session is resuming execution",
		Payload:   map[string]any{"approval_state": cloneMap(approvalState)},
	})

	validation := buildValidationEnvelope(mapFromAny(session.Context["preview_validation"]))
	if session.Mode == "auto" {
		return r.continueAutoApprovedSession(ctx, started, req, session, validation)
	}
	outputDir := firstNonEmpty(asString(payload["output_dir"]), asString(session.Context["output_dir"]))
	return r.continueExecuteApprovedSession(ctx, started, req, session, validation, outputDir)
}
