package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"appshell/backend/internal/engine"
)

func responsePtr(resp engine.Response) *engine.Response {
	return &resp
}

func (r *RuntimeRunner) runPlanningFlow(ctx context.Context, req engine.Request, started time.Time, mode string) (*planningResult, *engine.Response, error) {
	params, err := parsePlanningParams(cloneMap(req.Payload), req.Action)
	if err != nil {
		resp := r.errorResponseWithSafety(req.TaskID, started, ErrorInvalidInput, err.Error(), map[string]any{}, params.SessionID, "", mode, params.Goal, AgentPlan{}, defaultValidationResult(), defaultExecutionResult(mode == "auto"), defaultSafetyResult(), "")
		return nil, responsePtr(resp), nil
	}
	retrieveMode := resolveRetrieveMode(req.Payload)
	params.WorkspaceID = resolveWorkspaceID(params.WorkspaceID, params.CSVPath)
	preferenceRecord, _, preferenceSnapshot, err := resolvePreferenceSnapshot(r.store, params.WorkspaceID, params.UserPreferences)
	if err != nil {
		return nil, nil, err
	}
	params.RepairOverrides = applyPreferenceRepairDefaults(params.RepairOverrides, preferenceSnapshot)

	progress := planningProgressForMode(mode)
	session := newPlanningSession(req.TaskID, mode, params)
	session.Context["preference_snapshot"] = preferenceProfileToMap(preferenceSnapshot)
	session.Context["agent_retrieve_mode"] = retrieveMode
	if params.OutputDir != "" {
		session.Context["output_dir"] = params.OutputDir
	}
	if err := r.saveSession(session); err != nil {
		return nil, nil, err
	}
	if err := r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceSessionStarted,
		Summary:   fmt.Sprintf("Agent %s session started", mode),
		Payload:   map[string]any{"mode": mode, "goal": params.Goal, "csv_path": params.CSVPath},
	}); err != nil {
		return nil, nil, err
	}

	r.emitStage(req.TaskID, "agent_intent", "start", progress.IntentStart, "Agent is analyzing the user goal", map[string]any{"goal": params.Goal})
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentIntent,
		TraceType: TraceAgentDecision,
		Summary:   "Intent normalized to a deterministic planning goal",
		Payload:   map[string]any{"goal": params.Goal, "normalized": params.Goal},
	})
	r.emitStage(req.TaskID, "agent_intent", "complete", progress.IntentComplete, "Agent intent analysis completed", nil)

	scanPayload := buildScanPayload(params.CSVPath, params.ScanOverrides)
	r.emitStage(req.TaskID, "agent_profile", "start", progress.ProfileStart, "Agent is profiling the dataset", map[string]any{"csv_path": params.CSVPath})
	r.emitStage(req.TaskID, "agent_scan", "start", progress.ScanStart, "Agent is calling the scan tool", nil)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentProfile,
		TraceType: TraceToolCall,
		Summary:   "Calling engine.scan_table",
		Payload:   map[string]any{"tool_id": "engine.scan_table", "payload": cloneMap(scanPayload)},
	})
	scanResp, err := r.callTool(ctx, req.TaskID, "engine.scan_table", scanPayload)
	if err != nil {
		r.failSession(session, "Agent planning failed while calling the scan tool")
		return nil, nil, err
	}
	if strings.ToLower(strings.TrimSpace(scanResp.Status)) != "ok" {
		r.failSession(session, "Agent planning failed because the scan tool returned an error")
		safety := buildSafetyResult("rollback_failed", []string{"baseline_scan_failed"}, map[string]any{}, map[string]any{}, buildRollbackRecommendation(false, "Scan failed before planning"), map[string]any{"status": "not_run"}, "")
		resp := r.toolFailureResponseWithSafety(req.TaskID, started, session.SessionID, "", mode, params.Goal, AgentPlan{}, defaultValidationResult(), defaultExecutionResult(mode == "auto"), safety, scanResp, "engine.scan_table")
		return nil, responsePtr(resp), nil
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentProfile,
		TraceType: TraceToolResult,
		Summary:   "Scan tool completed",
		Payload:   map[string]any{"tool_id": "engine.scan_table", "result": cloneMap(scanResp.Result)},
	})
	scanResult := cloneMap(scanResp.Result)
	baseline := scanSummaryFromResult(params.CSVPath, scanResult)
	session.Context["scan_summary"] = cloneMap(mapFromAny(scanResult["scan_summary"]))
	session.Context["data_profile"] = cloneMap(mapFromAny(scanResult["data_profile"]))
	session.Context["baseline_scan"] = cloneMap(baseline)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(session); err != nil {
		return nil, nil, err
	}
	r.emitStage(req.TaskID, "agent_scan", "complete", progress.ScanComplete, "Agent scan completed", map[string]any{"issue_count": intFromAny(scanResult["issue_count"])})
	r.emitStage(req.TaskID, "agent_profile", "complete", progress.ProfileComplete, "Dataset profiling completed", nil)

	selectedIssueIDs, skippedIssues := selectRepairableIssues(scanResult)
	selectedIssueCatalog := buildSelectedIssueCatalog(scanResult, selectedIssueIDs)
	candidateColumns := buildCandidateColumns(selectedIssueCatalog)
	timeLikeColumns := detectTimeLikeColumns(scanResult, candidateColumns)
	approvalContext := buildPlanningApprovalContext(baseline, candidateColumns, timeLikeColumns, preferenceSnapshot)
	session.Context["workspace_id"] = params.WorkspaceID
	session.Context["preference_snapshot"] = preferenceProfileToMap(preferenceSnapshot)
	session.Context["selected_issue_catalog"] = cloneValue(selectedIssueCatalog)
	session.Context["candidate_columns"] = append([]string{}, candidateColumns...)
	session.Context["time_like_columns"] = append([]string{}, timeLikeColumns...)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(session); err != nil {
		return nil, nil, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentProfile,
		TraceType: TraceMemoryUpdated,
		Summary:   "Session memory snapshot updated with workspace preferences and issue catalog",
		Payload: map[string]any{
			"workspace_id":         params.WorkspaceID,
			"preference_snapshot":  preferenceProfileToMap(preferenceSnapshot),
			"candidate_columns":    append([]string{}, candidateColumns...),
			"time_like_columns":    append([]string{}, timeLikeColumns...),
			"selected_issue_count": len(selectedIssueCatalog),
			"preferences_saved_at": preferenceRecord.UpdatedAt.UTC().Format(time.RFC3339Nano),
		},
	})
	r.emitStage(req.TaskID, "agent_strategy", "start", progress.StrategyStart, "Agent is selecting repair tools", nil)
	r.emitStage(req.TaskID, "agent_strategy", "complete", progress.StrategyComplete, "Repairable issues selected", map[string]any{"selected_issue_count": len(selectedIssueIDs)})

	rulePreviewPayload := buildRulePreviewPayload(params.CSVPath, selectedIssueIDs, params.ScanOverrides, params.RepairOverrides, params.ColumnDependencies, params.OutputDir)
	gowerPreviewPayload := buildGowerPreviewPayload(params.CSVPath, selectedIssueIDs, params.ScanOverrides, params.ColumnDependencies, params.GowerOverrides, params.OutputDir, params.ModelDir)
	r.emitStage(req.TaskID, "agent_retrieve", "start", progress.RetrieveStart, "Agent is previewing rule and Gower candidates", map[string]any{"retrieve_mode": retrieveMode})

	previewSpecs := []previewToolSpec{
		{
			ToolID:      "engine.repair_batch",
			CallSummary: "Calling engine.repair_batch for rule preview",
			DoneSummary: "Rule preview completed",
			Payload:     rulePreviewPayload,
		},
		{
			ToolID:      "engine.repair_with_gower",
			CallSummary: "Calling engine.repair_with_gower for Gower preview",
			DoneSummary: "Gower preview completed",
			Payload:     gowerPreviewPayload,
		},
	}
	for _, spec := range previewSpecs {
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentRepairPlanner,
			TraceType: TraceToolCall,
			Summary:   spec.CallSummary,
			Payload:   map[string]any{"tool_id": spec.ToolID, "payload": cloneMap(spec.Payload), "retrieve_mode": retrieveMode},
		})
	}

	previewOutcomes := r.runPreviewTools(ctx, req.TaskID, previewSpecs, retrieveMode)
	rulePreviewResp := engine.Response{}
	gowerPreviewResp := engine.Response{}
	for _, outcome := range previewOutcomes {
		if outcome.Err != nil {
			if outcome.Spec.ToolID == "engine.repair_batch" {
				r.failSession(session, "Agent planning failed while calling rule preview")
			} else {
				r.failSession(session, "Agent planning failed while calling Gower preview")
			}
			return nil, nil, outcome.Err
		}
		_ = r.saveTrace(AgentTraceEvent{
			SessionID: session.SessionID,
			TaskID:    req.TaskID,
			AgentName: AgentRepairPlanner,
			TraceType: TraceToolResult,
			Summary:   outcome.Spec.DoneSummary,
			Payload:   map[string]any{"tool_id": outcome.Spec.ToolID, "result": cloneMap(outcome.Resp.Result), "retrieve_mode": retrieveMode},
		})
		switch outcome.Spec.ToolID {
		case "engine.repair_batch":
			rulePreviewResp = outcome.Resp
		case "engine.repair_with_gower":
			gowerPreviewResp = outcome.Resp
		}
	}
	if strings.ToLower(strings.TrimSpace(rulePreviewResp.Status)) != "ok" {
		r.failSession(session, "Rule preview returned an error")
		safety := buildSafetyResult("rollback_failed", []string{"rule_preview_failed"}, cloneMap(baseline), map[string]any{}, buildRollbackRecommendation(false, "Planning preview failed"), map[string]any{"status": "not_run"}, "")
		resp := r.toolFailureResponseWithSafety(req.TaskID, started, session.SessionID, "", mode, params.Goal, AgentPlan{}, defaultValidationResult(), defaultExecutionResult(mode == "auto"), safety, rulePreviewResp, "engine.repair_batch")
		return nil, responsePtr(resp), nil
	}
	if strings.ToLower(strings.TrimSpace(gowerPreviewResp.Status)) != "ok" {
		r.failSession(session, "Gower preview returned an error")
		safety := buildSafetyResult("rollback_failed", []string{"gower_preview_failed"}, cloneMap(baseline), map[string]any{}, buildRollbackRecommendation(false, "Planning preview failed"), map[string]any{"status": "not_run"}, "")
		resp := r.toolFailureResponseWithSafety(req.TaskID, started, session.SessionID, "", mode, params.Goal, AgentPlan{}, defaultValidationResult(), defaultExecutionResult(mode == "auto"), safety, gowerPreviewResp, "engine.repair_with_gower")
		return nil, responsePtr(resp), nil
	}
	r.emitStage(req.TaskID, "agent_retrieve", "complete", progress.RetrieveComplete, "Rule and Gower previews completed", nil)

	r.emitStage(req.TaskID, "agent_compare", "start", progress.CompareStart, "Agent is comparing rule, Gower, and hybrid candidates", nil)
	planInput := buildPlanningInput(
		session.SessionID,
		params.Goal,
		params,
		scanResult,
		selectedIssueIDs,
		skippedIssues,
		rulePreviewResp.Result,
		gowerPreviewResp.Result,
		preferenceProfileToMap(preferenceSnapshot),
		approvalContext,
	)
	plan, err := r.planner.BuildPlan(ctx, planInput)
	if err != nil {
		r.failSession(session, "Agent planning failed while building the repair plan")
		safety := buildSafetyResult("rollback_failed", []string{"planner_failed"}, cloneMap(baseline), map[string]any{}, buildRollbackRecommendation(false, "Planning failed"), map[string]any{"status": "not_run"}, "")
		resp := r.errorResponseWithSafety(req.TaskID, started, ErrorPlannerFailed, "Agent planner failed to build a repair plan", map[string]any{"reason": err.Error()}, session.SessionID, "", mode, params.Goal, AgentPlan{}, defaultValidationResult(), defaultExecutionResult(mode == "auto"), safety, "")
		return nil, responsePtr(resp), nil
	}
	plan = ensurePlanCognition(plan)
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentRepairPlanner,
		TraceType: TraceAgentDecision,
		Summary:   plan.ReasoningSummary,
		Payload: map[string]any{
			"plan_id":               plan.PlanID,
			"selected_candidate_id": plan.SelectedCandidateID,
			"selected_source":       plan.SelectedSource,
			"issue_source_map":      cloneMap(plan.IssueSourceMap),
		},
	})
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentRepairPlanner,
		TraceType: TraceCognitionTrace,
		Summary:   cognitionTraceSummary(plan.Cognition),
		Payload:   cognitionTracePayload(plan.Cognition, "plan_complete"),
	})
	r.emitStage(req.TaskID, "agent_compare", "complete", progress.CompareComplete, "Candidate comparison completed", map[string]any{"selected_source": plan.SelectedSource})

	skippedTypes := make([]string, 0, len(plan.SkippedIssues))
	skippedTypeSet := map[string]struct{}{}
	for _, skipped := range plan.SkippedIssues {
		if skipped.IssueType == "" {
			continue
		}
		if _, exists := skippedTypeSet[skipped.IssueType]; exists {
			continue
		}
		skippedTypeSet[skipped.IssueType] = struct{}{}
		skippedTypes = append(skippedTypes, skipped.IssueType)
	}

	session.LatestPlan = clonePlan(plan)
	session.Status = SessionStatusPlanned
	session.Context["selected_issue_ids"] = append([]string{}, plan.SelectedIssueIDs...)
	session.Context["skipped_issue_types"] = skippedTypes
	session.Context["latest_plan_id"] = plan.PlanID
	session.Context["issue_source_map"] = cloneMap(plan.IssueSourceMap)
	session.Context["cognition_state"] = cognitionStateToMap(plan.Cognition)
	riskAssessment := buildRiskAssessment(plan, baseline, selectedIssueCatalog, candidateColumns, timeLikeColumns, preferenceSnapshot)
	approvalState := buildApprovalState(func() string {
		if riskAssessment["required"] == true {
			return approvalStatusRequired
		}
		return approvalStatusNotRequired
	}(), riskAssessment["required"] == true, riskAssessment)
	updateSessionApprovalContext(&session, riskAssessment, approvalState)
	session.UpdatedAt = time.Now().UTC()
	if err := r.saveSession(session); err != nil {
		return nil, nil, err
	}
	_ = r.saveTrace(AgentTraceEvent{
		SessionID: session.SessionID,
		TaskID:    req.TaskID,
		AgentName: AgentSupervisor,
		TraceType: TraceMemoryUpdated,
		Summary:   "Risk assessment and approval state saved into session memory",
		Payload: map[string]any{
			"risk_assessment": cloneMap(riskAssessment),
			"approval_state":  cloneMap(approvalState),
			"cognition_state": cognitionStateToMap(plan.Cognition),
		},
	})
	r.emitStage(req.TaskID, "agent_plan", "complete", progress.PlanComplete, "Repair plan is ready", map[string]any{"plan_id": plan.PlanID})

	return &planningResult{
		Session:   session,
		Goal:      params.Goal,
		Plan:      plan,
		Baseline:  baseline,
		ScanInput: params,
	}, nil, nil
}
