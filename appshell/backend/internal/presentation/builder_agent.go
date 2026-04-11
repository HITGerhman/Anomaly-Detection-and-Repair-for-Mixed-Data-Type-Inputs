package presentation

import (
	"fmt"
	"strings"

	"appshell/backend/internal/agent"
)

func buildAgentBundle(agentBlock map[string]any, safety map[string]any, observability map[string]any) *Bundle {
	runMode := asString(agentBlock["run_mode"])
	plan := objectMap(agentBlock["plan"])
	explanation := objectMap(agentBlock["explanation"])
	validation := objectMap(agentBlock["validation"])
	execution := objectMap(agentBlock["execution"])
	traceSummary := objectMap(agentBlock["trace_summary"])
	cognition := agentCognition(plan, explanation, traceSummary)
	selectedSource := asString(plan["selected_source"])
	if selectedSource == "" {
		selectedSource = asString(execution["selected_source"])
	}

	return &Bundle{
		Version:  bundleVersion,
		Kind:     "agent",
		Headline: buildAgentHeadline(runMode, plan, validation, safety),
		Summary:  buildAgentSummary(runMode, plan, validation, execution, safety),
		Verdict:  agentVerdict(runMode, safety, validation),
		Highlights: []Highlight{
			{ID: "cognition_status", Label: "Cognition", Value: stringOrDash(cognitionStatusText(cognition)), Tone: cognitionTone(cognition)},
			{ID: "selected_issue_count", Label: "选中问题", Value: fmt.Sprintf("%d", len(sliceOfStrings(plan["selected_issue_ids"]))), Tone: "attention"},
			{ID: "candidate_count", Label: "候选方案", Value: fmt.Sprintf("%d", len(sliceOfMaps(plan["candidates"]))), Tone: "muted"},
			{ID: "final_verdict", Label: "最终裁决", Value: stringOrDash(asString(safety["final_verdict"])), Tone: agentVerdict(runMode, safety, validation)},
		},
		Sections: nonEmptySections([]Section{
			{
				ID:    "overview",
				Title: "会话概览",
				Body:  buildAgentSummary(runMode, plan, validation, execution, safety),
				Bullets: []string{
					fmt.Sprintf("运行模式：%s", stringOrDash(runMode)),
					fmt.Sprintf("选中来源：%s", stringOrDash(selectedSource)),
					fmt.Sprintf("计划编号：%s", stringOrDash(asString(agentBlock["plan_id"]))),
				},
				EvidenceRefs: []string{"plan", "validation", "execution"},
			},
			{
				ID:    "repair_strategy",
				Title: "策略说明",
				Body:  firstNonEmpty(asString(explanation["final_message"]), asString(plan["user_explanation"])),
				Bullets: compactStrings(
					firstNonEmpty(asString(explanation["summary"]), asString(plan["reasoning_summary"])),
					firstNonEmpty(asString(explanation["risk_note"]), asString(plan["risk_note"])),
					cognitionSummaryText(cognition),
					cognitionFallbackText(cognition),
					fmt.Sprintf("候选方案数：%d", len(sliceOfMaps(plan["candidates"]))),
					fmt.Sprintf("选中问题数：%d", len(sliceOfStrings(plan["selected_issue_ids"]))),
					fmt.Sprintf("跳过问题数：%d", len(sliceOfMaps(plan["skipped_issues"]))),
				),
				EvidenceRefs: []string{"plan"},
			},
			{
				ID:    "repair_impact",
				Title: "执行收益",
				Body:  buildAgentImpactText(validation, execution),
				Bullets: []string{
					fmt.Sprintf("预校验结果：%s", stringOrDash(asString(validation["message"]))),
					fmt.Sprintf("输出文件：%s", stringOrDash(asString(execution["output_csv"]))),
					fmt.Sprintf("回滚已应用：%s", boolLabel(execution["rollback_applied"])),
				},
				EvidenceRefs: []string{"validation", "execution"},
			},
			{
				ID:    "risk_and_safety",
				Title: "风险与安全",
				Body:  buildAgentSafetyText(safety),
				Bullets: []string{
					fmt.Sprintf("最终裁决：%s", stringOrDash(asString(safety["final_verdict"]))),
					fmt.Sprintf("风险标记数：%d", len(sliceOfStrings(safety["risk_flags"]))),
					fmt.Sprintf("失败产物快照：%s", stringOrDash(asString(safety["rejected_output_snapshot"]))),
				},
				EvidenceRefs: []string{"safety"},
			},
			{
				ID:    "next_steps",
				Title: "下一步建议",
				Body:  buildAgentNextStepsText(runMode, safety),
				Bullets: []string{
					"保留原始 CSV 不变，优先检查输出文件和 presentation.json。",
					"需要进一步复盘时，可结合 agent trace 与回滚信息定位问题。",
				},
			},
		}),
		Charts: nonEmptyCharts([]ChartSpec{
			{
				ID:         "repair_source_breakdown",
				Kind:       "stacked_bar",
				Title:      "候选来源分布",
				Subtitle:   "对比 rule / gower / hybrid 方案构成",
				EmptyState: "暂无候选来源数据",
				Data:       map[string]any{"series": buildCandidateSourceSeries(plan)},
			},
			{
				ID:         "validation_verdict_timeline",
				Kind:       "timeline",
				Title:      "验证裁决时间线",
				Subtitle:   "查看 preview 与 post-execute 两段验证结果",
				EmptyState: "暂无验证事件",
				Data:       map[string]any{"events": buildValidationTimeline(validation, safety)},
			},
			{
				ID:         "trace_stage_timeline",
				Kind:       "timeline",
				Title:      "执行轨迹概览",
				Subtitle:   "结合观测信息和 trace 摘要查看执行轨迹",
				EmptyState: "暂无轨迹信息",
				Data:       map[string]any{"events": buildTraceTimeline(observability, traceSummary)},
			},
			{
				ID:         "rollback_summary",
				Kind:       "spotlight_card",
				Title:      "回滚摘要",
				Subtitle:   "仅在回滚或回滚失败路径显示",
				EmptyState: "本次执行未触发回滚",
				Data:       buildRollbackSpotlight(safety),
			},
			{
				ID:         "safety_risk_delta",
				Kind:       "comparison_bar",
				Title:      "安全风险变化",
				Subtitle:   "对比基线与后验扫描的风险变化",
				EmptyState: "暂无可对比的安全数据",
				Data:       buildSafetyDeltaSeries(safety),
			},
		}),
		Artifacts: map[string]any{
			"output_csv":               asString(execution["output_csv"]),
			"rollback_manifest":        asString(objectMap(execution["rollback"])["manifest_path"]),
			"rejected_output_snapshot": asString(safety["rejected_output_snapshot"]),
		},
	}
}

func buildAgentSessionBundle(snapshot agent.AgentSessionSnapshot) *Bundle {
	agentBlock := map[string]any{
		"session_id":    snapshot.SessionID,
		"plan_id":       snapshot.LatestPlan.PlanID,
		"run_mode":      snapshot.Mode,
		"goal":          snapshot.UserGoal,
		"plan":          snapshot.LatestPlan,
		"explanation":   buildAgentExplanationBlock(snapshot.LatestPlan),
		"trace_summary": snapshot.TraceSummary,
		"validation":    objectMap(snapshot.Context["preview_validation"]),
		"execution":     objectMap(snapshot.Context["execution_artifacts"]),
	}
	safety := map[string]any{
		"final_verdict":            asString(snapshot.Context["final_verdict"]),
		"baseline_scan_summary":    objectMap(snapshot.Context["baseline_scan"]),
		"post_scan_summary":        objectMap(snapshot.Context["post_scan"]),
		"rollback_execution":       objectMap(snapshot.Context["rollback_summary"]),
		"rejected_output_snapshot": asString(snapshot.Context["rejected_output_snapshot"]),
	}
	if postValidation := objectMap(snapshot.Context["post_validation"]); len(postValidation) > 0 {
		agentBlock["validation"] = map[string]any{
			"status":       asString(postValidation["status"]),
			"message":      asString(postValidation["message"]),
			"can_execute":  true,
			"preview":      objectMap(snapshot.Context["preview_validation"]),
			"post_execute": postValidation,
		}
	}
	return buildAgentBundle(agentBlock, safety, nil)
}

func buildAgentHeadline(runMode string, plan map[string]any, validation map[string]any, safety map[string]any) string {
	if runMode == "plan" {
		return fmt.Sprintf("已生成修复计划，共准备 %d 个候选方案。", len(sliceOfMaps(plan["candidates"])))
	}
	verdict := asString(safety["final_verdict"])
	switch verdict {
	case "accepted":
		return "自动闭环已完成，结果通过验证并保留输出。"
	case "rolled_back":
		return "自动闭环已触发回滚，系统已恢复到安全状态。"
	case "rollback_failed":
		return "自动闭环检测到风险，但自动回滚失败，需要人工复核。"
	}
	if asString(validation["status"]) == "rejected" {
		return "计划已生成，但验证拒绝执行当前方案。"
	}
	return "Agent 会话已完成。"
}

func buildAgentSummary(runMode string, plan map[string]any, validation map[string]any, execution map[string]any, safety map[string]any) string {
	if runMode == "plan" {
		return fmt.Sprintf("当前选择的修复来源为 %s，后续可直接执行或切换候选方案。", stringOrDash(asString(plan["selected_source"])))
	}
	if verdict := asString(safety["final_verdict"]); verdict != "" {
		return fmt.Sprintf("本次会话最终裁决为 %s，输出文件为 %s。", verdict, stringOrDash(asString(execution["output_csv"])))
	}
	return fmt.Sprintf("当前预校验结论：%s。", stringOrDash(asString(validation["message"])))
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func compactStrings(values ...string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func buildAgentImpactText(validation map[string]any, execution map[string]any) string {
	post := objectMap(validation["post_execute"])
	if len(post) > 0 {
		return stringOrDash(asString(post["message"]))
	}
	if asString(execution["status"]) == "executed" {
		return fmt.Sprintf("本次执行已输出修复文件 %s。", stringOrDash(asString(execution["output_csv"])))
	}
	return stringOrDash(asString(validation["message"]))
}

func buildAgentSafetyText(safety map[string]any) string {
	verdict := asString(safety["final_verdict"])
	switch verdict {
	case "accepted":
		return "后验复扫通过，系统确认输出结果相对基线更安全。"
	case "rolled_back":
		return "后验验证未通过，系统已自动回滚输出文件并保留失败快照。"
	case "rollback_failed":
		return "系统识别到风险，但未能完成自动回滚，需要立即人工复核。"
	case "validation_rejected":
		return "系统在 preview 阶段即拒绝执行，避免了高风险写入。"
	default:
		return "当前会话未进入完整的安全闭环。"
	}
}

func buildAgentNextStepsText(runMode string, safety map[string]any) string {
	if runMode == "plan" {
		return "可以直接执行当前计划，也可以结合候选来源分布选择更偏保守或更偏近邻的方案。"
	}
	if asString(safety["final_verdict"]) == "accepted" {
		return "建议直接查看修复收益图和风险变化图，再决定是否导出产物。"
	}
	return "建议优先查看回滚摘要、失败快照与轨迹时间线，确认问题来源后再重试。"
}

func buildCandidateSourceSeries(plan map[string]any) []map[string]any {
	candidates := sliceOfMaps(plan["candidates"])
	counts := map[string]int{}
	for _, item := range candidates {
		counts[asString(item["source"])]++
	}
	return []map[string]any{
		{"label": "rule", "value": counts["rule"], "tone": "neutral"},
		{"label": "gower", "value": counts["gower"], "tone": "attention"},
		{"label": "hybrid", "value": counts["hybrid"], "tone": "positive"},
	}
}

func buildValidationTimeline(validation map[string]any, safety map[string]any) []map[string]any {
	events := make([]map[string]any, 0, 3)
	preview := objectMap(validation["preview"])
	if len(preview) > 0 {
		events = append(events, map[string]any{
			"label": "preview",
			"value": stringOrDash(asString(preview["status"])),
			"tone":  validationTone(asString(preview["status"]), false),
			"hint":  stringOrDash(asString(preview["message"])),
		})
	} else if len(validation) > 0 {
		events = append(events, map[string]any{
			"label": "preview",
			"value": stringOrDash(asString(validation["status"])),
			"tone":  validationTone(asString(validation["status"]), false),
			"hint":  stringOrDash(asString(validation["message"])),
		})
	}
	post := objectMap(validation["post_execute"])
	if len(post) > 0 {
		events = append(events, map[string]any{
			"label": "post_execute",
			"value": stringOrDash(asString(post["status"])),
			"tone":  validationTone(asString(post["status"]), false),
			"hint":  stringOrDash(asString(post["message"])),
		})
	}
	if verdict := asString(safety["final_verdict"]); verdict != "" {
		events = append(events, map[string]any{
			"label": "final",
			"value": verdict,
			"tone":  validationTone(verdict, true),
			"hint":  "系统最终裁决",
		})
	}
	return events
}

func validationTone(status string, verdict bool) string {
	status = strings.ToLower(strings.TrimSpace(status))
	switch status {
	case "accepted", "checked":
		return "positive"
	case "rolled_back", "rollback_failed":
		return "warning"
	case "validation_rejected", "rejected":
		return "danger"
	}
	if verdict {
		return "attention"
	}
	return "muted"
}

func buildTraceTimeline(observability map[string]any, traceSummary map[string]any) []map[string]any {
	events := make([]map[string]any, 0, 8)
	stageDurations := objectMap(observability["stage_durations_ms"])
	for _, key := range mapKeysSorted(stageDurations) {
		events = append(events, map[string]any{
			"label": key,
			"value": fmt.Sprintf("%d ms", asInt(stageDurations[key])),
			"tone":  "muted",
			"hint":  "阶段耗时",
		})
	}
	if cognition := objectMap(traceSummary["cognition"]); len(cognition) > 0 {
		events = append(events, map[string]any{
			"label": firstNonEmpty(asString(cognition["provider"]), "cognition"),
			"value": stringOrDash(asString(cognition["status"])),
			"tone":  cognitionTone(cognition),
			"hint":  stringOrDash(firstNonEmpty(asString(cognition["last_summary"]), cognitionFallbackText(cognition))),
		})
	}
	if len(events) == 0 {
		for _, name := range sliceOfStrings(traceSummary["agent_names"]) {
			events = append(events, map[string]any{
				"label": name,
				"value": "seen",
				"tone":  "muted",
				"hint":  "出现在本次 trace 中",
			})
		}
	}
	return events
}

func buildRollbackSpotlight(safety map[string]any) map[string]any {
	verdict := asString(safety["final_verdict"])
	if verdict != "rolled_back" && verdict != "rollback_failed" {
		return nil
	}
	rollback := objectMap(safety["rollback_execution"])
	return map[string]any{
		"final_verdict": verdict,
		"status":        asString(rollback["status"]),
		"snapshot":      asString(safety["rejected_output_snapshot"]),
	}
}

func buildSafetyDeltaSeries(safety map[string]any) map[string]any {
	baseline := objectMap(safety["baseline_scan_summary"])
	post := objectMap(safety["post_scan_summary"])
	if len(baseline) == 0 || len(post) == 0 {
		return nil
	}
	return map[string]any{
		"series": []map[string]any{
			{"label": "问题总数", "before": asInt(baseline["issue_count"]), "after": asInt(post["issue_count"]), "delta": asInt(post["issue_count"]) - asInt(baseline["issue_count"])},
			{"label": "高风险问题", "before": asInt(baseline["high_risk_issue_count"]), "after": asInt(post["high_risk_issue_count"]), "delta": asInt(post["high_risk_issue_count"]) - asInt(baseline["high_risk_issue_count"])},
			{"label": "总风险分", "before": asFloat(baseline["total_issue_score"]), "after": asFloat(post["total_issue_score"]), "delta": asFloat(post["total_issue_score"]) - asFloat(baseline["total_issue_score"])},
		},
	}
}

func agentVerdict(runMode string, safety map[string]any, validation map[string]any) string {
	if runMode == "plan" {
		return "planned"
	}
	if verdict := asString(safety["final_verdict"]); verdict != "" {
		return verdict
	}
	return asString(validation["status"])
}

func buildAgentExplanationBlock(plan agent.AgentPlan) map[string]any {
	cognition := objectMap(plan.Cognition)
	mode := "deterministic"
	switch asString(cognition["status"]) {
	case agent.CognitionStatusEngaged:
		mode = "langgraph_llm"
	case agent.CognitionStatusDegraded:
		mode = "langgraph_degraded"
	case agent.CognitionStatusFallback, agent.CognitionStatusDisabled, agent.CognitionStatusUnavailable:
		if asString(cognition["provider"]) == agent.CognitionProviderLangGraph || asString(cognition["fallback_reason_code"]) != "" {
			mode = "langgraph_fallback"
		}
	}
	return map[string]any{
		"mode":          mode,
		"summary":       strings.TrimSpace(plan.ReasoningSummary),
		"final_message": strings.TrimSpace(plan.UserExplanation),
		"short_bullets": append([]string{}, plan.ExplanationBullets...),
		"reason_codes":  append([]string{}, plan.ReasonCodes...),
		"risk_note":     strings.TrimSpace(plan.RiskNote),
		"cognition":     cognition,
	}
}

func agentCognition(plan map[string]any, explanation map[string]any, traceSummary map[string]any) map[string]any {
	if cognition := objectMap(explanation["cognition"]); len(cognition) > 0 {
		return cognition
	}
	if cognition := objectMap(plan["cognition"]); len(cognition) > 0 {
		return cognition
	}
	return objectMap(traceSummary["cognition"])
}

func cognitionStatusText(cognition map[string]any) string {
	provider := asString(cognition["provider"])
	status := asString(cognition["status"])
	switch {
	case provider != "" && status != "":
		return fmt.Sprintf("%s/%s", provider, status)
	case status != "":
		return status
	case provider != "":
		return provider
	default:
		return ""
	}
}

func cognitionSummaryText(cognition map[string]any) string {
	return firstNonEmpty(asString(cognition["summary"]), asString(cognition["last_summary"]))
}

func cognitionFallbackText(cognition map[string]any) string {
	reason := asString(cognition["fallback_reason_code"])
	message := asString(cognition["fallback_message"])
	if reason == "" && message == "" {
		return ""
	}
	if reason == "" {
		return message
	}
	if message == "" {
		return fmt.Sprintf("Fallback: %s", reason)
	}
	return fmt.Sprintf("Fallback: %s (%s)", reason, message)
}

func cognitionTone(cognition map[string]any) string {
	switch asString(cognition["status"]) {
	case agent.CognitionStatusEngaged:
		return "positive"
	case agent.CognitionStatusDegraded:
		return "attention"
	case agent.CognitionStatusFallback, agent.CognitionStatusDisabled, agent.CognitionStatusUnavailable:
		return "warning"
	default:
		return "muted"
	}
}
