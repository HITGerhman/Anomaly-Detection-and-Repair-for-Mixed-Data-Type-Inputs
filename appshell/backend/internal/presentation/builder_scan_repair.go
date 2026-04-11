package presentation

import (
	"fmt"
	"strings"

	"appshell/backend/internal/engine"
)

func buildScanBundle(result map[string]any) *Bundle {
	issues := sliceOfMaps(result["issues"])
	thumbnails := sliceOfMaps(result["column_thumbnails"])
	summary := objectMap(result["scan_summary"])
	profile := objectMap(result["data_profile"])
	issueCount := asInt(result["issue_count"])
	highRiskColumns := sliceOfStrings(summary["high_risk_columns"])
	repairable := countRepairableIssues(issues)
	verdict := "clean"
	if issueCount > 0 {
		verdict = "attention"
	}
	headline := "未检测到异常。"
	summaryText := "当前扫描结果较为干净，可以直接进入后续验证或导出。"
	if issueCount > 0 {
		headline = fmt.Sprintf("检测到 %d 个异常问题，建议优先查看高风险列。", issueCount)
		summaryText = fmt.Sprintf("本次扫描识别到 %d 个问题，其中 %d 个问题属于当前可修复范围。", issueCount, repairable)
	}

	return &Bundle{
		Version:  bundleVersion,
		Kind:     "scan",
		Headline: headline,
		Summary:  summaryText,
		Verdict:  verdict,
		Highlights: []Highlight{
			{ID: "issue_count", Label: "异常问题", Value: fmt.Sprintf("%d", issueCount), Tone: verdict, Hint: "扫描发现的问题总数"},
			{ID: "high_risk_columns", Label: "高风险列", Value: fmt.Sprintf("%d", len(highRiskColumns)), Tone: "attention", Hint: "优先排查的列"},
			{ID: "repairable_issue_count", Label: "可修复问题", Value: fmt.Sprintf("%d", repairable), Tone: "positive", Hint: "可直接进入修复流程的问题数"},
		},
		Sections: nonEmptySections([]Section{
			{
				ID:    "overview",
				Title: "总体判断",
				Body:  summaryText,
				Bullets: []string{
					fmt.Sprintf("扫描行数：%d", asInt(profile["rows"])),
					fmt.Sprintf("扫描列数：%d", asInt(profile["columns"])),
					fmt.Sprintf("高风险列数量：%d", len(highRiskColumns)),
				},
				EvidenceRefs: []string{"scan_summary", "data_profile"},
			},
			{
				ID:    "anomaly_findings",
				Title: "异常分布",
				Body:  buildScanFindingsText(issueCount, highRiskColumns, issues),
				Bullets: []string{
					buildIssueTypeBullet(issues),
					buildRiskColumnBullet(highRiskColumns),
					fmt.Sprintf("可修复问题数：%d", repairable),
				},
				EvidenceRefs: []string{"issues", "column_thumbnails"},
			},
			{
				ID:    "next_steps",
				Title: "下一步建议",
				Body:  buildScanNextStepText(issueCount, repairable),
				Bullets: []string{
					"先查看风险排序和异常缩略图，再决定是否批量修复。",
					"若需要快速闭环，可直接进入自动修复或 agent 自动闭环流程。",
				},
			},
		}),
		Charts: nonEmptyCharts([]ChartSpec{
			{
				ID:         "column_risk_ranking",
				Kind:       "ranked_bar",
				Title:      "列风险排序",
				Subtitle:   "按风险分数查看最值得优先处理的列",
				EmptyState: "暂无列风险数据",
				Data:       map[string]any{"series": topColumnRiskSeries(thumbnails, 8)},
			},
			{
				ID:         "issue_type_distribution",
				Kind:       "stacked_bar",
				Title:      "问题类型分布",
				Subtitle:   "查看缺失值、离群值和类别问题的占比",
				EmptyState: "暂无问题类型分布",
				Data:       map[string]any{"series": issueTypeSeries(issues)},
			},
			{
				ID:         "anomaly_density_heatmap",
				Kind:       "heatmap_grid",
				Title:      "异常密度热图",
				Subtitle:   "沿用当前列缩略图，快速定位异常热点",
				EmptyState: "暂无异常缩略图",
				Data: map[string]any{
					"column_thumbnails": thumbnails,
					"row_count":         asInt(profile["rows"]),
				},
			},
			{
				ID:         "high_risk_column_spotlight",
				Kind:       "spotlight_card",
				Title:      "高风险列聚焦",
				Subtitle:   "给出当前最值得优先排查的一列",
				EmptyState: "暂无高风险列",
				Data:       topRiskColumn(thumbnails),
			},
		}),
		Artifacts: map[string]any{
			"csv_path": asString(result["csv_path"]),
		},
	}
}

func buildRepairBundle(action string, result map[string]any) *Bundle {
	comparison := objectMap(result["comparison"])
	before := asInt(comparison["before_issue_count"])
	after := asInt(comparison["after_issue_count"])
	resolved := asInt(comparison["resolved_issue_count"])
	changed := asInt(comparison["changed_cell_count"])
	if before == 0 && after == 0 && resolved == 0 && changed == 0 && len(comparison) == 0 {
		before = asInt(result["scan_issue_count"])
		after = maxInt(0, before-asInt(result["applied_issue_count"]))
		resolved = maxInt(0, before-after)
	}
	applied := asInt(result["applied_issue_count"])
	skipped := len(sliceOfMaps(result["skipped_issues"]))
	source := "rule"
	if strings.TrimSpace(action) == string(engine.ActionRepairWithGower) {
		source = "gower"
	}
	if selected := asString(result["selected_source"]); selected != "" {
		source = selected
	}
	headline := fmt.Sprintf("本次修复共解决 %d 个问题，修复后剩余 %d 个问题。", resolved, after)
	if before == 0 && applied == 0 {
		headline = "本次修复未执行任何变更。"
	}
	summaryText := fmt.Sprintf("修复来源为 %s，本次共修改 %d 个单元格，跳过 %d 个问题。", strings.ToUpper(source), changed, skipped)

	return &Bundle{
		Version:  bundleVersion,
		Kind:     "repair",
		Headline: headline,
		Summary:  summaryText,
		Verdict:  repairVerdict(before, after, applied),
		Highlights: []Highlight{
			{ID: "resolved_issue_count", Label: "已解决问题", Value: fmt.Sprintf("%d", resolved), Tone: "positive", Hint: "修复后减少的问题数"},
			{ID: "changed_cell_count", Label: "修改单元格", Value: fmt.Sprintf("%d", changed), Tone: "attention", Hint: "本次被修改的数据单元格数"},
			{ID: "skipped_issue_count", Label: "跳过问题", Value: fmt.Sprintf("%d", skipped), Tone: "muted", Hint: "未执行修复的问题数"},
		},
		Sections: nonEmptySections([]Section{
			{
				ID:    "overview",
				Title: "修复概览",
				Body:  headline,
				Bullets: []string{
					fmt.Sprintf("修复来源：%s", strings.ToUpper(source)),
					fmt.Sprintf("输出文件：%s", stringOrDash(asString(result["output_csv"]))),
					fmt.Sprintf("已写出结果：%s", boolLabel(result["write_output"])),
				},
				EvidenceRefs: []string{"comparison", "output_csv"},
			},
			{
				ID:    "repair_strategy",
				Title: "修复策略",
				Body:  buildRepairStrategyText(source, result),
				Bullets: []string{
					fmt.Sprintf("已选问题：%d", asInt(result["selected_issue_count"])),
					fmt.Sprintf("已修复问题：%d", applied),
					fmt.Sprintf("跳过问题：%d", skipped),
				},
				EvidenceRefs: []string{"applied_repairs", "skipped_issues", "neighbor_evidence"},
			},
			{
				ID:    "repair_impact",
				Title: "修复收益",
				Body:  summaryText,
				Bullets: []string{
					fmt.Sprintf("问题数变化：%d -> %d", before, after),
					fmt.Sprintf("已减少问题：%d", resolved),
					fmt.Sprintf("修改单元格：%d", changed),
				},
				EvidenceRefs: []string{"comparison"},
			},
			{
				ID:    "risk_and_safety",
				Title: "风险与安全",
				Body:  buildRepairSafetyText(result),
				Bullets: []string{
					fmt.Sprintf("回滚清单：%s", stringOrDash(asString(objectMap(result["rollback"])["manifest_path"]))),
					"旧修复路径默认不额外复扫；若需要风险复核，可使用 agent 自动闭环模式。",
				},
				EvidenceRefs: []string{"rollback"},
			},
			{
				ID:    "next_steps",
				Title: "下一步建议",
				Body:  "可以继续复扫结果文件，或切换到 agent 自动闭环模式进行验证优先的安全执行。",
				Bullets: []string{
					"若结果符合预期，可导出 CSV 和 presentation.json 作为交付物。",
					"若对局部修复不满意，可依据回滚清单恢复输出文件。",
				},
			},
		}),
		Charts: nonEmptyCharts([]ChartSpec{
			{
				ID:         "before_after_issue_comparison",
				Kind:       "comparison_bar",
				Title:      "修复前后问题数对比",
				Subtitle:   "直接查看修复是否有效减少问题",
				EmptyState: "暂无修复对比数据",
				Data: map[string]any{
					"series": []map[string]any{
						{"label": "修复前", "value": before, "tone": "before"},
						{"label": "修复后", "value": after, "tone": "after"},
					},
					"delta": resolved,
				},
			},
			{
				ID:         "repaired_vs_skipped_breakdown",
				Kind:       "stacked_bar",
				Title:      "修复与跳过分布",
				Subtitle:   "区分已应用修复和保留问题",
				EmptyState: "暂无修复明细",
				Data: map[string]any{
					"series": []map[string]any{
						{"label": "已修复", "value": applied, "tone": "positive"},
						{"label": "已跳过", "value": skipped, "tone": "warning"},
					},
				},
			},
			{
				ID:         "column_issue_delta",
				Kind:       "comparison_bar",
				Title:      "列级问题变化",
				Subtitle:   "按列查看修复前后的问题变化",
				EmptyState: "暂无列级对比数据",
				Data:       map[string]any{"series": buildColumnDeltaSeries(comparison)},
			},
			{
				ID:         "repair_source_breakdown",
				Kind:       "stacked_bar",
				Title:      "修复来源分布",
				Subtitle:   "区分规则、Gower 或混合路径的贡献",
				EmptyState: "当前结果不包含多来源修复信息",
				Data:       map[string]any{"series": buildRepairSourceSeries(source, result)},
			},
		}),
		Artifacts: map[string]any{
			"output_csv":        asString(result["output_csv"]),
			"rollback_manifest": asString(objectMap(result["rollback"])["manifest_path"]),
		},
	}
}

func buildScanFindingsText(issueCount int, highRiskColumns []string, issues []map[string]any) string {
	if issueCount == 0 {
		return "本次扫描没有发现需要处理的异常列。"
	}
	parts := []string{fmt.Sprintf("本次共识别 %d 个问题。", issueCount)}
	if len(highRiskColumns) > 0 {
		parts = append(parts, fmt.Sprintf("高风险列集中在 %s。", strings.Join(highRiskColumns, "、")))
	}
	if bullet := buildIssueTypeBullet(issues); bullet != "" {
		parts = append(parts, bullet)
	}
	return strings.Join(parts, "")
}

func buildScanNextStepText(issueCount int, repairable int) string {
	if issueCount == 0 {
		return "可以直接导出本次扫描结果，或切换到其他数据集继续验证。"
	}
	if repairable > 0 {
		return "当前已有可直接修复的问题，建议先查看图表，再进入批量修复或 agent 自动闭环。"
	}
	return "当前问题更多偏向诊断型异常，建议先结合高风险列和问题详情做人工判断。"
}

func buildRepairStrategyText(source string, result map[string]any) string {
	if source == "gower" {
		return "本次修复使用 Gower 邻居检索生成候选值，适合混合类型近邻替代场景。"
	}
	if source == "hybrid" {
		return "本次修复采用规则与 Gower 双路混合执行，按问题级来源协同落地。"
	}
	return "本次修复以规则路径为主，优先使用稳定的确定性修复策略。"
}

func buildRepairSafetyText(result map[string]any) string {
	rollback := objectMap(result["rollback"])
	if asString(rollback["manifest_path"]) != "" {
		return "本次结果已生成回滚清单，必要时可以恢复输出文件。"
	}
	return "本次结果未附带完整回滚清单，建议在高风险数据集上切换到 agent 自动闭环模式。"
}
