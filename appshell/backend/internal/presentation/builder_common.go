package presentation

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
)

const bundleVersion = "stage4.v1"

func EnrichTaskResult(req engine.Request, result map[string]any) (string, error) {
	if len(result) == 0 {
		return "", nil
	}

	switch strings.TrimSpace(req.Action) {
	case string(engine.ActionScanFile):
		bundle := buildScanBundle(result)
		if bundle == nil {
			return "", nil
		}
		artifactPath, err := maybeWriteArtifact(req, result, false, bundle)
		result["presentation"] = toMap(*bundle)
		if artifactPath != "" {
			result["presentation_artifact"] = artifactPath
		}
		return artifactPath, err
	case string(engine.ActionRepairBatch), string(engine.ActionRepairWithGower), string(engine.ActionRepairWithMissForest):
		bundle := buildRepairBundle(req.Action, result)
		if bundle == nil {
			return "", nil
		}
		artifactPath, err := maybeWriteArtifact(req, result, false, bundle)
		result["presentation"] = toMap(*bundle)
		if artifactPath != "" {
			result["presentation_artifact"] = artifactPath
		}
		return artifactPath, err
	case agent.ActionSessionPlan, agent.ActionSessionExecute, agent.ActionSessionAuto:
		agentBlock := objectMap(result["agent"])
		if len(agentBlock) == 0 {
			return "", nil
		}
		bundle := buildAgentBundle(agentBlock, objectMap(result["safety"]), objectMap(result["observability"]))
		if bundle == nil {
			return "", nil
		}
		artifactPath, err := maybeWriteArtifact(req, result, true, bundle)
		agentBlock["presentation"] = toMap(*bundle)
		if artifactPath != "" {
			agentBlock["presentation_artifact"] = artifactPath
		}
		result["agent"] = agentBlock
		return artifactPath, err
	default:
		return "", nil
	}
}

func EnrichAgentSessionSnapshot(snapshot *agent.AgentSessionSnapshot) error {
	if snapshot == nil {
		return nil
	}
	bundle := buildAgentSessionBundle(*snapshot)
	if bundle == nil {
		return nil
	}
	snapshot.Presentation = toMap(*bundle)
	artifactPath, err := maybeWriteSnapshotArtifact(snapshot, bundle)
	if artifactPath != "" {
		snapshot.PresentationArtifact = artifactPath
	}
	return err
}

func countRepairableIssues(issues []map[string]any) int {
	total := 0
	for _, item := range issues {
		switch asString(item["issue_type"]) {
		case "missing_values", "numeric_outlier", "rare_category":
			total++
		}
	}
	return total
}

func buildIssueTypeBullet(issues []map[string]any) string {
	series := issueTypeSeries(issues)
	if len(series) == 0 {
		return ""
	}
	parts := make([]string, 0, len(series))
	for _, item := range series {
		parts = append(parts, fmt.Sprintf("%s %d 个", asString(item["label"]), asInt(item["value"])))
	}
	return "问题类型分布：" + strings.Join(parts, "，")
}

func buildRiskColumnBullet(columns []string) string {
	if len(columns) == 0 {
		return "当前没有高风险列。"
	}
	return "高风险列：" + strings.Join(columns, "、")
}

func buildColumnDeltaSeries(comparison map[string]any) []map[string]any {
	before := objectMap(comparison["before_column_issue_counts"])
	after := objectMap(comparison["after_column_issue_counts"])
	keys := map[string]struct{}{}
	for _, bucket := range []map[string]any{before, after} {
		for key := range bucket {
			keys[key] = struct{}{}
		}
	}
	names := make([]string, 0, len(keys))
	for key := range keys {
		names = append(names, key)
	}
	if len(names) == 0 {
		return nil
	}
	sort.Strings(names)
	out := make([]map[string]any, 0, len(names))
	for _, key := range names {
		beforeValue := asInt(before[key])
		afterValue := asInt(after[key])
		out = append(out, map[string]any{
			"label":  key,
			"before": beforeValue,
			"after":  afterValue,
			"delta":  afterValue - beforeValue,
		})
	}
	return out
}

func buildRepairSourceSeries(source string, result map[string]any) []map[string]any {
	if source == "hybrid" {
		issueMap := objectMap(result["issue_source_map"])
		counts := map[string]int{}
		for _, value := range issueMap {
			counts[asString(value)]++
		}
		return []map[string]any{
			{"label": "rule", "value": counts["rule"], "tone": "neutral"},
			{"label": "gower", "value": counts["gower"], "tone": "attention"},
			{"label": "missforest", "value": counts["missforest"], "tone": "attention"},
			{"label": "hybrid", "value": counts["hybrid"], "tone": "positive"},
		}
	}
	if source == "gower" || len(sliceOfMaps(result["neighbor_evidence"])) > 0 {
		return []map[string]any{
			{"label": "rule", "value": 0, "tone": "neutral"},
			{"label": "gower", "value": asInt(result["applied_issue_count"]), "tone": "attention"},
		}
	}
	if source == "missforest" || len(sliceOfMaps(result["model_evidence"])) > 0 {
		return []map[string]any{
			{"label": "rule", "value": 0, "tone": "neutral"},
			{"label": "gower", "value": 0, "tone": "neutral"},
			{"label": "missforest", "value": asInt(result["applied_issue_count"]), "tone": "attention"},
		}
	}
	return []map[string]any{
		{"label": "rule", "value": asInt(result["applied_issue_count"]), "tone": "positive"},
		{"label": "gower", "value": 0, "tone": "neutral"},
		{"label": "missforest", "value": 0, "tone": "neutral"},
	}
}

func repairVerdict(before int, after int, applied int) string {
	if before == 0 && applied == 0 {
		return "neutral"
	}
	if after < before {
		return "improved"
	}
	if applied > 0 {
		return "changed"
	}
	return "neutral"
}

func fileArtifactPath(outputCSV string) string {
	if strings.TrimSpace(outputCSV) == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(outputCSV), "presentation.json")
}
