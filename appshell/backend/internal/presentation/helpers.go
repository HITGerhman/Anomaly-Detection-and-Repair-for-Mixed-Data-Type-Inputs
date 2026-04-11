package presentation

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

func asString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	case fmt.Stringer:
		return strings.TrimSpace(typed.String())
	default:
		return strings.TrimSpace(fmt.Sprint(value))
	}
}

func asInt(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float32:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func asFloat(value any) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int32:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return 0
	}
}

func cloneMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = cloneValue(value)
	}
	return out
}

func cloneValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		out := make([]any, len(typed))
		for idx, item := range typed {
			out[idx] = cloneValue(item)
		}
		return out
	case []string:
		out := make([]string, len(typed))
		copy(out, typed)
		return out
	default:
		return typed
	}
}

func objectMap(value any) map[string]any {
	if value == nil {
		return nil
	}
	if typed, ok := value.(map[string]any); ok {
		return cloneMap(typed)
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil
	}
	return out
}

func sliceOfMaps(value any) []map[string]any {
	if value == nil {
		return nil
	}
	switch typed := value.(type) {
	case []map[string]any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			out = append(out, cloneMap(item))
		}
		return out
	case []any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if mapped := objectMap(item); mapped != nil {
				out = append(out, mapped)
			}
		}
		return out
	default:
		payload, err := json.Marshal(value)
		if err != nil {
			return nil
		}
		var out []map[string]any
		if err := json.Unmarshal(payload, &out); err != nil {
			return nil
		}
		return out
	}
}

func sliceOfStrings(value any) []string {
	if value == nil {
		return nil
	}
	switch typed := value.(type) {
	case []string:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			item = strings.TrimSpace(item)
			if item != "" {
				out = append(out, item)
			}
		}
		return out
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			text := asString(item)
			if text != "" {
				out = append(out, text)
			}
		}
		return out
	default:
		return nil
	}
}

func toMap(value any) map[string]any {
	payload, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func nonEmptySections(items []Section) []Section {
	out := make([]Section, 0, len(items))
	for _, item := range items {
		if strings.TrimSpace(item.Title) == "" && strings.TrimSpace(item.Body) == "" && len(item.Bullets) == 0 {
			continue
		}
		out = append(out, item)
	}
	return out
}

func nonEmptyCharts(items []ChartSpec) []ChartSpec {
	out := make([]ChartSpec, 0, len(items))
	for _, item := range items {
		if strings.TrimSpace(item.ID) == "" || len(item.Data) == 0 {
			continue
		}
		out = append(out, item)
	}
	return out
}

func mapKeysSorted(input map[string]any) []string {
	keys := make([]string, 0, len(input))
	for key := range input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func stringOrDash(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return strings.TrimSpace(value)
}

func boolLabel(value any) string {
	switch typed := value.(type) {
	case bool:
		if typed {
			return "是"
		}
	case string:
		if strings.EqualFold(strings.TrimSpace(typed), "true") {
			return "是"
		}
	}
	return "否"
}

func maxInt(values ...int) int {
	best := 0
	for _, item := range values {
		if item > best {
			best = item
		}
	}
	return best
}

func countHighRiskIssues(issues []map[string]any) int {
	total := 0
	for _, item := range issues {
		if asString(item["risk_level"]) == "high" {
			total++
		}
	}
	return total
}

func totalIssueScore(issues []map[string]any) float64 {
	total := 0.0
	for _, item := range issues {
		total += asFloat(item["issue_score"])
	}
	return total
}

func issueTypeSeries(issues []map[string]any) []map[string]any {
	counts := map[string]int{}
	for _, item := range issues {
		key := asString(item["issue_type"])
		if key == "" {
			key = "unknown"
		}
		counts[key]++
	}
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]map[string]any, 0, len(keys))
	for _, key := range keys {
		out = append(out, map[string]any{
			"label": key,
			"value": counts[key],
		})
	}
	return out
}

func topColumnRiskSeries(thumbnails []map[string]any, limit int) []map[string]any {
	items := make([]map[string]any, 0, len(thumbnails))
	for _, item := range thumbnails {
		items = append(items, cloneMap(item))
	}
	sort.SliceStable(items, func(i, j int) bool {
		left := asFloat(items[i]["risk_score"])
		right := asFloat(items[j]["risk_score"])
		if left == right {
			return asString(items[i]["column"]) < asString(items[j]["column"])
		}
		return left > right
	})
	if limit > 0 && len(items) > limit {
		items = items[:limit]
	}
	series := make([]map[string]any, 0, len(items))
	for _, item := range items {
		series = append(series, map[string]any{
			"label":          asString(item["column"]),
			"value":          asFloat(item["risk_score"]),
			"issue_count":    asInt(item["issue_count"]),
			"anomaly_points": asInt(item["anomaly_points"]),
			"tone":           asString(item["risk_level"]),
		})
	}
	return series
}

func topRiskColumn(thumbnails []map[string]any) map[string]any {
	if len(thumbnails) == 0 {
		return nil
	}
	series := topColumnRiskSeries(thumbnails, 1)
	if len(series) == 0 {
		return nil
	}
	label := asString(series[0]["label"])
	for _, item := range thumbnails {
		if asString(item["column"]) == label {
			return cloneMap(item)
		}
	}
	return cloneMap(thumbnails[0])
}
