package agent

import (
	"fmt"
	"sort"

	"appshell/backend/internal/engine"
)

type ToolSpec struct {
	ToolID  string `json:"tool_id"`
	Action  string `json:"action"`
	Summary string `json:"summary"`
}

type ToolRegistry struct {
	specs map[string]ToolSpec
}

func NewToolRegistry() *ToolRegistry {
	specs := []ToolSpec{
		{ToolID: "engine.health", Action: string(engine.ActionHealth), Summary: "Runtime health and dependency snapshot."},
		{ToolID: "engine.train_model", Action: string(engine.ActionTrain), Summary: "Train and persist the LightGBM model."},
		{ToolID: "engine.repair_sample", Action: string(engine.ActionRepair), Summary: "Repair a single anomaly sample."},
		{ToolID: "engine.scan_table", Action: string(engine.ActionScanFile), Summary: "Scan a table and return issue summaries."},
		{ToolID: "engine.repair_batch", Action: string(engine.ActionRepairBatch), Summary: "Repair selected issue ids in batch."},
		{ToolID: "engine.repair_with_gower", Action: string(engine.ActionRepairWithGower), Summary: "Repair selected issue ids with Gower neighbor retrieval."},
		{ToolID: "engine.repair_with_missforest", Action: string(engine.ActionRepairWithMissForest), Summary: "Repair selected issue ids with iterative MissForest random forest imputation."},
		{ToolID: "engine.rollback_batch", Action: string(engine.ActionRollbackRepairBatch), Summary: "Rollback a previous batch repair."},
	}
	items := make(map[string]ToolSpec, len(specs))
	for _, spec := range specs {
		items[spec.ToolID] = spec
	}
	return &ToolRegistry{specs: items}
}

func (r *ToolRegistry) Get(toolID string) (ToolSpec, bool) {
	if r == nil {
		return ToolSpec{}, false
	}
	spec, ok := r.specs[toolID]
	return spec, ok
}

func (r *ToolRegistry) MustGet(toolID string) (ToolSpec, error) {
	spec, ok := r.Get(toolID)
	if !ok {
		return ToolSpec{}, fmt.Errorf("tool not found: %s", toolID)
	}
	return spec, nil
}

func (r *ToolRegistry) All() []ToolSpec {
	if r == nil {
		return nil
	}
	keys := make([]string, 0, len(r.specs))
	for key := range r.specs {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	out := make([]ToolSpec, 0, len(keys))
	for _, key := range keys {
		out = append(out, r.specs[key])
	}
	return out
}
