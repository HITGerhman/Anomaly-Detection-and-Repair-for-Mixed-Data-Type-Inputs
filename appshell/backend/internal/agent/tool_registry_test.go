package agent

import "testing"

func TestToolRegistryIncludesStableTools(t *testing.T) {
	registry := NewToolRegistry()

	cases := map[string]string{
		"engine.health":            "health",
		"engine.train_model":       "train",
		"engine.repair_sample":     "repair",
		"engine.scan_table":        "scan_file",
		"engine.repair_batch":      "repair_batch",
		"engine.repair_with_gower": "repair_with_gower",
		"engine.rollback_batch":    "rollback_repair_batch",
	}

	for toolID, action := range cases {
		spec, ok := registry.Get(toolID)
		if !ok {
			t.Fatalf("expected tool %s to exist", toolID)
		}
		if spec.Action != action {
			t.Fatalf("unexpected action for %s: got=%s want=%s", toolID, spec.Action, action)
		}
	}
	if len(registry.All()) != len(cases) {
		t.Fatalf("unexpected tool count: got=%d want=%d", len(registry.All()), len(cases))
	}
}
