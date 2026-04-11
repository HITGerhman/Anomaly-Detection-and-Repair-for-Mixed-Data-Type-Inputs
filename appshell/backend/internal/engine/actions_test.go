package engine

import "testing"

func TestKnownActionsAreStableAndUnique(t *testing.T) {
	got := KnownActions()
	want := []ActionName{
		ActionHealth,
		ActionTrain,
		ActionRepair,
		ActionScanFile,
		ActionRepairBatch,
		ActionRepairWithGower,
		ActionRollbackRepairBatch,
	}

	if len(got) != len(want) {
		t.Fatalf("unexpected action count: got=%d want=%d", len(got), len(want))
	}

	seen := map[ActionName]bool{}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected action order at index %d: got=%s want=%s", i, got[i], want[i])
		}
		if seen[got[i]] {
			t.Fatalf("duplicate action returned: %s", got[i])
		}
		seen[got[i]] = true
	}
}

func TestKnownActionStringsMatchStableNames(t *testing.T) {
	got := KnownActionStrings()
	want := []string{
		"health",
		"train",
		"repair",
		"scan_file",
		"repair_batch",
		"repair_with_gower",
		"rollback_repair_batch",
	}

	if len(got) != len(want) {
		t.Fatalf("unexpected string action count: got=%d want=%d", len(got), len(want))
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected action string at index %d: got=%s want=%s", i, got[i], want[i])
		}
	}
}
