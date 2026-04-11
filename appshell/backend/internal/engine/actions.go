package engine

type ActionName string

const (
	ActionHealth              ActionName = "health"
	ActionTrain               ActionName = "train"
	ActionRepair              ActionName = "repair"
	ActionScanFile            ActionName = "scan_file"
	ActionRepairBatch         ActionName = "repair_batch"
	ActionRepairWithGower     ActionName = "repair_with_gower"
	ActionRollbackRepairBatch ActionName = "rollback_repair_batch"
)

var knownActions = [...]ActionName{
	ActionHealth,
	ActionTrain,
	ActionRepair,
	ActionScanFile,
	ActionRepairBatch,
	ActionRepairWithGower,
	ActionRollbackRepairBatch,
}

func (a ActionName) String() string {
	return string(a)
}

func KnownActions() []ActionName {
	out := make([]ActionName, len(knownActions))
	copy(out, knownActions[:])
	return out
}

func KnownActionStrings() []string {
	items := KnownActions()
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, item.String())
	}
	return out
}
