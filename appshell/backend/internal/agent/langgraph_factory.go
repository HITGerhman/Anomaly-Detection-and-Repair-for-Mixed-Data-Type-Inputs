package agent

// NewPhaseBPlannerStack preserves the existing planner stack wiring while
// allowing the LangGraph decorator to upgrade from mock cognition to Phase C
// LLM-backed cognition.
func NewPhaseBPlannerStack(engineScript string) (Planner, *LangGraphSidecarManager) {
	config := ResolveLangGraphConfig(engineScript)
	client := NewLangGraphClient(config.BaseURL(), config.RequestTimeout)
	manager := NewLangGraphSidecarManager(config, client)
	planner := NewLangGraphPlanner(NewDeterministicPlanner(), manager, client)
	return planner, manager
}
