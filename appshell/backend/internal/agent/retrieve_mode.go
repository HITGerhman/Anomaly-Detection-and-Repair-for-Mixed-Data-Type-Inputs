package agent

import (
	"os"
	"strings"
)

const (
	RetrieveModeSequential = "sequential"
	RetrieveModeParallel   = "parallel"
)

func normalizeRetrieveMode(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case RetrieveModeSequential:
		return RetrieveModeSequential
	case "", RetrieveModeParallel:
		return RetrieveModeParallel
	default:
		return RetrieveModeParallel
	}
}

func resolveRetrieveMode(payload map[string]any) string {
	if raw := asString(payload["agent_retrieve_mode"]); raw != "" {
		return normalizeRetrieveMode(raw)
	}
	return normalizeRetrieveMode(os.Getenv("APPSHELL_AGENT_RETRIEVE_MODE"))
}

func retrieveModeRunsInParallel(mode string) bool {
	return normalizeRetrieveMode(mode) == RetrieveModeParallel
}
