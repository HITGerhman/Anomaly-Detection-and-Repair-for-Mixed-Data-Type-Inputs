package agent

import "time"

type AgentPreferenceProfile struct {
	ConservativeMode           bool     `json:"conservative_mode"`
	AvoidTimeColumns           bool     `json:"avoid_time_columns"`
	ProtectedColumns           []string `json:"protected_columns"`
	RequireApprovalForHighRisk bool     `json:"require_approval_for_high_risk"`
}

type AgentPreferenceRecord struct {
	WorkspaceID string                 `json:"workspace_id"`
	Profile     AgentPreferenceProfile `json:"profile"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

func DefaultPreferenceProfile() AgentPreferenceProfile {
	return defaultPreferenceProfile()
}

func NormalizePreferenceProfile(profile AgentPreferenceProfile) AgentPreferenceProfile {
	return normalizePreferenceProfile(profile)
}

func PreferenceProfileFromMap(raw map[string]any) AgentPreferenceProfile {
	return mergePreferenceProfile(defaultPreferenceProfile(), parsePreferenceOverrides(raw))
}

func PreferenceProfileToMap(profile AgentPreferenceProfile) map[string]any {
	return preferenceProfileToMap(profile)
}

func ResolveWorkspaceID(explicitWorkspaceID string, csvPath string) string {
	return resolveWorkspaceID(explicitWorkspaceID, csvPath)
}
