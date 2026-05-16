package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	defaultLangGraphHost           = "127.0.0.1"
	defaultLangGraphPort           = 58331
	defaultLangGraphStartupTimeout = 5 * time.Second
	defaultLangGraphRequestTimeout = 3 * time.Second
)

type LangGraphConfig struct {
	Enabled        bool
	Host           string
	Port           int
	ScriptPath     string
	PythonBin      string
	StartupTimeout time.Duration
	RequestTimeout time.Duration
}

func (c LangGraphConfig) BaseURL() string {
	host := strings.TrimSpace(c.Host)
	if host == "" {
		host = defaultLangGraphHost
	}
	port := c.Port
	if port <= 0 {
		port = defaultLangGraphPort
	}
	return fmt.Sprintf("http://%s:%d", host, port)
}

type LangGraphHealth struct {
	Status      string `json:"status"`
	Service     string `json:"service"`
	PlannerMode string `json:"planner_mode"`
	LLMMode     string `json:"llm_mode"`
	Model       string `json:"model"`
	Ready       bool   `json:"ready"`
	GraphID     string `json:"graph_id"`
	Version     string `json:"version"`
}

type LangGraphCandidatePreview struct {
	CandidateID      string         `json:"candidate_id"`
	Source           string         `json:"source"`
	Comparison       map[string]any `json:"comparison"`
	Score            map[string]any `json:"score,omitempty"`
	SelectedIssueIDs []string       `json:"selected_issue_ids"`
	ToolSequence     []string       `json:"tool_sequence"`
	Summary          string         `json:"summary"`
}

type LangGraphPlanRequest struct {
	SessionID         string                      `json:"session_id"`
	Goal              string                      `json:"goal"`
	ScanSummary       map[string]any              `json:"scan_summary"`
	CandidatePreviews []LangGraphCandidatePreview `json:"candidate_previews"`
	SafetyContext     map[string]any              `json:"safety_context"`
	ApprovalContext   map[string]any              `json:"approval_context"`
	UserPreferences   map[string]any              `json:"user_preferences"`
	OutputConstraints map[string]any              `json:"output_constraints"`
}

type LangGraphPlanResponse struct {
	StrategyLabel       string   `json:"strategy_label"`
	SelectedCandidateID string   `json:"selected_candidate_id"`
	ReasonCodes         []string `json:"reason_codes"`
	RiskNote            string   `json:"risk_note"`
	IntentLabel         string   `json:"intent_label"`
	OneSentenceSummary  string   `json:"one_sentence_summary"`
	ShortBullets        []string `json:"short_bullets"`
	ApprovalNeeded      bool     `json:"approval_needed"`
}

type LangGraphExplainRequest struct {
	SessionID         string                    `json:"session_id"`
	Goal              string                    `json:"goal"`
	SelectedCandidate LangGraphCandidatePreview `json:"selected_candidate"`
	StrategyLabel     string                    `json:"strategy_label"`
	ReasonCodes       []string                  `json:"reason_codes"`
	RiskNote          string                    `json:"risk_note"`
	ValidationPreview map[string]any            `json:"validation_preview"`
	SafetyContext     map[string]any            `json:"safety_context"`
	ApprovalContext   map[string]any            `json:"approval_context"`
	OutputConstraints map[string]any            `json:"output_constraints"`
}

type LangGraphExplainResponse struct {
	Summary      string   `json:"summary"`
	FinalMessage string   `json:"final_message"`
	ShortBullets []string `json:"short_bullets"`
	ReasonCodes  []string `json:"reason_codes"`
	RiskNote     string   `json:"risk_note"`
}

func defaultLangGraphPythonBin() string {
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func resolveLangGraphScript(engineScript string) string {
	engineDir := filepath.Dir(strings.TrimSpace(engineScript))
	coreDir := filepath.Dir(engineDir)
	if strings.TrimSpace(coreDir) == "" {
		return filepath.Clean(filepath.Join("..", "..", "core", "langgraph_sidecar", "main.py"))
	}
	return filepath.Join(coreDir, "langgraph_sidecar", "main.py")
}

func parseEnvBool(raw string, fallback bool) bool {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "":
		return fallback
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func parseEnvInt(raw string, fallback int) int {
	value := strings.TrimSpace(raw)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func parseEnvDurationMS(raw string, fallback time.Duration) time.Duration {
	millis := parseEnvInt(raw, int(fallback.Milliseconds()))
	if millis <= 0 {
		return fallback
	}
	return time.Duration(millis) * time.Millisecond
}

func ResolveLangGraphConfig(engineScript string) LangGraphConfig {
	config := LangGraphConfig{
		Enabled:        parseEnvBool("", true),
		Host:           defaultLangGraphHost,
		Port:           defaultLangGraphPort,
		ScriptPath:     resolveLangGraphScript(engineScript),
		PythonBin:      defaultLangGraphPythonBin(),
		StartupTimeout: defaultLangGraphStartupTimeout,
		RequestTimeout: defaultLangGraphRequestTimeout,
	}

	config.Enabled = parseEnvBool(strings.TrimSpace(os.Getenv("APPSHELL_LANGGRAPH_ENABLED")), true)
	if host := strings.TrimSpace(os.Getenv("APPSHELL_LANGGRAPH_HOST")); host != "" {
		config.Host = host
	}
	config.Port = parseEnvInt(os.Getenv("APPSHELL_LANGGRAPH_PORT"), defaultLangGraphPort)
	if script := strings.TrimSpace(os.Getenv("APPSHELL_LANGGRAPH_SCRIPT")); script != "" {
		config.ScriptPath = script
	}
	if pythonBin := strings.TrimSpace(os.Getenv("APPSHELL_LANGGRAPH_PYTHON_BIN")); pythonBin != "" {
		config.PythonBin = pythonBin
	}
	config.StartupTimeout = parseEnvDurationMS(os.Getenv("APPSHELL_LANGGRAPH_STARTUP_TIMEOUT_MS"), defaultLangGraphStartupTimeout)
	config.RequestTimeout = parseEnvDurationMS(os.Getenv("APPSHELL_LANGGRAPH_REQUEST_TIMEOUT_MS"), defaultLangGraphRequestTimeout)
	return config
}
