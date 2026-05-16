package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
	"appshell/backend/internal/task"
)

const startupChecksBlockedMessage = "启动自检未通过，请先修复阻塞问题并重新自检。"

type StartupCheckItem struct {
	Key       string         `json:"key"`
	Label     string         `json:"label"`
	Status    string         `json:"status"`
	Blocking  bool           `json:"blocking"`
	Message   string         `json:"message"`
	Path      string         `json:"path,omitempty"`
	Detail    map[string]any `json:"detail,omitempty"`
	AutoFixed bool           `json:"auto_fixed,omitempty"`
}

type StartupCheckSummary struct {
	Passed   int `json:"passed"`
	Warnings int `json:"warnings"`
	Failed   int `json:"failed"`
}

type StartupCheckReport struct {
	OverallStatus string              `json:"overall_status"`
	CanEnter      bool                `json:"can_enter"`
	CheckedAt     string              `json:"checked_at"`
	Items         []StartupCheckItem  `json:"items"`
	Summary       StartupCheckSummary `json:"summary"`
	Raw           map[string]any      `json:"raw,omitempty"`
}

func (a *App) RunStartupChecks() (StartupCheckReport, error) {
	report := StartupCheckReport{
		CheckedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Items:     make([]StartupCheckItem, 0, 7),
		Raw: map[string]any{
			"paths": map[string]any{},
		},
	}

	rawPaths := map[string]any{
		"engine_script":       a.engineScript,
		"task_history_sqlite": a.taskDBPath,
	}

	engineItem := checkEngineScript(a.engineScript)
	report.Items = append(report.Items, engineItem)

	var (
		healthItem StartupCheckItem
		depsItem   StartupCheckItem
	)

	if engineItem.Status == "pass" {
		healthResp, healthErr := runEngineHealthCheck(a.engineScript)
		if healthErr != nil {
			healthItem = StartupCheckItem{
				Key:      "engine_health",
				Label:    "Python 引擎健康检查",
				Status:   "fail",
				Blocking: true,
				Message:  "Python 引擎健康检查失败。",
				Path:     a.engineScript,
				Detail: map[string]any{
					"reason": healthErr.Error(),
				},
			}
			depsItem = StartupCheckItem{
				Key:      "runtime_dependencies",
				Label:    "运行时依赖",
				Status:   "fail",
				Blocking: true,
				Message:  "未能验证运行时依赖，因为 Python 引擎健康检查失败。",
				Detail: map[string]any{
					"reason": healthErr.Error(),
				},
			}
		} else {
			report.Raw["engine_health"] = healthResp.Result
			healthItem = buildEngineHealthItem(a.engineScript, healthResp.Result)
			depsItem = buildRuntimeDependencyItem(healthResp.Result)
		}
	} else {
		reason := "未能验证 Python 引擎健康检查，因为引擎脚本路径不可用。"
		healthItem = StartupCheckItem{
			Key:      "engine_health",
			Label:    "Python 引擎健康检查",
			Status:   "fail",
			Blocking: true,
			Message:  reason,
			Path:     a.engineScript,
			Detail: map[string]any{
				"reason": engineItem.Message,
			},
		}
		depsItem = StartupCheckItem{
			Key:      "runtime_dependencies",
			Label:    "运行时依赖",
			Status:   "fail",
			Blocking: true,
			Message:  "未能验证运行时依赖，因为引擎脚本不可用。",
			Detail: map[string]any{
				"reason": engineItem.Message,
			},
		}
	}
	report.Items = append(report.Items, healthItem, depsItem)

	a.mu.Lock()
	langGraphManager := a.ensureLangGraphManagerLocked()
	a.mu.Unlock()
	langGraphItem := checkLangGraphSidecar(langGraphManager)
	report.Items = append(report.Items, langGraphItem)
	report.Raw["langgraph_sidecar"] = clonePayload(langGraphItem.Detail)

	sqliteItem := checkTaskHistorySQLite(a.taskDBPath)
	report.Items = append(report.Items, sqliteItem)

	resultsDir, resultsErr := resolveResultsOutputRoot()
	rawPaths["results_output_root"] = resultsDir
	resultsItem := checkResultsOutputRoot(resultsDir, resultsErr)
	report.Items = append(report.Items, resultsItem)

	modelCandidates, modelCandidateErr := resolveDefaultModelCandidates()
	rawPaths["model_candidates"] = modelCandidates
	modelItem := checkModelArtifacts(modelCandidates, modelCandidateErr)
	report.Items = append(report.Items, modelItem)

	report.Raw["paths"] = rawPaths
	finalizeStartupCheckReport(&report)

	a.mu.Lock()
	defer a.mu.Unlock()

	if !report.CanEnter {
		a.closeServiceLocked()
		a.lastStartupReport = report
		return report, nil
	}

	if err := a.ensureServiceLocked(); err != nil {
		report = mutateStartupReportFailure(report, "task_history_sqlite", "任务历史服务初始化失败。", map[string]any{
			"reason":  err.Error(),
			"db_path": a.taskDBPath,
		})
		a.closeServiceLocked()
		a.lastStartupReport = report
		return report, nil
	}

	a.lastStartupReport = report
	return report, nil
}

func (a *App) startupBlockedError() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.startupBlockedErrorLocked()
}

func (a *App) startupBlockedErrorLocked() error {
	if len(a.lastStartupReport.Items) == 0 {
		return fmt.Errorf("%s", startupChecksBlockedMessage)
	}
	return fmt.Errorf("%s", startupChecksBlockedMessage)
}

func (a *App) ensureServiceLocked() error {
	if a.service != nil {
		return nil
	}
	historyStore, err := task.NewSQLiteHistoryStore(a.taskDBPath, historyKeepFromEnv(100))
	if err != nil {
		return fmt.Errorf("init task history store failed: %w", err)
	}
	agentStore, err := agent.NewSQLiteStore(a.taskDBPath)
	if err != nil {
		_ = historyStore.Close()
		return fmt.Errorf("init agent session store failed: %w", err)
	}
	baseRunner := engine.NewRunner(a.engineScript)
	manager := a.ensureLangGraphManagerLocked()
	client := agent.NewLangGraphClient(manager.Config().BaseURL(), manager.Config().RequestTimeout)
	planner := agent.NewLangGraphPlanner(agent.NewMockPlanner(), manager, client)
	runner := agent.NewRuntimeRunner(baseRunner, agentStore, planner)
	a.service = task.NewServiceWithConfig(runner, task.Config{
		HistoryStore: historyStore,
	})
	a.agentStore = agentStore
	return nil
}

func (a *App) closeServiceLocked() {
	if a.service == nil {
		if a.agentStore != nil {
			_ = a.agentStore.Close()
			a.agentStore = nil
		}
		if a.langGraphManager != nil {
			_ = a.langGraphManager.Close()
			a.langGraphManager = nil
		}
		return
	}
	a.service.Close()
	a.service = nil
	if a.agentStore != nil {
		_ = a.agentStore.Close()
		a.agentStore = nil
	}
	if a.langGraphManager != nil {
		_ = a.langGraphManager.Close()
		a.langGraphManager = nil
	}
}

func (a *App) getService() (*task.Service, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.service == nil {
		return nil, a.startupBlockedErrorLocked()
	}
	return a.service, nil
}

func (a *App) getAgentStore() (agent.SessionStore, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.service == nil || a.agentStore == nil {
		return nil, a.startupBlockedErrorLocked()
	}
	return a.agentStore, nil
}

func checkEngineScript(engineScript string) StartupCheckItem {
	item := StartupCheckItem{
		Key:      "engine_script",
		Label:    "Python 引擎脚本",
		Blocking: true,
		Path:     engineScript,
	}

	clean := strings.TrimSpace(engineScript)
	if clean == "" {
		item.Status = "fail"
		item.Message = "Python 引擎脚本路径为空。"
		return item
	}

	info, err := os.Stat(clean)
	if err != nil {
		item.Status = "fail"
		item.Message = "Python 引擎脚本不存在。"
		item.Detail = map[string]any{"reason": err.Error()}
		return item
	}
	if info.IsDir() {
		item.Status = "fail"
		item.Message = "Python 引擎脚本路径指向目录，而不是文件。"
		return item
	}

	item.Status = "pass"
	item.Message = "Python 引擎脚本路径可用。"
	return item
}

func runEngineHealthCheck(engineScript string) (engine.Response, error) {
	runner := engine.NewRunner(engineScript)
	runner.DefaultTimeout = 20 * time.Second

	resp, err := runner.Run(context.Background(), engine.Request{
		TaskID:  fmt.Sprintf("startup-health-%d", time.Now().UnixNano()),
		Action:  "health",
		Payload: map[string]any{},
	})
	if err != nil {
		return engine.Response{}, err
	}
	if strings.ToLower(strings.TrimSpace(resp.Status)) != "ok" {
		if resp.Error != nil {
			return engine.Response{}, fmt.Errorf("%s: %s", strings.TrimSpace(resp.Error.Code), strings.TrimSpace(resp.Error.Message))
		}
		return engine.Response{}, fmt.Errorf("engine health returned status=%s", resp.Status)
	}
	return resp, nil
}

func buildEngineHealthItem(engineScript string, result map[string]any) StartupCheckItem {
	pythonVersion := stringFromAny(result["python"])
	platformName := stringFromAny(result["platform"])
	message := "Python 引擎健康检查通过。"
	if pythonVersion != "" || platformName != "" {
		message = fmt.Sprintf("Python 引擎健康检查通过（Python %s）。", fallbackString(pythonVersion, "未知版本"))
	}

	detail := map[string]any{
		"python":       platformNameOrDash(pythonVersion),
		"platform":     platformNameOrDash(platformName),
		"project_root": platformNameOrDash(stringFromAny(result["project_root"])),
		"actions":      result["actions"],
	}

	return StartupCheckItem{
		Key:      "engine_health",
		Label:    "Python 引擎健康检查",
		Status:   "pass",
		Blocking: true,
		Message:  message,
		Path:     engineScript,
		Detail:   detail,
	}
}

func buildRuntimeDependencyItem(result map[string]any) StartupCheckItem {
	rawDeps, ok := result["dependencies"].(map[string]any)
	if !ok || len(rawDeps) == 0 {
		return StartupCheckItem{
			Key:      "runtime_dependencies",
			Label:    "运行时依赖",
			Status:   "fail",
			Blocking: true,
			Message:  "未从 Python 引擎健康检查中读取到依赖信息。",
		}
	}

	summary := make([]string, 0, len(rawDeps))
	detail := map[string]any{
		"dependencies": rawDeps,
	}

	for _, name := range []string{"pandas", "numpy", "lightgbm", "scikit-learn", "joblib"} {
		depMap, ok := rawDeps[name].(map[string]any)
		if !ok {
			return StartupCheckItem{
				Key:      "runtime_dependencies",
				Label:    "运行时依赖",
				Status:   "fail",
				Blocking: true,
				Message:  fmt.Sprintf("运行时依赖信息缺失：%s。", name),
				Detail:   detail,
			}
		}
		if strings.ToLower(stringFromAny(depMap["status"])) != "ok" {
			return StartupCheckItem{
				Key:      "runtime_dependencies",
				Label:    "运行时依赖",
				Status:   "fail",
				Blocking: true,
				Message:  fmt.Sprintf("运行时依赖不可用：%s。", name),
				Detail:   detail,
			}
		}
		summary = append(summary, fmt.Sprintf("%s %s", name, fallbackString(stringFromAny(depMap["version"]), "unknown")))
	}

	return StartupCheckItem{
		Key:      "runtime_dependencies",
		Label:    "运行时依赖",
		Status:   "pass",
		Blocking: true,
		Message:  fmt.Sprintf("运行时依赖检查通过（%s）。", strings.Join(summary, " / ")),
		Detail:   detail,
	}
}

func checkTaskHistorySQLite(dbPath string) StartupCheckItem {
	item := StartupCheckItem{
		Key:      "task_history_sqlite",
		Label:    "SQLite 任务历史",
		Status:   "fail",
		Blocking: true,
		Path:     dbPath,
	}

	clean := strings.TrimSpace(dbPath)
	if clean == "" {
		item.Message = "SQLite 任务历史路径为空。"
		return item
	}

	parent := filepath.Dir(clean)
	existedBefore := dirExists(parent)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		item.Message = "无法创建 SQLite 任务历史目录。"
		item.Detail = map[string]any{"reason": err.Error()}
		return item
	}
	item.AutoFixed = !existedBefore

	store, err := task.NewSQLiteHistoryStore(clean, historyKeepFromEnv(100))
	if err != nil {
		item.Message = "SQLite 任务历史初始化失败。"
		item.Detail = map[string]any{"reason": err.Error()}
		return item
	}
	defer func() {
		_ = store.Close()
	}()

	item.Status = "pass"
	item.Message = "SQLite 任务历史可用。"
	if item.AutoFixed {
		item.Message = "SQLite 任务历史目录已自动创建，且初始化成功。"
	}
	item.Detail = map[string]any{
		"db_path": clean,
	}
	return item
}

func resolveResultsOutputRoot() (string, error) {
	if _, ok := packagedRuntimeDir(); ok {
		root, err := packagedDataRoot()
		if err != nil {
			return "", err
		}
		return filepath.Join(root, "outputs", "results"), nil
	}
	return filepath.Abs(filepath.Join("..", "..", "outputs", "results"))
}

func checkResultsOutputRoot(path string, resolveErr error) StartupCheckItem {
	item := StartupCheckItem{
		Key:      "results_output_root",
		Label:    "结果输出目录",
		Status:   "fail",
		Blocking: true,
		Path:     path,
	}

	if resolveErr != nil {
		item.Message = "无法解析结果输出目录。"
		item.Detail = map[string]any{"reason": resolveErr.Error()}
		return item
	}

	info, err := os.Stat(path)
	existedBefore := err == nil && info.IsDir()
	if err == nil && !info.IsDir() {
		item.Message = "结果输出根路径存在，但它不是目录。"
		return item
	}
	if err != nil && !os.IsNotExist(err) {
		item.Message = "无法访问结果输出目录。"
		item.Detail = map[string]any{"reason": err.Error()}
		return item
	}

	if err := os.MkdirAll(path, 0o755); err != nil {
		item.Message = "无法创建结果输出目录。"
		item.Detail = map[string]any{"reason": err.Error()}
		return item
	}
	item.AutoFixed = !existedBefore

	probe, err := os.CreateTemp(path, "startup-check-*.tmp")
	if err != nil {
		item.Message = "结果输出目录不可写。"
		item.Detail = map[string]any{"reason": err.Error()}
		return item
	}
	probePath := probe.Name()
	_ = probe.Close()
	_ = os.Remove(probePath)

	item.Status = "pass"
	item.Message = "结果输出目录可用且可写。"
	if item.AutoFixed {
		item.Message = "结果输出目录已自动创建，且写入探针成功。"
	}
	item.Detail = map[string]any{
		"path": path,
	}
	return item
}

func resolveDefaultModelCandidates() ([]string, error) {
	candidates := []string{
		filepath.Join("..", "..", "outputs", "results", "wails_repair"),
		filepath.Join("..", "..", "data", "processed"),
	}
	if _, ok := packagedRuntimeDir(); ok {
		root, err := packagedDataRoot()
		if err != nil {
			return nil, err
		}
		candidates = []string{
			filepath.Join(root, "outputs", "results", "wails_repair"),
			filepath.Join(root, "data", "processed"),
		}
	}

	out := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		abs, err := filepath.Abs(candidate)
		if err != nil {
			return nil, err
		}
		out = append(out, abs)
	}
	return out, nil
}

func checkModelArtifacts(candidates []string, resolveErr error) StartupCheckItem {
	item := StartupCheckItem{
		Key:      "model_artifacts",
		Label:    "默认模型产物",
		Status:   "warning",
		Blocking: false,
	}

	if resolveErr != nil {
		item.Message = "无法解析默认模型目录，但这不会阻塞启动。"
		item.Detail = map[string]any{"reason": resolveErr.Error()}
		return item
	}

	requiredFiles := []string{"model_lgb.pkl", "test_data.pkl", "normal_data.pkl"}
	details := make([]map[string]any, 0, len(candidates))

	for _, candidate := range candidates {
		candidateInfo := map[string]any{
			"path": candidate,
		}
		info, err := os.Stat(candidate)
		if err != nil {
			candidateInfo["status"] = "missing"
			candidateInfo["exists"] = false
			details = append(details, candidateInfo)
			continue
		}
		if !info.IsDir() {
			candidateInfo["status"] = "invalid"
			candidateInfo["exists"] = true
			candidateInfo["reason"] = "path is not a directory"
			details = append(details, candidateInfo)
			continue
		}

		missingFiles := make([]string, 0, len(requiredFiles))
		for _, name := range requiredFiles {
			if _, err := os.Stat(filepath.Join(candidate, name)); err != nil {
				missingFiles = append(missingFiles, name)
			}
		}
		if len(missingFiles) == 0 {
			candidateInfo["status"] = "ready"
			candidateInfo["exists"] = true
			details = append(details, candidateInfo)
			item.Status = "pass"
			item.Message = fmt.Sprintf("已检测到可用模型产物目录：%s。", candidate)
			item.Path = candidate
			item.Detail = map[string]any{
				"selected_path": candidate,
				"candidates":    details,
			}
			return item
		}

		candidateInfo["status"] = "partial"
		candidateInfo["exists"] = true
		candidateInfo["missing_files"] = missingFiles
		details = append(details, candidateInfo)
	}

	item.Message = "尚未检测到完整的默认模型产物目录；训练完成后再执行修复即可。"
	item.Detail = map[string]any{
		"candidates": details,
	}
	return item
}

func finalizeStartupCheckReport(report *StartupCheckReport) {
	summary := StartupCheckSummary{}
	hasWarning := false
	hasFail := false

	for _, item := range report.Items {
		switch strings.ToLower(strings.TrimSpace(item.Status)) {
		case "pass":
			summary.Passed++
		case "warning":
			summary.Warnings++
			hasWarning = true
		default:
			summary.Failed++
			hasFail = true
		}
	}

	report.Summary = summary
	report.CanEnter = !hasFail
	switch {
	case hasFail:
		report.OverallStatus = "failed"
	case hasWarning:
		report.OverallStatus = "warning"
	default:
		report.OverallStatus = "ok"
	}
}

func mutateStartupReportFailure(report StartupCheckReport, key string, message string, detail map[string]any) StartupCheckReport {
	found := false
	for idx := range report.Items {
		if report.Items[idx].Key != key {
			continue
		}
		report.Items[idx].Status = "fail"
		report.Items[idx].Blocking = true
		report.Items[idx].Message = message
		report.Items[idx].Detail = detail
		found = true
		break
	}
	if !found {
		report.Items = append(report.Items, StartupCheckItem{
			Key:      key,
			Label:    key,
			Status:   "fail",
			Blocking: true,
			Message:  message,
			Detail:   detail,
		})
	}
	finalizeStartupCheckReport(&report)
	return report
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func (a *App) ensureLangGraphManagerLocked() *agent.LangGraphSidecarManager {
	if a.langGraphManager != nil {
		return a.langGraphManager
	}
	config := agent.ResolveLangGraphConfig(a.engineScript)
	client := agent.NewLangGraphClient(config.BaseURL(), config.RequestTimeout)
	a.langGraphManager = agent.NewLangGraphSidecarManager(config, client)
	return a.langGraphManager
}

func checkLangGraphSidecar(manager *agent.LangGraphSidecarManager) StartupCheckItem {
	config := manager.Config()
	detail := map[string]any{
		"base_url":         config.BaseURL(),
		"script_path":      config.ScriptPath,
		"planner_mode":     "fallback",
		"llm_mode":         "unavailable",
		"model":            "",
		"python_bin":       config.PythonBin,
		"managed":          manager.Managed(),
		"fallback_active":  true,
		"cognition_status": agent.CognitionStatusFallback,
		"provider":         agent.CognitionProviderDeterministic,
	}
	item := StartupCheckItem{
		Key:      "langgraph_sidecar",
		Label:    "LangGraph Sidecar",
		Blocking: false,
		Path:     config.ScriptPath,
		Detail:   detail,
	}
	if !config.Enabled {
		item.Status = "warning"
		item.Message = "LangGraph Sidecar 已关闭，系统将继续回退到 deterministic planner。"
		detail["reason"] = agent.CognitionFallbackDisabled
		detail["fallback_reason_code"] = agent.CognitionFallbackDisabled
		detail["fallback_message"] = "LangGraph sidecar is disabled, so deterministic planning stays active."
		detail["cognition_status"] = agent.CognitionStatusDisabled
		return item
	}

	ctx, cancel := context.WithTimeout(context.Background(), config.StartupTimeout)
	defer cancel()
	health, err := manager.EnsureHealthy(ctx)
	if err != nil {
		status, reasonCode, message := agent.ClassifyLangGraphAvailabilityError(err)
		item.Status = "warning"
		item.Message = "LangGraph Sidecar 预热失败，系统将继续回退到 deterministic planner。"
		detail["reason"] = err.Error()
		detail["fallback_reason_code"] = reasonCode
		detail["fallback_message"] = message
		detail["cognition_status"] = status
		detail["managed"] = manager.Managed()
		return item
	}

	item.Status = "pass"
	item.Message = "LangGraph Sidecar 已就绪，可用于 mock planning。"
	detail["status"] = health.Status
	detail["service"] = health.Service
	detail["planner_mode"] = health.PlannerMode
	detail["llm_mode"] = health.LLMMode
	detail["model"] = health.Model
	detail["graph_id"] = health.GraphID
	detail["version"] = health.Version
	detail["managed"] = manager.Managed()
	detail["fallback_active"] = !strings.EqualFold(strings.TrimSpace(health.PlannerMode), "llm")
	detail["provider"] = agent.CognitionProviderLangGraph
	detail["cognition_status"] = agent.CognitionStatusEngaged
	if !strings.EqualFold(strings.TrimSpace(health.PlannerMode), "llm") {
		item.Status = "warning"
		detail["provider"] = agent.CognitionProviderDeterministic
		detail["cognition_status"] = agent.CognitionStatusFallback
		detail["fallback_reason_code"] = agent.CognitionFallbackPlannerMode
		detail["fallback_message"] = "LangGraph reported fallback planner mode, so deterministic planning stays active."
	}
	return item
}

func stringFromAny(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	default:
		return strings.TrimSpace(fmt.Sprint(value))
	}
}

func fallbackString(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func platformNameOrDash(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return value
}
