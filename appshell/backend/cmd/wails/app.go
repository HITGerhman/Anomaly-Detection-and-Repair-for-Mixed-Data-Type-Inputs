package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
	"appshell/backend/internal/presentation"
	"appshell/backend/internal/task"
	"github.com/wailsapp/wails/v2/pkg/runtime"
)

type App struct {
	ctx context.Context

	mu                sync.Mutex
	service           *task.Service
	agentStore        agent.SessionStore
	langGraphManager  *agent.LangGraphSidecarManager
	engineScript      string
	taskDBPath        string
	lastStartupReport StartupCheckReport
}

func NewApp(engineScript string) (*App, error) {
	absEngine, err := filepath.Abs(engineScript)
	if err != nil {
		return nil, fmt.Errorf("resolve engine path failed: %w", err)
	}

	dbPath, err := resolveTaskDBPath()
	if err != nil {
		return nil, err
	}

	return &App{
		engineScript: absEngine,
		taskDBPath:   dbPath,
	}, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.autoAdjustWindowSize()
}

func (a *App) shutdown(context.Context) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closeServiceLocked()
}

func clampInt(v int, minValue int, maxValue int) int {
	if v < minValue {
		return minValue
	}
	if v > maxValue {
		return maxValue
	}
	return v
}

func pickBestScreen(screens []runtime.Screen) runtime.Screen {
	if len(screens) == 0 {
		return runtime.Screen{}
	}

	for _, screen := range screens {
		if screen.IsCurrent {
			return screen
		}
	}
	for _, screen := range screens {
		if screen.IsPrimary {
			return screen
		}
	}
	return screens[0]
}

func (a *App) autoAdjustWindowSize() {
	if a.ctx == nil {
		return
	}

	screens, err := runtime.ScreenGetAll(a.ctx)
	if err != nil || len(screens) == 0 {
		return
	}

	screen := pickBestScreen(screens)
	screenWidth := screen.Size.Width
	screenHeight := screen.Size.Height
	if screenWidth <= 0 {
		screenWidth = screen.Width
	}
	if screenHeight <= 0 {
		screenHeight = screen.Height
	}
	if screenWidth <= 0 || screenHeight <= 0 {
		return
	}

	targetWidth := clampInt(int(float64(screenWidth)*0.72), 960, 1440)
	targetHeight := clampInt(int(float64(screenHeight)*0.84), 700, 1040)

	runtime.WindowSetMinSize(a.ctx, 900, 640)
	runtime.WindowSetSize(a.ctx, targetWidth, targetHeight)
	runtime.WindowCenter(a.ctx)
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	case fmt.Stringer:
		return strings.TrimSpace(x.String())
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func resolveTaskDBPath() (string, error) {
	if raw := strings.TrimSpace(os.Getenv("APPSHELL_TASK_DB")); raw != "" {
		abs, err := filepath.Abs(raw)
		if err != nil {
			return "", fmt.Errorf("resolve APPSHELL_TASK_DB failed: %w", err)
		}
		return abs, nil
	}

	abs, err := filepath.Abs(filepath.Join("..", "..", "outputs", "appshell", "task_history.sqlite"))
	if err != nil {
		return "", fmt.Errorf("resolve default task history path failed: %w", err)
	}
	return abs, nil
}

func historyKeepFromEnv(fallback int) int {
	raw := strings.TrimSpace(os.Getenv("APPSHELL_TASK_HISTORY_KEEP"))
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}

func timeoutFromPayload(payload map[string]any, fallback time.Duration) time.Duration {
	raw := payload["timeout_ms"]
	if raw == nil {
		return fallback
	}

	switch v := raw.(type) {
	case float64:
		if v <= 0 {
			return fallback
		}
		return time.Duration(v) * time.Millisecond
	case int:
		if v <= 0 {
			return fallback
		}
		return time.Duration(v) * time.Millisecond
	case int64:
		if v <= 0 {
			return fallback
		}
		return time.Duration(v) * time.Millisecond
	case string:
		v = strings.TrimSpace(v)
		if v == "" {
			return fallback
		}
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}

	return fallback
}

func clonePayload(payload map[string]any) map[string]any {
	if payload == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(payload))
	for key, value := range payload {
		out[key] = value
	}
	return out
}

func normalizeRequest(payload map[string]any) (engine.Request, time.Duration, error) {
	if payload == nil {
		payload = map[string]any{}
	}

	action := asString(payload["action"])
	if action == "" {
		action = string(engine.ActionTrain)
	}
	actionName := engine.ActionName(action)
	taskID := asString(payload["task_id"])
	timeout := timeoutFromPayload(payload, 90*time.Second)

	req := engine.Request{
		TaskID:  taskID,
		Action:  action,
		Payload: map[string]any{},
	}

	copyOptional := func(keys ...string) {
		for _, key := range keys {
			if value, ok := payload[key]; ok {
				req.Payload[key] = value
			}
		}
	}

	// If caller already provides a nested payload object, use it directly.
	if nested, ok := payload["payload"].(map[string]any); ok {
		req.Payload = nested
		return req, timeout, nil
	}

	// MVP path: flatten action-specific fields.
	if actionName == engine.ActionTrain {
		csvPath := asString(payload["csv_path"])
		targetCol := asString(payload["target_col"])
		outputDir := asString(payload["output_dir"])
		taskType := asString(payload["task_type"])

		if csvPath == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: csv_path")
		}
		if targetCol == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: target_col")
		}

		req.Payload["csv_path"] = csvPath
		req.Payload["target_col"] = targetCol
		if outputDir != "" {
			req.Payload["output_dir"] = outputDir
		}
		if taskType != "" {
			req.Payload["task_type"] = taskType
		}
		return req, timeout, nil
	}

	if actionName == engine.ActionRepair {
		modelDir := asString(payload["model_dir"])
		if modelDir == "" {
			modelDir = asString(payload["output_dir"])
		}
		if modelDir == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: model_dir")
		}

		req.Payload["model_dir"] = modelDir

		if sampleIndex, ok := payload["sample_index"]; ok {
			req.Payload["sample_index"] = sampleIndex
		} else {
			req.Payload["sample_index"] = 0
		}
		if dryRun, ok := payload["dry_run"]; ok {
			req.Payload["dry_run"] = dryRun
		}
		if maxChanges, ok := payload["max_changes"]; ok {
			req.Payload["max_changes"] = maxChanges
		}
		if kNeighbors, ok := payload["k_neighbors"]; ok {
			req.Payload["k_neighbors"] = kNeighbors
		}
		if outputDir := asString(payload["output_dir"]); outputDir != "" {
			req.Payload["output_dir"] = outputDir
		}
		if immutableColumns, ok := payload["immutable_columns"]; ok {
			req.Payload["immutable_columns"] = immutableColumns
		}
		if numericBounds, ok := payload["numeric_bounds"]; ok {
			req.Payload["numeric_bounds"] = numericBounds
		}
		return req, timeout, nil
	}

	if actionName == engine.ActionScanFile {
		csvPath := asString(payload["csv_path"])
		if csvPath == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: csv_path")
		}
		req.Payload["csv_path"] = csvPath
		copyOptional(
			"max_bins",
			"max_issues",
			"numeric_iqr_factor",
			"robust_z_threshold",
			"rare_ratio_threshold",
			"rare_count_floor",
			"min_numeric_samples",
			"min_categorical_samples",
			"preview_limit",
			"enable_time_series_shift",
			"time_series_z_threshold",
			"time_series_min_points",
			"enable_cross_column_consistency",
			"consistency_rules",
			"enable_duplicate_record",
			"duplicate_subset",
			"auto_pair_constraints",
			"scan_config",
		)
		return req, timeout, nil
	}

	if actionName == engine.ActionRepairBatch {
		csvPath := asString(payload["csv_path"])
		if csvPath == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: csv_path")
		}
		req.Payload["csv_path"] = csvPath

		if issueIDs, ok := payload["issue_ids"]; ok {
			req.Payload["issue_ids"] = issueIDs
		} else {
			req.Payload["issue_ids"] = []string{}
		}
		if outputCSV, ok := payload["output_csv"]; ok {
			req.Payload["output_csv"] = outputCSV
		}
		if outputDir, ok := payload["output_dir"]; ok {
			req.Payload["output_dir"] = outputDir
		}
		if writeOutput, ok := payload["write_output"]; ok {
			req.Payload["write_output"] = writeOutput
		}
		if planOnly, ok := payload["plan_only"]; ok {
			req.Payload["plan_only"] = planOnly
		}
		if enableRollback, ok := payload["enable_rollback"]; ok {
			req.Payload["enable_rollback"] = enableRollback
		}
		if rollbackDir, ok := payload["rollback_dir"]; ok {
			req.Payload["rollback_dir"] = rollbackDir
		}
		if repairStrategy, ok := payload["repair_strategy"]; ok {
			req.Payload["repair_strategy"] = repairStrategy
		}
		if columnDependencies, ok := payload["column_dependencies"]; ok {
			req.Payload["column_dependencies"] = columnDependencies
		}
		copyOptional(
			"max_bins",
			"max_issues",
			"numeric_iqr_factor",
			"robust_z_threshold",
			"rare_ratio_threshold",
			"rare_count_floor",
			"min_numeric_samples",
			"min_categorical_samples",
			"preview_limit",
			"enable_time_series_shift",
			"time_series_z_threshold",
			"time_series_min_points",
			"enable_cross_column_consistency",
			"consistency_rules",
			"enable_duplicate_record",
			"duplicate_subset",
			"auto_pair_constraints",
			"scan_config",
		)
		return req, timeout, nil
	}

	if actionName == engine.ActionRepairWithGower {
		csvPath := asString(payload["csv_path"])
		if csvPath == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: csv_path")
		}
		req.Payload["csv_path"] = csvPath

		if issueIDs, ok := payload["issue_ids"]; ok {
			req.Payload["issue_ids"] = issueIDs
		} else {
			req.Payload["issue_ids"] = []string{}
		}
		if outputCSV, ok := payload["output_csv"]; ok {
			req.Payload["output_csv"] = outputCSV
		}
		if outputDir, ok := payload["output_dir"]; ok {
			req.Payload["output_dir"] = outputDir
		}
		if writeOutput, ok := payload["write_output"]; ok {
			req.Payload["write_output"] = writeOutput
		}
		if planOnly, ok := payload["plan_only"]; ok {
			req.Payload["plan_only"] = planOnly
		}
		if enableRollback, ok := payload["enable_rollback"]; ok {
			req.Payload["enable_rollback"] = enableRollback
		}
		if rollbackDir, ok := payload["rollback_dir"]; ok {
			req.Payload["rollback_dir"] = rollbackDir
		}
		if columnDependencies, ok := payload["column_dependencies"]; ok {
			req.Payload["column_dependencies"] = columnDependencies
		}
		if modelDir, ok := payload["model_dir"]; ok {
			req.Payload["model_dir"] = modelDir
		}
		if gowerStrategy, ok := payload["gower_strategy"]; ok {
			req.Payload["gower_strategy"] = gowerStrategy
		}
		copyOptional(
			"max_bins",
			"max_issues",
			"numeric_iqr_factor",
			"robust_z_threshold",
			"rare_ratio_threshold",
			"rare_count_floor",
			"min_numeric_samples",
			"min_categorical_samples",
			"preview_limit",
			"enable_time_series_shift",
			"time_series_z_threshold",
			"time_series_min_points",
			"enable_cross_column_consistency",
			"consistency_rules",
			"enable_duplicate_record",
			"duplicate_subset",
			"auto_pair_constraints",
			"scan_config",
		)
		return req, timeout, nil
	}

	if actionName == engine.ActionRollbackRepairBatch {
		manifestPath := asString(payload["manifest_path"])
		if manifestPath == "" {
			return engine.Request{}, 0, fmt.Errorf("missing required field: manifest_path")
		}
		req.Payload["manifest_path"] = manifestPath
		copyOptional("restore_target", "target_csv")
		return req, timeout, nil
	}

	return req, timeout, nil
}

func (a *App) RunTask(payload map[string]any) (task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return task.Task{}, err
	}

	req, timeout, err := normalizeRequest(payload)
	if err != nil {
		return task.Task{}, err
	}

	taskID, err := service.RunTask(req, timeout)
	if err != nil {
		return task.Task{}, err
	}

	snapshot, ok := service.GetTaskStatus(taskID)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found after submit: %s", taskID)
	}
	return *snapshot, nil
}

func (a *App) GetTaskStatus(taskID string) (task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return task.Task{}, err
	}
	id := strings.TrimSpace(taskID)
	if id == "" {
		return task.Task{}, fmt.Errorf("task id is required")
	}

	snapshot, ok := service.GetTaskStatus(id)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found: %s", id)
	}
	return *snapshot, nil
}

func (a *App) CancelTask(taskID string) (bool, error) {
	service, err := a.getService()
	if err != nil {
		return false, err
	}
	id := strings.TrimSpace(taskID)
	if id == "" {
		return false, fmt.Errorf("task id is required")
	}
	return service.CancelTask(id), nil
}

func (a *App) ListTaskHistory(limit int) ([]task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 20
	}
	return service.ListRecentTasks(limit)
}

func (a *App) RunAgentSession(payload map[string]any) (task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return task.Task{}, err
	}

	reqPayload := clonePayload(payload)
	timeout := timeoutFromPayload(reqPayload, 90*time.Second)
	taskID, err := service.RunTask(engine.Request{
		TaskID:  asString(reqPayload["task_id"]),
		Action:  agent.ActionSessionPlan,
		Payload: reqPayload,
	}, timeout)
	if err != nil {
		return task.Task{}, err
	}

	snapshot, ok := service.GetTaskStatus(taskID)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found after submit: %s", taskID)
	}
	return *snapshot, nil
}

func (a *App) ExecuteAgentPlan(payload map[string]any) (task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return task.Task{}, err
	}

	reqPayload := clonePayload(payload)
	timeout := timeoutFromPayload(reqPayload, 90*time.Second)
	taskID, err := service.RunTask(engine.Request{
		TaskID:  asString(reqPayload["task_id"]),
		Action:  agent.ActionSessionExecute,
		Payload: reqPayload,
	}, timeout)
	if err != nil {
		return task.Task{}, err
	}

	snapshot, ok := service.GetTaskStatus(taskID)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found after submit: %s", taskID)
	}
	return *snapshot, nil
}

func (a *App) RunAgentAutofixSession(payload map[string]any) (task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return task.Task{}, err
	}

	reqPayload := clonePayload(payload)
	timeout := timeoutFromPayload(reqPayload, 90*time.Second)
	taskID, err := service.RunTask(engine.Request{
		TaskID:  asString(reqPayload["task_id"]),
		Action:  agent.ActionSessionAuto,
		Payload: reqPayload,
	}, timeout)
	if err != nil {
		return task.Task{}, err
	}

	snapshot, ok := service.GetTaskStatus(taskID)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found after submit: %s", taskID)
	}
	return *snapshot, nil
}

func (a *App) ApproveAgentSession(payload map[string]any) (task.Task, error) {
	service, err := a.getService()
	if err != nil {
		return task.Task{}, err
	}

	reqPayload := clonePayload(payload)
	timeout := timeoutFromPayload(reqPayload, 90*time.Second)
	taskID, err := service.RunTask(engine.Request{
		TaskID:  asString(reqPayload["task_id"]),
		Action:  agent.ActionSessionApprove,
		Payload: reqPayload,
	}, timeout)
	if err != nil {
		return task.Task{}, err
	}

	snapshot, ok := service.GetTaskStatus(taskID)
	if !ok {
		return task.Task{}, fmt.Errorf("task not found after submit: %s", taskID)
	}
	return *snapshot, nil
}

func (a *App) GetAgentPreferences(workspaceID string, csvPath string) (agent.AgentPreferenceRecord, error) {
	store, err := a.getAgentStore()
	if err != nil {
		return agent.AgentPreferenceRecord{}, err
	}

	resolvedWorkspaceID := agent.ResolveWorkspaceID(workspaceID, csvPath)
	if strings.TrimSpace(resolvedWorkspaceID) == "" {
		return agent.AgentPreferenceRecord{}, fmt.Errorf("workspace_id or csv_path is required")
	}

	record, ok, err := store.GetPreferences(context.Background(), resolvedWorkspaceID)
	if err != nil {
		return agent.AgentPreferenceRecord{}, err
	}
	if ok {
		record.WorkspaceID = resolvedWorkspaceID
		record.Profile = agent.NormalizePreferenceProfile(record.Profile)
		return record, nil
	}
	return agent.AgentPreferenceRecord{
		WorkspaceID: resolvedWorkspaceID,
		Profile:     agent.DefaultPreferenceProfile(),
	}, nil
}

func (a *App) SaveAgentPreferences(payload map[string]any) (agent.AgentPreferenceRecord, error) {
	store, err := a.getAgentStore()
	if err != nil {
		return agent.AgentPreferenceRecord{}, err
	}

	reqPayload := clonePayload(payload)
	resolvedWorkspaceID := agent.ResolveWorkspaceID(asString(reqPayload["workspace_id"]), asString(reqPayload["csv_path"]))
	if strings.TrimSpace(resolvedWorkspaceID) == "" {
		return agent.AgentPreferenceRecord{}, fmt.Errorf("workspace_id or csv_path is required")
	}
	profilePayload := map[string]any{}
	if nested, ok := reqPayload["profile"].(map[string]any); ok {
		profilePayload = clonePayload(nested)
	} else {
		profilePayload = reqPayload
	}
	record := agent.AgentPreferenceRecord{
		WorkspaceID: resolvedWorkspaceID,
		Profile:     agent.PreferenceProfileFromMap(profilePayload),
		UpdatedAt:   time.Now().UTC(),
	}
	if err := store.SavePreferences(context.Background(), record); err != nil {
		return agent.AgentPreferenceRecord{}, err
	}
	return record, nil
}

func (a *App) GetAgentSession(sessionID string) (agent.AgentSessionSnapshot, error) {
	store, err := a.getAgentStore()
	if err != nil {
		return agent.AgentSessionSnapshot{}, err
	}

	id := strings.TrimSpace(sessionID)
	if id == "" {
		return agent.AgentSessionSnapshot{}, fmt.Errorf("session id is required")
	}

	session, ok, err := store.GetSession(context.Background(), id)
	if err != nil {
		return agent.AgentSessionSnapshot{}, err
	}
	if !ok {
		return agent.AgentSessionSnapshot{}, fmt.Errorf("agent session not found: %s", id)
	}
	trace, err := store.ListTrace(context.Background(), id)
	if err != nil {
		return agent.AgentSessionSnapshot{}, err
	}
	snapshot := session.Snapshot(agentTraceSummary(trace))
	if err := presentation.EnrichAgentSessionSnapshot(&snapshot); err != nil {
		return agent.AgentSessionSnapshot{}, err
	}
	return snapshot, nil
}

func (a *App) ListAgentTrace(sessionID string) ([]agent.AgentTraceEvent, error) {
	store, err := a.getAgentStore()
	if err != nil {
		return nil, err
	}

	id := strings.TrimSpace(sessionID)
	if id == "" {
		return nil, fmt.Errorf("session id is required")
	}
	return store.ListTrace(context.Background(), id)
}

// Backward-compatible alias used by previous frontend template.
func (a *App) RunTrainTask(payload map[string]any) (task.Task, error) {
	if payload == nil {
		payload = map[string]any{}
	}
	payload["action"] = string(engine.ActionTrain)
	return a.RunTask(payload)
}

func (a *App) SelectCSV() (string, error) {
	if a.ctx == nil {
		return "", fmt.Errorf("runtime is not initialized")
	}

	path, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select CSV File",
		Filters: []runtime.FileFilter{
			{
				DisplayName: "CSV",
				Pattern:     "*.csv",
			},
		},
	})
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(path), nil
}

func (a *App) SelectOutputDir() (string, error) {
	if a.ctx == nil {
		return "", fmt.Errorf("runtime is not initialized")
	}

	path, err := runtime.OpenDirectoryDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select Output Directory",
	})
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(path), nil
}

func listCSVColumnsFromFile(csvPath string) ([]string, error) {
	absPath, err := resolveExistingFilePath(csvPath)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(absPath)
	if err != nil {
		return nil, fmt.Errorf("open csv failed: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1

	record, err := reader.Read()
	if err == io.EOF {
		return nil, fmt.Errorf("csv file is empty")
	}
	if err != nil {
		return nil, fmt.Errorf("read csv header failed: %w", err)
	}
	if len(record) == 0 {
		return nil, fmt.Errorf("csv header is empty")
	}

	columns := make([]string, 0, len(record))
	seen := make(map[string]struct{}, len(record))

	for i, raw := range record {
		col := strings.TrimSpace(raw)
		if i == 0 {
			col = strings.TrimPrefix(col, "\ufeff")
		}
		if col == "" {
			col = fmt.Sprintf("column_%d", i+1)
		}
		if _, ok := seen[col]; ok {
			continue
		}
		seen[col] = struct{}{}
		columns = append(columns, col)
	}

	if len(columns) == 0 {
		return nil, fmt.Errorf("no available columns found in csv header")
	}

	return columns, nil
}

func (a *App) ListCSVColumns(csvPath string) ([]string, error) {
	return listCSVColumnsFromFile(csvPath)
}

func resolveExistingFilePath(rawPath string) (string, error) {
	clean := strings.TrimSpace(rawPath)
	if clean == "" {
		return "", fmt.Errorf("csv path is required")
	}

	candidates := make([]string, 0, 6)
	seen := map[string]struct{}{}
	pushCandidate := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		normalized := filepath.Clean(path)
		if _, ok := seen[normalized]; ok {
			return
		}
		seen[normalized] = struct{}{}
		candidates = append(candidates, normalized)
	}

	if filepath.IsAbs(clean) {
		pushCandidate(clean)
	} else {
		pushCandidate(clean)
		pushCandidate(filepath.Join("..", clean))
		pushCandidate(filepath.Join("..", "..", clean))
	}

	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		pushCandidate(filepath.Join(exeDir, clean))
		pushCandidate(filepath.Join(exeDir, "..", clean))
		pushCandidate(filepath.Join(exeDir, "..", "..", clean))
	}

	for _, candidate := range candidates {
		abs, err := filepath.Abs(candidate)
		if err != nil {
			continue
		}
		info, err := os.Stat(abs)
		if err == nil && !info.IsDir() {
			return abs, nil
		}
	}

	return "", fmt.Errorf("csv file not found: %s", clean)
}

func agentTraceSummary(events []agent.AgentTraceEvent) agent.TraceSummary {
	return agent.SummarizeTraceEvents(events)
}
