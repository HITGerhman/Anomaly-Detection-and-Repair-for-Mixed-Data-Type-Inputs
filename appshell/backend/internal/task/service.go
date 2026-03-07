package task

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"appshell/backend/internal/engine"
	"appshell/backend/internal/observability"
)

const (
	StatusPending   = "pending"
	StatusRunning   = "running"
	StatusSucceeded = "succeeded"
	StatusFailed    = "failed"
	StatusCanceled  = "canceled"
	StatusTimedOut  = "timed_out"
)

type Task struct {
	ID        string          `json:"id"`
	Status    string          `json:"status"`
	Request   engine.Request  `json:"request"`
	Response  engine.Response `json:"response"`
	Error     string          `json:"error"`
	Progress  TaskProgress    `json:"progress"`
	CreatedAt time.Time       `json:"created_at"`
	StartedAt time.Time       `json:"started_at"`
	EndedAt   time.Time       `json:"ended_at"`
	TimeoutMS int64           `json:"timeout_ms"`
}

type TaskProgress struct {
	CurrentStage     string               `json:"current_stage"`
	ProgressPercent  int                  `json:"progress_percent"`
	LastMessage      string               `json:"last_message"`
	UpdatedAtMS      int64                `json:"updated_at_ms"`
	StageStartedAtMS map[string]int64     `json:"stage_started_at_ms,omitempty"`
	StageDurationsMS map[string]int64     `json:"stage_durations_ms,omitempty"`
	BottleneckStage  string               `json:"bottleneck_stage,omitempty"`
	BottleneckMS     int64                `json:"bottleneck_ms,omitempty"`
	Failure          *TaskFailureLocation `json:"failure,omitempty"`
	Events           []TaskProgressEvent  `json:"events,omitempty"`
	Meta             map[string]any       `json:"meta,omitempty"`
}

type TaskFailureLocation struct {
	Stage     string `json:"stage,omitempty"`
	Message   string `json:"message,omitempty"`
	File      string `json:"file,omitempty"`
	Column    string `json:"column,omitempty"`
	Rule      string `json:"rule,omitempty"`
	ErrorCode string `json:"error_code,omitempty"`
}

type TaskProgressEvent struct {
	AtMS      int64  `json:"at_ms"`
	Stage     string `json:"stage"`
	Phase     string `json:"phase"`
	Progress  int    `json:"progress"`
	Message   string `json:"message"`
	File      string `json:"file,omitempty"`
	Column    string `json:"column,omitempty"`
	Rule      string `json:"rule,omitempty"`
	ErrorCode string `json:"error_code,omitempty"`
}

type Runner interface {
	Run(ctx context.Context, req engine.Request) (engine.Response, error)
}

type Config struct {
	MaxConcurrency int
	QueueSize      int
	HistoryStore   HistoryStore
}

func defaultConfig() Config {
	return Config{
		MaxConcurrency: 3,
		QueueSize:      128,
	}
}

type runtimeInfo struct {
	ctx    context.Context
	cancel context.CancelFunc
}

type Service struct {
	runner  Runner
	cfg     Config
	history HistoryStore

	mu      sync.RWMutex
	tasks   map[string]*Task
	runtime map[string]runtimeInfo

	queue  chan string
	stopCh chan struct{}
	wg     sync.WaitGroup
	once   sync.Once
	seq    uint64
}

func NewService(runner Runner) *Service {
	return NewServiceWithConfig(runner, Config{})
}

func NewServiceWithConfig(runner Runner, cfg Config) *Service {
	def := defaultConfig()
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = def.MaxConcurrency
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = def.QueueSize
	}

	s := &Service{
		runner:  runner,
		cfg:     cfg,
		history: cfg.HistoryStore,
		tasks:   make(map[string]*Task),
		runtime: make(map[string]runtimeInfo),
		queue:   make(chan string, cfg.QueueSize),
		stopCh:  make(chan struct{}),
	}

	if observerRunner, ok := runner.(interface {
		SetStderrObserver(engine.StderrObserver)
	}); ok {
		observerRunner.SetStderrObserver(s.onRunnerStderr)
	}

	for i := 0; i < cfg.MaxConcurrency; i++ {
		s.wg.Add(1)
		go s.worker()
	}

	return s
}

func (s *Service) Close() {
	s.once.Do(func() {
		close(s.stopCh)
		s.wg.Wait()

		s.mu.Lock()
		defer s.mu.Unlock()
		for id, rt := range s.runtime {
			rt.cancel()
			delete(s.runtime, id)
		}

		if s.history != nil {
			if err := s.history.Close(); err != nil {
				observability.Warn("task_history_close_failed", map[string]any{
					"error": err.Error(),
				})
			}
		}
	})
}

func (s *Service) worker() {
	defer s.wg.Done()
	for {
		select {
		case <-s.stopCh:
			return
		case taskID := <-s.queue:
			s.execute(taskID)
		}
	}
}

func createTaskContext(timeout time.Duration) (context.Context, context.CancelFunc) {
	baseCtx, cancelBase := context.WithCancel(context.Background())
	ctx := baseCtx
	cancelTimeout := func() {}

	if timeout > 0 {
		var timeoutCancel context.CancelFunc
		ctx, timeoutCancel = context.WithTimeout(baseCtx, timeout)
		cancelTimeout = timeoutCancel
	}

	return ctx, func() {
		cancelTimeout()
		cancelBase()
	}
}

func isTerminalStatus(status string) bool {
	switch status {
	case StatusSucceeded, StatusFailed, StatusCanceled, StatusTimedOut:
		return true
	default:
		return false
	}
}

const maxProgressEvents = 240

func newTaskProgressSnapshot() TaskProgress {
	return TaskProgress{
		CurrentStage:     "",
		ProgressPercent:  0,
		LastMessage:      "",
		UpdatedAtMS:      time.Now().UnixMilli(),
		StageStartedAtMS: map[string]int64{},
		StageDurationsMS: map[string]int64{},
		BottleneckStage:  "",
		BottleneckMS:     0,
		Failure:          nil,
		Events:           make([]TaskProgressEvent, 0, 16),
		Meta:             map[string]any{},
	}
}

func toStringAny(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case fmt.Stringer:
		return x.String()
	default:
		if v == nil {
			return ""
		}
		return fmt.Sprint(v)
	}
}

func toIntAny(v any, fallback int) int {
	switch x := v.(type) {
	case int:
		return x
	case int32:
		return int(x)
	case int64:
		return int(x)
	case float32:
		return int(x)
	case float64:
		return int(x)
	case string:
		if x == "" {
			return fallback
		}
		var parsed int
		if _, err := fmt.Sscanf(x, "%d", &parsed); err == nil {
			return parsed
		}
		return fallback
	default:
		return fallback
	}
}

func stageDisplayName(stage string) string {
	text := stage
	switch text {
	case "validate_input":
		return "参数校验"
	case "load_csv":
		return "读取数据"
	case "train_model":
		return "模型训练"
	case "evaluate_metrics":
		return "评估指标"
	case "save_artifacts":
		return "保存产物"
	case "scan_columns":
		return "扫描列异常"
	case "build_summary":
		return "整理结果"
	case "load_model":
		return "加载模型"
	case "repair_search":
		return "修复搜索"
	case "apply_repairs":
		return "应用修复"
	case "write_output":
		return "写出结果"
	case "complete":
		return "完成"
	default:
		if text == "" {
			return "未知阶段"
		}
		return text
	}
}

func (s *Service) RunTask(req engine.Request, timeout time.Duration) (string, error) {
	if s.runner == nil {
		return "", fmt.Errorf("runner is nil")
	}

	taskID := req.TaskID
	if taskID == "" {
		taskID = s.nextTaskID()
		req.TaskID = taskID
	}

	ctx, cancel := createTaskContext(timeout)
	now := time.Now()
	task := &Task{
		ID:        taskID,
		Status:    StatusPending,
		Request:   req,
		Progress:  newTaskProgressSnapshot(),
		CreatedAt: now,
		TimeoutMS: int64(timeout / time.Millisecond),
	}

	s.mu.Lock()
	if _, exists := s.tasks[taskID]; exists {
		s.mu.Unlock()
		cancel()
		return "", fmt.Errorf("task id already exists: %s", taskID)
	}
	s.tasks[taskID] = task
	s.runtime[taskID] = runtimeInfo{
		ctx:    ctx,
		cancel: cancel,
	}
	s.mu.Unlock()
	submittedSnapshot := *task

	select {
	case s.queue <- taskID:
		s.persistTask(submittedSnapshot)
		observability.Info("task_submitted", map[string]any{
			"task_id":    taskID,
			"action":     req.Action,
			"status":     submittedSnapshot.Status,
			"timeout_ms": submittedSnapshot.TimeoutMS,
		})
		return taskID, nil
	default:
		s.mu.Lock()
		delete(s.tasks, taskID)
		delete(s.runtime, taskID)
		s.mu.Unlock()
		cancel()
		observability.Warn("task_submit_failed", map[string]any{
			"task_id": taskID,
			"action":  req.Action,
			"reason":  "task queue is full",
		})
		return "", fmt.Errorf("task queue is full")
	}
}

func (s *Service) nextTaskID() string {
	seq := atomic.AddUint64(&s.seq, 1)
	return fmt.Sprintf("task-%d-%d", time.Now().UnixNano(), seq)
}

func (s *Service) execute(taskID string) {
	s.mu.Lock()
	task := s.tasks[taskID]
	rt, ok := s.runtime[taskID]
	if task == nil || !ok {
		s.mu.Unlock()
		return
	}
	if task.Status == StatusCanceled {
		s.cleanupRuntimeLocked(taskID)
		s.mu.Unlock()
		return
	}
	task.Status = StatusRunning
	task.StartedAt = time.Now()
	if task.Progress.StageStartedAtMS == nil {
		task.Progress = newTaskProgressSnapshot()
	}
	task.Progress.CurrentStage = "任务执行"
	if task.Progress.ProgressPercent < 5 {
		task.Progress.ProgressPercent = 5
	}
	task.Progress.LastMessage = "任务已进入执行阶段"
	task.Progress.UpdatedAtMS = task.StartedAt.UnixMilli()
	runningSnapshot := *task
	req := task.Request
	ctx := rt.ctx
	s.mu.Unlock()
	s.persistTask(runningSnapshot)
	observability.Info("task_started", map[string]any{
		"task_id": taskID,
		"action":  req.Action,
		"status":  runningSnapshot.Status,
	})

	resp, err := s.runner.Run(ctx, req)
	endedAt := time.Now()

	s.mu.Lock()
	defer s.mu.Unlock()

	stored := s.tasks[taskID]
	if stored == nil {
		s.cleanupRuntimeLocked(taskID)
		return
	}
	if isTerminalStatus(stored.Status) {
		if stored.EndedAt.IsZero() {
			stored.EndedAt = endedAt
		}
		s.cleanupRuntimeLocked(taskID)
		return
	}

	stored.EndedAt = endedAt

	switch {
	case errors.Is(ctx.Err(), context.Canceled) || errors.Is(err, context.Canceled):
		stored.Status = StatusCanceled
		if stored.Error == "" {
			stored.Error = "canceled"
		}
		s.finalizeTaskProgressLocked(stored)
		s.persistTask(*stored)
		observability.Warn("task_finished", map[string]any{
			"task_id": taskID,
			"status":  stored.Status,
			"error":   stored.Error,
		})
		s.cleanupRuntimeLocked(taskID)
		return
	case errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded):
		stored.Status = StatusTimedOut
		if stored.Error == "" {
			stored.Error = "timeout"
		}
		s.finalizeTaskProgressLocked(stored)
		s.persistTask(*stored)
		observability.Warn("task_finished", map[string]any{
			"task_id": taskID,
			"status":  stored.Status,
			"error":   stored.Error,
		})
		s.cleanupRuntimeLocked(taskID)
		return
	case err != nil:
		stored.Status = StatusFailed
		stored.Error = err.Error()
		s.finalizeTaskProgressLocked(stored)
		s.persistTask(*stored)
		observability.Error("task_finished", map[string]any{
			"task_id": taskID,
			"status":  stored.Status,
			"error":   stored.Error,
		})
		s.cleanupRuntimeLocked(taskID)
		return
	}

	stored.Response = resp
	if resp.Status == "ok" {
		stored.Status = StatusSucceeded
	} else {
		stored.Status = StatusFailed
		if resp.Error != nil {
			stored.Error = resp.Error.Code + ": " + resp.Error.Message
		}
	}
	s.finalizeTaskProgressLocked(stored)
	if stored.Response.Result == nil {
		stored.Response.Result = map[string]any{}
	}
	stored.Response.Result["observability"] = map[string]any{
		"current_stage":      stored.Progress.CurrentStage,
		"progress_percent":   stored.Progress.ProgressPercent,
		"stage_durations_ms": stored.Progress.StageDurationsMS,
		"bottleneck_stage":   stored.Progress.BottleneckStage,
		"bottleneck_ms":      stored.Progress.BottleneckMS,
		"failure":            stored.Progress.Failure,
		"last_message":       stored.Progress.LastMessage,
	}
	s.persistTask(*stored)
	observability.Info("task_finished", map[string]any{
		"task_id": taskID,
		"status":  stored.Status,
		"error":   stored.Error,
	})
	s.cleanupRuntimeLocked(taskID)
}

func (s *Service) onRunnerStderr(event engine.StderrEvent) {
	if s == nil || strings.TrimSpace(event.TaskID) == "" {
		return
	}

	parsed := event.Parsed
	if len(parsed) == 0 {
		return
	}

	eventName := strings.TrimSpace(toStringAny(parsed["event"]))
	if eventName == "" {
		return
	}

	stage := strings.TrimSpace(toStringAny(parsed["stage"]))
	phase := strings.ToLower(strings.TrimSpace(toStringAny(parsed["phase"])))
	message := strings.TrimSpace(toStringAny(parsed["message"]))
	progress := toIntAny(parsed["progress"], -1)
	if progress < 0 {
		progress = toIntAny(parsed["progress_percent"], -1)
	}

	if eventName == "engine_request_received" {
		stage = "validate_input"
		phase = "start"
		if progress < 0 {
			progress = 2
		}
	}
	if eventName == "engine_request_succeeded" {
		stage = "complete"
		phase = "complete"
		if progress < 0 {
			progress = 100
		}
	}
	if eventName == "engine_request_failed" || eventName == "engine_request_crashed" {
		if stage == "" {
			stage = "complete"
		}
		phase = "error"
		if progress < 0 {
			progress = 100
		}
	}

	if eventName != "stage_progress" && !strings.HasPrefix(eventName, "engine_request_") {
		return
	}

	atMS := event.ObservedAt.UnixMilli()
	if tsText := strings.TrimSpace(toStringAny(parsed["timestamp"])); tsText != "" {
		if ts, err := time.Parse(time.RFC3339Nano, tsText); err == nil {
			atMS = ts.UnixMilli()
		}
	}
	if stage == "" {
		stage = "unknown"
	}
	if phase == "" {
		phase = "info"
	}
	stageLabel := stageDisplayName(stage)
	file := strings.TrimSpace(toStringAny(parsed["file"]))
	if file == "" {
		file = strings.TrimSpace(toStringAny(parsed["csv_path"]))
	}
	column := strings.TrimSpace(toStringAny(parsed["column"]))
	if column == "" {
		column = strings.TrimSpace(toStringAny(parsed["target_col"]))
	}
	rule := strings.TrimSpace(toStringAny(parsed["rule"]))
	if rule == "" {
		rule = strings.TrimSpace(toStringAny(parsed["rule_name"]))
	}
	errorCode := strings.TrimSpace(toStringAny(parsed["error_code"]))
	if errorCode == "" {
		errorCode = strings.TrimSpace(toStringAny(parsed["code"]))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	task, ok := s.tasks[event.TaskID]
	if !ok || task == nil {
		return
	}
	if task.Progress.StageStartedAtMS == nil {
		task.Progress = newTaskProgressSnapshot()
	}

	if progress >= 0 {
		if progress > 100 {
			progress = 100
		}
		if progress > task.Progress.ProgressPercent {
			task.Progress.ProgressPercent = progress
		}
	}

	if phase == "start" {
		task.Progress.CurrentStage = stageLabel
		task.Progress.StageStartedAtMS[stage] = atMS
	}
	if phase == "complete" || phase == "done" || phase == "success" || phase == "error" {
		if startedMS, exists := task.Progress.StageStartedAtMS[stage]; exists && startedMS > 0 {
			duration := atMS - startedMS
			if duration < 0 {
				duration = 0
			}
			task.Progress.StageDurationsMS[stage] = duration
			if duration > task.Progress.BottleneckMS {
				task.Progress.BottleneckMS = duration
				task.Progress.BottleneckStage = stageLabel
			}
			delete(task.Progress.StageStartedAtMS, stage)
		}
	}

	if phase == "error" {
		task.Progress.Failure = &TaskFailureLocation{
			Stage:     stageLabel,
			Message:   message,
			File:      file,
			Column:    column,
			Rule:      rule,
			ErrorCode: errorCode,
		}
	}

	if message == "" {
		message = stageLabel
	}
	task.Progress.LastMessage = message
	task.Progress.UpdatedAtMS = atMS
	task.Progress.Meta["event"] = eventName

	progressEvent := TaskProgressEvent{
		AtMS:      atMS,
		Stage:     stageLabel,
		Phase:     phase,
		Progress:  max(progress, 0),
		Message:   message,
		File:      file,
		Column:    column,
		Rule:      rule,
		ErrorCode: errorCode,
	}
	task.Progress.Events = append(task.Progress.Events, progressEvent)
	if len(task.Progress.Events) > maxProgressEvents {
		task.Progress.Events = task.Progress.Events[len(task.Progress.Events)-maxProgressEvents:]
	}
}

func (s *Service) finalizeTaskProgressLocked(task *Task) {
	if task == nil {
		return
	}
	if task.Progress.StageStartedAtMS == nil {
		task.Progress = newTaskProgressSnapshot()
	}

	endMS := time.Now().UnixMilli()
	if !task.EndedAt.IsZero() {
		endMS = task.EndedAt.UnixMilli()
	}

	for stage, startedMS := range task.Progress.StageStartedAtMS {
		if startedMS <= 0 {
			continue
		}
		duration := endMS - startedMS
		if duration < 0 {
			duration = 0
		}
		if duration > task.Progress.StageDurationsMS[stage] {
			task.Progress.StageDurationsMS[stage] = duration
		}
	}
	task.Progress.StageStartedAtMS = map[string]int64{}

	var maxStage string
	var maxDuration int64
	for stage, duration := range task.Progress.StageDurationsMS {
		if duration > maxDuration {
			maxDuration = duration
			maxStage = stageDisplayName(stage)
		}
	}
	task.Progress.BottleneckMS = maxDuration
	task.Progress.BottleneckStage = maxStage

	if task.Progress.ProgressPercent < 100 {
		task.Progress.ProgressPercent = 100
	}
	if task.Progress.CurrentStage == "" {
		task.Progress.CurrentStage = stageDisplayName("complete")
	}
	if task.Progress.LastMessage == "" {
		if task.Status == StatusSucceeded {
			task.Progress.LastMessage = "任务完成"
		} else {
			task.Progress.LastMessage = "任务结束"
		}
	}
	if task.Progress.Failure == nil && task.Status != StatusSucceeded {
		task.Progress.Failure = &TaskFailureLocation{
			Stage:   task.Progress.CurrentStage,
			Message: task.Error,
		}
	}
	task.Progress.UpdatedAtMS = endMS
}

func hydrateTaskProgressFromResponse(task *Task) {
	if task == nil {
		return
	}
	if task.Progress.StageDurationsMS != nil && (task.Progress.ProgressPercent > 0 || len(task.Progress.Events) > 0) {
		return
	}
	result := task.Response.Result
	if result == nil {
		if task.Progress.StageDurationsMS == nil {
			task.Progress = newTaskProgressSnapshot()
		}
		return
	}
	rawObs, ok := result["observability"]
	if !ok {
		if task.Progress.StageDurationsMS == nil {
			task.Progress = newTaskProgressSnapshot()
		}
		return
	}
	obs, ok := rawObs.(map[string]any)
	if !ok {
		if task.Progress.StageDurationsMS == nil {
			task.Progress = newTaskProgressSnapshot()
		}
		return
	}

	task.Progress = newTaskProgressSnapshot()
	task.Progress.CurrentStage = strings.TrimSpace(toStringAny(obs["current_stage"]))
	task.Progress.ProgressPercent = toIntAny(obs["progress_percent"], 0)
	task.Progress.LastMessage = strings.TrimSpace(toStringAny(obs["last_message"]))
	task.Progress.BottleneckStage = strings.TrimSpace(toStringAny(obs["bottleneck_stage"]))
	task.Progress.BottleneckMS = int64(toIntAny(obs["bottleneck_ms"], 0))

	if rawDurations, ok := obs["stage_durations_ms"].(map[string]any); ok {
		for key, value := range rawDurations {
			stage := strings.TrimSpace(key)
			if stage == "" {
				continue
			}
			task.Progress.StageDurationsMS[stage] = int64(toIntAny(value, 0))
		}
	}
	if rawFailure, ok := obs["failure"].(map[string]any); ok {
		task.Progress.Failure = &TaskFailureLocation{
			Stage:     strings.TrimSpace(toStringAny(rawFailure["stage"])),
			Message:   strings.TrimSpace(toStringAny(rawFailure["message"])),
			File:      strings.TrimSpace(toStringAny(rawFailure["file"])),
			Column:    strings.TrimSpace(toStringAny(rawFailure["column"])),
			Rule:      strings.TrimSpace(toStringAny(rawFailure["rule"])),
			ErrorCode: strings.TrimSpace(toStringAny(rawFailure["error_code"])),
		}
	}
	if task.Progress.ProgressPercent <= 0 && isTerminalStatus(task.Status) {
		task.Progress.ProgressPercent = 100
	}
	task.Progress.UpdatedAtMS = time.Now().UnixMilli()
}

func (s *Service) cleanupRuntimeLocked(taskID string) {
	rt, ok := s.runtime[taskID]
	if !ok {
		return
	}
	delete(s.runtime, taskID)
	rt.cancel()
}

func (s *Service) CancelTask(taskID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	task, ok := s.tasks[taskID]
	if !ok {
		return false
	}
	if isTerminalStatus(task.Status) {
		return false
	}

	rt, ok := s.runtime[taskID]
	if !ok {
		return false
	}

	switch task.Status {
	case StatusPending:
		task.Status = StatusCanceled
		task.Error = "canceled"
		task.EndedAt = time.Now()
		s.finalizeTaskProgressLocked(task)
		delete(s.runtime, taskID)
		rt.cancel()
		s.persistTask(*task)
		observability.Warn("task_canceled", map[string]any{
			"task_id": taskID,
			"status":  task.Status,
		})
	case StatusRunning:
		task.Error = "canceled"
		task.Progress.LastMessage = "任务已收到取消请求"
		task.Progress.UpdatedAtMS = time.Now().UnixMilli()
		rt.cancel()
		observability.Warn("task_cancel_requested", map[string]any{
			"task_id": taskID,
			"status":  task.Status,
		})
	default:
		return false
	}

	return true
}

func (s *Service) GetTaskStatus(taskID string) (*Task, bool) {
	s.mu.RLock()
	task, ok := s.tasks[taskID]
	if ok {
		copyTask := *task
		hydrateTaskProgressFromResponse(&copyTask)
		s.mu.RUnlock()
		return &copyTask, true
	}
	s.mu.RUnlock()

	if s.history == nil {
		return nil, false
	}
	hTask, ok, err := s.history.GetTask(context.Background(), taskID)
	if err != nil {
		observability.Warn("task_history_lookup_failed", map[string]any{
			"task_id": taskID,
			"error":   err.Error(),
		})
		return nil, false
	}
	hydrateTaskProgressFromResponse(hTask)
	return hTask, ok
}

func (s *Service) List() []Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Task, 0, len(s.tasks))
	for _, task := range s.tasks {
		copyTask := *task
		hydrateTaskProgressFromResponse(&copyTask)
		out = append(out, copyTask)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

func (s *Service) ListRecentTasks(limit int) ([]Task, error) {
	if s.history != nil {
		items, err := s.history.ListRecentTasks(context.Background(), limit)
		if err != nil {
			return nil, err
		}
		for i := range items {
			hydrateTaskProgressFromResponse(&items[i])
		}
		return items, nil
	}

	items := s.List()
	if limit <= 0 || limit >= len(items) {
		return items, nil
	}
	return items[:limit], nil
}

func (s *Service) persistTask(task Task) {
	if s.history == nil {
		return
	}
	if err := s.history.SaveTask(context.Background(), task); err != nil {
		observability.Warn("task_history_save_failed", map[string]any{
			"task_id": task.ID,
			"status":  task.Status,
			"error":   err.Error(),
		})
	}
}

// Start is kept for backward compatibility.
func (s *Service) Start(req engine.Request, timeout time.Duration) (string, error) {
	return s.RunTask(req, timeout)
}

// Cancel is kept for backward compatibility.
func (s *Service) Cancel(taskID string) bool {
	return s.CancelTask(taskID)
}

// Get is kept for backward compatibility.
func (s *Service) Get(taskID string) (*Task, bool) {
	return s.GetTaskStatus(taskID)
}
