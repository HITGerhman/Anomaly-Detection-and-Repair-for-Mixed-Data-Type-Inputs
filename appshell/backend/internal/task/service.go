package task

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"appshell/backend/internal/engine"
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
	CreatedAt time.Time       `json:"created_at"`
	StartedAt time.Time       `json:"started_at"`
	EndedAt   time.Time       `json:"ended_at"`
	TimeoutMS int64           `json:"timeout_ms"`
}

type Runner interface {
	Run(ctx context.Context, req engine.Request) (engine.Response, error)
}

type Config struct {
	MaxConcurrency int
	QueueSize      int
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
	runner Runner
	cfg    Config

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
		tasks:   make(map[string]*Task),
		runtime: make(map[string]runtimeInfo),
		queue:   make(chan string, cfg.QueueSize),
		stopCh:  make(chan struct{}),
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

	select {
	case s.queue <- taskID:
		return taskID, nil
	default:
		s.mu.Lock()
		delete(s.tasks, taskID)
		delete(s.runtime, taskID)
		s.mu.Unlock()
		cancel()
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
	req := task.Request
	ctx := rt.ctx
	s.mu.Unlock()

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
		s.cleanupRuntimeLocked(taskID)
		return
	case errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded):
		stored.Status = StatusTimedOut
		if stored.Error == "" {
			stored.Error = "timeout"
		}
		s.cleanupRuntimeLocked(taskID)
		return
	case err != nil:
		stored.Status = StatusFailed
		stored.Error = err.Error()
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
	s.cleanupRuntimeLocked(taskID)
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
		delete(s.runtime, taskID)
		rt.cancel()
	case StatusRunning:
		task.Error = "canceled"
		rt.cancel()
	default:
		return false
	}

	return true
}

func (s *Service) GetTaskStatus(taskID string) (*Task, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	task, ok := s.tasks[taskID]
	if !ok {
		return nil, false
	}

	copyTask := *task
	return &copyTask, true
}

func (s *Service) List() []Task {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Task, 0, len(s.tasks))
	for _, task := range s.tasks {
		out = append(out, *task)
	}
	return out
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
