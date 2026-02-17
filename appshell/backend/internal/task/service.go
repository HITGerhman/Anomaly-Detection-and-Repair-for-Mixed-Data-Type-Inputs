package task

import (
	"context"
	"fmt"
	"sync"
	"time"

	"appshell/backend/internal/engine"
)

const (
	StatusPending   = "pending"
	StatusRunning   = "running"
	StatusSucceeded = "succeeded"
	StatusFailed    = "failed"
	StatusCanceled  = "canceled"
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
}

type Service struct {
	runner  *engine.Runner
	mu      sync.RWMutex
	tasks   map[string]*Task
	cancels map[string]context.CancelFunc
}

func NewService(runner *engine.Runner) *Service {
	return &Service{
		runner:  runner,
		tasks:   make(map[string]*Task),
		cancels: make(map[string]context.CancelFunc),
	}
}

func (s *Service) Start(req engine.Request, timeout time.Duration) (string, error) {
	if s.runner == nil {
		return "", fmt.Errorf("runner is nil")
	}

	taskID := req.TaskID
	if taskID == "" {
		taskID = fmt.Sprintf("task-%d", time.Now().UnixNano())
		req.TaskID = taskID
	}

	ctx := context.Background()
	var cancel context.CancelFunc
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, timeout)
	} else {
		ctx, cancel = context.WithCancel(ctx)
	}

	t := &Task{
		ID:        taskID,
		Status:    StatusPending,
		Request:   req,
		CreatedAt: time.Now(),
	}

	s.mu.Lock()
	s.tasks[taskID] = t
	s.cancels[taskID] = cancel
	s.mu.Unlock()

	go func() {
		s.setRunning(taskID)
		resp, err := s.runner.Run(ctx, req)
		endedAt := time.Now()

		s.mu.Lock()
		defer s.mu.Unlock()
		defer delete(s.cancels, taskID)

		stored := s.tasks[taskID]
		if stored == nil {
			cancel()
			return
		}
		stored.EndedAt = endedAt

		if ctx.Err() != nil {
			if ctx.Err() == context.DeadlineExceeded {
				stored.Status = StatusFailed
				stored.Error = "timeout"
			} else {
				stored.Status = StatusCanceled
				stored.Error = "canceled"
			}
			cancel()
			return
		}

		if err != nil {
			stored.Status = StatusFailed
			stored.Error = err.Error()
			cancel()
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
		cancel()
	}()

	return taskID, nil
}

func (s *Service) setRunning(taskID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if t := s.tasks[taskID]; t != nil {
		t.Status = StatusRunning
		t.StartedAt = time.Now()
	}
}

func (s *Service) Cancel(taskID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	cancel, ok := s.cancels[taskID]
	if !ok {
		return false
	}
	cancel()
	return true
}

func (s *Service) Get(taskID string) (*Task, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	t, ok := s.tasks[taskID]
	if !ok {
		return nil, false
	}
	cpy := *t
	return &cpy, true
}

func (s *Service) List() []Task {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Task, 0, len(s.tasks))
	for _, t := range s.tasks {
		out = append(out, *t)
	}
	return out
}
