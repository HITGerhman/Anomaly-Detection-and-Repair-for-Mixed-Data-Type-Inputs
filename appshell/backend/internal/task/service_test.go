package task

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"appshell/backend/internal/engine"
)

type fakeRunner struct {
	delay         time.Duration
	blockUntilCtx bool
	running       int32
	maxRunning    int32
}

func (r *fakeRunner) Run(ctx context.Context, req engine.Request) (engine.Response, error) {
	current := atomic.AddInt32(&r.running, 1)
	defer atomic.AddInt32(&r.running, -1)

	for {
		max := atomic.LoadInt32(&r.maxRunning)
		if current <= max || atomic.CompareAndSwapInt32(&r.maxRunning, max, current) {
			break
		}
	}

	if r.blockUntilCtx {
		<-ctx.Done()
		return engine.Response{}, ctx.Err()
	}

	if r.delay > 0 {
		select {
		case <-time.After(r.delay):
		case <-ctx.Done():
			return engine.Response{}, ctx.Err()
		}
	}

	return engine.Response{
		TaskID: req.TaskID,
		Status: "ok",
		Result: map[string]any{"ok": true},
	}, nil
}

func waitForStatus(t *testing.T, svc *Service, taskID string, timeout time.Duration, expected ...string) *Task {
	t.Helper()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		task, ok := svc.GetTaskStatus(taskID)
		if ok {
			for _, status := range expected {
				if task.Status == status {
					return task
				}
			}
		}
		time.Sleep(20 * time.Millisecond)
	}

	task, _ := svc.GetTaskStatus(taskID)
	if task == nil {
		t.Fatalf("task not found: %s", taskID)
	}
	t.Fatalf("task %s status=%s, expected one of %v", taskID, task.Status, expected)
	return nil
}

func TestRunTaskSupportsAtLeastThreeConcurrentTasks(t *testing.T) {
	runner := &fakeRunner{delay: 300 * time.Millisecond}
	svc := NewServiceWithConfig(runner, Config{
		MaxConcurrency: 3,
		QueueSize:      32,
	})
	defer svc.Close()

	taskIDs := make([]string, 0, 6)
	for i := 0; i < 6; i++ {
		taskID, err := svc.RunTask(engine.Request{
			Action:  "health",
			Payload: map[string]any{},
		}, 3*time.Second)
		if err != nil {
			t.Fatalf("RunTask failed: %v", err)
		}
		taskIDs = append(taskIDs, taskID)
	}

	for _, taskID := range taskIDs {
		waitForStatus(t, svc, taskID, 5*time.Second, StatusSucceeded)
	}

	if got := atomic.LoadInt32(&runner.maxRunning); got < 3 {
		t.Fatalf("expected max concurrent running >= 3, got %d", got)
	}
}

func TestCancelTaskTransitionsToCanceledWithinTwoSeconds(t *testing.T) {
	runner := &fakeRunner{blockUntilCtx: true}
	svc := NewServiceWithConfig(runner, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
	})
	defer svc.Close()

	taskID, err := svc.RunTask(engine.Request{
		Action:  "train",
		Payload: map[string]any{"x": 1},
	}, 10*time.Second)
	if err != nil {
		t.Fatalf("RunTask failed: %v", err)
	}

	waitForStatus(t, svc, taskID, 1500*time.Millisecond, StatusRunning)

	start := time.Now()
	if ok := svc.CancelTask(taskID); !ok {
		t.Fatalf("CancelTask returned false")
	}

	waitForStatus(t, svc, taskID, 2*time.Second, StatusCanceled)
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("canceled transition exceeded 2s: %s", elapsed)
	}
}

func TestTimedOutTaskTransitionsToTimedOutAndCanBeRecycled(t *testing.T) {
	runner := &fakeRunner{delay: 700 * time.Millisecond}
	svc := NewServiceWithConfig(runner, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
	})
	defer svc.Close()

	taskID, err := svc.RunTask(engine.Request{
		Action:  "train",
		Payload: map[string]any{"x": 1},
	}, 250*time.Millisecond)
	if err != nil {
		t.Fatalf("RunTask failed: %v", err)
	}

	waitForStatus(t, svc, taskID, 2*time.Second, StatusTimedOut)

	nextTaskID, err := svc.RunTask(engine.Request{
		Action:  "health",
		Payload: map[string]any{},
	}, 2*time.Second)
	if err != nil {
		t.Fatalf("RunTask after timeout failed: %v", err)
	}
	waitForStatus(t, svc, nextTaskID, 2*time.Second, StatusSucceeded)
}
