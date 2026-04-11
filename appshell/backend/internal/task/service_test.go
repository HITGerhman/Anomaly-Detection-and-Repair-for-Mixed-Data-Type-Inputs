package task

import (
	"context"
	"fmt"
	"path/filepath"
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

type progressRunner struct {
	observer engine.StderrObserver
	fail     bool
	stage    string
}

func (r *progressRunner) SetStderrObserver(observer engine.StderrObserver) {
	r.observer = observer
}

func (r *progressRunner) Run(ctx context.Context, req engine.Request) (engine.Response, error) {
	stageName := r.stage
	if stageName == "" {
		stageName = "load_csv"
	}
	if r.observer != nil {
		now := time.Now()
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     stageName,
				"phase":     "start",
				"progress":  12,
				"message":   "开始读取文件",
				"timestamp": now.Format(time.RFC3339Nano),
				"file":      "demo.csv",
			},
			ObservedAt: now,
		})
		time.Sleep(20 * time.Millisecond)
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     stageName,
				"phase":     "complete",
				"progress":  48,
				"message":   "读取完成",
				"timestamp": time.Now().Format(time.RFC3339Nano),
				"file":      "demo.csv",
			},
			ObservedAt: time.Now(),
		})
	}

	if r.fail {
		if r.observer != nil {
			r.observer(engine.StderrEvent{
				TaskID: req.TaskID,
				Parsed: map[string]any{
					"event":      "stage_progress",
					"stage":      "apply_repairs",
					"phase":      "error",
					"progress":   100,
					"message":    "规则冲突",
					"file":       "demo.csv",
					"column":     "age",
					"rule":       "age<=retire_age",
					"error_code": "REPAIR_BATCH_FAILED",
					"timestamp":  time.Now().Format(time.RFC3339Nano),
				},
				ObservedAt: time.Now(),
			})
		}
		return engine.Response{}, fmt.Errorf("simulated runner failure")
	}

	if r.observer != nil {
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     "complete",
				"phase":     "complete",
				"progress":  100,
				"message":   "任务完成",
				"timestamp": time.Now().Format(time.RFC3339Nano),
			},
			ObservedAt: time.Now(),
		})
	}
	return engine.Response{
		TaskID: req.TaskID,
		Status: "ok",
		Result: map[string]any{"ok": true},
	}, nil
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

func TestTaskHistoryCanBeLoadedAfterServiceRestart(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "task_history.sqlite")

	history1, err := NewSQLiteHistoryStore(dbPath, 100)
	if err != nil {
		t.Fatalf("create history store failed: %v", err)
	}
	svc1 := NewServiceWithConfig(&fakeRunner{}, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
		HistoryStore:   history1,
	})

	taskID, err := svc1.RunTask(engine.Request{
		Action:  "health",
		Payload: map[string]any{},
	}, 2*time.Second)
	if err != nil {
		t.Fatalf("RunTask failed: %v", err)
	}
	waitForStatus(t, svc1, taskID, 2*time.Second, StatusSucceeded)
	svc1.Close()

	history2, err := NewSQLiteHistoryStore(dbPath, 100)
	if err != nil {
		t.Fatalf("reopen history store failed: %v", err)
	}
	svc2 := NewServiceWithConfig(&fakeRunner{}, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
		HistoryStore:   history2,
	})
	defer svc2.Close()

	taskSnapshot, ok := svc2.GetTaskStatus(taskID)
	if !ok {
		t.Fatalf("expected task %s to be found in persisted history", taskID)
	}
	if taskSnapshot.Status != StatusSucceeded {
		t.Fatalf("expected persisted status=%s got=%s", StatusSucceeded, taskSnapshot.Status)
	}
}

func TestTaskHistoryKeepsOnlyRecentNRecords(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "task_history.sqlite")
	history, err := NewSQLiteHistoryStore(dbPath, 2)
	if err != nil {
		t.Fatalf("create history store failed: %v", err)
	}

	svc := NewServiceWithConfig(&fakeRunner{}, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
		HistoryStore:   history,
	})
	defer svc.Close()

	taskIDs := make([]string, 0, 3)
	for i := 0; i < 3; i++ {
		taskID, err := svc.RunTask(engine.Request{
			Action:  "health",
			Payload: map[string]any{"index": i},
		}, 2*time.Second)
		if err != nil {
			t.Fatalf("RunTask failed: %v", err)
		}
		taskIDs = append(taskIDs, taskID)
		waitForStatus(t, svc, taskID, 2*time.Second, StatusSucceeded)
	}

	items, err := svc.ListRecentTasks(10)
	if err != nil {
		t.Fatalf("ListRecentTasks failed: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 records after trim, got %d", len(items))
	}

	got := map[string]bool{}
	for _, item := range items {
		got[item.ID] = true
	}
	if got[taskIDs[0]] {
		t.Fatalf("oldest task should have been trimmed: %s", taskIDs[0])
	}
	if !got[taskIDs[1]] || !got[taskIDs[2]] {
		t.Fatalf("newest tasks should remain after trim")
	}
}

func TestTaskProgressStreamIsCapturedFromRunnerEvents(t *testing.T) {
	runner := &progressRunner{}
	svc := NewServiceWithConfig(runner, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
	})
	defer svc.Close()

	taskID, err := svc.RunTask(engine.Request{
		Action:  "scan_file",
		Payload: map[string]any{"csv_path": "demo.csv"},
	}, 2*time.Second)
	if err != nil {
		t.Fatalf("RunTask failed: %v", err)
	}
	waitForStatus(t, svc, taskID, 2*time.Second, StatusSucceeded)

	task, ok := svc.GetTaskStatus(taskID)
	if !ok {
		t.Fatalf("task not found: %s", taskID)
	}
	if task.Progress.ProgressPercent != 100 {
		t.Fatalf("expected progress 100, got %d", task.Progress.ProgressPercent)
	}
	if len(task.Progress.Events) < 2 {
		t.Fatalf("expected >=2 progress events, got %d", len(task.Progress.Events))
	}
	if task.Progress.BottleneckStage == "" {
		t.Fatalf("expected bottleneck stage to be computed")
	}
	if len(task.Progress.StageDurationsMS) == 0 {
		t.Fatalf("expected stage durations to be present")
	}
	obs := task.Response.Result["observability"]
	if obs == nil {
		t.Fatalf("expected observability summary in response result")
	}
}

func TestTaskFailureLocationCapturedFromRunnerEvents(t *testing.T) {
	runner := &progressRunner{fail: true}
	svc := NewServiceWithConfig(runner, Config{
		MaxConcurrency: 1,
		QueueSize:      8,
	})
	defer svc.Close()

	taskID, err := svc.RunTask(engine.Request{
		Action:  "repair_batch",
		Payload: map[string]any{"csv_path": "demo.csv"},
	}, 2*time.Second)
	if err != nil {
		t.Fatalf("RunTask failed: %v", err)
	}
	waitForStatus(t, svc, taskID, 2*time.Second, StatusFailed)

	task, ok := svc.GetTaskStatus(taskID)
	if !ok {
		t.Fatalf("task not found: %s", taskID)
	}
	if task.Progress.Failure == nil {
		t.Fatalf("expected failure location from stage events")
	}
	if task.Progress.Failure.Column != "age" {
		t.Fatalf("expected failure column=age, got %s", task.Progress.Failure.Column)
	}
	if task.Progress.Failure.Rule == "" {
		t.Fatalf("expected failure rule to be captured")
	}
}

func TestAgentStageDisplayNameCapturedFromRunnerEvents(t *testing.T) {
	cases := []struct {
		stage string
		want  string
	}{
		{stage: "agent_plan", want: "Agent Plan"},
		{stage: "agent_rescan", want: "Agent Rescan"},
		{stage: "agent_post_validate", want: "Agent Post Validate"},
		{stage: "agent_rollback", want: "Agent Rollback"},
	}

	for _, tc := range cases {
		t.Run(tc.stage, func(t *testing.T) {
			runner := &progressRunner{stage: tc.stage}
			svc := NewServiceWithConfig(runner, Config{
				MaxConcurrency: 1,
				QueueSize:      8,
			})
			defer svc.Close()

			taskID, err := svc.RunTask(engine.Request{
				Action:  "agent.session.plan",
				Payload: map[string]any{"csv_path": "demo.csv"},
			}, 2*time.Second)
			if err != nil {
				t.Fatalf("RunTask failed: %v", err)
			}
			waitForStatus(t, svc, taskID, 2*time.Second, StatusSucceeded)

			taskSnapshot, ok := svc.GetTaskStatus(taskID)
			if !ok {
				t.Fatalf("task not found: %s", taskID)
			}
			found := false
			for _, event := range taskSnapshot.Progress.Events {
				if event.Stage == tc.want {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("expected agent stage display name %q in progress events", tc.want)
			}
		})
	}
}
