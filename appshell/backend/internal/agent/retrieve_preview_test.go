package agent

import (
	"context"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"appshell/backend/internal/engine"
)

type retrievePreviewTestRunner struct {
	ruleDelay         time.Duration
	gowerDelay        time.Duration
	previewRunning    int32
	maxPreviewRunning int32
	observer          engine.StderrObserver
	mu                sync.Mutex
	calls             []engine.Request
}

func (r *retrievePreviewTestRunner) SetStderrObserver(observer engine.StderrObserver) {
	r.observer = observer
}

func (r *retrievePreviewTestRunner) Run(ctx context.Context, req engine.Request) (engine.Response, error) {
	r.mu.Lock()
	r.calls = append(r.calls, req)
	r.mu.Unlock()

	switch req.Action {
	case string(engine.ActionScanFile):
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"issue_count": 2,
				"issues": []any{
					map[string]any{"issue_id": "issue-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.7},
					map[string]any{"issue_id": "issue-2", "issue_type": "rare_category", "column": "city", "risk_level": "medium", "issue_score": 0.3},
				},
				"scan_summary": map[string]any{"total_issues": 2},
				"data_profile": map[string]any{"rows": 10, "columns": 2},
			},
		}, nil
	case string(engine.ActionRepairBatch):
		return r.previewResponse(ctx, req.TaskID, r.ruleDelay, map[string]any{
			"comparison": map[string]any{
				"before_issue_count":   2,
				"after_issue_count":    1,
				"resolved_issue_count": 1,
				"changed_cell_count":   1,
			},
			"applied_repairs": []any{
				map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.72},
			},
		})
	case string(engine.ActionRepairWithGower):
		return r.previewResponse(ctx, req.TaskID, r.gowerDelay, map[string]any{
			"comparison": map[string]any{
				"before_issue_count":   2,
				"after_issue_count":    0,
				"resolved_issue_count": 2,
				"changed_cell_count":   2,
			},
			"applied_repairs": []any{
				map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.91},
				map[string]any{"issue_id": "issue-2", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.88},
			},
			"neighbor_evidence": []any{
				map[string]any{"issue_id": "issue-1", "candidate_confidence": 0.91},
				map[string]any{"issue_id": "issue-2", "candidate_confidence": 0.88},
			},
		})
	default:
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{"action": req.Action},
		}, nil
	}
}

func (r *retrievePreviewTestRunner) previewResponse(ctx context.Context, taskID string, delay time.Duration, result map[string]any) (engine.Response, error) {
	current := atomic.AddInt32(&r.previewRunning, 1)
	defer atomic.AddInt32(&r.previewRunning, -1)
	for {
		maxObserved := atomic.LoadInt32(&r.maxPreviewRunning)
		if current <= maxObserved || atomic.CompareAndSwapInt32(&r.maxPreviewRunning, maxObserved, current) {
			break
		}
	}

	select {
	case <-time.After(delay):
	case <-ctx.Done():
		return engine.Response{}, ctx.Err()
	}

	return engine.Response{
		TaskID: taskID,
		Status: "ok",
		Result: result,
	}, nil
}

func TestRetrievePreviewParallelModeReducesWallClockAndOverlapsCalls(t *testing.T) {
	seqElapsed, seqMax := runPlanForRetrieveMode(t, RetrieveModeSequential)
	parElapsed, parMax := runPlanForRetrieveMode(t, RetrieveModeParallel)

	if seqMax != 1 {
		t.Fatalf("expected sequential retrieve preview max concurrency = 1, got %d", seqMax)
	}
	if parMax < 2 {
		t.Fatalf("expected parallel retrieve preview max concurrency >= 2, got %d", parMax)
	}
	if parElapsed >= seqElapsed {
		t.Fatalf("expected parallel retrieve preview to be faster than sequential, got parallel=%s sequential=%s", parElapsed, seqElapsed)
	}
	if seqElapsed-parElapsed < 120*time.Millisecond {
		t.Fatalf("expected parallel retrieve preview to save at least 120ms, got parallel=%s sequential=%s", parElapsed, seqElapsed)
	}
}

func runPlanForRetrieveMode(t *testing.T, mode string) (time.Duration, int32) {
	t.Helper()
	t.Setenv("APPSHELL_AGENT_RETRIEVE_MODE", mode)

	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	base := &retrievePreviewTestRunner{
		ruleDelay:  220 * time.Millisecond,
		gowerDelay: 220 * time.Millisecond,
	}
	runner := NewRuntimeRunner(base, store, NewMockPlanner())

	started := time.Now()
	resp, err := runner.Run(t.Context(), engine.Request{
		TaskID: "task-plan-" + mode,
		Action: ActionSessionPlan,
		Payload: map[string]any{
			"csv_path": "demo.csv",
		},
	})
	elapsed := time.Since(started)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if resp.Status != "ok" {
		t.Fatalf("expected ok response, got %s", resp.Status)
	}
	return elapsed, atomic.LoadInt32(&base.maxPreviewRunning)
}
