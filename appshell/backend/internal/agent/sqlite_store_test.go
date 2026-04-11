package agent

import (
	"fmt"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestSQLiteStoreRoundTrip(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	session := AgentSession{
		SessionID:     "session-1",
		RootTaskID:    "task-root",
		CurrentTaskID: "task-root",
		Status:        SessionStatusPlanned,
		Mode:          "plan",
		UserGoal:      DefaultUserGoal,
		Context: map[string]any{
			"csv_path": "demo.csv",
		},
		LatestPlan: AgentPlan{
			PlanID:           "plan-1",
			Status:           "planned",
			SelectedIssueIDs: []string{"i-1"},
			ProposedToolID:   "engine.repair_batch",
			ProposedPayload:  map[string]any{"csv_path": "demo.csv"},
		},
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}
	if err := store.SaveSession(t.Context(), session); err != nil {
		t.Fatalf("SaveSession failed: %v", err)
	}

	event, err := store.SaveTraceEvent(t.Context(), AgentTraceEvent{
		SessionID: "session-1",
		TaskID:    "task-root",
		AgentName: AgentSupervisor,
		TraceType: TraceSessionStarted,
		Summary:   "started",
		Payload:   map[string]any{"mode": "plan"},
	})
	if err != nil {
		t.Fatalf("SaveTraceEvent failed: %v", err)
	}
	if event.Seq != 1 {
		t.Fatalf("unexpected trace seq: %d", event.Seq)
	}

	loaded, ok, err := store.GetSession(t.Context(), "session-1")
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected session to exist")
	}
	if loaded.LatestPlan.PlanID != "plan-1" {
		t.Fatalf("unexpected plan id: %s", loaded.LatestPlan.PlanID)
	}

	trace, err := store.ListTrace(t.Context(), "session-1")
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	if len(trace) != 1 {
		t.Fatalf("expected 1 trace event, got %d", len(trace))
	}
	if trace[0].Summary != "started" {
		t.Fatalf("unexpected trace summary: %s", trace[0].Summary)
	}
}

func TestSQLiteStorePreferencesRoundTripAndReopen(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "agent.sqlite")
	store, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}

	record := AgentPreferenceRecord{
		WorkspaceID: "workspace-1",
		Profile: AgentPreferenceProfile{
			ConservativeMode:           true,
			AvoidTimeColumns:           false,
			ProtectedColumns:           []string{"id", "created_at", "id"},
			RequireApprovalForHighRisk: true,
		},
		UpdatedAt: time.Now().UTC(),
	}
	if err := store.SavePreferences(t.Context(), record); err != nil {
		t.Fatalf("SavePreferences failed: %v", err)
	}

	loaded, ok, err := store.GetPreferences(t.Context(), "workspace-1")
	if err != nil {
		t.Fatalf("GetPreferences failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected workspace preferences to exist")
	}
	if !loaded.Profile.ConservativeMode || loaded.Profile.AvoidTimeColumns {
		t.Fatalf("unexpected preference profile: %#v", loaded.Profile)
	}
	if len(loaded.Profile.ProtectedColumns) != 2 {
		t.Fatalf("expected protected columns to be normalized, got %#v", loaded.Profile.ProtectedColumns)
	}

	record.Profile = AgentPreferenceProfile{
		ConservativeMode:           false,
		AvoidTimeColumns:           true,
		ProtectedColumns:           []string{"event_time"},
		RequireApprovalForHighRisk: false,
	}
	record.UpdatedAt = time.Now().UTC().Add(time.Minute)
	if err := store.SavePreferences(t.Context(), record); err != nil {
		t.Fatalf("SavePreferences overwrite failed: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	reopened, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("reopen NewSQLiteStore failed: %v", err)
	}
	defer reopened.Close()

	reloaded, ok, err := reopened.GetPreferences(t.Context(), "workspace-1")
	if err != nil {
		t.Fatalf("reopened GetPreferences failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected reopened workspace preferences to exist")
	}
	if reloaded.Profile.ConservativeMode || !reloaded.Profile.AvoidTimeColumns || reloaded.Profile.RequireApprovalForHighRisk {
		t.Fatalf("unexpected reopened preference profile: %#v", reloaded.Profile)
	}
	if len(reloaded.Profile.ProtectedColumns) != 1 || reloaded.Profile.ProtectedColumns[0] != "event_time" {
		t.Fatalf("unexpected reopened protected columns: %#v", reloaded.Profile.ProtectedColumns)
	}
}

func TestSQLiteStoreSaveTraceEventConcurrentWriters(t *testing.T) {
	store, err := NewSQLiteStore(filepath.Join(t.TempDir(), "agent.sqlite"))
	if err != nil {
		t.Fatalf("NewSQLiteStore failed: %v", err)
	}
	defer store.Close()

	const writerCount = 12
	errCh := make(chan error, writerCount)
	seqCh := make(chan int, writerCount)
	start := make(chan struct{})

	var wg sync.WaitGroup
	for i := 0; i < writerCount; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			<-start
			event, saveErr := store.SaveTraceEvent(t.Context(), AgentTraceEvent{
				SessionID: "session-concurrent",
				TaskID:    fmt.Sprintf("task-%02d", index),
				AgentName: AgentSupervisor,
				TraceType: TraceAgentDecision,
				Summary:   fmt.Sprintf("event-%02d", index),
				Payload:   map[string]any{"index": index},
			})
			if saveErr != nil {
				errCh <- saveErr
				return
			}
			seqCh <- event.Seq
		}(i)
	}

	close(start)
	wg.Wait()
	close(errCh)
	close(seqCh)

	for saveErr := range errCh {
		if saveErr != nil {
			t.Fatalf("SaveTraceEvent concurrent write failed: %v", saveErr)
		}
	}

	seqs := make([]int, 0, writerCount)
	for seq := range seqCh {
		seqs = append(seqs, seq)
	}
	if len(seqs) != writerCount {
		t.Fatalf("expected %d trace seq values, got %d", writerCount, len(seqs))
	}

	sort.Ints(seqs)
	for idx, seq := range seqs {
		want := idx + 1
		if seq != want {
			t.Fatalf("unexpected trace seq ordering at index %d: got %d want %d", idx, seq, want)
		}
	}

	trace, err := store.ListTrace(t.Context(), "session-concurrent")
	if err != nil {
		t.Fatalf("ListTrace failed: %v", err)
	}
	if len(trace) != writerCount {
		t.Fatalf("expected %d persisted trace events, got %d", writerCount, len(trace))
	}
}
