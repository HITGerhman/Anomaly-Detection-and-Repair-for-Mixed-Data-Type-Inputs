package agent

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

type SQLiteStore struct {
	db      *sql.DB
	writeMu sync.Mutex
}

func NewSQLiteStore(dbPath string) (*SQLiteStore, error) {
	path := filepath.Clean(dbPath)
	if path == "" || path == "." {
		return nil, fmt.Errorf("invalid sqlite db path")
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create sqlite dir failed: %w", err)
	}

	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)", filepath.ToSlash(path))
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite failed: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	store := &SQLiteStore{db: db}
	if err := store.initSchema(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *SQLiteStore) initSchema(ctx context.Context) error {
	const ddl = `
CREATE TABLE IF NOT EXISTS agent_sessions (
	session_id TEXT PRIMARY KEY,
	root_task_id TEXT NOT NULL,
	current_task_id TEXT NOT NULL,
	status TEXT NOT NULL,
	mode TEXT NOT NULL,
	user_goal TEXT NOT NULL,
	context_json TEXT NOT NULL,
	latest_plan_json TEXT NOT NULL,
	created_at_unix_ms INTEGER NOT NULL,
	updated_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated_at ON agent_sessions(updated_at_unix_ms DESC);

CREATE TABLE IF NOT EXISTS agent_trace (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id TEXT NOT NULL,
	task_id TEXT NOT NULL,
	seq INTEGER NOT NULL,
	agent_name TEXT NOT NULL,
	trace_type TEXT NOT NULL,
	summary TEXT NOT NULL,
	payload_json TEXT NOT NULL,
	created_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_trace_session_seq ON agent_trace(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_agent_trace_task_seq ON agent_trace(task_id, seq);

CREATE TABLE IF NOT EXISTS agent_preferences (
	workspace_id TEXT PRIMARY KEY,
	profile_json TEXT NOT NULL,
	updated_at_unix_ms INTEGER NOT NULL
);
`
	if _, err := s.db.ExecContext(ctx, ddl); err != nil {
		return fmt.Errorf("init agent schema failed: %w", err)
	}
	return nil
}

func (s *SQLiteStore) SaveSession(ctx context.Context, session AgentSession) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("sqlite store is not initialized")
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	if session.CreatedAt.IsZero() {
		session.CreatedAt = time.Now().UTC()
	}
	if session.UpdatedAt.IsZero() {
		session.UpdatedAt = session.CreatedAt
	}

	contextJSON, err := json.Marshal(cloneMap(session.Context))
	if err != nil {
		return fmt.Errorf("marshal session context failed: %w", err)
	}
	planJSON, err := json.Marshal(clonePlan(session.LatestPlan))
	if err != nil {
		return fmt.Errorf("marshal session plan failed: %w", err)
	}

	const upsert = `
INSERT INTO agent_sessions (
	session_id, root_task_id, current_task_id, status, mode, user_goal,
	context_json, latest_plan_json, created_at_unix_ms, updated_at_unix_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(session_id) DO UPDATE SET
	root_task_id=excluded.root_task_id,
	current_task_id=excluded.current_task_id,
	status=excluded.status,
	mode=excluded.mode,
	user_goal=excluded.user_goal,
	context_json=excluded.context_json,
	latest_plan_json=excluded.latest_plan_json,
	created_at_unix_ms=excluded.created_at_unix_ms,
	updated_at_unix_ms=excluded.updated_at_unix_ms;
`
	_, err = s.db.ExecContext(
		ctx,
		upsert,
		session.SessionID,
		session.RootTaskID,
		session.CurrentTaskID,
		session.Status,
		session.Mode,
		session.UserGoal,
		string(contextJSON),
		string(planJSON),
		timeToUnixMS(session.CreatedAt),
		timeToUnixMS(session.UpdatedAt),
	)
	if err != nil {
		return fmt.Errorf("upsert agent session failed: %w", err)
	}
	return nil
}

func (s *SQLiteStore) GetSession(ctx context.Context, sessionID string) (*AgentSession, bool, error) {
	if s == nil || s.db == nil {
		return nil, false, fmt.Errorf("sqlite store is not initialized")
	}

	const query = `
SELECT
	session_id, root_task_id, current_task_id, status, mode, user_goal,
	context_json, latest_plan_json, created_at_unix_ms, updated_at_unix_ms
FROM agent_sessions
WHERE session_id = ?;
`
	var (
		id            string
		rootTaskID    string
		currentTaskID string
		status        string
		mode          string
		userGoal      string
		contextJSON   string
		planJSON      string
		createdAtMS   int64
		updatedAtMS   int64
	)
	err := s.db.QueryRowContext(ctx, query, sessionID).Scan(
		&id,
		&rootTaskID,
		&currentTaskID,
		&status,
		&mode,
		&userGoal,
		&contextJSON,
		&planJSON,
		&createdAtMS,
		&updatedAtMS,
	)
	if err == sql.ErrNoRows {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("query agent session failed: %w", err)
	}

	session, err := scanSession(id, rootTaskID, currentTaskID, status, mode, userGoal, contextJSON, planJSON, createdAtMS, updatedAtMS)
	if err != nil {
		return nil, false, err
	}
	return session, true, nil
}

func (s *SQLiteStore) SaveTraceEvent(ctx context.Context, event AgentTraceEvent) (AgentTraceEvent, error) {
	if s == nil || s.db == nil {
		return AgentTraceEvent{}, fmt.Errorf("sqlite store is not initialized")
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now().UTC()
	}

	payloadJSON, err := json.Marshal(cloneMap(event.Payload))
	if err != nil {
		return AgentTraceEvent{}, fmt.Errorf("marshal trace payload failed: %w", err)
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return AgentTraceEvent{}, fmt.Errorf("begin trace tx failed: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	seq := event.Seq
	if seq <= 0 {
		if err := tx.QueryRowContext(
			ctx,
			`SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_trace WHERE session_id = ?;`,
			event.SessionID,
		).Scan(&seq); err != nil {
			return AgentTraceEvent{}, fmt.Errorf("allocate trace seq failed: %w", err)
		}
	}

	result, err := tx.ExecContext(
		ctx,
		`INSERT INTO agent_trace (
			session_id, task_id, seq, agent_name, trace_type, summary, payload_json, created_at_unix_ms
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?);`,
		event.SessionID,
		event.TaskID,
		seq,
		event.AgentName,
		event.TraceType,
		event.Summary,
		string(payloadJSON),
		timeToUnixMS(event.CreatedAt),
	)
	if err != nil {
		return AgentTraceEvent{}, fmt.Errorf("insert trace event failed: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return AgentTraceEvent{}, fmt.Errorf("commit trace event failed: %w", err)
	}

	id, _ := result.LastInsertId()
	event.ID = id
	event.Seq = seq
	event.Payload = cloneMap(event.Payload)
	return event, nil
}

func (s *SQLiteStore) ListTrace(ctx context.Context, sessionID string) ([]AgentTraceEvent, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("sqlite store is not initialized")
	}

	rows, err := s.db.QueryContext(
		ctx,
		`SELECT id, session_id, task_id, seq, agent_name, trace_type, summary, payload_json, created_at_unix_ms
		 FROM agent_trace
		 WHERE session_id = ?
		 ORDER BY seq ASC, id ASC;`,
		sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("query agent trace failed: %w", err)
	}
	defer rows.Close()

	out := make([]AgentTraceEvent, 0, 32)
	for rows.Next() {
		var (
			event         AgentTraceEvent
			payloadJSON   string
			createdAtUnix int64
		)
		if err := rows.Scan(
			&event.ID,
			&event.SessionID,
			&event.TaskID,
			&event.Seq,
			&event.AgentName,
			&event.TraceType,
			&event.Summary,
			&payloadJSON,
			&createdAtUnix,
		); err != nil {
			return nil, fmt.Errorf("scan trace event failed: %w", err)
		}
		event.CreatedAt = unixMSToTime(createdAtUnix)
		if strings.TrimSpace(payloadJSON) != "" && strings.TrimSpace(payloadJSON) != "null" {
			event.Payload = map[string]any{}
			if err := json.Unmarshal([]byte(payloadJSON), &event.Payload); err != nil {
				return nil, fmt.Errorf("unmarshal trace payload failed: %w", err)
			}
		} else {
			event.Payload = map[string]any{}
		}
		out = append(out, event)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate trace rows failed: %w", err)
	}
	return out, nil
}

func (s *SQLiteStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *SQLiteStore) SavePreferences(ctx context.Context, record AgentPreferenceRecord) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("sqlite store is not initialized")
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	workspaceID := strings.TrimSpace(record.WorkspaceID)
	if workspaceID == "" {
		return fmt.Errorf("workspace id is required")
	}
	if record.UpdatedAt.IsZero() {
		record.UpdatedAt = time.Now().UTC()
	}

	profileJSON, err := json.Marshal(normalizePreferenceProfile(record.Profile))
	if err != nil {
		return fmt.Errorf("marshal preference profile failed: %w", err)
	}

	const upsert = `
INSERT INTO agent_preferences (
	workspace_id, profile_json, updated_at_unix_ms
) VALUES (?, ?, ?)
ON CONFLICT(workspace_id) DO UPDATE SET
	profile_json=excluded.profile_json,
	updated_at_unix_ms=excluded.updated_at_unix_ms;
`
	_, err = s.db.ExecContext(
		ctx,
		upsert,
		workspaceID,
		string(profileJSON),
		timeToUnixMS(record.UpdatedAt),
	)
	if err != nil {
		return fmt.Errorf("upsert agent preferences failed: %w", err)
	}
	return nil
}

func (s *SQLiteStore) GetPreferences(ctx context.Context, workspaceID string) (AgentPreferenceRecord, bool, error) {
	if s == nil || s.db == nil {
		return AgentPreferenceRecord{}, false, fmt.Errorf("sqlite store is not initialized")
	}

	id := strings.TrimSpace(workspaceID)
	if id == "" {
		return AgentPreferenceRecord{}, false, nil
	}

	const query = `
SELECT workspace_id, profile_json, updated_at_unix_ms
FROM agent_preferences
WHERE workspace_id = ?;
`
	var (
		profileJSON string
		updatedAtMS int64
	)
	err := s.db.QueryRowContext(ctx, query, id).Scan(&workspaceID, &profileJSON, &updatedAtMS)
	if err == sql.ErrNoRows {
		return AgentPreferenceRecord{}, false, nil
	}
	if err != nil {
		return AgentPreferenceRecord{}, false, fmt.Errorf("query agent preferences failed: %w", err)
	}

	profile := AgentPreferenceProfile{}
	if strings.TrimSpace(profileJSON) != "" && strings.TrimSpace(profileJSON) != "null" {
		if err := json.Unmarshal([]byte(profileJSON), &profile); err != nil {
			return AgentPreferenceRecord{}, false, fmt.Errorf("unmarshal preference profile failed: %w", err)
		}
	}

	return AgentPreferenceRecord{
		WorkspaceID: strings.TrimSpace(workspaceID),
		Profile:     normalizePreferenceProfile(profile),
		UpdatedAt:   unixMSToTime(updatedAtMS),
	}, true, nil
}

func scanSession(
	id string,
	rootTaskID string,
	currentTaskID string,
	status string,
	mode string,
	userGoal string,
	contextJSON string,
	planJSON string,
	createdAtMS int64,
	updatedAtMS int64,
) (*AgentSession, error) {
	contextMap := map[string]any{}
	if strings.TrimSpace(contextJSON) != "" {
		if err := json.Unmarshal([]byte(contextJSON), &contextMap); err != nil {
			return nil, fmt.Errorf("unmarshal session context failed: %w", err)
		}
	}

	var plan AgentPlan
	if strings.TrimSpace(planJSON) != "" && strings.TrimSpace(planJSON) != "null" {
		if err := json.Unmarshal([]byte(planJSON), &plan); err != nil {
			return nil, fmt.Errorf("unmarshal session plan failed: %w", err)
		}
	}

	return &AgentSession{
		SessionID:     id,
		RootTaskID:    rootTaskID,
		CurrentTaskID: currentTaskID,
		Status:        status,
		Mode:          mode,
		UserGoal:      userGoal,
		Context:       contextMap,
		LatestPlan:    plan,
		CreatedAt:     unixMSToTime(createdAtMS),
		UpdatedAt:     unixMSToTime(updatedAtMS),
	}, nil
}

func timeToUnixMS(ts time.Time) int64 {
	if ts.IsZero() {
		return 0
	}
	return ts.UTC().UnixMilli()
}

func unixMSToTime(ms int64) time.Time {
	if ms <= 0 {
		return time.Time{}
	}
	return time.UnixMilli(ms).UTC()
}
