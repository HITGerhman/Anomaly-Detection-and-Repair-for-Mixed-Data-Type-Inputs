package task

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"appshell/backend/internal/engine"
	_ "modernc.org/sqlite"
)

type SQLiteHistoryStore struct {
	db         *sql.DB
	keepRecent int
}

func NewSQLiteHistoryStore(dbPath string, keepRecent int) (*SQLiteHistoryStore, error) {
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

	store := &SQLiteHistoryStore{
		db:         db,
		keepRecent: keepRecent,
	}
	if err := store.initSchema(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *SQLiteHistoryStore) initSchema(ctx context.Context) error {
	const ddl = `
CREATE TABLE IF NOT EXISTS task_history (
	id TEXT PRIMARY KEY,
	status TEXT NOT NULL,
	request_json TEXT NOT NULL,
	response_json TEXT NOT NULL,
	error_text TEXT NOT NULL,
	created_at_unix_ms INTEGER NOT NULL,
	started_at_unix_ms INTEGER NOT NULL,
	ended_at_unix_ms INTEGER NOT NULL,
	timeout_ms INTEGER NOT NULL,
	updated_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_history_updated_at ON task_history(updated_at_unix_ms DESC);`
	_, err := s.db.ExecContext(ctx, ddl)
	if err != nil {
		return fmt.Errorf("init task_history schema failed: %w", err)
	}
	return nil
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

func (s *SQLiteHistoryStore) SaveTask(ctx context.Context, task Task) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("sqlite history store is not initialized")
	}

	reqJSON, err := json.Marshal(task.Request)
	if err != nil {
		return fmt.Errorf("marshal task request failed: %w", err)
	}
	respJSON, err := json.Marshal(task.Response)
	if err != nil {
		return fmt.Errorf("marshal task response failed: %w", err)
	}

	updatedAt := time.Now().UTC().UnixMilli()
	const upsert = `
INSERT INTO task_history (
	id, status, request_json, response_json, error_text,
	created_at_unix_ms, started_at_unix_ms, ended_at_unix_ms, timeout_ms, updated_at_unix_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
	status=excluded.status,
	request_json=excluded.request_json,
	response_json=excluded.response_json,
	error_text=excluded.error_text,
	created_at_unix_ms=excluded.created_at_unix_ms,
	started_at_unix_ms=excluded.started_at_unix_ms,
	ended_at_unix_ms=excluded.ended_at_unix_ms,
	timeout_ms=excluded.timeout_ms,
	updated_at_unix_ms=excluded.updated_at_unix_ms;
`

	_, err = s.db.ExecContext(
		ctx,
		upsert,
		task.ID,
		task.Status,
		string(reqJSON),
		string(respJSON),
		task.Error,
		timeToUnixMS(task.CreatedAt),
		timeToUnixMS(task.StartedAt),
		timeToUnixMS(task.EndedAt),
		task.TimeoutMS,
		updatedAt,
	)
	if err != nil {
		return fmt.Errorf("upsert task history failed: %w", err)
	}

	if s.keepRecent > 0 {
		const trim = `
DELETE FROM task_history
WHERE id NOT IN (
	SELECT id FROM task_history
	ORDER BY updated_at_unix_ms DESC
	LIMIT ?
);`
		if _, err := s.db.ExecContext(ctx, trim, s.keepRecent); err != nil {
			return fmt.Errorf("trim task history failed: %w", err)
		}
	}

	return nil
}

func scanTask(
	id string,
	status string,
	requestJSON string,
	responseJSON string,
	errorText string,
	createdAtMS int64,
	startedAtMS int64,
	endedAtMS int64,
	timeoutMS int64,
) (*Task, error) {
	var req engine.Request
	if err := json.Unmarshal([]byte(requestJSON), &req); err != nil {
		return nil, fmt.Errorf("unmarshal request failed: %w", err)
	}

	var resp engine.Response
	if err := json.Unmarshal([]byte(responseJSON), &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}

	return &Task{
		ID:        id,
		Status:    status,
		Request:   req,
		Response:  resp,
		Error:     errorText,
		CreatedAt: unixMSToTime(createdAtMS),
		StartedAt: unixMSToTime(startedAtMS),
		EndedAt:   unixMSToTime(endedAtMS),
		TimeoutMS: timeoutMS,
	}, nil
}

func (s *SQLiteHistoryStore) GetTask(ctx context.Context, taskID string) (*Task, bool, error) {
	if s == nil || s.db == nil {
		return nil, false, fmt.Errorf("sqlite history store is not initialized")
	}

	const query = `
SELECT
	id, status, request_json, response_json, error_text,
	created_at_unix_ms, started_at_unix_ms, ended_at_unix_ms, timeout_ms
FROM task_history
WHERE id = ?;`

	var (
		id, status, reqJSON, respJSON, errText string
		createdAtMS, startedAtMS               int64
		endedAtMS, timeoutMS                   int64
	)
	err := s.db.QueryRowContext(ctx, query, taskID).Scan(
		&id, &status, &reqJSON, &respJSON, &errText,
		&createdAtMS, &startedAtMS, &endedAtMS, &timeoutMS,
	)
	if err == sql.ErrNoRows {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("query task history failed: %w", err)
	}

	task, err := scanTask(id, status, reqJSON, respJSON, errText, createdAtMS, startedAtMS, endedAtMS, timeoutMS)
	if err != nil {
		return nil, false, err
	}
	return task, true, nil
}

func (s *SQLiteHistoryStore) ListRecentTasks(ctx context.Context, limit int) ([]Task, error) {
	if s == nil || s.db == nil {
		return nil, fmt.Errorf("sqlite history store is not initialized")
	}
	if limit <= 0 {
		limit = 20
	}

	const query = `
SELECT
	id, status, request_json, response_json, error_text,
	created_at_unix_ms, started_at_unix_ms, ended_at_unix_ms, timeout_ms
FROM task_history
ORDER BY updated_at_unix_ms DESC
LIMIT ?;`
	rows, err := s.db.QueryContext(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("query recent task history failed: %w", err)
	}
	defer rows.Close()

	out := make([]Task, 0, limit)
	for rows.Next() {
		var (
			id, status, reqJSON, respJSON, errText string
			createdAtMS, startedAtMS               int64
			endedAtMS, timeoutMS                   int64
		)
		if err := rows.Scan(
			&id, &status, &reqJSON, &respJSON, &errText,
			&createdAtMS, &startedAtMS, &endedAtMS, &timeoutMS,
		); err != nil {
			return nil, fmt.Errorf("scan task history row failed: %w", err)
		}
		task, err := scanTask(id, status, reqJSON, respJSON, errText, createdAtMS, startedAtMS, endedAtMS, timeoutMS)
		if err != nil {
			return nil, err
		}
		out = append(out, *task)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate task history rows failed: %w", err)
	}
	return out, nil
}

func (s *SQLiteHistoryStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}
