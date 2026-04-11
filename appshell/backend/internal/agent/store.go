package agent

import "context"

type SessionStore interface {
	SaveSession(ctx context.Context, session AgentSession) error
	GetSession(ctx context.Context, sessionID string) (*AgentSession, bool, error)
	SaveTraceEvent(ctx context.Context, event AgentTraceEvent) (AgentTraceEvent, error)
	ListTrace(ctx context.Context, sessionID string) ([]AgentTraceEvent, error)
	SavePreferences(ctx context.Context, record AgentPreferenceRecord) error
	GetPreferences(ctx context.Context, workspaceID string) (AgentPreferenceRecord, bool, error)
	Close() error
}
