package task

import "context"

type HistoryStore interface {
	SaveTask(ctx context.Context, task Task) error
	GetTask(ctx context.Context, taskID string) (*Task, bool, error)
	ListRecentTasks(ctx context.Context, limit int) ([]Task, error)
	Close() error
}
