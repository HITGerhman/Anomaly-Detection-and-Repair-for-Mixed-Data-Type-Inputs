package engine

type Request struct {
	TaskID  string         `json:"task_id"`
	Action  string         `json:"action"`
	Payload map[string]any `json:"payload"`
}

type ErrorBody struct {
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Details map[string]any `json:"details"`
}

type Response struct {
	TaskID     string         `json:"task_id"`
	Status     string         `json:"status"`
	Result     map[string]any `json:"result"`
	Error      *ErrorBody     `json:"error"`
	Timestamp  string         `json:"timestamp"`
	DurationMS int            `json:"duration_ms"`
}
