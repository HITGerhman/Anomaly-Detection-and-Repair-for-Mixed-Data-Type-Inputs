package observability

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var (
	initOnce sync.Once
	writeMu  sync.Mutex
	logFile  *os.File
)

func initLogger() {
	logPath := strings.TrimSpace(os.Getenv("APPSHELL_GO_LOG_FILE"))
	if logPath == "" {
		return
	}

	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		return
	}

	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	logFile = f
}

func writeLine(line []byte) {
	writeMu.Lock()
	defer writeMu.Unlock()

	_, _ = os.Stderr.Write(line)
	if logFile != nil {
		_, _ = logFile.Write(line)
	}
}

func Log(level string, event string, fields map[string]any) {
	initOnce.Do(initLogger)

	payload := map[string]any{
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
		"level":     strings.ToLower(strings.TrimSpace(level)),
		"event":     strings.TrimSpace(event),
	}
	for k, v := range fields {
		payload[k] = v
	}

	line, err := json.Marshal(payload)
	if err != nil {
		line = []byte(fmt.Sprintf(`{"timestamp":"%s","level":"error","event":"log_marshal_failed","reason":%q}`,
			time.Now().UTC().Format(time.RFC3339Nano), err.Error()))
	}
	line = append(line, '\n')
	writeLine(line)
}

func Info(event string, fields map[string]any) {
	Log("info", event, fields)
}

func Warn(event string, fields map[string]any) {
	Log("warning", event, fields)
}

func Error(event string, fields map[string]any) {
	Log("error", event, fields)
}
