package agent

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"appshell/backend/internal/observability"
)

type LangGraphSidecarManager struct {
	mu      sync.Mutex
	config  LangGraphConfig
	checker healthChecker
	cmd     *exec.Cmd
	waitCh  chan error
}

func NewLangGraphSidecarManager(config LangGraphConfig, checker healthChecker) *LangGraphSidecarManager {
	if checker == nil {
		checker = NewLangGraphClient(config.BaseURL(), config.RequestTimeout)
	}
	return &LangGraphSidecarManager{
		config:  config,
		checker: checker,
	}
}

func (m *LangGraphSidecarManager) Config() LangGraphConfig {
	if m == nil {
		return LangGraphConfig{}
	}
	return m.config
}

func (m *LangGraphSidecarManager) Managed() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.refreshProcessStateLocked()
	return m.cmd != nil
}

func (m *LangGraphSidecarManager) EnsureHealthy(ctx context.Context) (LangGraphHealth, error) {
	if m == nil {
		return LangGraphHealth{}, fmt.Errorf("langgraph sidecar manager is nil")
	}
	if !m.config.Enabled {
		return LangGraphHealth{}, fmt.Errorf("langgraph sidecar is disabled")
	}

	health, err := m.healthWithRequestTimeout(ctx)
	if err == nil {
		return health, nil
	}

	if strings.TrimSpace(m.config.ScriptPath) == "" {
		return LangGraphHealth{}, fmt.Errorf("langgraph sidecar script path is empty")
	}
	if _, statErr := os.Stat(m.config.ScriptPath); statErr != nil {
		return LangGraphHealth{}, fmt.Errorf("langgraph sidecar script unavailable: %w", statErr)
	}

	m.mu.Lock()
	m.refreshProcessStateLocked()
	alreadyManaged := m.cmd != nil
	m.mu.Unlock()

	if !alreadyManaged && m.portOccupiedByNonSidecar() {
		return LangGraphHealth{}, fmt.Errorf("langgraph sidecar port is occupied by another process")
	}

	m.mu.Lock()
	m.refreshProcessStateLocked()
	if m.cmd == nil {
		if startErr := m.startLocked(); startErr != nil {
			m.mu.Unlock()
			return LangGraphHealth{}, startErr
		}
	}
	m.mu.Unlock()

	return m.waitUntilHealthy(ctx)
}

func (m *LangGraphSidecarManager) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cmd == nil {
		return nil
	}
	cmd := m.cmd
	waitCh := m.waitCh
	m.cmd = nil
	m.waitCh = nil
	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
	if waitCh != nil {
		select {
		case <-waitCh:
		case <-time.After(2 * time.Second):
		}
	}
	return nil
}

func (m *LangGraphSidecarManager) healthWithRequestTimeout(ctx context.Context) (LangGraphHealth, error) {
	timeout := m.config.RequestTimeout
	if timeout <= 0 {
		timeout = defaultLangGraphRequestTimeout
	}
	healthCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return m.checker.Health(healthCtx)
}

func (m *LangGraphSidecarManager) portOccupiedByNonSidecar() bool {
	address := net.JoinHostPort(strings.TrimSpace(m.config.Host), strconv.Itoa(m.config.Port))
	conn, err := net.DialTimeout("tcp", address, 300*time.Millisecond)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func (m *LangGraphSidecarManager) startLocked() error {
	pythonBin := strings.TrimSpace(m.config.PythonBin)
	if pythonBin == "" {
		pythonBin = defaultLangGraphPythonBin()
	}
	scriptPath := strings.TrimSpace(m.config.ScriptPath)
	cmd := exec.Command(pythonBin, scriptPath, "--host", m.config.Host, "--port", strconv.Itoa(m.config.Port))
	cmd.Dir = filepath.Dir(scriptPath)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start langgraph sidecar failed: %w", err)
	}

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
		close(waitCh)
	}()

	m.cmd = cmd
	m.waitCh = waitCh
	observability.Info("langgraph_sidecar_started", map[string]any{
		"pid":        cmd.Process.Pid,
		"base_url":   m.config.BaseURL(),
		"script":     scriptPath,
		"python_bin": pythonBin,
	})
	return nil
}

func (m *LangGraphSidecarManager) refreshProcessStateLocked() {
	if m.cmd == nil || m.waitCh == nil {
		return
	}
	select {
	case err, ok := <-m.waitCh:
		if ok && err != nil {
			observability.Warn("langgraph_sidecar_exited", map[string]any{
				"base_url": m.config.BaseURL(),
				"error":    err.Error(),
			})
		}
		m.cmd = nil
		m.waitCh = nil
	default:
	}
}

func (m *LangGraphSidecarManager) waitUntilHealthy(ctx context.Context) (LangGraphHealth, error) {
	timeout := m.config.StartupTimeout
	if timeout <= 0 {
		timeout = defaultLangGraphStartupTimeout
	}
	waitCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	var lastErr error
	for {
		health, err := m.healthWithRequestTimeout(waitCtx)
		if err == nil {
			return health, nil
		}
		lastErr = err

		m.mu.Lock()
		m.refreshProcessStateLocked()
		exited := m.cmd == nil
		m.mu.Unlock()
		if exited {
			if lastErr == nil {
				lastErr = fmt.Errorf("langgraph sidecar exited before becoming healthy")
			}
			return LangGraphHealth{}, lastErr
		}

		select {
		case <-waitCtx.Done():
			if lastErr == nil {
				lastErr = waitCtx.Err()
			}
			return LangGraphHealth{}, lastErr
		case <-ticker.C:
		}
	}
}
