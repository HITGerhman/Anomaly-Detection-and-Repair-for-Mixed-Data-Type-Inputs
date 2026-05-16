package main

import (
	"appshell/backend/internal/observability"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

func resolveFrontendDir() (string, error) {
	candidates := []string{
		os.Getenv("APPSHELL_FRONTEND_DIR"),
	}
	if root, ok := packagedRuntimeDir(); ok {
		candidates = append(candidates, filepath.Join(root, "frontend"))
	}
	if exeDir, ok := executableDir(); ok {
		candidates = append(candidates,
			filepath.Join(exeDir, "frontend"),
			filepath.Join(exeDir, "..", "frontend"),
			filepath.Join(exeDir, "..", "..", "frontend"),
		)
	}
	candidates = append(candidates,
		"../frontend",
		"../../frontend",
		"appshell/frontend",
	)

	for _, candidate := range candidates {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		abs, err := filepath.Abs(candidate)
		if err != nil {
			continue
		}
		info, err := os.Stat(abs)
		if err == nil && info.IsDir() {
			return abs, nil
		}
	}

	return "", fmt.Errorf("frontend directory not found, set APPSHELL_FRONTEND_DIR")
}

func resolveDefaultEnginePath() string {
	if raw := strings.TrimSpace(os.Getenv("APPSHELL_ENGINE_PATH")); raw != "" {
		return raw
	}
	if root, ok := packagedRuntimeDir(); ok {
		return packagedEnginePath(root)
	}

	candidates := []string{"../core/python_engine/engine_main.py"}
	if exeDir, ok := executableDir(); ok {
		candidates = append(candidates,
			filepath.Join(exeDir, "python_engine", packagedEngineFileName()),
			filepath.Join(exeDir, "..", "core", "python_engine", "engine_main.py"),
			filepath.Join(exeDir, "..", "..", "core", "python_engine", "engine_main.py"),
		)
	}
	candidates = append(candidates,
		"../../core/python_engine/engine_main.py",
		"appshell/core/python_engine/engine_main.py",
	)

	for _, candidate := range candidates {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		abs, err := filepath.Abs(candidate)
		if err == nil && fileExists(abs) {
			return abs
		}
	}
	return "../core/python_engine/engine_main.py"
}

func ensureDefaultGoLogFile() {
	if strings.TrimSpace(os.Getenv("APPSHELL_GO_LOG_FILE")) != "" {
		return
	}

	if _, ok := packagedRuntimeDir(); ok {
		root, err := packagedDataRoot()
		if err != nil {
			return
		}
		_ = os.Setenv("APPSHELL_GO_LOG_FILE", filepath.Join(root, "outputs", "appshell", "go_backend.log"))
		return
	}

	abs, err := filepath.Abs(filepath.Join("..", "..", "outputs", "appshell", "go_backend.log"))
	if err != nil {
		return
	}
	_ = os.Setenv("APPSHELL_GO_LOG_FILE", abs)
}

func main() {
	ensureDefaultGoLogFile()

	engineScript := flag.String("engine", resolveDefaultEnginePath(), "Path to python engine script or packaged engine executable")
	title := flag.String("title", "Anomaly AppShell", "Window title")
	flag.Parse()

	observability.Info("wails_main_start", map[string]any{
		"engine_script": *engineScript,
		"title":         *title,
	})

	app, err := NewApp(*engineScript)
	if err != nil {
		observability.Error("wails_app_init_failed", map[string]any{"reason": err.Error()})
		fmt.Fprintf(os.Stderr, "init app failed: %v\n", err)
		os.Exit(1)
	}

	frontendDir, err := resolveFrontendDir()
	if err != nil {
		observability.Error("wails_frontend_resolve_failed", map[string]any{"reason": err.Error()})
		fmt.Fprintf(os.Stderr, "resolve frontend failed: %v\n", err)
		os.Exit(1)
	}

	assets := os.DirFS(frontendDir)
	if _, err := fs.Stat(assets, "index.html"); err != nil {
		observability.Error("wails_frontend_index_missing", map[string]any{
			"frontend_dir": frontendDir,
			"reason":       err.Error(),
		})
		fmt.Fprintf(os.Stderr, "frontend index.html not found in %s: %v\n", frontendDir, err)
		os.Exit(1)
	}

	observability.Info("wails_runtime_paths_resolved", map[string]any{
		"engine_script": *engineScript,
		"frontend_dir":  frontendDir,
	})

	err = wails.Run(&options.App{
		Title:         *title,
		Width:         1080,
		Height:        760,
		MinWidth:      900,
		MinHeight:     640,
		DisableResize: false,
		AssetServer:   &assetserver.Options{Assets: assets},
		OnStartup:     app.startup,
		OnShutdown:    app.shutdown,
		Bind:          []any{app},
	})
	if err != nil {
		observability.Error("wails_run_failed", map[string]any{"reason": err.Error()})
		fmt.Fprintf(os.Stderr, "wails run failed: %v\n", err)
		os.Exit(1)
	}
	observability.Info("wails_run_returned", map[string]any{})
}
