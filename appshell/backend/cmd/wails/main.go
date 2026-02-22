package main

import (
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
		"../frontend",
		"../../frontend",
		"appshell/frontend",
	}

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

func ensureDefaultGoLogFile() {
	if strings.TrimSpace(os.Getenv("APPSHELL_GO_LOG_FILE")) != "" {
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

	engineScript := flag.String("engine", "../core/python_engine/engine_main.py", "Path to python engine script")
	title := flag.String("title", "Anomaly AppShell", "Window title")
	flag.Parse()

	app, err := NewApp(*engineScript)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init app failed: %v\n", err)
		os.Exit(1)
	}

	frontendDir, err := resolveFrontendDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve frontend failed: %v\n", err)
		os.Exit(1)
	}

	assets := os.DirFS(frontendDir)
	if _, err := fs.Stat(assets, "index.html"); err != nil {
		fmt.Fprintf(os.Stderr, "frontend index.html not found in %s: %v\n", frontendDir, err)
		os.Exit(1)
	}

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
		fmt.Fprintf(os.Stderr, "wails run failed: %v\n", err)
		os.Exit(1)
	}
}
