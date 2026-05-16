package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const packagedAppDataDirName = "AnomalyDetectionRepair"

func executableDir() (string, bool) {
	exePath, err := os.Executable()
	if err != nil {
		return "", false
	}
	abs, err := filepath.Abs(exePath)
	if err != nil {
		return "", false
	}
	return filepath.Dir(abs), true
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func packagedEngineFileName() string {
	if runtime.GOOS == "windows" {
		return "anomaly_engine.exe"
	}
	return "anomaly_engine"
}

func packagedEnginePath(root string) string {
	return filepath.Join(root, "python_engine", packagedEngineFileName())
}

func packagedRuntimeDir() (string, bool) {
	exeDir, ok := executableDir()
	if !ok {
		return "", false
	}
	if !fileExists(filepath.Join(exeDir, "frontend", "index.html")) {
		return "", false
	}
	if !fileExists(packagedEnginePath(exeDir)) {
		return "", false
	}
	return exeDir, true
}

func packagedDataRoot() (string, error) {
	if raw := strings.TrimSpace(os.Getenv("APPSHELL_DATA_ROOT")); raw != "" {
		return filepath.Abs(raw)
	}
	if raw := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); raw != "" {
		return filepath.Join(raw, packagedAppDataDirName), nil
	}
	configDir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("resolve user config dir failed: %w", err)
	}
	return filepath.Join(configDir, packagedAppDataDirName), nil
}

func resolvePackagedWritablePath(pathText string) string {
	clean := strings.TrimSpace(pathText)
	if clean == "" || filepath.IsAbs(clean) {
		return clean
	}
	if _, ok := packagedRuntimeDir(); !ok {
		return clean
	}
	root, err := packagedDataRoot()
	if err != nil {
		return clean
	}
	return filepath.Join(root, clean)
}

func resolvePackagedWritableValue(value any) any {
	text, ok := value.(string)
	if !ok {
		return value
	}
	return resolvePackagedWritablePath(text)
}
