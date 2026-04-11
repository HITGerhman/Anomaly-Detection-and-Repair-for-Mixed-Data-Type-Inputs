package presentation

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
)

func maybeWriteArtifact(req engine.Request, result map[string]any, isAgent bool, bundle *Bundle) (string, error) {
	path := presentationArtifactPath(req, result, isAgent)
	if strings.TrimSpace(path) == "" || bundle == nil {
		return "", nil
	}
	if bundle.Artifacts == nil {
		bundle.Artifacts = map[string]any{}
	}
	bundle.Artifacts["presentation_json"] = path
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	payload, err := json.MarshalIndent(bundle, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(payload, '\n'), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func maybeWriteSnapshotArtifact(snapshot *agent.AgentSessionSnapshot, bundle *Bundle) (string, error) {
	if snapshot == nil || bundle == nil {
		return "", nil
	}
	path := snapshotArtifactPath(*snapshot)
	if strings.TrimSpace(path) == "" {
		return "", nil
	}
	if bundle.Artifacts == nil {
		bundle.Artifacts = map[string]any{}
	}
	bundle.Artifacts["presentation_json"] = path
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	payload, err := json.MarshalIndent(bundle, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(payload, '\n'), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func presentationArtifactPath(req engine.Request, result map[string]any, isAgent bool) string {
	if isAgent {
		agentBlock := objectMap(result["agent"])
		execution := objectMap(agentBlock["execution"])
		if outputCSV := asString(execution["output_csv"]); outputCSV != "" {
			return fileArtifactPath(outputCSV)
		}
		if outputDir := asString(req.Payload["output_dir"]); outputDir != "" {
			return filepath.Join(outputDir, "presentation.json")
		}
		return ""
	}
	if outputCSV := asString(result["output_csv"]); outputCSV != "" {
		return fileArtifactPath(outputCSV)
	}
	if outputDir := asString(result["output_dir"]); outputDir != "" {
		return filepath.Join(outputDir, "presentation.json")
	}
	if outputDir := asString(req.Payload["output_dir"]); outputDir != "" {
		return filepath.Join(outputDir, "presentation.json")
	}
	return ""
}

func snapshotArtifactPath(snapshot agent.AgentSessionSnapshot) string {
	if snapshot.PresentationArtifact != "" {
		return snapshot.PresentationArtifact
	}
	execution := objectMap(snapshot.Context["execution_artifacts"])
	if outputCSV := asString(execution["output_csv"]); outputCSV != "" {
		return fileArtifactPath(outputCSV)
	}
	if outputDir := asString(snapshot.Context["output_dir"]); outputDir != "" {
		return filepath.Join(outputDir, "presentation.json")
	}
	return ""
}
