package main

import (
	"testing"
	"time"
)

func TestNormalizeRequestForTrainPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action":     "train",
		"csv_path":   "data/raw/demo.csv",
		"target_col": "stroke",
		"output_dir": "outputs/results/mvp",
		"timeout_ms": 12000,
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}

	if req.Action != "train" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["csv_path"] != "data/raw/demo.csv" {
		t.Fatalf("unexpected csv_path: %v", req.Payload["csv_path"])
	}
	if req.Payload["target_col"] != "stroke" {
		t.Fatalf("unexpected target_col: %v", req.Payload["target_col"])
	}
	if req.Payload["output_dir"] != "outputs/results/mvp" {
		t.Fatalf("unexpected output_dir: %v", req.Payload["output_dir"])
	}
	if timeout != 12*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}

func TestNormalizeRequestMissingTrainFields(t *testing.T) {
	_, _, err := normalizeRequest(map[string]any{
		"action": "train",
	})
	if err == nil {
		t.Fatalf("expected error for missing fields")
	}
}

func TestNormalizeRequestUsesNestedPayload(t *testing.T) {
	req, timeout, err := normalizeRequest(map[string]any{
		"action": "health",
		"payload": map[string]any{
			"x": 1,
		},
		"timeout_ms": "3s",
	})
	if err != nil {
		t.Fatalf("normalizeRequest failed: %v", err)
	}
	if req.Action != "health" {
		t.Fatalf("unexpected action: %s", req.Action)
	}
	if req.Payload["x"] != 1 {
		t.Fatalf("unexpected nested payload: %v", req.Payload["x"])
	}
	if timeout != 3*time.Second {
		t.Fatalf("unexpected timeout: %s", timeout)
	}
}
