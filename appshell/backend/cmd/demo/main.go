package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"appshell/backend/internal/engine"
	"appshell/backend/internal/task"
)

func mustJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("<marshal-error: %v>", err)
	}
	return string(b)
}

func main() {
	action := flag.String("action", "health", "Action to run: health or train")
	csv := flag.String("csv", "", "CSV path for train action")
	target := flag.String("target", "", "Target column for train action")
	outputDir := flag.String("output", "", "Output directory for model artifacts")
	engineScript := flag.String("engine", "../core/python_engine/engine_main.py", "Path to python engine script")
	timeout := flag.Duration("timeout", 90*time.Second, "Task timeout")
	flag.Parse()

	absEngine, err := filepath.Abs(*engineScript)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve engine path failed: %v\n", err)
		os.Exit(1)
	}

	runner := engine.NewRunner(absEngine)
	svc := task.NewService(runner)

	req := engine.Request{
		Action:  *action,
		Payload: map[string]any{},
	}

	if *action == "train" {
		if *csv == "" || *target == "" {
			fmt.Fprintln(os.Stderr, "for train action, -csv and -target are required")
			os.Exit(1)
		}
		req.Payload["csv_path"] = *csv
		req.Payload["target_col"] = *target
		if *outputDir != "" {
			req.Payload["output_dir"] = *outputDir
		}
	}

	taskID, err := svc.Start(req, *timeout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "start task failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("task started: %s\n", taskID)

	ticker := time.NewTicker(400 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		t, ok := svc.Get(taskID)
		if !ok {
			fmt.Fprintf(os.Stderr, "task not found: %s\n", taskID)
			os.Exit(1)
		}

		fmt.Printf("status=%s\n", t.Status)
		switch t.Status {
		case task.StatusSucceeded:
			fmt.Println("response:")
			fmt.Println(mustJSON(t.Response))
			return
		case task.StatusFailed, task.StatusCanceled:
			fmt.Println("error:")
			fmt.Println(t.Error)
			if t.Response.TaskID != "" {
				fmt.Println("response:")
				fmt.Println(mustJSON(t.Response))
			}
			os.Exit(1)
		}
	}
}
