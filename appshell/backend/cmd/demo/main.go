package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
	"appshell/backend/internal/task"
)

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

func mustJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("<marshal-error: %v>", err)
	}
	return string(b)
}

func buildRequest(
	action string,
	csv string,
	target string,
	outputDir string,
	goal string,
	sessionID string,
	planID string,
	modelDir string,
	sampleIndex int,
	maxChanges int,
	kNeighbors int,
	index int,
	parallel int,
) (engine.Request, error) {
	normalizedAction := strings.TrimSpace(action)
	req := engine.Request{
		Action:  normalizedAction,
		Payload: map[string]any{},
	}

	switch normalizedAction {
	case agent.ActionSessionPlan:
		if csv == "" {
			return engine.Request{}, fmt.Errorf("for agent.session.plan, -csv is required")
		}
		req.Payload["csv_path"] = csv
		if goal != "" {
			req.Payload["user_goal"] = goal
		}
		if sessionID != "" {
			req.Payload["session_id"] = sessionID
		}
		if outputDir != "" {
			req.Payload["output_dir"] = outputDir
		}
		return req, nil
	case agent.ActionSessionAuto:
		if csv == "" {
			return engine.Request{}, fmt.Errorf("for agent.session.auto, -csv is required")
		}
		req.Payload["csv_path"] = csv
		if goal != "" {
			req.Payload["user_goal"] = goal
		}
		if sessionID != "" {
			req.Payload["session_id"] = sessionID
		}
		if outputDir != "" {
			req.Payload["output_dir"] = outputDir
		}
		if modelDir != "" {
			req.Payload["model_dir"] = modelDir
		}
		return req, nil
	case agent.ActionSessionExecute:
		if sessionID == "" || planID == "" {
			return engine.Request{}, fmt.Errorf("for agent.session.execute, -session-id and -plan-id are required")
		}
		req.Payload["session_id"] = sessionID
		req.Payload["plan_id"] = planID
		if outputDir != "" {
			req.Payload["output_dir"] = outputDir
		}
		return req, nil
	}

	switch engine.ActionName(normalizedAction) {
	case engine.ActionTrain:
		if csv == "" || target == "" {
			return engine.Request{}, fmt.Errorf("for train action, -csv and -target are required")
		}
		req.Payload["csv_path"] = csv
		req.Payload["target_col"] = target

		if outputDir != "" {
			if parallel > 1 {
				req.Payload["output_dir"] = filepath.Join(outputDir, fmt.Sprintf("task-%d", index))
			} else {
				req.Payload["output_dir"] = outputDir
			}
		}
	case engine.ActionRepair:
		if modelDir == "" {
			return engine.Request{}, fmt.Errorf("for repair action, -model-dir is required")
		}
		req.Payload["model_dir"] = modelDir
		req.Payload["sample_index"] = sampleIndex
		if maxChanges > 0 {
			req.Payload["max_changes"] = maxChanges
		}
		if kNeighbors > 0 {
			req.Payload["k_neighbors"] = kNeighbors
		}
		if outputDir != "" {
			if parallel > 1 {
				req.Payload["output_dir"] = filepath.Join(outputDir, fmt.Sprintf("task-%d", index))
			} else {
				req.Payload["output_dir"] = outputDir
			}
		}
	case engine.ActionRepairWithGower:
		if csv == "" {
			return engine.Request{}, fmt.Errorf("for repair_with_gower action, -csv is required")
		}
		req.Payload["csv_path"] = csv
		req.Payload["issue_ids"] = []string{}
		req.Payload["write_output"] = true
		if modelDir != "" {
			req.Payload["model_dir"] = modelDir
		}
		if outputDir != "" {
			if parallel > 1 {
				req.Payload["output_dir"] = filepath.Join(outputDir, fmt.Sprintf("task-%d", index))
			} else {
				req.Payload["output_dir"] = outputDir
			}
		}
	}

	return req, nil
}

func main() {
	ensureDefaultGoLogFile()

	actionChoices := append(engine.KnownActionStrings(), agent.ActionSessionPlan, agent.ActionSessionExecute, agent.ActionSessionAuto)
	action := flag.String("action", string(engine.ActionHealth), "Action to run: "+strings.Join(actionChoices, ", "))
	csv := flag.String("csv", "", "CSV path for train action")
	target := flag.String("target", "", "Target column for train action")
	modelDir := flag.String("model-dir", "", "Model artifacts directory for repair action")
	goal := flag.String("goal", "", "User goal for agent.session.plan / agent.session.auto")
	sessionID := flag.String("session-id", "", "Session id for agent.session.plan / agent.session.auto / agent.session.execute")
	planID := flag.String("plan-id", "", "Plan id for agent.session.execute")
	sampleIndex := flag.Int("sample-index", 0, "Sample index in test_data.pkl for repair action")
	maxChanges := flag.Int("max-changes", 3, "Maximum number of feature edits for repair action")
	kNeighbors := flag.Int("k-neighbors", 9, "Nearest healthy neighbors for repair action")
	outputDir := flag.String("output", "", "Base output directory for model artifacts")
	engineScript := flag.String("engine", "../core/python_engine/engine_main.py", "Path to python engine script")
	historyDB := flag.String("history-db", "../../outputs/appshell/task_history.sqlite", "Path to task history sqlite db")
	historyKeep := flag.Int("history-keep", 100, "Keep only latest N history records (<=0 means no trim)")
	timeout := flag.Duration("timeout", 90*time.Second, "Task timeout")
	parallel := flag.Int("parallel", 1, "Number of tasks to submit")
	cancelAfter := flag.Duration("cancel-after", 0, "Cancel the first task after this duration")
	flag.Parse()

	if *parallel <= 0 {
		fmt.Fprintln(os.Stderr, "-parallel must be >= 1")
		os.Exit(1)
	}

	absEngine, err := filepath.Abs(*engineScript)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve engine path failed: %v\n", err)
		os.Exit(1)
	}

	absHistoryDB, err := filepath.Abs(*historyDB)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve history db path failed: %v\n", err)
		os.Exit(1)
	}
	historyStore, err := task.NewSQLiteHistoryStore(absHistoryDB, *historyKeep)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init history store failed: %v\n", err)
		os.Exit(1)
	}
	agentStore, err := agent.NewSQLiteStore(absHistoryDB)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init agent store failed: %v\n", err)
		os.Exit(1)
	}
	baseRunner := engine.NewRunner(absEngine)
	if *timeout > 0 {
		baseRunner.DefaultTimeout = *timeout
	}
	planner, langGraphManager := agent.NewPhaseBPlannerStack(absEngine)
	runner := agent.NewRuntimeRunner(baseRunner, agentStore, planner)
	svc := task.NewServiceWithConfig(runner, task.Config{
		HistoryStore: historyStore,
	})
	defer svc.Close()
	defer agentStore.Close()
	if langGraphManager != nil {
		defer langGraphManager.Close()
	}

	taskIDs := make([]string, 0, *parallel)
	for i := 0; i < *parallel; i++ {
		req, err := buildRequest(
			*action,
			*csv,
			*target,
			*outputDir,
			*goal,
			*sessionID,
			*planID,
			*modelDir,
			*sampleIndex,
			*maxChanges,
			*kNeighbors,
			i+1,
			*parallel,
		)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}

		taskID, err := svc.RunTask(req, *timeout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "submit task failed: %v\n", err)
			os.Exit(1)
		}
		taskIDs = append(taskIDs, taskID)
		fmt.Printf("task submitted: %s\n", taskID)
	}

	if *cancelAfter > 0 && len(taskIDs) > 0 {
		firstTaskID := taskIDs[0]
		go func() {
			time.Sleep(*cancelAfter)
			ok := svc.CancelTask(firstTaskID)
			fmt.Printf("cancel requested: task=%s success=%v\n", firstTaskID, ok)
		}()
	}

	pending := make(map[string]struct{}, len(taskIDs))
	for _, id := range taskIDs {
		pending[id] = struct{}{}
	}
	hasFailure := false

	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		for id := range pending {
			t, ok := svc.GetTaskStatus(id)
			if !ok {
				fmt.Fprintf(os.Stderr, "task not found: %s\n", id)
				os.Exit(1)
			}

			fmt.Printf("task=%s status=%s\n", id, t.Status)
			switch t.Status {
			case task.StatusSucceeded:
				fmt.Printf("task=%s response:\n%s\n", id, mustJSON(t.Response))
				delete(pending, id)
			case task.StatusFailed, task.StatusCanceled, task.StatusTimedOut:
				hasFailure = true
				fmt.Printf("task=%s error=%s\n", id, t.Error)
				if t.Response.TaskID != "" {
					fmt.Printf("task=%s response:\n%s\n", id, mustJSON(t.Response))
				}
				delete(pending, id)
			}
		}

		if len(pending) == 0 {
			if hasFailure {
				os.Exit(1)
			}
			return
		}
	}
}
