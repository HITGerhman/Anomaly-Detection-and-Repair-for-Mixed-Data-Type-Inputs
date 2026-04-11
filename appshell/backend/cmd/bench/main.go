package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"appshell/backend/internal/agent"
	"appshell/backend/internal/engine"
	"appshell/backend/internal/task"
)

type benchmarkReport struct {
	GeneratedAt string                    `json:"generated_at"`
	Hostname    string                    `json:"hostname,omitempty"`
	WorkingDir  string                    `json:"working_dir"`
	Config      benchmarkConfig           `json:"config"`
	Scheduler   []schedulerBenchmarkCase  `json:"scheduler,omitempty"`
	Approval    *approvalBenchmarkResult  `json:"approval,omitempty"`
	AgentPlan   *agentPlanBenchmarkResult `json:"agent_plan,omitempty"`
}

type benchmarkConfig struct {
	Scenario             string `json:"scenario"`
	SchedulerTasks       int    `json:"scheduler_tasks"`
	SchedulerDelayMS     int    `json:"scheduler_delay_ms"`
	SchedulerConcurrency []int  `json:"scheduler_concurrency"`
	ApprovalIterations   int    `json:"approval_iterations"`
	AgentPlanWarmups     int    `json:"agent_plan_warmups"`
	AgentPlanIterations  int    `json:"agent_plan_iterations"`
	AgentPlanCSV         string `json:"agent_plan_csv,omitempty"`
	AgentPlanModelDir    string `json:"agent_plan_model_dir,omitempty"`
	AgentRetrieveMode    string `json:"agent_retrieve_mode,omitempty"`
	LangGraphEnabled     bool   `json:"langgraph_enabled"`
	PythonBin            string `json:"python_bin,omitempty"`
	OutputPath           string `json:"output_path"`
}

type schedulerBenchmarkCase struct {
	Kind                string  `json:"kind"`
	Workers             int     `json:"workers"`
	Tasks               int     `json:"tasks"`
	SimulatedDelayMS    float64 `json:"simulated_delay_ms"`
	TotalDurationMS     float64 `json:"total_duration_ms"`
	ThroughputTPS       float64 `json:"throughput_tps"`
	AvgQueueWaitMS      float64 `json:"avg_queue_wait_ms"`
	P95QueueWaitMS      float64 `json:"p95_queue_wait_ms"`
	AvgRuntimeMS        float64 `json:"avg_runtime_ms"`
	P95RuntimeMS        float64 `json:"p95_runtime_ms"`
	AvgEndToEndMS       float64 `json:"avg_end_to_end_ms"`
	P95EndToEndMS       float64 `json:"p95_end_to_end_ms"`
	MaxRunningObserved  int     `json:"max_running_observed"`
	SpeedupVsSingleNode float64 `json:"speedup_vs_single_worker,omitempty"`
}

type approvalBenchmarkResult struct {
	Kind                string  `json:"kind"`
	Iterations          int     `json:"iterations"`
	SuccessCount        int     `json:"success_count"`
	FailureCount        int     `json:"failure_count"`
	SuccessRate         float64 `json:"success_rate"`
	AvgPauseLatencyMS   float64 `json:"avg_pause_latency_ms"`
	P95PauseLatencyMS   float64 `json:"p95_pause_latency_ms"`
	AvgResumeLatencyMS  float64 `json:"avg_resume_latency_ms"`
	P95ResumeLatencyMS  float64 `json:"p95_resume_latency_ms"`
	AvgRoundTripMS      float64 `json:"avg_round_trip_ms"`
	P95RoundTripMS      float64 `json:"p95_round_trip_ms"`
	TracePersistencePct float64 `json:"trace_persistence_pct"`
}

type agentPlanBenchmarkResult struct {
	Kind                 string             `json:"kind"`
	WarmupCount          int                `json:"warmup_count"`
	IterationCount       int                `json:"iteration_count"`
	CSVPath              string             `json:"csv_path"`
	ModelDir             string             `json:"model_dir,omitempty"`
	RetrieveMode         string             `json:"retrieve_mode"`
	LangGraphEnabled     bool               `json:"langgraph_enabled"`
	AvgTotalMS           float64            `json:"avg_total_ms"`
	P95TotalMS           float64            `json:"p95_total_ms"`
	AvgQueueWaitMS       float64            `json:"avg_queue_wait_ms"`
	P95QueueWaitMS       float64            `json:"p95_queue_wait_ms"`
	AvgEndToEndMS        float64            `json:"avg_end_to_end_ms"`
	P95EndToEndMS        float64            `json:"p95_end_to_end_ms"`
	AvgStageDurationsMS  map[string]float64 `json:"avg_stage_durations_ms"`
	SlowestStage         string             `json:"slowest_stage"`
	SlowestStageAvgMS    float64            `json:"slowest_stage_avg_ms"`
	SelectedSourceCounts map[string]int     `json:"selected_source_counts"`
}

type syntheticLatencyRunner struct {
	delay      time.Duration
	running    int32
	maxRunning int32
}

func (r *syntheticLatencyRunner) Run(ctx context.Context, req engine.Request) (engine.Response, error) {
	current := atomic.AddInt32(&r.running, 1)
	defer atomic.AddInt32(&r.running, -1)

	for {
		maxObserved := atomic.LoadInt32(&r.maxRunning)
		if current <= maxObserved || atomic.CompareAndSwapInt32(&r.maxRunning, maxObserved, current) {
			break
		}
	}

	select {
	case <-time.After(r.delay):
	case <-ctx.Done():
		return engine.Response{}, ctx.Err()
	}

	return engine.Response{
		TaskID:     req.TaskID,
		Status:     "ok",
		Result:     map[string]any{"ok": true},
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		DurationMS: int(r.delay.Milliseconds()),
	}, nil
}

type syntheticApprovalRunner struct {
	mu                   sync.Mutex
	tempDir              string
	sourceCSV            string
	observer             engine.StderrObserver
	calls                []engine.Request
	highRiskColumns      []string
	previewCanExecute    bool
	postScanAccept       bool
	omitRollbackMetadata bool
}

func (r *syntheticApprovalRunner) SetStderrObserver(observer engine.StderrObserver) {
	r.observer = observer
}

func (r *syntheticApprovalRunner) Run(_ context.Context, req engine.Request) (engine.Response, error) {
	r.mu.Lock()
	r.calls = append(r.calls, req)
	r.mu.Unlock()

	if r.observer != nil {
		r.observer(engine.StderrEvent{
			TaskID: req.TaskID,
			Parsed: map[string]any{
				"event":     "stage_progress",
				"stage":     "load_csv",
				"phase":     "start",
				"progress":  12,
				"message":   "synthetic benchmark tool started",
				"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
			},
			ObservedAt: time.Now(),
		})
	}

	switch req.Action {
	case string(engine.ActionScanFile):
		return r.scanResponse(req), nil
	case string(engine.ActionRepairBatch):
		return r.repairResponse(req, "rule"), nil
	case string(engine.ActionRepairWithGower):
		return r.repairResponse(req, "gower"), nil
	case string(engine.ActionRollbackRepairBatch):
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"restored_csv":   asString(req.Payload["target_csv"]),
				"restore_target": asString(req.Payload["restore_target"]),
			},
			Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		}, nil
	default:
		return engine.Response{
			TaskID:     req.TaskID,
			Status:     "ok",
			Result:     map[string]any{"action": req.Action},
			Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
			DurationMS: 1,
		}, nil
	}
}

func (r *syntheticApprovalRunner) scanResponse(req engine.Request) engine.Response {
	csvPath := asString(req.Payload["csv_path"])
	if csvPath != "" && filepath.Clean(csvPath) != filepath.Clean(r.sourceCSV) {
		issues := []any{}
		issueCount := 0
		if !r.postScanAccept {
			issues = []any{
				map[string]any{"issue_id": "post-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.9},
			}
			issueCount = len(issues)
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"issue_count": issueCount,
				"issues":      issues,
				"scan_summary": map[string]any{
					"total_issues": issueCount,
				},
				"data_profile": map[string]any{
					"rows":    1,
					"columns": 2,
				},
			},
			Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
			DurationMS: 1,
		}
	}

	return engine.Response{
		TaskID: req.TaskID,
		Status: "ok",
		Result: map[string]any{
			"issue_count": 2,
			"issues": []any{
				map[string]any{"issue_id": "issue-1", "issue_type": "missing_values", "column": "age", "risk_level": "high", "issue_score": 0.8},
				map[string]any{"issue_id": "issue-2", "issue_type": "rare_category", "column": "city", "risk_level": "medium", "issue_score": 0.4},
			},
			"scan_summary": map[string]any{
				"total_issues":      2,
				"high_risk_columns": append([]string{}, r.highRiskColumns...),
			},
			"data_profile": map[string]any{
				"rows":    1,
				"columns": 2,
			},
		},
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		DurationMS: 1,
	}
}

func (r *syntheticApprovalRunner) repairResponse(req engine.Request, source string) engine.Response {
	planOnly, _ := req.Payload["plan_only"].(bool)
	if planOnly {
		if !r.previewCanExecute {
			return engine.Response{
				TaskID: req.TaskID,
				Status: "ok",
				Result: map[string]any{
					"comparison": map[string]any{
						"before_issue_count":   2,
						"after_issue_count":    2,
						"resolved_issue_count": 0,
						"changed_cell_count":   0,
					},
					"skipped_issues": []any{
						map[string]any{"issue_id": "issue-1", "reason": "strategy_disabled"},
						map[string]any{"issue_id": "issue-2", "reason": "strategy_disabled"},
					},
				},
				Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
				DurationMS: 1,
			}
		}

		afterIssueCount := 1
		resolvedIssueCount := 1
		if source == "gower" {
			afterIssueCount = 0
			resolvedIssueCount = 2
		}
		return engine.Response{
			TaskID: req.TaskID,
			Status: "ok",
			Result: map[string]any{
				"comparison": map[string]any{
					"before_issue_count":   2,
					"after_issue_count":    afterIssueCount,
					"resolved_issue_count": resolvedIssueCount,
					"changed_cell_count":   resolvedIssueCount,
				},
				"applied_repairs": []any{
					map[string]any{"issue_id": "issue-1", "resolved_count": 1, "rows_touched": 1, "candidate_confidence": 0.9},
				},
			},
			Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
			DurationMS: 1,
		}
	}

	outputCSV := asString(req.Payload["output_csv"])
	if outputCSV == "" {
		outputCSV = filepath.Join(r.tempDir, source+".repaired.csv")
	}
	if err := os.MkdirAll(filepath.Dir(outputCSV), 0o755); err != nil {
		return engine.Response{
			TaskID: req.TaskID,
			Status: "error",
			Error: &engine.ErrorBody{
				Code:    "MKDIR_FAILED",
				Message: err.Error(),
			},
			Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		}
	}
	content := []byte("age,city\n20,a\n")
	_ = os.WriteFile(outputCSV, content, 0o644)

	rollbackDir := asString(req.Payload["rollback_dir"])
	if rollbackDir == "" {
		rollbackDir = filepath.Join(r.tempDir, ".rollback")
	}
	_ = os.MkdirAll(rollbackDir, 0o755)
	baseName := strings.TrimSuffix(filepath.Base(outputCSV), filepath.Ext(outputCSV))
	manifestPath := filepath.Join(rollbackDir, baseName+".json")
	backupCSV := filepath.Join(rollbackDir, baseName+".backup.csv")
	_ = os.WriteFile(manifestPath, []byte("{}"), 0o644)
	_ = os.WriteFile(backupCSV, content, 0o644)

	rollback := map[string]any{}
	if !r.omitRollbackMetadata {
		rollback = map[string]any{
			"rollback_id":      baseName,
			"manifest_path":    manifestPath,
			"backup_csv":       backupCSV,
			"manifest_version": 2,
			"source_tool_id": map[string]string{
				"rule":  "engine.repair_batch",
				"gower": "engine.repair_with_gower",
			}[source],
		}
	}

	return engine.Response{
		TaskID: req.TaskID,
		Status: "ok",
		Result: map[string]any{
			"output_csv":          outputCSV,
			"applied_issue_count": 2,
			"rollback":            rollback,
			"comparison": map[string]any{
				"before_issue_count":   2,
				"after_issue_count":    0,
				"resolved_issue_count": 2,
				"changed_cell_count":   2,
			},
		},
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		DurationMS: 1,
	}
}

func main() {
	scenario := flag.String("scenario", "all", "Benchmark scenario: all, synthetic, scheduler, approval, agent-plan")
	schedulerTasks := flag.Int("scheduler-tasks", 60, "Synthetic scheduler task count")
	schedulerDelayMS := flag.Int("scheduler-delay-ms", 40, "Synthetic per-task runtime in milliseconds")
	schedulerConcurrency := flag.String("scheduler-concurrency", "1,3,6", "Comma-separated worker counts for synthetic scheduler benchmark")
	approvalIterations := flag.Int("approval-iterations", 30, "Synthetic approval/resume iterations")
	planWarmups := flag.Int("plan-warmups", 1, "Warmup iterations for end-to-end agent planning benchmark")
	planIterations := flag.Int("plan-iterations", 3, "Measured iterations for end-to-end agent planning benchmark")
	planCSV := flag.String("plan-csv", "../../data/raw/simple_obvious_anomaly.csv", "CSV path for end-to-end agent planning benchmark")
	planModelDir := flag.String("plan-model-dir", "../../outputs/results/wails_mvp", "Model dir for end-to-end agent planning benchmark")
	agentRetrieveMode := flag.String("agent-retrieve-mode", "parallel", "Retrieve preview mode for agent planning benchmark: sequential or parallel")
	pythonBin := flag.String("python-bin", "", "Python executable used by the engine and LangGraph sidecar")
	langGraphEnabled := flag.Bool("langgraph-enabled", false, "Whether to enable LangGraph during end-to-end planning benchmark")
	outputPath := flag.String("output", "", "Optional JSON output path")
	flag.Parse()

	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve working directory failed: %v\n", err)
		os.Exit(1)
	}

	concurrencySet, err := parseConcurrencyList(*schedulerConcurrency)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid -scheduler-concurrency: %v\n", err)
		os.Exit(1)
	}
	if *schedulerTasks <= 0 || *schedulerDelayMS <= 0 || *approvalIterations <= 0 || *planWarmups < 0 || *planIterations < 0 {
		fmt.Fprintln(os.Stderr, "benchmark counts must be positive and warmups must be >= 0")
		os.Exit(1)
	}

	if strings.TrimSpace(*outputPath) == "" {
		*outputPath = filepath.Clean(filepath.Join("..", "..", "..", "outputs", "results", fmt.Sprintf("backend_benchmark_%s.json", time.Now().UTC().Format("20060102_150405"))))
	}
	if err := os.MkdirAll(filepath.Dir(*outputPath), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "create output directory failed: %v\n", err)
		os.Exit(1)
	}

	if !*langGraphEnabled {
		_ = os.Setenv("APPSHELL_LANGGRAPH_ENABLED", "0")
	} else {
		_ = os.Setenv("APPSHELL_LANGGRAPH_ENABLED", "1")
	}
	if strings.TrimSpace(*pythonBin) != "" {
		_ = os.Setenv("APPSHELL_LANGGRAPH_PYTHON_BIN", strings.TrimSpace(*pythonBin))
	}

	report := benchmarkReport{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Hostname:    hostname(),
		WorkingDir:  cwd,
		Config: benchmarkConfig{
			Scenario:             strings.TrimSpace(*scenario),
			SchedulerTasks:       *schedulerTasks,
			SchedulerDelayMS:     *schedulerDelayMS,
			SchedulerConcurrency: concurrencySet,
			ApprovalIterations:   *approvalIterations,
			AgentPlanWarmups:     *planWarmups,
			AgentPlanIterations:  *planIterations,
			AgentPlanCSV:         *planCSV,
			AgentPlanModelDir:    *planModelDir,
			AgentRetrieveMode:    strings.ToLower(strings.TrimSpace(*agentRetrieveMode)),
			LangGraphEnabled:     *langGraphEnabled,
			PythonBin:            strings.TrimSpace(*pythonBin),
			OutputPath:           *outputPath,
		},
	}

	normalizedScenario := normalizeScenario(*scenario)
	if normalizedScenario == "" {
		fmt.Fprintf(os.Stderr, "unsupported scenario %q\n", *scenario)
		os.Exit(1)
	}

	if normalizedScenario == "all" || normalizedScenario == "synthetic" || normalizedScenario == "scheduler" {
		report.Scheduler, err = runSchedulerBenchmarks(concurrencySet, *schedulerTasks, time.Duration(*schedulerDelayMS)*time.Millisecond)
		if err != nil {
			fmt.Fprintf(os.Stderr, "scheduler benchmark failed: %v\n", err)
			os.Exit(1)
		}
	}
	if normalizedScenario == "all" || normalizedScenario == "synthetic" || normalizedScenario == "approval" {
		report.Approval, err = runApprovalBenchmark(*approvalIterations)
		if err != nil {
			fmt.Fprintf(os.Stderr, "approval benchmark failed: %v\n", err)
			os.Exit(1)
		}
	}
	if normalizedScenario == "all" || normalizedScenario == "agent-plan" {
		report.AgentPlan, err = runAgentPlanBenchmark(*planCSV, *planModelDir, strings.TrimSpace(*pythonBin), strings.ToLower(strings.TrimSpace(*agentRetrieveMode)), *langGraphEnabled, *planWarmups, *planIterations)
		if err != nil {
			fmt.Fprintf(os.Stderr, "agent-plan benchmark failed: %v\n", err)
			os.Exit(1)
		}
	}

	payload, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal report failed: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*outputPath, append(payload, '\n'), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write report failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(payload))
	fmt.Printf("\nSaved benchmark report to %s\n", *outputPath)
}

func runSchedulerBenchmarks(concurrencySet []int, taskCount int, runnerDelay time.Duration) ([]schedulerBenchmarkCase, error) {
	results := make([]schedulerBenchmarkCase, 0, len(concurrencySet))
	var baselineDuration float64

	for _, workers := range concurrencySet {
		runner := &syntheticLatencyRunner{delay: runnerDelay}
		svc := task.NewServiceWithConfig(runner, task.Config{
			MaxConcurrency: workers,
			QueueSize:      taskCount + workers + 4,
		})

		ids := make([]string, 0, taskCount)
		submittedAt := time.Now()
		for i := 0; i < taskCount; i++ {
			taskID, err := svc.RunTask(engine.Request{
				Action: string(engine.ActionHealth),
				Payload: map[string]any{
					"task_index": i,
				},
			}, 5*time.Second)
			if err != nil {
				svc.Close()
				return nil, err
			}
			ids = append(ids, taskID)
		}

		snapshots, err := waitForAllTasks(svc, ids, 30*time.Second)
		svc.Close()
		if err != nil {
			return nil, err
		}

		queueWaits := make([]float64, 0, len(snapshots))
		runtimes := make([]float64, 0, len(snapshots))
		endToEnd := make([]float64, 0, len(snapshots))
		latestEnd := submittedAt
		for _, snapshot := range snapshots {
			if snapshot.StartedAt.After(latestEnd) {
				latestEnd = snapshot.StartedAt
			}
			if snapshot.EndedAt.After(latestEnd) {
				latestEnd = snapshot.EndedAt
			}
			queueWaits = append(queueWaits, snapshot.StartedAt.Sub(snapshot.CreatedAt).Seconds()*1000)
			runtimes = append(runtimes, snapshot.EndedAt.Sub(snapshot.StartedAt).Seconds()*1000)
			endToEnd = append(endToEnd, snapshot.EndedAt.Sub(snapshot.CreatedAt).Seconds()*1000)
		}
		totalDurationMS := latestEnd.Sub(submittedAt).Seconds() * 1000
		result := schedulerBenchmarkCase{
			Kind:               "synthetic_scheduler",
			Workers:            workers,
			Tasks:              taskCount,
			SimulatedDelayMS:   float64(runnerDelay.Milliseconds()),
			TotalDurationMS:    round2(totalDurationMS),
			ThroughputTPS:      round2(float64(taskCount) / math.Max(totalDurationMS/1000.0, 0.001)),
			AvgQueueWaitMS:     round2(avg(queueWaits)),
			P95QueueWaitMS:     round2(percentile(queueWaits, 95)),
			AvgRuntimeMS:       round2(avg(runtimes)),
			P95RuntimeMS:       round2(percentile(runtimes, 95)),
			AvgEndToEndMS:      round2(avg(endToEnd)),
			P95EndToEndMS:      round2(percentile(endToEnd, 95)),
			MaxRunningObserved: int(atomic.LoadInt32(&runner.maxRunning)),
		}
		if workers == 1 {
			baselineDuration = result.TotalDurationMS
		}
		if baselineDuration > 0 {
			result.SpeedupVsSingleNode = round2(baselineDuration / math.Max(result.TotalDurationMS, 0.001))
		}
		results = append(results, result)
	}

	return results, nil
}

func runApprovalBenchmark(iterations int) (*approvalBenchmarkResult, error) {
	tempDir, err := os.MkdirTemp("", "appshell-bench-approval-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	sourceCSV := filepath.Join(tempDir, "source.csv")
	if err := os.WriteFile(sourceCSV, []byte("age,city\n20,a\n"), 0o644); err != nil {
		return nil, err
	}

	store, err := agent.NewSQLiteStore(filepath.Join(tempDir, "agent.sqlite"))
	if err != nil {
		return nil, err
	}
	defer store.Close()

	base := &syntheticApprovalRunner{
		tempDir:           tempDir,
		sourceCSV:         sourceCSV,
		highRiskColumns:   []string{"age"},
		previewCanExecute: true,
		postScanAccept:    true,
	}
	runner := agent.NewRuntimeRunner(base, store, agent.NewMockPlanner())

	pauseLatencies := make([]float64, 0, iterations)
	resumeLatencies := make([]float64, 0, iterations)
	roundTrips := make([]float64, 0, iterations)
	tracePersistence := 0
	successCount := 0

	for i := 0; i < iterations; i++ {
		pauseStarted := time.Now()
		paused, err := runner.Run(context.Background(), engine.Request{
			TaskID: fmt.Sprintf("bench-auto-%02d", i),
			Action: agent.ActionSessionAuto,
			Payload: map[string]any{
				"csv_path": sourceCSV,
			},
		})
		pauseDurationMS := time.Since(pauseStarted).Seconds() * 1000
		if err != nil || paused.Status != "ok" {
			continue
		}

		agentBlock := mapFromAny(paused.Result["agent"])
		plan, ok := agentBlock["plan"].(agent.AgentPlan)
		if !ok {
			continue
		}
		sessionID := asString(agentBlock["session_id"])
		if sessionID == "" || plan.PlanID == "" {
			continue
		}

		resumeStarted := time.Now()
		resumed, err := runner.Run(context.Background(), engine.Request{
			TaskID: fmt.Sprintf("bench-approve-%02d", i),
			Action: agent.ActionSessionApprove,
			Payload: map[string]any{
				"session_id": sessionID,
				"plan_id":    plan.PlanID,
				"decision":   "approve",
			},
		})
		resumeDurationMS := time.Since(resumeStarted).Seconds() * 1000
		if err != nil || resumed.Status != "ok" {
			continue
		}

		safety := mapFromAny(resumed.Result["safety"])
		if asString(safety["final_verdict"]) != "accepted" {
			continue
		}

		trace, err := store.ListTrace(context.Background(), sessionID)
		if err == nil && len(trace) > 0 {
			tracePersistence++
		}

		successCount++
		pauseLatencies = append(pauseLatencies, pauseDurationMS)
		resumeLatencies = append(resumeLatencies, resumeDurationMS)
		roundTrips = append(roundTrips, pauseDurationMS+resumeDurationMS)
	}

	result := &approvalBenchmarkResult{
		Kind:                "synthetic_approval_resume",
		Iterations:          iterations,
		SuccessCount:        successCount,
		FailureCount:        iterations - successCount,
		SuccessRate:         round4(float64(successCount) / math.Max(float64(iterations), 1)),
		AvgPauseLatencyMS:   round2(avg(pauseLatencies)),
		P95PauseLatencyMS:   round2(percentile(pauseLatencies, 95)),
		AvgResumeLatencyMS:  round2(avg(resumeLatencies)),
		P95ResumeLatencyMS:  round2(percentile(resumeLatencies, 95)),
		AvgRoundTripMS:      round2(avg(roundTrips)),
		P95RoundTripMS:      round2(percentile(roundTrips, 95)),
		TracePersistencePct: round4(float64(tracePersistence) / math.Max(float64(iterations), 1)),
	}
	return result, nil
}

func runAgentPlanBenchmark(csvPath string, modelDir string, pythonBin string, retrieveMode string, langGraphEnabled bool, warmups int, iterations int) (*agentPlanBenchmarkResult, error) {
	if iterations == 0 {
		return &agentPlanBenchmarkResult{
			Kind:             "e2e_agent_plan",
			WarmupCount:      warmups,
			IterationCount:   0,
			CSVPath:          csvPath,
			ModelDir:         modelDir,
			RetrieveMode:     retrieveMode,
			LangGraphEnabled: langGraphEnabled,
		}, nil
	}

	absCSV, err := filepath.Abs(csvPath)
	if err != nil {
		return nil, err
	}
	absModelDir := ""
	if strings.TrimSpace(modelDir) != "" {
		absModelDir, err = filepath.Abs(modelDir)
		if err != nil {
			return nil, err
		}
	}
	engineScript, err := filepath.Abs(filepath.Join("..", "core", "python_engine", "engine_main.py"))
	if err != nil {
		return nil, err
	}

	tempDir, err := os.MkdirTemp("", "appshell-bench-plan-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tempDir)

	historyDB := filepath.Join(tempDir, "task_history.sqlite")
	historyStore, err := task.NewSQLiteHistoryStore(historyDB, 200)
	if err != nil {
		return nil, err
	}
	defer historyStore.Close()

	agentStore, err := agent.NewSQLiteStore(historyDB)
	if err != nil {
		return nil, err
	}
	defer agentStore.Close()

	baseRunner := engine.NewRunner(engineScript)
	if strings.TrimSpace(pythonBin) != "" {
		baseRunner.PythonBin = strings.TrimSpace(pythonBin)
	}
	planner, langGraphManager := agent.NewPhaseBPlannerStack(engineScript)
	if langGraphManager != nil {
		defer langGraphManager.Close()
	}

	runtimeRunner := agent.NewRuntimeRunner(baseRunner, agentStore, planner)
	svc := task.NewServiceWithConfig(runtimeRunner, task.Config{
		MaxConcurrency: 1,
		QueueSize:      iterations + warmups + 4,
		HistoryStore:   historyStore,
	})
	defer svc.Close()

	totalDurations := make([]float64, 0, iterations)
	queueWaits := make([]float64, 0, iterations)
	endToEnd := make([]float64, 0, iterations)
	stageTotals := map[string][]float64{}
	selectedSourceCounts := map[string]int{}

	runOnce := func(index int, measured bool) error {
		taskID, err := svc.RunTask(engine.Request{
			Action: agent.ActionSessionPlan,
			Payload: map[string]any{
				"csv_path":            absCSV,
				"model_dir":           absModelDir,
				"agent_retrieve_mode": retrieveMode,
			},
		}, 3*time.Minute)
		if err != nil {
			return err
		}
		snapshots, err := waitForAllTasks(svc, []string{taskID}, 3*time.Minute)
		if err != nil {
			return err
		}
		snapshot := snapshots[0]
		if snapshot.Status != task.StatusSucceeded {
			return fmt.Errorf("plan task %s ended with status %s", taskID, snapshot.Status)
		}
		if !measured {
			return nil
		}

		totalDurations = append(totalDurations, float64(snapshot.Response.DurationMS))
		queueWaits = append(queueWaits, snapshot.StartedAt.Sub(snapshot.CreatedAt).Seconds()*1000)
		endToEnd = append(endToEnd, snapshot.EndedAt.Sub(snapshot.CreatedAt).Seconds()*1000)
		for stage, duration := range snapshot.Progress.StageDurationsMS {
			stageTotals[stage] = append(stageTotals[stage], float64(duration))
		}
		agentBlock := mapFromAny(snapshot.Response.Result["agent"])
		plan, ok := agentBlock["plan"].(agent.AgentPlan)
		if ok {
			selectedSourceCounts[strings.TrimSpace(plan.SelectedSource)]++
		}
		return nil
	}

	for i := 0; i < warmups; i++ {
		if err := runOnce(i, false); err != nil {
			return nil, err
		}
	}
	for i := 0; i < iterations; i++ {
		if err := runOnce(i, true); err != nil {
			return nil, err
		}
	}

	avgStages := make(map[string]float64, len(stageTotals))
	slowestStage := ""
	slowestStageAvgMS := 0.0
	for stage, values := range stageTotals {
		stageAvg := round2(avg(values))
		avgStages[stage] = stageAvg
		if stageAvg > slowestStageAvgMS {
			slowestStageAvgMS = stageAvg
			slowestStage = stage
		}
	}

	return &agentPlanBenchmarkResult{
		Kind:                 "e2e_agent_plan",
		WarmupCount:          warmups,
		IterationCount:       iterations,
		CSVPath:              absCSV,
		ModelDir:             absModelDir,
		RetrieveMode:         retrieveMode,
		LangGraphEnabled:     langGraphEnabled,
		AvgTotalMS:           round2(avg(totalDurations)),
		P95TotalMS:           round2(percentile(totalDurations, 95)),
		AvgQueueWaitMS:       round2(avg(queueWaits)),
		P95QueueWaitMS:       round2(percentile(queueWaits, 95)),
		AvgEndToEndMS:        round2(avg(endToEnd)),
		P95EndToEndMS:        round2(percentile(endToEnd, 95)),
		AvgStageDurationsMS:  avgStages,
		SlowestStage:         slowestStage,
		SlowestStageAvgMS:    round2(slowestStageAvgMS),
		SelectedSourceCounts: selectedSourceCounts,
	}, nil
}

func waitForAllTasks(svc *task.Service, ids []string, timeout time.Duration) ([]task.Task, error) {
	deadline := time.Now().Add(timeout)
	pending := map[string]struct{}{}
	for _, id := range ids {
		pending[id] = struct{}{}
	}
	results := map[string]task.Task{}

	for len(pending) > 0 {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out while waiting for %d benchmark tasks", len(pending))
		}
		for id := range pending {
			snapshot, ok := svc.GetTaskStatus(id)
			if !ok || snapshot == nil {
				continue
			}
			switch snapshot.Status {
			case task.StatusSucceeded, task.StatusFailed, task.StatusCanceled, task.StatusTimedOut:
				results[id] = *snapshot
				delete(pending, id)
			}
		}
		time.Sleep(5 * time.Millisecond)
	}

	out := make([]task.Task, 0, len(ids))
	for _, id := range ids {
		out = append(out, results[id])
	}
	return out, nil
}

func parseConcurrencyList(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		text := strings.TrimSpace(part)
		if text == "" {
			continue
		}
		value, err := strconv.Atoi(text)
		if err != nil || value <= 0 {
			return nil, fmt.Errorf("invalid worker count %q", text)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one worker count is required")
	}
	sort.Ints(out)
	return out, nil
}

func normalizeScenario(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "all", "synthetic", "scheduler", "approval", "agent-plan":
		return strings.ToLower(strings.TrimSpace(raw))
	default:
		return ""
	}
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]float64{}, values...)
	sort.Float64s(ordered)
	if len(ordered) == 1 {
		return ordered[0]
	}
	position := (p / 100) * float64(len(ordered)-1)
	lower := int(math.Floor(position))
	upper := int(math.Ceil(position))
	if lower == upper {
		return ordered[lower]
	}
	weight := position - float64(lower)
	return ordered[lower]*(1-weight) + ordered[upper]*weight
}

func avg(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func round2(value float64) float64 {
	return math.Round(value*100) / 100
}

func round4(value float64) float64 {
	return math.Round(value*10000) / 10000
}

func hostname() string {
	name, err := os.Hostname()
	if err != nil {
		return ""
	}
	return name
}

func asString(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	default:
		if value == nil {
			return ""
		}
		return fmt.Sprint(value)
	}
}

func mapFromAny(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}
