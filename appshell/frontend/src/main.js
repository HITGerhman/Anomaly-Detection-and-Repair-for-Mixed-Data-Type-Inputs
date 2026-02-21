const wizardCard = document.getElementById("wizard-card");
const wizardInner = document.getElementById("wizard-inner");
const frontKicker = document.getElementById("front-kicker");
const frontTitle = document.getElementById("front-title");
const frontSubtitle = document.getElementById("front-subtitle");
const configView = document.getElementById("config-view");
const resultView = document.getElementById("result-view");

const form = document.getElementById("train-form");
const runBtn = document.getElementById("run-btn");
const cancelBtn = document.getElementById("cancel-btn");
const retryBtn = document.getElementById("retry-btn");
const newTaskBtn = document.getElementById("new-task-btn");
const chooseCsvBtn = document.getElementById("choose-csv-btn");
const chooseOutputBtn = document.getElementById("choose-output-btn");
const csvInput = document.getElementById("csv-path");
const targetInput = document.getElementById("target-col");
const outputInput = document.getElementById("output-dir");
const timeoutInput = document.getElementById("timeout-ms");
const csvFileInput = document.getElementById("csv-file-input");

const statusPill = document.getElementById("status-pill");
const statusMessage = document.getElementById("status-message");
const progressFill = document.getElementById("progress-fill");
const taskIdLabel = document.getElementById("task-id-label");
const eventLog = document.getElementById("event-log");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");
const errorHint = document.getElementById("error-hint");

const resultSummary = document.getElementById("result-summary");
const metricsTableBody = document.getElementById("metrics-table-body");
const resultBox = document.getElementById("result-box");
const exportJsonBtn = document.getElementById("export-json-btn");
const exportCsvBtn = document.getElementById("export-csv-btn");
const copyJsonBtn = document.getElementById("copy-json-btn");

const STEP_CONFIG = "config";
const STEP_PROGRESS = "progress";
const STEP_RESULT = "result";

const STEP_ROTATE_DEG = {
  [STEP_CONFIG]: 0,
  [STEP_PROGRESS]: 180,
  [STEP_RESULT]: 360,
};

const STATUS_PROGRESS = {
  idle: 0,
  pending: 16,
  running: 72,
  succeeded: 100,
  failed: 100,
  canceled: 100,
  timed_out: 100,
};

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled", "timed_out"]);

const state = {
  currentStep: STEP_CONFIG,
  currentTaskId: "",
  currentTask: null,
  lastPayload: null,
  availableColumns: [],
  pollingToken: 0,
  mockTasks: new Map(),
};

function hasBinding(methodName) {
  return Boolean(window?.go?.main?.App?.[methodName]);
}

function setWizardStep(step) {
  const normalized = STEP_ROTATE_DEG[step] === undefined ? STEP_CONFIG : step;
  state.currentStep = normalized;

  if (wizardCard) {
    wizardCard.dataset.step = normalized;
  }
  if (wizardInner) {
    wizardInner.style.transform = `rotateY(${STEP_ROTATE_DEG[normalized]}deg)`;
  }

  const isResult = normalized === STEP_RESULT;
  if (configView) {
    configView.classList.toggle("hidden", isResult);
  }
  if (resultView) {
    resultView.classList.toggle("hidden", !isResult);
  }

  if (normalized === STEP_CONFIG) {
    frontKicker.textContent = "STEP 1";
    frontTitle.textContent = "Parameter Setup";
    frontSubtitle.textContent = "Configure training parameters and start the task.";
  } else if (normalized === STEP_RESULT) {
    frontKicker.textContent = "STEP 3";
    frontTitle.textContent = "Result Review & Export";
    frontSubtitle.textContent = "Task finished. Review results and export artifacts.";
  }
}

function emitFrontendLog(message, taskId = "") {
  const payload = {
    timestamp: new Date().toISOString(),
    layer: "frontend",
    event: "ui_event",
    task_id: taskId || "",
    message: String(message || ""),
  };
  try {
    console.info(JSON.stringify(payload));
  } catch {
    console.info("frontend ui_event", payload);
  }
}

function addEvent(message, taskId = state.currentTaskId) {
  if (!eventLog) {
    return;
  }
  const li = document.createElement("li");
  const time = new Date();
  const hh = String(time.getHours()).padStart(2, "0");
  const mm = String(time.getMinutes()).padStart(2, "0");
  const ss = String(time.getSeconds()).padStart(2, "0");
  const normalizedTaskId = (taskId || "").trim();
  const visibleMessage = normalizedTaskId ? `[${normalizedTaskId}] ${message}` : message;
  li.innerHTML = `<span class="event-time">${hh}:${mm}:${ss}</span><span>${visibleMessage}</span>`;
  eventLog.prepend(li);
  emitFrontendLog(message, normalizedTaskId);
}

function setStatus(status, message) {
  const normalized = (status || "idle").toLowerCase();
  const progress = STATUS_PROGRESS[normalized] ?? 0;

  if (statusPill) {
    statusPill.className = `status-pill ${normalized}`;
    statusPill.textContent = normalized;
  }
  if (statusMessage) {
    statusMessage.textContent = message || normalized;
  }
  if (progressFill) {
    progressFill.className = `progress-fill ${normalized === "running" ? "running" : normalized}`;
    progressFill.style.width = `${progress}%`;
  }
}

function setTaskId(taskId) {
  if (taskIdLabel) {
    taskIdLabel.textContent = `Task: ${taskId || "-"}`;
  }
}

function setRunningUi(isRunning) {
  if (runBtn) runBtn.disabled = isRunning;
  if (cancelBtn) cancelBtn.disabled = !isRunning || !state.currentTaskId;
  if (chooseCsvBtn) chooseCsvBtn.disabled = isRunning;
  if (chooseOutputBtn) chooseOutputBtn.disabled = isRunning;
  if (retryBtn) retryBtn.disabled = isRunning || !state.lastPayload;
  if (csvInput) csvInput.disabled = isRunning;
  if (targetInput) targetInput.disabled = isRunning;
  if (outputInput) outputInput.disabled = isRunning;
  if (timeoutInput) timeoutInput.disabled = isRunning;
  if (newTaskBtn) newTaskBtn.disabled = isRunning;
}

function showError(message, hint = "Please check parameters/environment and retry.") {
  if (errorPanel) errorPanel.classList.remove("hidden");
  if (errorMessage) errorMessage.textContent = message;
  if (errorHint) errorHint.textContent = hint;
}

function clearError() {
  if (errorPanel) errorPanel.classList.add("hidden");
  if (errorMessage) errorMessage.textContent = "-";
  if (errorHint) errorHint.textContent = "Fix issues and click 'Retry Last Task'.";
}

function resetResultView() {
  if (resultSummary) resultSummary.innerHTML = "";
  if (metricsTableBody) metricsTableBody.innerHTML = "";
  if (resultBox) resultBox.textContent = "{}";
  if (exportJsonBtn) exportJsonBtn.disabled = true;
  if (exportCsvBtn) exportCsvBtn.disabled = true;
  if (copyJsonBtn) copyJsonBtn.textContent = "Copy";
}

function renderSummary(task) {
  if (!resultSummary) {
    return;
  }
  const response = task?.response || {};
  const result = response?.result || {};
  const profile = result?.data_profile || {};
  const artifacts = result?.artifacts || {};

  const rows = [
    ["Task ID", task?.id || "-"],
    ["Status", task?.status || "-"],
    ["Rows", profile?.rows ?? "-"],
    ["Columns", profile?.columns ?? "-"],
    ["Target", profile?.target_col ?? "-"],
    ["Output Dir", artifacts?.output_dir ?? "-"],
  ];

  resultSummary.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${String(v)}</dd>`).join("");
}

function renderMetrics(task) {
  if (!metricsTableBody) {
    return;
  }
  const metrics = task?.response?.result?.metrics || {};
  const keys = ["accuracy", "precision", "recall", "f1", "auc"];
  const rows = [];

  for (const key of keys) {
    if (metrics[key] !== undefined) {
      rows.push([key, Number(metrics[key]).toFixed(6)]);
    }
  }

  if (rows.length === 0) {
    metricsTableBody.innerHTML = `<tr><td colspan="2">No metrics yet</td></tr>`;
    return;
  }

  metricsTableBody.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

function renderTask(task) {
  state.currentTask = task;
  setTaskId(task?.id || "");

  const status = (task?.status || "idle").toLowerCase();
  let message = `Task status: ${status}`;
  if (task?.error) {
    message += ` (${task.error})`;
  }
  setStatus(status, message);

  renderSummary(task);
  renderMetrics(task);
  if (resultBox) {
    resultBox.textContent = JSON.stringify(task || {}, null, 2);
  }

  if (task && TERMINAL_STATUSES.has(status)) {
    if (exportJsonBtn) exportJsonBtn.disabled = false;
    if (exportCsvBtn) exportCsvBtn.disabled = false;
  }
}

function collectPayload() {
  return {
    action: "train",
    csv_path: (csvInput?.value || "").trim(),
    target_col: String(targetInput?.value || "").trim(),
    output_dir: (outputInput?.value || "").trim(),
    timeout_ms: Number(timeoutInput?.value) || 90000,
  };
}

function validatePayload(payload) {
  if (!payload.csv_path) {
    return "CSV file path is required.";
  }
  if (!payload.target_col) {
    return "Target column is required.";
  }
  if (!Number.isFinite(payload.timeout_ms) || payload.timeout_ms < 1000) {
    return "Timeout must be a number >= 1000 (ms).";
  }
  return "";
}

function extractEngineErrorDetails(task) {
  const details = task?.response?.error?.details;
  if (!details || typeof details !== "object") {
    return { reason: "", suggestion: "" };
  }
  const reason = typeof details.reason === "string" ? details.reason.trim() : "";
  const suggestion = typeof details.suggestion === "string" ? details.suggestion.trim() : "";
  return { reason, suggestion };
}

function toReadableError(task, err) {
  if (err) {
    return `request failed: ${String(err)}`;
  }
  if (!task) {
    return "task returned empty result";
  }

  if (task?.response?.error?.message) {
    const code = task.response.error.code || "UNKNOWN";
    const message = String(task.response.error.message || "").trim();
    const details = extractEngineErrorDetails(task);
    if (details.reason) {
      return `Engine error [${code}] ${message}; reason: ${details.reason}`;
    }
    return `Engine error [${code}] ${message}`;
  }

  if (task.error) {
    return task.error;
  }
  return `task ended with status: ${task.status}`;
}

function saveFile(name, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function copyResultJson() {
  if (!state.currentTask) {
    addEvent("No task result available to copy.");
    return;
  }

  const text = `${JSON.stringify(state.currentTask, null, 2)}\n`;
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const temp = document.createElement("textarea");
      temp.value = text;
      temp.style.position = "fixed";
      temp.style.opacity = "0";
      document.body.appendChild(temp);
      temp.focus();
      temp.select();
      document.execCommand("copy");
      document.body.removeChild(temp);
    }
    if (copyJsonBtn) {
      copyJsonBtn.textContent = "Copied";
      setTimeout(() => {
        copyJsonBtn.textContent = "Copy";
      }, 1200);
    }
    addEvent("Copied full response JSON.");
  } catch (err) {
    addEvent(`Copy failed: ${String(err)}`);
  }
}

function exportResultJson() {
  if (!state.currentTask) {
    return;
  }
  saveFile(
    `${state.currentTask.id || "task"}-result.json`,
    `${JSON.stringify(state.currentTask, null, 2)}\n`,
    "application/json"
  );
}

function exportMetricsCsv() {
  const metrics = state.currentTask?.response?.result?.metrics || {};
  const lines = ["metric,value"];
  for (const [k, v] of Object.entries(metrics)) {
    let value = v;
    if (typeof value === "object") {
      value = JSON.stringify(value);
    }
    const safe = String(value).replaceAll('"', '""');
    lines.push(`${k},"${safe}"`);
  }
  saveFile(`${state.currentTask?.id || "task"}-metrics.csv`, `${lines.join("\n")}\n`, "text/csv;charset=utf-8");
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function apiRunTask(payload) {
  if (hasBinding("RunTask")) {
    return window.go.main.App.RunTask(payload);
  }
  return mockRunTask(payload);
}

async function apiGetTaskStatus(taskId) {
  if (hasBinding("GetTaskStatus")) {
    return window.go.main.App.GetTaskStatus(taskId);
  }
  return mockGetTaskStatus(taskId);
}

async function apiCancelTask(taskId) {
  if (hasBinding("CancelTask")) {
    return window.go.main.App.CancelTask(taskId);
  }
  return mockCancelTask(taskId);
}

async function apiSelectCsv() {
  if (hasBinding("SelectCSV")) {
    return window.go.main.App.SelectCSV();
  }
  return "";
}

async function apiSelectOutputDir() {
  if (hasBinding("SelectOutputDir")) {
    return window.go.main.App.SelectOutputDir();
  }
  return "";
}

async function apiListTaskHistory(limit = 20) {
  if (hasBinding("ListTaskHistory")) {
    return window.go.main.App.ListTaskHistory(limit);
  }
  return [];
}

async function apiListCsvColumns(csvPath) {
  if (hasBinding("ListCSVColumns")) {
    return window.go.main.App.ListCSVColumns(csvPath);
  }
  return mockListCsvColumns(csvPath);
}

async function mockRunTask(payload) {
  const id = `mock-task-${Date.now()}`;
  const createdAt = Date.now();
  state.mockTasks.set(id, {
    id,
    payload,
    createdAt,
    status: "pending",
    canceled: false,
  });

  return {
    id,
    status: "pending",
    request: {
      action: "train",
      payload,
    },
    response: {},
    error: "",
  };
}

async function mockGetTaskStatus(taskId) {
  const item = state.mockTasks.get(taskId);
  if (!item) {
    throw new Error(`mock task not found: ${taskId}`);
  }

  const elapsed = Date.now() - item.createdAt;
  if (item.canceled) {
    item.status = "canceled";
    return {
      id: taskId,
      status: "canceled",
      request: {
        action: "train",
        payload: item.payload,
      },
      response: {},
      error: "canceled",
    };
  }

  if (elapsed < 700) {
    item.status = "pending";
  } else if (elapsed < 2600) {
    item.status = "running";
  } else {
    item.status = "succeeded";
  }

  if (item.status !== "succeeded") {
    return {
      id: taskId,
      status: item.status,
      request: {
        action: "train",
        payload: item.payload,
      },
      response: {},
      error: "",
    };
  }

  return {
    id: taskId,
    status: "succeeded",
    request: {
      action: "train",
      payload: item.payload,
    },
    response: {
      task_id: taskId,
      status: "ok",
      result: {
        data_profile: {
          rows: 5110,
          columns: 12,
          target_col: item.payload.target_col,
        },
        metrics: {
          accuracy: 0.949119,
          precision: 0.333333,
          recall: 0.04,
          f1: 0.929694,
          auc: 0.811708,
        },
        artifacts: {
          output_dir: item.payload.output_dir || "outputs/results/wails_mvp",
        },
      },
      error: null,
      timestamp: new Date().toISOString(),
      duration_ms: 3600,
    },
    error: "",
  };
}

async function mockCancelTask(taskId) {
  const item = state.mockTasks.get(taskId);
  if (!item) {
    return false;
  }
  item.canceled = true;
  return true;
}

async function mockListCsvColumns() {
  return [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
  ];
}

function setTargetOptions(columns, preferred = "") {
  const previous = String(preferred || targetInput?.value || "").trim();
  const items = Array.isArray(columns)
    ? columns.map((v) => String(v || "").trim()).filter((v) => v.length > 0)
    : [];

  const unique = [];
  const seen = new Set();
  for (const col of items) {
    if (seen.has(col)) {
      continue;
    }
    seen.add(col);
    unique.push(col);
  }

  if (!targetInput) {
    state.availableColumns = unique;
    return;
  }

  targetInput.innerHTML = "";

  if (unique.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No available columns";
    targetInput.appendChild(option);
    targetInput.value = "";
    state.availableColumns = [];
    return;
  }

  for (const col of unique) {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    targetInput.appendChild(option);
  }

  let selected = unique[0];
  if (previous && unique.includes(previous)) {
    selected = previous;
  } else if (unique.includes("stroke")) {
    selected = "stroke";
  }

  targetInput.value = selected;
  state.availableColumns = unique;
}

async function refreshColumnsForCsv(csvPath, source = "manual path input") {
  const path = String(csvPath || "").trim();
  if (!path) {
    setTargetOptions([]);
    return;
  }

  try {
    const columns = await apiListCsvColumns(path);
    setTargetOptions(columns);
    addEvent(`Loaded ${state.availableColumns.length} selectable columns (${source}).`);
  } catch (err) {
    setTargetOptions([]);
    addEvent(`Failed to load columns: ${String(err)}`);
  }
}

async function pollTask(taskId) {
  const token = Date.now();
  state.pollingToken = token;

  while (state.pollingToken === token && state.currentTaskId === taskId) {
    let snapshot;
    try {
      snapshot = await apiGetTaskStatus(taskId);
    } catch (err) {
      setRunningUi(false);
      showError(`Status polling failed: ${String(err)}`, "Please verify backend runtime and retry.");
      addEvent(`Polling error: ${String(err)}`);
      return;
    }

    renderTask(snapshot);
    const status = (snapshot?.status || "").toLowerCase();

    if (TERMINAL_STATUSES.has(status)) {
      setRunningUi(false);
      setWizardStep(STEP_RESULT);

      if (status === "succeeded") {
        clearError();
        addEvent("Task completed successfully. You can export results now.");
      } else {
        const readable = toReadableError(snapshot);
        const details = extractEngineErrorDetails(snapshot);
        const hint = details.suggestion || "Fix parameters and click 'Retry Last Task'.";
        showError(readable, hint);
        addEvent(`Task finished (${status}): ${readable}`);
      }
      return;
    }

    await delay(450);
  }
}

async function startTask(payload) {
  const invalidReason = validatePayload(payload);
  if (invalidReason) {
    showError(invalidReason, "Please complete required fields first.");
    addEvent(`Payload validation failed: ${invalidReason}`);
    return;
  }

  state.lastPayload = payload;
  state.currentTaskId = "";
  clearError();
  resetResultView();
  setTaskId("");
  setStatus("pending", "Task status: pending");
  setWizardStep(STEP_PROGRESS);

  let submitted;
  try {
    submitted = await apiRunTask(payload);
  } catch (err) {
    const msg = `Failed to start task: ${String(err)}`;
    showError(msg, "Check Python engine path, CSV path, and dependencies.");
    addEvent(msg);
    setWizardStep(STEP_CONFIG);
    setRunningUi(false);
    return;
  }

  state.currentTaskId = submitted.id;
  setRunningUi(true);
  addEvent(`Task submitted: ${submitted.id}`);
  renderTask(submitted);
  pollTask(submitted.id);
}

async function loadRecentHistory() {
  try {
    const tasks = await apiListTaskHistory(10);
    if (!Array.isArray(tasks) || tasks.length === 0) {
      addEvent("No persisted task history found.");
      return;
    }

    const latest = tasks[0];
    state.currentTaskId = latest?.id || "";
    renderTask(latest);

    const status = (latest?.status || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) {
      setWizardStep(STEP_RESULT);
      setRunningUi(false);
    } else if (latest?.id) {
      setWizardStep(STEP_PROGRESS);
      setRunningUi(true);
      pollTask(latest.id);
    }

    addEvent(`Loaded ${tasks.length} history task(s).`, state.currentTaskId);
  } catch (err) {
    addEvent(`Failed to load task history: ${String(err)}`);
  }
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await startTask(collectPayload());
  });
}

if (cancelBtn) {
  cancelBtn.addEventListener("click", async () => {
    if (!state.currentTaskId) {
      return;
    }

    cancelBtn.disabled = true;
    try {
      const ok = await apiCancelTask(state.currentTaskId);
      addEvent(ok ? "Cancel request sent." : "Cancel request had no effect (task may have ended).");
    } catch (err) {
      showError(`Failed to cancel task: ${String(err)}`, "Please retry shortly.");
      addEvent(`Cancel failed: ${String(err)}`);
    }
  });
}

if (retryBtn) {
  retryBtn.addEventListener("click", async () => {
    if (!state.lastPayload) {
      return;
    }
    addEvent("Retrying task with previous payload.");
    await startTask({ ...state.lastPayload });
  });
}

if (newTaskBtn) {
  newTaskBtn.addEventListener("click", () => {
    state.currentTaskId = "";
    state.currentTask = null;
    setTaskId("");
    setStatus("idle", "Waiting for task start");
    clearError();
    setRunningUi(false);
    setWizardStep(STEP_CONFIG);
    addEvent("Back to parameter step. Ready for a new task.");
  });
}

if (chooseCsvBtn) {
  chooseCsvBtn.addEventListener("click", async () => {
    const nativePicker = hasBinding("SelectCSV");
    try {
      const selected = await apiSelectCsv();
      if (selected) {
        if (csvInput) {
          csvInput.value = selected;
        }
        addEvent(`Selected CSV: ${selected}`);
        await refreshColumnsForCsv(selected, "file picker");
        return;
      }
      if (nativePicker) {
        addEvent("CSV selection canceled.");
        return;
      }
    } catch (err) {
      addEvent(`System file picker unavailable: ${String(err)}`);
    }

    if (csvFileInput) {
      csvFileInput.click();
    }
  });
}

if (csvFileInput) {
  csvFileInput.addEventListener("change", () => {
    const file = csvFileInput.files?.[0];
    if (!file) {
      return;
    }
    if (csvInput) {
      csvInput.value = file.name;
    }
    addEvent(`Browser-mode selected file: ${file.name}`);
    refreshColumnsForCsv(file.name, "browser file picker");
  });
}

if (csvInput) {
  csvInput.addEventListener("change", () => {
    refreshColumnsForCsv(csvInput.value, "path changed");
  });

  csvInput.addEventListener("blur", () => {
    refreshColumnsForCsv(csvInput.value, "path blur");
  });
}

if (chooseOutputBtn) {
  chooseOutputBtn.addEventListener("click", async () => {
    const nativePicker = hasBinding("SelectOutputDir");
    try {
      const selected = await apiSelectOutputDir();
      if (selected) {
        if (outputInput) {
          outputInput.value = selected;
        }
        addEvent(`Selected output directory: ${selected}`);
        return;
      }
      if (nativePicker) {
        addEvent("Output directory selection canceled.");
        return;
      }
    } catch (err) {
      addEvent(`Directory picker unavailable: ${String(err)}`);
    }

    const current = outputInput ? outputInput.value.trim() : "";
    const manual = window.prompt("Enter output directory", current);
    if (manual && manual.trim()) {
      if (outputInput) {
        outputInput.value = manual.trim();
      }
      addEvent(`Manually set output directory: ${manual.trim()}`);
    }
  });
}

if (exportJsonBtn) {
  exportJsonBtn.addEventListener("click", exportResultJson);
}
if (exportCsvBtn) {
  exportCsvBtn.addEventListener("click", exportMetricsCsv);
}
if (copyJsonBtn) {
  copyJsonBtn.addEventListener("click", copyResultJson);
}

setWizardStep(STEP_CONFIG);
setStatus("idle", "Waiting for task start");
setTaskId("");
setRunningUi(false);
resetResultView();
clearError();
addEvent("Frontend ready. Configure parameters and run a task.");
refreshColumnsForCsv(csvInput?.value || "", "initial load");
loadRecentHistory();
