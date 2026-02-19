const form = document.getElementById("train-form");
const runBtn = document.getElementById("run-btn");
const cancelBtn = document.getElementById("cancel-btn");
const retryBtn = document.getElementById("retry-btn");
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
  currentTaskId: "",
  currentTask: null,
  lastPayload: null,
  pollingToken: 0,
  mockTasks: new Map(),
};

function hasBinding(methodName) {
  return Boolean(window?.go?.main?.App?.[methodName]);
}

function addEvent(message) {
  const li = document.createElement("li");
  const time = new Date();
  const hh = String(time.getHours()).padStart(2, "0");
  const mm = String(time.getMinutes()).padStart(2, "0");
  const ss = String(time.getSeconds()).padStart(2, "0");
  li.innerHTML = `<span class="event-time">${hh}:${mm}:${ss}</span><span>${message}</span>`;
  eventLog.prepend(li);
}

function setStatus(status, message) {
  const normalized = (status || "idle").toLowerCase();
  const progress = STATUS_PROGRESS[normalized] ?? 0;

  statusPill.className = `status-pill ${normalized}`;
  statusPill.textContent = normalized;
  statusMessage.textContent = message || normalized;

  progressFill.className = `progress-fill ${normalized === "running" ? "running" : normalized}`;
  progressFill.style.width = `${progress}%`;
}

function setTaskId(taskId) {
  taskIdLabel.textContent = `Task: ${taskId || "-"}`;
}

function setRunningUi(isRunning) {
  runBtn.disabled = isRunning;
  cancelBtn.disabled = !isRunning || !state.currentTaskId;
  chooseCsvBtn.disabled = isRunning;
  chooseOutputBtn.disabled = isRunning;
  retryBtn.disabled = isRunning || !state.lastPayload;
}

function showError(message, hint = "请检查参数或环境后重试。") {
  errorPanel.classList.remove("hidden");
  errorMessage.textContent = message;
  errorHint.textContent = hint;
}

function clearError() {
  errorPanel.classList.add("hidden");
  errorMessage.textContent = "-";
  errorHint.textContent = "可修复后点击“重试上次参数”。";
}

function resetResultView() {
  resultSummary.innerHTML = "";
  metricsTableBody.innerHTML = "";
  resultBox.textContent = "{}";
  exportJsonBtn.disabled = true;
  exportCsvBtn.disabled = true;
}

function renderSummary(task) {
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

  resultSummary.innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${String(v)}</dd>`)
    .join("");
}

function renderMetrics(task) {
  const metrics = task?.response?.result?.metrics || {};
  const keys = ["accuracy", "precision", "recall", "f1", "auc"];
  const rows = [];

  for (const key of keys) {
    if (metrics[key] !== undefined) {
      rows.push([key, Number(metrics[key]).toFixed(6)]);
    }
  }

  if (rows.length === 0) {
    metricsTableBody.innerHTML = `<tr><td colspan="2">暂无指标</td></tr>`;
    return;
  }

  metricsTableBody.innerHTML = rows
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

function renderTask(task) {
  state.currentTask = task;
  setTaskId(task?.id || "");

  const status = (task?.status || "idle").toLowerCase();
  let message = `任务状态：${status}`;
  if (task?.error) {
    message += `（${task.error}）`;
  }
  setStatus(status, message);

  renderSummary(task);
  renderMetrics(task);
  resultBox.textContent = JSON.stringify(task || {}, null, 2);

  if (task && TERMINAL_STATUSES.has(status)) {
    exportJsonBtn.disabled = false;
    exportCsvBtn.disabled = false;
  }
}

function collectPayload() {
  return {
    action: "train",
    csv_path: csvInput.value.trim(),
    target_col: targetInput.value.trim(),
    output_dir: outputInput.value.trim(),
    timeout_ms: Number(timeoutInput.value) || 90000,
  };
}

function validatePayload(payload) {
  if (!payload.csv_path) {
    return "CSV 文件路径不能为空";
  }
  if (!payload.target_col) {
    return "目标列不能为空";
  }
  if (!Number.isFinite(payload.timeout_ms) || payload.timeout_ms < 1000) {
    return "超时必须是 >= 1000 的数字（毫秒）";
  }
  return "";
}

function toReadableError(task, err) {
  if (err) {
    return `请求失败：${String(err)}`;
  }
  if (!task) {
    return "任务未返回有效结果";
  }
  if (task?.response?.error?.message) {
    const code = task.response.error.code || "UNKNOWN";
    return `引擎错误 [${code}] ${task.response.error.message}`;
  }
  if (task.error) {
    return task.error;
  }
  return `任务结束状态：${task.status}`;
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
  saveFile(
    `${state.currentTask?.id || "task"}-metrics.csv`,
    `${lines.join("\n")}\n`,
    "text/csv;charset=utf-8"
  );
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

async function pollTask(taskId) {
  const token = Date.now();
  state.pollingToken = token;

  while (state.pollingToken === token && state.currentTaskId === taskId) {
    let snapshot;
    try {
      snapshot = await apiGetTaskStatus(taskId);
    } catch (err) {
      setRunningUi(false);
      showError(`状态轮询失败：${String(err)}`, "请检查后端运行状态后点击重试。");
      addEvent(`轮询异常：${String(err)}`);
      return;
    }

    renderTask(snapshot);
    const status = (snapshot?.status || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) {
      setRunningUi(false);

      if (status === "succeeded") {
        clearError();
        addEvent("任务成功完成，可导出结果。");
      } else {
        const readable = toReadableError(snapshot);
        showError(readable, "你可以修正参数后点击“重试上次参数”。");
        addEvent(`任务结束（${status}）：${readable}`);
      }
      return;
    }

    await delay(450);
  }
}

async function startTask(payload) {
  const invalidReason = validatePayload(payload);
  if (invalidReason) {
    showError(invalidReason, "请先完成参数配置。");
    addEvent(`参数校验失败：${invalidReason}`);
    return;
  }

  state.lastPayload = payload;
  clearError();
  resetResultView();

  let submitted;
  try {
    submitted = await apiRunTask(payload);
  } catch (err) {
    const msg = `启动任务失败：${String(err)}`;
    showError(msg, "请检查 Python 引擎路径、CSV 路径和依赖状态。");
    addEvent(msg);
    return;
  }

  state.currentTaskId = submitted.id;
  setRunningUi(true);
  addEvent(`任务已提交：${submitted.id}`);
  renderTask(submitted);
  pollTask(submitted.id);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await startTask(collectPayload());
});

cancelBtn.addEventListener("click", async () => {
  if (!state.currentTaskId) {
    return;
  }

  cancelBtn.disabled = true;
  try {
    const ok = await apiCancelTask(state.currentTaskId);
    addEvent(ok ? "已发送取消请求。" : "取消请求未生效（任务可能已结束）。");
  } catch (err) {
    showError(`取消任务失败：${String(err)}`, "请稍后重试。");
    addEvent(`取消失败：${String(err)}`);
  }
});

retryBtn.addEventListener("click", async () => {
  if (!state.lastPayload) {
    return;
  }
  addEvent("使用上次参数重试任务。");
  await startTask({ ...state.lastPayload });
});

chooseCsvBtn.addEventListener("click", async () => {
  const nativePicker = hasBinding("SelectCSV");
  try {
    const selected = await apiSelectCsv();
    if (selected) {
      csvInput.value = selected;
      addEvent(`已选择 CSV：${selected}`);
      return;
    }
    if (nativePicker) {
      addEvent("已取消 CSV 选择。");
      return;
    }
  } catch (err) {
    addEvent(`系统文件选择器不可用：${String(err)}`);
  }

  csvFileInput.click();
});

csvFileInput.addEventListener("change", () => {
  const file = csvFileInput.files?.[0];
  if (!file) {
    return;
  }
  csvInput.value = file.name;
  addEvent(`浏览器模式已选择文件：${file.name}`);
});

chooseOutputBtn.addEventListener("click", async () => {
  const nativePicker = hasBinding("SelectOutputDir");
  try {
    const selected = await apiSelectOutputDir();
    if (selected) {
      outputInput.value = selected;
      addEvent(`已选择输出目录：${selected}`);
      return;
    }
    if (nativePicker) {
      addEvent("已取消输出目录选择。");
      return;
    }
  } catch (err) {
    addEvent(`目录选择器不可用：${String(err)}`);
  }

  const manual = window.prompt("请输入输出目录", outputInput.value.trim());
  if (manual && manual.trim()) {
    outputInput.value = manual.trim();
    addEvent(`手动设置输出目录：${manual.trim()}`);
  }
});

exportJsonBtn.addEventListener("click", exportResultJson);
exportCsvBtn.addEventListener("click", exportMetricsCsv);

setStatus("idle", "等待任务开始");
setTaskId("");
setRunningUi(false);
resetResultView();
addEvent("前端已就绪。可配置参数并启动任务。");
