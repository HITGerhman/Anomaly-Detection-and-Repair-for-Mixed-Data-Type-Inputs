const wheelOrbit = document.getElementById("wheel-orbit");
const wizardCard = document.getElementById("wizard-card");

const cardKicker = document.getElementById("card-kicker");
const cardTitle = document.getElementById("card-title");
const cardSubtitle = document.getElementById("card-subtitle");

const stageFill = document.getElementById("stage-fill");
const stageSteps = document.getElementById("stage-steps");

const stepConfig = document.getElementById("step-config");
const stepProgress = document.getElementById("step-progress");
const stepResult = document.getElementById("step-result");
const stepRepair = document.getElementById("step-repair");
const responseShell = document.getElementById("response-shell");

const detectForm = document.getElementById("detect-form");
const repairForm = document.getElementById("repair-form");

const runDetectBtn = document.getElementById("run-detect-btn");
const retryDetectBtn = document.getElementById("retry-detect-btn");
const cancelBtn = document.getElementById("cancel-btn");

const chooseCsvBtn = document.getElementById("choose-csv-btn");
const chooseModelBtn = document.getElementById("choose-model-btn");
const chooseOutputBtn = document.getElementById("choose-output-btn");

const gotoRepairBtn = document.getElementById("goto-repair-btn");
const newDetectBtn = document.getElementById("new-detect-btn");
const backResultBtn = document.getElementById("back-result-btn");
const runRepairBtn = document.getElementById("run-repair-btn");

const csvPathInput = document.getElementById("csv-path");
const modelDirInput = document.getElementById("model-dir");
const targetColInput = document.getElementById("target-col");
const retrainModelInput = document.getElementById("retrain-model");
const sampleIndexInput = document.getElementById("sample-index");
const timeoutInput = document.getElementById("timeout-ms");
const outputInput = document.getElementById("output-dir");
const maxChangesInput = document.getElementById("max-changes");
const kNeighborsInput = document.getElementById("k-neighbors");
const csvFileInput = document.getElementById("csv-file-input");

const statusPill = document.getElementById("status-pill");
const statusMessage = document.getElementById("status-message");
const progressFill = document.getElementById("progress-fill");
const taskIdLabel = document.getElementById("task-id-label");
const eventLog = document.getElementById("event-log");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");
const errorHint = document.getElementById("error-hint");

const detectionSummary = document.getElementById("detection-summary");
const detectionMessage = document.getElementById("detection-message");
const repairSummary = document.getElementById("repair-summary");

const resultBox = document.getElementById("result-box");
const copyJsonBtn = document.getElementById("copy-json-btn");
const exportJsonBtn = document.getElementById("export-json-btn");
const exportCsvBtn = document.getElementById("export-csv-btn");

const STEP_CONFIG = "config";
const STEP_PROGRESS = "progress";
const STEP_RESULT = "result";
const STEP_REPAIR = "repair";

const STEP_ORDER = [STEP_CONFIG, STEP_PROGRESS, STEP_RESULT, STEP_REPAIR];

const STEP_META = {
  [STEP_CONFIG]: {
    kicker: "STEP 1",
    title: "检测参数配置",
    subtitle: "选择模型目录和样本，先进行异常检测。",
  },
  [STEP_PROGRESS]: {
    kicker: "STEP 2",
    title: "任务执行中",
    subtitle: "任务正在运行，请稍候。",
  },
  [STEP_RESULT]: {
    kicker: "STEP 3",
    title: "检测结果",
    subtitle: "若样本为异常，可继续进入修复阶段。",
  },
  [STEP_REPAIR]: {
    kicker: "STEP 4",
    title: "修复界面",
    subtitle: "设置修复策略并执行修复。",
  },
};

const WHEEL_SPIN_DEG = 34;
const WHEEL_SPIN_DURATION_MS = 720;

const STATUS_PROGRESS = {
  idle: 0,
  pending: 14,
  running: 68,
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
  pendingIntent: "",
  pollingToken: 0,
  stepAnimating: false,
  queuedStep: "",
  detectionResult: null,
  lastDetectPayload: null,
  lastRepairPayload: null,
  availableColumns: [],
  mockTasks: new Map(),
};

function hasBinding(methodName) {
  return Boolean(window?.go?.main?.App?.[methodName]);
}

function formatNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return String(value ?? "-");
  }
  return n.toFixed(6);
}

function setTaskId(taskId) {
  if (taskIdLabel) {
    taskIdLabel.textContent = `Task: ${taskId || "-"}`;
  }
}

function setStatus(status, message) {
  const normalized = String(status || "idle").toLowerCase();
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
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const tid = String(taskId || "").trim();
  const visible = tid ? `[${tid}] ${message}` : message;
  li.innerHTML = `<span class="event-time">${hh}:${mm}:${ss}</span><span>${visible}</span>`;
  eventLog.prepend(li);
  emitFrontendLog(message, tid);
}

function showError(message, hint = "请检查参数后重试。") {
  if (errorPanel) errorPanel.classList.remove("hidden");
  if (errorMessage) errorMessage.textContent = message;
  if (errorHint) errorHint.textContent = hint;
}

function clearError() {
  if (errorPanel) errorPanel.classList.add("hidden");
  if (errorMessage) errorMessage.textContent = "-";
  if (errorHint) errorHint.textContent = "修正后可重试。";
}

function setRunningUi(isRunning) {
  if (runDetectBtn) runDetectBtn.disabled = isRunning;
  if (retryDetectBtn) retryDetectBtn.disabled = isRunning || !state.lastDetectPayload;
  if (cancelBtn) cancelBtn.disabled = !isRunning || !state.currentTaskId;
  if (chooseCsvBtn) chooseCsvBtn.disabled = isRunning;
  if (chooseModelBtn) chooseModelBtn.disabled = isRunning;
  if (chooseOutputBtn) chooseOutputBtn.disabled = isRunning;

  if (csvPathInput) csvPathInput.disabled = isRunning;
  if (modelDirInput) modelDirInput.disabled = isRunning;
  if (targetColInput) targetColInput.disabled = isRunning;
  if (retrainModelInput) retrainModelInput.disabled = isRunning;
  if (sampleIndexInput) sampleIndexInput.disabled = isRunning;
  if (timeoutInput) timeoutInput.disabled = isRunning;
  if (outputInput) outputInput.disabled = isRunning;

  if (runRepairBtn) runRepairBtn.disabled = isRunning;
  if (maxChangesInput) maxChangesInput.disabled = isRunning;
  if (kNeighborsInput) kNeighborsInput.disabled = isRunning;
  if (gotoRepairBtn) gotoRepairBtn.disabled = isRunning;
  if (newDetectBtn) newDetectBtn.disabled = isRunning;
  if (backResultBtn) backResultBtn.disabled = isRunning;
}

function setStepVisibility(step) {
  const visibleMap = {
    [STEP_CONFIG]: stepConfig,
    [STEP_PROGRESS]: stepProgress,
    [STEP_RESULT]: stepResult,
    [STEP_REPAIR]: stepRepair,
  };
  for (const [key, el] of Object.entries(visibleMap)) {
    if (!el) continue;
    el.classList.toggle("hidden", key !== step);
  }
  if (responseShell) {
    responseShell.classList.toggle("hidden", !(step === STEP_RESULT || step === STEP_REPAIR));
  }
}

function setStageProgress(step) {
  const idx = STEP_ORDER.indexOf(step);
  const ratio = idx <= 0 ? 0 : idx / (STEP_ORDER.length - 1);
  if (stageFill) {
    stageFill.style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;
  }
  if (!stageSteps) {
    return;
  }
  const items = Array.from(stageSteps.querySelectorAll("li"));
  items.forEach((item, i) => {
    item.classList.remove("active", "completed");
    if (i < idx) {
      item.classList.add("completed");
    } else if (i === idx) {
      item.classList.add("active");
    }
  });
}

function applyStepNow(step) {
  const normalized = STEP_META[step] ? step : STEP_CONFIG;
  state.currentStep = normalized;
  const meta = STEP_META[normalized];
  if (cardKicker) cardKicker.textContent = meta.kicker;
  if (cardTitle) cardTitle.textContent = meta.title;
  if (cardSubtitle) cardSubtitle.textContent = meta.subtitle;
  setStageProgress(normalized);
  setStepVisibility(normalized);
}

function clearWheelPose() {
  if (wheelOrbit) {
    wheelOrbit.style.transform = "rotate(0deg)";
    wheelOrbit.style.opacity = "1";
  }
}

function cancelWheelAnimations() {
  if (!wheelOrbit || typeof wheelOrbit.getAnimations !== "function") {
    return;
  }
  for (const animation of wheelOrbit.getAnimations()) {
    animation.cancel();
  }
}

function animateOrbitTransform(fromDeg, toDeg, fromOpacity, toOpacity, durationMs) {
  if (!wheelOrbit || typeof wheelOrbit.animate !== "function") {
    if (wheelOrbit) {
      wheelOrbit.style.transform = `rotate(${toDeg}deg)`;
      wheelOrbit.style.opacity = String(toOpacity);
    }
    return Promise.resolve();
  }

  const animation = wheelOrbit.animate(
    [
      { transform: `rotate(${fromDeg}deg)`, opacity: fromOpacity },
      { transform: `rotate(${toDeg}deg)`, opacity: toOpacity },
    ],
    {
      duration: durationMs,
      easing: "cubic-bezier(0.175, 0.885, 0.32, 1.275)",
      fill: "forwards",
    }
  );

  return animation.finished
    .catch(() => {})
    .then(() => {
      if (!wheelOrbit) {
        return;
      }
      wheelOrbit.style.transform = `rotate(${toDeg}deg)`;
      wheelOrbit.style.opacity = String(toOpacity);
    });
}

function stepIndex(step) {
  const idx = STEP_ORDER.indexOf(step);
  return idx >= 0 ? idx : 0;
}

function stepDirection(fromStep, toStep) {
  const fromIdx = stepIndex(fromStep);
  const toIdx = stepIndex(toStep);
  if (toIdx === fromIdx) {
    return -1;
  }
  // Forward transitions rotate left; backward transitions rotate right.
  return toIdx > fromIdx ? -1 : 1;
}

async function transitionWizardStep(step, immediate = false) {
  const normalized = STEP_META[step] ? step : STEP_CONFIG;
  if (immediate || !wizardCard || !wheelOrbit) {
    cancelWheelAnimations();
    applyStepNow(normalized);
    clearWheelPose();
    if (wizardCard) {
      wizardCard.classList.remove("is-animating");
      wizardCard.style.pointerEvents = "";
    }
    state.stepAnimating = false;
    state.queuedStep = "";
    return;
  }
  if (state.stepAnimating) {
    state.queuedStep = normalized;
    return;
  }
  if (state.currentStep === normalized) {
    applyStepNow(normalized);
    return;
  }

  state.stepAnimating = true;
  state.queuedStep = "";
  wizardCard.classList.add("is-animating");
  wizardCard.style.pointerEvents = "none";

  const halfDuration = Math.round(WHEEL_SPIN_DURATION_MS / 2);
  const direction = stepDirection(state.currentStep, normalized);
  await animateOrbitTransform(0, direction * WHEEL_SPIN_DEG, 1, 0.03, halfDuration);
  applyStepNow(normalized);
  await animateOrbitTransform(-direction * WHEEL_SPIN_DEG, 0, 0.03, 1, halfDuration);

  wizardCard.classList.remove("is-animating");
  wizardCard.style.pointerEvents = "";
  state.stepAnimating = false;
  clearWheelPose();

  if (state.queuedStep && state.queuedStep !== state.currentStep) {
    const queued = state.queuedStep;
    state.queuedStep = "";
    await transitionWizardStep(queued, false);
  }
}

function setWizardStep(step, options = {}) {
  const immediate = Boolean(options?.immediate);
  void transitionWizardStep(step, immediate);
}

function resetResultPanels() {
  if (detectionSummary) detectionSummary.innerHTML = "";
  if (repairSummary) repairSummary.innerHTML = "";
  if (detectionMessage) detectionMessage.textContent = "-";
  if (gotoRepairBtn) gotoRepairBtn.classList.add("hidden");
  if (resultBox) resultBox.textContent = "{}";
  if (copyJsonBtn) copyJsonBtn.textContent = "复制";
  if (exportJsonBtn) exportJsonBtn.disabled = true;
  if (exportCsvBtn) exportCsvBtn.disabled = true;
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
    addEvent("没有可复制的任务结果。");
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
      copyJsonBtn.textContent = "已复制";
      setTimeout(() => {
        copyJsonBtn.textContent = "复制";
      }, 1200);
    }
    addEvent("已复制完整响应 JSON。", state.currentTask?.id || "");
  } catch (err) {
    addEvent(`复制失败: ${String(err)}`);
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

function exportResultCsv() {
  const result = state.currentTask?.response?.result || {};
  const summary = result?.repair_summary || {};
  const lines = ["key,value"];
  for (const [k, v] of Object.entries(summary)) {
    const safe = String(v).replaceAll('"', '""');
    lines.push(`${k},"${safe}"`);
  }
  if (Array.isArray(result?.repair_changes)) {
    lines.push("");
    lines.push("feature,before,after,score_delta");
    for (const item of result.repair_changes) {
      const feature = String(item?.feature ?? "").replaceAll('"', '""');
      const before = String(item?.before ?? "").replaceAll('"', '""');
      const after = String(item?.after ?? "").replaceAll('"', '""');
      const delta = String(item?.score_delta ?? "").replaceAll('"', '""');
      lines.push(`"${feature}","${before}","${after}","${delta}"`);
    }
  }
  saveFile(`${state.currentTask?.id || "task"}-repair.csv`, `${lines.join("\n")}\n`, "text/csv;charset=utf-8");
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
    return `Engine error [${code}] ${String(task.response.error.message)}`;
  }
  if (task.error) {
    return String(task.error);
  }
  return `task ended with status: ${task.status}`;
}

function isRepairTask(task) {
  return String(task?.request?.action || "").toLowerCase() === "repair";
}

function isDryRunTask(task) {
  return Boolean(task?.request?.payload?.dry_run);
}

function renderDetectionResult(result) {
  const summary = result?.repair_summary || {};
  const rows = [
    ["sample_index", result?.sample_index ?? "-"],
    ["status", summary?.status ?? "-"],
    ["before_pred", summary?.before_pred ?? "-"],
    ["before_score", formatNumber(summary?.before_score ?? "-")],
    ["dry_run", String(summary?.dry_run ?? result?.dry_run ?? true)],
  ];
  if (detectionSummary) {
    detectionSummary.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${String(v)}</dd>`).join("");
  }

  const hasAnomaly = Number(summary?.before_pred) === 1;
  if (hasAnomaly) {
    if (detectionMessage) {
      detectionMessage.textContent = "检测到异常样本，可进入修复阶段。";
    }
    if (gotoRepairBtn) {
      gotoRepairBtn.classList.remove("hidden");
      gotoRepairBtn.disabled = false;
    }
  } else {
    if (detectionMessage) {
      detectionMessage.textContent = "该样本未检测到异常，无需修复。";
    }
    if (gotoRepairBtn) {
      gotoRepairBtn.classList.add("hidden");
    }
  }
}

function renderDetectionFailure(reason, task = null, phase = "检测") {
  const taskId = String(task?.id || state.currentTaskId || "-");
  const status = String(task?.status || "failed").toLowerCase();
  const errorCode = String(task?.response?.error?.code || "-");
  const suggestion = String(task?.response?.error?.details?.suggestion || "").trim();
  const rows = [
    ["phase", phase],
    ["task_id", taskId],
    ["status", status],
    ["error_code", errorCode],
    ["can_repair", "false"],
  ];
  if (detectionSummary) {
    detectionSummary.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${String(v)}</dd>`).join("");
  }

  let message = `${phase}失败：${String(reason || "未知错误")}。当前无法判断是否可修复。`;
  if (suggestion) {
    message += ` 建议：${suggestion}`;
  }
  if (detectionMessage) {
    detectionMessage.textContent = message;
  }
  if (gotoRepairBtn) {
    gotoRepairBtn.classList.add("hidden");
    gotoRepairBtn.disabled = true;
  }
}

function renderRepairResult(result) {
  const summary = result?.repair_summary || {};
  const rows = [
    ["status", summary?.status ?? "-"],
    ["success", String(summary?.success ?? "-")],
    ["before_pred", summary?.before_pred ?? "-"],
    ["after_pred", summary?.after_pred ?? "-"],
    ["before_score", formatNumber(summary?.before_score ?? "-")],
    ["after_score", formatNumber(summary?.after_score ?? "-")],
    ["score_reduction", formatNumber(summary?.score_reduction ?? "-")],
    ["applied_changes", summary?.applied_changes ?? "-"],
  ];
  if (repairSummary) {
    repairSummary.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${String(v)}</dd>`).join("");
  }
}

function renderTask(task) {
  state.currentTask = task;
  setTaskId(task?.id || "");

  const status = String(task?.status || "idle").toLowerCase();
  let message = `任务状态: ${status}`;
  if (task?.error) {
    message += ` (${task.error})`;
  }
  setStatus(status, message);

  if (resultBox) {
    resultBox.textContent = JSON.stringify(task || {}, null, 2);
  }

  if (task && TERMINAL_STATUSES.has(status)) {
    if (exportJsonBtn) exportJsonBtn.disabled = false;
    if (exportCsvBtn) exportCsvBtn.disabled = false;
  }

  if (!isRepairTask(task)) {
    return;
  }

  const result = task?.response?.result || {};
  if (isDryRunTask(task)) {
    if (status === "succeeded") {
      renderDetectionResult(result);
      state.detectionResult = result;
    } else if (TERMINAL_STATUSES.has(status)) {
      const readable = toReadableError(task);
      renderDetectionFailure(readable, task, "检测");
      state.detectionResult = null;
    }
  } else {
    renderRepairResult(result);
  }
}

function collectDetectPayload() {
  const retrainBeforeDetect = Boolean(retrainModelInput?.checked);
  return {
    action: "repair",
    csv_path: String(csvPathInput?.value || "").trim(),
    target_col: String(targetColInput?.value || "").trim(),
    model_dir: String(modelDirInput?.value || "").trim(),
    sample_index: Math.trunc(Number(sampleIndexInput?.value) || 0),
    output_dir: String(outputInput?.value || "").trim(),
    timeout_ms: Math.trunc(Number(timeoutInput?.value) || 90000),
    retrain_before_detect: retrainBeforeDetect,
    dry_run: true,
    max_changes: 0,
    k_neighbors: 9,
  };
}

function collectRepairPayload() {
  const sampleIndex = Math.trunc(Number(sampleIndexInput?.value) || 0);
  return {
    action: "repair",
    model_dir: String(modelDirInput?.value || "").trim(),
    sample_index: sampleIndex,
    output_dir: String(outputInput?.value || "").trim(),
    timeout_ms: Math.trunc(Number(timeoutInput?.value) || 90000),
    dry_run: false,
    max_changes: Math.trunc(Number(maxChangesInput?.value) || 3),
    k_neighbors: Math.trunc(Number(kNeighborsInput?.value) || 9),
  };
}

function validateDetectPayload(payload) {
  if (!payload.model_dir) {
    return "模型目录不能为空。";
  }
  if (!Number.isInteger(payload.sample_index) || payload.sample_index < 0) {
    return "样本索引必须是 >= 0 的整数。";
  }
  if (!Number.isInteger(payload.timeout_ms) || payload.timeout_ms < 1000) {
    return "超时必须是 >= 1000 的整数(ms)。";
  }
  if (payload.retrain_before_detect) {
    if (!String(payload.csv_path || "").trim()) {
      return "启用检测前重训时，CSV 文件路径不能为空。";
    }
    if (!String(payload.target_col || "").trim()) {
      return "启用检测前重训时，请选择目标列。";
    }
  }
  return "";
}

function collectTrainPayloadFromDetect(detectPayload) {
  return {
    action: "train",
    csv_path: String(detectPayload?.csv_path || "").trim(),
    target_col: String(detectPayload?.target_col || "").trim(),
    output_dir: String(detectPayload?.model_dir || "").trim(),
    timeout_ms: Math.trunc(Number(detectPayload?.timeout_ms) || 90000),
  };
}

function validateTrainPayload(payload) {
  if (!String(payload?.csv_path || "").trim()) {
    return "CSV 文件路径不能为空。";
  }
  if (!String(payload?.target_col || "").trim()) {
    return "目标列不能为空。";
  }
  if (!String(payload?.output_dir || "").trim()) {
    return "模型目录不能为空。";
  }
  if (!Number.isInteger(payload.timeout_ms) || payload.timeout_ms < 1000) {
    return "超时必须是 >= 1000 的整数(ms)。";
  }
  return "";
}

function validateRepairPayload(payload) {
  const base = validateDetectPayload(payload);
  if (base) {
    return base;
  }
  if (!Number.isInteger(payload.max_changes) || payload.max_changes < 1) {
    return "最多修改字段必须是 >= 1 的整数。";
  }
  if (!Number.isInteger(payload.k_neighbors) || payload.k_neighbors < 3) {
    return "邻居数量必须是 >= 3 的整数。";
  }
  return "";
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

async function apiListCsvColumns(csvPath) {
  if (hasBinding("ListCSVColumns")) {
    return window.go.main.App.ListCSVColumns(csvPath);
  }
  return mockListCsvColumns(csvPath);
}

async function apiListTaskHistory(limit = 20) {
  if (hasBinding("ListTaskHistory")) {
    return window.go.main.App.ListTaskHistory(limit);
  }
  return [];
}

async function mockRunTask(payload) {
  const id = `mock-task-${Date.now()}`;
  state.mockTasks.set(id, {
    id,
    payload,
    createdAt: Date.now(),
    canceled: false,
  });

  return {
    id,
    status: "pending",
    request: {
      action: payload.action,
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
    return {
      id: taskId,
      status: "canceled",
      request: { action: "repair", payload: item.payload },
      response: {},
      error: "canceled",
    };
  }

  if (elapsed < 700) {
    return {
      id: taskId,
      status: "pending",
      request: { action: "repair", payload: item.payload },
      response: {},
      error: "",
    };
  }

  if (elapsed < 2600) {
    return {
      id: taskId,
      status: "running",
      request: { action: "repair", payload: item.payload },
      response: {},
      error: "",
    };
  }

  const sampleIndex = Math.trunc(Number(item.payload.sample_index) || 0);
  const isDryRun = Boolean(item.payload.dry_run);

  if (isDryRun) {
    const isAnomaly = sampleIndex % 2 === 1;
    return {
      id: taskId,
      status: "succeeded",
      request: { action: "repair", payload: item.payload },
      response: {
        task_id: taskId,
        status: "ok",
        result: {
          dry_run: true,
          sample_index: sampleIndex,
          repair_summary: {
            status: isAnomaly ? "anomaly_detected" : "already_normal",
            success: !isAnomaly,
            before_pred: isAnomaly ? 1 : 0,
            after_pred: isAnomaly ? 1 : 0,
            before_score: isAnomaly ? 0.892431 : 0.082312,
            after_score: isAnomaly ? 0.892431 : 0.082312,
            score_reduction: 0,
            applied_changes: 0,
            dry_run: true,
          },
          repair_changes: [],
        },
        error: null,
        timestamp: new Date().toISOString(),
        duration_ms: 2100,
      },
      error: "",
    };
  }

  return {
    id: taskId,
    status: "succeeded",
    request: { action: "repair", payload: item.payload },
    response: {
      task_id: taskId,
      status: "ok",
      result: {
        dry_run: false,
        sample_index: sampleIndex,
        repair_summary: {
          status: "repaired",
          success: true,
          before_pred: 1,
          after_pred: 0,
          before_score: 0.892431,
          after_score: 0.461202,
          score_reduction: 0.431229,
          applied_changes: 2,
          dry_run: false,
        },
        repair_changes: [
          {
            feature: "avg_glucose_level",
            before: 193.44,
            after: 124.31,
            score_delta: 0.286901,
          },
          {
            feature: "heart_disease",
            before: 1,
            after: 0,
            score_delta: 0.144328,
          },
        ],
      },
      error: null,
      timestamp: new Date().toISOString(),
      duration_ms: 3300,
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
  if (!targetColInput) {
    state.availableColumns = Array.isArray(columns) ? [...columns] : [];
    return;
  }

  const clean = Array.isArray(columns)
    ? columns.map((v) => String(v || "").trim()).filter((v) => v.length > 0)
    : [];
  const uniq = [];
  const seen = new Set();
  for (const col of clean) {
    if (seen.has(col)) continue;
    seen.add(col);
    uniq.push(col);
  }
  state.availableColumns = uniq;

  targetColInput.innerHTML = "";

  if (uniq.length === 0) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "无可用列";
    targetColInput.appendChild(placeholder);
    targetColInput.value = "";
    return;
  }

  for (const col of uniq) {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    targetColInput.appendChild(option);
  }

  let selected = String(preferred || targetColInput.value || "").trim();
  if (!selected || !uniq.includes(selected)) {
    selected = uniq.includes("stroke") ? "stroke" : uniq[0];
  }
  targetColInput.value = selected;
}

async function refreshColumnsForCsv(csvPath, source = "manual path input") {
  const path = String(csvPath || "").trim();
  if (!path) {
    setTargetOptions([]);
    return;
  }

  try {
    const columns = await apiListCsvColumns(path);
    const before = targetColInput ? targetColInput.value : "";
    setTargetOptions(columns, before);
    addEvent(`已加载 ${state.availableColumns.length} 个目标列(${source})。`);
  } catch (err) {
    setTargetOptions([]);
    addEvent(`读取列失败: ${String(err)}`);
  }
}

function successStepForIntent(intent) {
  if (intent === "repair") {
    return STEP_REPAIR;
  }
  if (intent === "detect") {
    return STEP_RESULT;
  }
  return "";
}

function successMessageForIntent(intent) {
  if (intent === "repair") {
    return "修复任务完成。";
  }
  if (intent === "train") {
    return "训练任务完成。";
  }
  return "检测任务完成。";
}

async function pollTask(taskId, intent, options = {}) {
  const successStep = options.successStep === undefined ? successStepForIntent(intent) : options.successStep;
  const failureStep = options.failureStep === undefined ? successStepForIntent(intent) : options.failureStep;
  const token = Date.now();
  state.pollingToken = token;

  while (state.pollingToken === token && state.currentTaskId === taskId) {
    let snapshot;
    try {
      snapshot = await apiGetTaskStatus(taskId);
    } catch (err) {
      setRunningUi(false);
      const msg = `状态轮询失败: ${String(err)}`;
      showError(msg, "请检查后端连接后重试。");
      if (intent === "detect" || intent === "train") {
        renderDetectionFailure(msg, null, intent === "train" ? "训练" : "检测");
        if (failureStep) {
          setWizardStep(failureStep);
        }
      }
      addEvent(msg);
      return null;
    }

    renderTask(snapshot);
    const status = String(snapshot?.status || "").toLowerCase();

    if (TERMINAL_STATUSES.has(status)) {
      if (!options.keepRunningUi) {
        setRunningUi(false);
      }

      if (status === "succeeded") {
        clearError();
        if (successStep) {
          setWizardStep(successStep);
        }
        if (!options.suppressSuccessEvent) {
          addEvent(successMessageForIntent(intent), taskId);
        }
      } else {
        const readable = toReadableError(snapshot);
        showError(readable, "请调整参数后重试。");
        if (intent === "detect" || intent === "train") {
          renderDetectionFailure(readable, snapshot, intent === "train" ? "训练" : "检测");
        }
        if (failureStep) {
          setWizardStep(failureStep);
        }
        if (!options.suppressFailureEvent) {
          addEvent(`任务结束(${status}): ${readable}`, taskId);
        }
      }
      return snapshot;
    }

    await delay(450);
  }

  return null;
}

async function startTask(payload, intent, options = {}) {
  let invalidReason = "";
  if (intent === "repair") {
    invalidReason = validateRepairPayload(payload);
  } else if (intent === "train") {
    invalidReason = validateTrainPayload(payload);
  } else {
    invalidReason = validateDetectPayload(payload);
  }
  if (invalidReason) {
    showError(invalidReason, "请先修正参数。");
    if (intent === "detect" || intent === "train") {
      renderDetectionFailure(invalidReason, null, intent === "train" ? "训练" : "检测");
      setWizardStep(STEP_RESULT);
    }
    addEvent(`参数校验失败: ${invalidReason}`);
    return null;
  }

  clearError();
  setTaskId("");
  setStatus("pending", "任务状态: pending");
  if (!options.skipProgressStep) {
    setWizardStep(STEP_PROGRESS);
  }

  if (intent === "detect") {
    state.lastDetectPayload = payload;
    state.detectionResult = null;
    if (gotoRepairBtn) gotoRepairBtn.classList.add("hidden");
  } else if (intent === "repair") {
    state.lastRepairPayload = payload;
  }

  let submitted;
  try {
    submitted = await apiRunTask(payload);
  } catch (err) {
    const msg = `任务启动失败: ${String(err)}`;
    showError(msg, "请检查引擎路径与参数。",);
    if (intent === "detect" || intent === "train") {
      renderDetectionFailure(msg, null, intent === "train" ? "训练" : "检测");
    }
    addEvent(msg);
    const fallbackStep =
      options.fallbackStep === undefined
        ? (intent === "repair" ? STEP_REPAIR : (intent === "detect" || intent === "train" ? STEP_RESULT : STEP_CONFIG))
        : options.fallbackStep;
    if (fallbackStep) {
      setWizardStep(fallbackStep);
    }
    setRunningUi(false);
    return null;
  }

  state.currentTaskId = submitted.id;
  state.pendingIntent = intent;
  setRunningUi(true);
  addEvent(`任务已提交(${intent}): ${submitted.id}`);
  renderTask(submitted);
  return pollTask(submitted.id, intent, options.pollOptions || {});
}

async function startDetectWorkflow(rawPayload) {
  const detectPayload = { ...rawPayload };

  if (detectPayload.retrain_before_detect) {
    const trainPayload = collectTrainPayloadFromDetect(detectPayload);
    const trainInvalid = validateTrainPayload(trainPayload);
    if (trainInvalid) {
      showError(trainInvalid, "请先修正参数。");
      renderDetectionFailure(trainInvalid, null, "训练");
      setWizardStep(STEP_RESULT);
      addEvent(`训练参数校验失败: ${trainInvalid}`);
      return;
    }

    addEvent(`检测前重训已启用，目标列: ${trainPayload.target_col || "-"}`);
    const trainTask = await startTask(trainPayload, "train", {
      fallbackStep: STEP_RESULT,
      pollOptions: {
        successStep: "",
        failureStep: STEP_RESULT,
      },
    });
    const trainStatus = String(trainTask?.status || "").toLowerCase();
    if (trainStatus !== "succeeded") {
      return;
    }
    addEvent("重训完成，开始异常检测。", trainTask?.id || "");
  }

  const taskPayload = {
    ...detectPayload,
    action: "repair",
    dry_run: true,
    max_changes: 0,
    k_neighbors: 9,
  };
  await startTask(taskPayload, "detect", {
    fallbackStep: STEP_RESULT,
    pollOptions: {
      successStep: STEP_RESULT,
      failureStep: STEP_RESULT,
    },
  });
}

async function chooseDirectory(targetInput, promptText, eventText) {
  const nativePicker = hasBinding("SelectOutputDir");
  try {
    const selected = await apiSelectOutputDir();
    if (selected) {
      if (targetInput) {
        targetInput.value = selected;
      }
      addEvent(`${eventText}: ${selected}`);
      return;
    }
    if (nativePicker) {
      addEvent(`${eventText}已取消。`);
      return;
    }
  } catch (err) {
    addEvent(`目录选择器不可用: ${String(err)}`);
  }

  const current = targetInput ? targetInput.value.trim() : "";
  const manual = window.prompt(promptText, current);
  if (manual && manual.trim()) {
    if (targetInput) {
      targetInput.value = manual.trim();
    }
    addEvent(`${eventText}(手动): ${manual.trim()}`);
  }
}

async function loadRecentHistory() {
  try {
    const tasks = await apiListTaskHistory(10);
    if (!Array.isArray(tasks) || tasks.length === 0) {
      addEvent("未找到历史任务。");
      return;
    }

    const latest = tasks[0];
    if (!isRepairTask(latest)) {
      addEvent("最新历史任务不是 repair 类型，已忽略回放。", latest?.id || "");
      return;
    }

    state.currentTaskId = latest?.id || "";
    renderTask(latest);
    const status = String(latest?.status || "").toLowerCase();
    const intent = isDryRunTask(latest) ? "detect" : "repair";

    if (TERMINAL_STATUSES.has(status)) {
      setRunningUi(false);
      setWizardStep(intent === "repair" ? STEP_REPAIR : STEP_RESULT);
    } else if (latest?.id) {
      setRunningUi(true);
      setWizardStep(STEP_PROGRESS);
      pollTask(latest.id, intent);
    }

    addEvent(`已加载 ${tasks.length} 条历史任务。`, state.currentTaskId);
  } catch (err) {
    addEvent(`加载历史任务失败: ${String(err)}`);
  }
}

if (detectForm) {
  detectForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    resetResultPanels();
    await startDetectWorkflow(collectDetectPayload());
  });
}

if (repairForm) {
  repairForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await startTask(collectRepairPayload(), "repair");
  });
}

if (retryDetectBtn) {
  retryDetectBtn.addEventListener("click", async () => {
    if (!state.lastDetectPayload) {
      return;
    }
    addEvent("使用上次参数重试检测。", state.currentTaskId);
    await startDetectWorkflow({ ...state.lastDetectPayload });
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
      addEvent(ok ? "取消请求已发送。" : "取消无效（任务可能已结束）。", state.currentTaskId);
    } catch (err) {
      showError(`取消失败: ${String(err)}`);
      addEvent(`取消失败: ${String(err)}`, state.currentTaskId);
    }
  });
}

if (gotoRepairBtn) {
  gotoRepairBtn.addEventListener("click", () => {
    setWizardStep(STEP_REPAIR);
    addEvent("进入修复界面。", state.currentTaskId);
  });
}

if (backResultBtn) {
  backResultBtn.addEventListener("click", () => {
    setWizardStep(STEP_RESULT);
  });
}

if (newDetectBtn) {
  newDetectBtn.addEventListener("click", () => {
    state.currentTaskId = "";
    state.currentTask = null;
    state.pendingIntent = "";
    state.detectionResult = null;
    setTaskId("");
    setStatus("idle", "等待任务开始");
    clearError();
    setRunningUi(false);
    resetResultPanels();
    setWizardStep(STEP_CONFIG);
    addEvent("已返回参数配置，可开始新检测。");
  });
}

if (chooseModelBtn) {
  chooseModelBtn.addEventListener("click", async () => {
    await chooseDirectory(modelDirInput, "输入模型目录", "已选择模型目录");
  });
}

if (chooseCsvBtn) {
  chooseCsvBtn.addEventListener("click", async () => {
    const nativePicker = hasBinding("SelectCSV");
    try {
      const selected = await apiSelectCsv();
      if (selected) {
        if (csvPathInput) {
          csvPathInput.value = selected;
        }
        addEvent(`已选择 CSV 文件: ${selected}`);
        await refreshColumnsForCsv(selected, "文件选择器");
        return;
      }
      if (nativePicker) {
        addEvent("CSV 选择已取消。");
        return;
      }
    } catch (err) {
      addEvent(`CSV 选择器不可用: ${String(err)}`);
    }

    if (csvFileInput) {
      csvFileInput.click();
    }
  });
}

if (csvFileInput) {
  csvFileInput.addEventListener("change", async () => {
    const file = csvFileInput.files?.[0];
    if (!file) {
      return;
    }
    if (csvPathInput) {
      csvPathInput.value = file.name;
    }
    addEvent(`浏览器模式选择文件: ${file.name}`);
    await refreshColumnsForCsv(file.name, "浏览器文件选择");
  });
}

if (csvPathInput) {
  csvPathInput.addEventListener("change", async () => {
    await refreshColumnsForCsv(csvPathInput.value, "路径变更");
  });
  csvPathInput.addEventListener("blur", async () => {
    await refreshColumnsForCsv(csvPathInput.value, "路径失焦");
  });
}

if (chooseOutputBtn) {
  chooseOutputBtn.addEventListener("click", async () => {
    await chooseDirectory(outputInput, "输入输出目录", "已选择输出目录");
  });
}

if (copyJsonBtn) {
  copyJsonBtn.addEventListener("click", copyResultJson);
}
if (exportJsonBtn) {
  exportJsonBtn.addEventListener("click", exportResultJson);
}
if (exportCsvBtn) {
  exportCsvBtn.addEventListener("click", exportResultCsv);
}

setWizardStep(STEP_CONFIG);
setStatus("idle", "等待任务开始");
setTaskId("");
setRunningUi(false);
resetResultPanels();
clearError();
addEvent("前端已就绪，请先执行检测。");
refreshColumnsForCsv(csvPathInput?.value || "", "初始加载");
loadRecentHistory();
