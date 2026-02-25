const wheelFrame = document.getElementById("wheel-frame");
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
const progressAdvanced = document.getElementById("progress-advanced");
const resultAdvanced = document.getElementById("result-advanced");

const detectForm = document.getElementById("detect-form");

const advancedToggleBtn = document.getElementById("advanced-toggle-btn");
const runDetectBtn = document.getElementById("run-detect-btn");
const retryDetectBtn = document.getElementById("retry-detect-btn");
const cancelBtn = document.getElementById("cancel-btn");
const chooseCsvBtn = document.getElementById("choose-csv-btn");
const chooseOutputBtn = document.getElementById("choose-output-btn");
const newDetectBtn = document.getElementById("new-detect-btn");
const runAutoRepairBtn = document.getElementById("run-auto-repair-btn");
const runBatchRepairBtn = document.getElementById("run-batch-repair-btn");
const backResultBtn = document.getElementById("back-result-btn");
const newRepairDetectBtn = document.getElementById("new-repair-detect-btn");

const csvPathInput = document.getElementById("csv-path");
const targetColRow = document.getElementById("target-col-row");
const targetColInput = document.getElementById("target-col");
const autoModeInput = document.getElementById("auto-mode");
const maxBinsInput = document.getElementById("max-bins");
const matrixDensityInput = document.getElementById("matrix-density");
const matrixDensityValue = document.getElementById("matrix-density-value");
const maxIssuesInput = document.getElementById("max-issues");
const timeoutInput = document.getElementById("timeout-ms");
const outputInput = document.getElementById("output-dir");
const configAdvanced = document.getElementById("config-advanced");
const configBasicNote = document.getElementById("config-basic-note");
const configSummary = document.getElementById("config-summary");
const configRecommend = document.getElementById("config-recommend");
const configLast = document.getElementById("config-last");
const configFlowline = document.getElementById("config-flowline");
const csvFileInput = document.getElementById("csv-file-input");

const statusPill = document.getElementById("status-pill");
const statusIcon = document.getElementById("status-icon");
const statusMessage = document.getElementById("status-message");
const progressFill = document.getElementById("progress-fill");
const phaseHints = document.getElementById("phase-hints");
const progressTimeline = document.getElementById("progress-timeline");
const progressMetrics = document.getElementById("progress-metrics");
const taskIdLabel = document.getElementById("task-id-label");
const eventLog = document.getElementById("event-log");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");
const errorHint = document.getElementById("error-hint");

const detectionSummary = document.getElementById("detection-summary");
const detectionMessage = document.getElementById("detection-message");
const repairableOverview = document.getElementById("repairable-overview");
const nextActionText = document.getElementById("next-action-text");
const selectedIssuePill = document.getElementById("selected-issue-pill");
const issueMapList = document.getElementById("issue-map-list");
const mapScaleNote = document.getElementById("map-scale-note");

const issueDetailEmpty = document.getElementById("issue-detail-empty");
const issueDetailContent = document.getElementById("issue-detail-content");
const issueDetailSummary = document.getElementById("issue-detail-summary");
const issueDetailMessage = document.getElementById("issue-detail-message");
const issueCompareBody = document.getElementById("issue-compare-body");
const issueCompareNote = document.getElementById("issue-compare-note");
const issueDetailCheckbox = document.getElementById("issue-detail-checkbox");

const repairSummary = document.getElementById("repair-summary");
const repairDetailList = document.getElementById("repair-detail-list");
const repairBeforeBar = document.getElementById("repair-before-bar");
const repairAfterBar = document.getElementById("repair-after-bar");
const repairBeforeValue = document.getElementById("repair-before-value");
const repairAfterValue = document.getElementById("repair-after-value");
const repairReductionText = document.getElementById("repair-reduction-text");
const kpiAppliedCount = document.getElementById("kpi-applied-count");
const kpiSkippedCount = document.getElementById("kpi-skipped-count");
const kpiCellsCount = document.getElementById("kpi-cells-count");

const resultBox = document.getElementById("result-box");
const toggleResponseBtn = document.getElementById("toggle-response-btn");
const copyJsonBtn = document.getElementById("copy-json-btn");
const exportJsonBtn = document.getElementById("export-json-btn");
const exportCsvBtn = document.getElementById("export-csv-btn");
const exportJsonSideBtn = document.getElementById("export-json-side-btn");
const exportCsvSideBtn = document.getElementById("export-csv-side-btn");

const STEP_CONFIG = "config";
const STEP_PROGRESS = "progress";
const STEP_RESULT = "result";
const STEP_REPAIR = "repair";
const STEP_ORDER = [STEP_CONFIG, STEP_PROGRESS, STEP_RESULT, STEP_REPAIR];

const INTENT_SCAN = "scan";
const INTENT_REPAIR = "repair_batch";
const INTENT_TRAIN = "train";

const STEP_META = {
  [STEP_CONFIG]: {
    kicker: "STEP 1",
    title: "检测参数配置",
    subtitle: "选择 CSV 并设置扫描参数，对全列执行异常检测。",
  },
  [STEP_PROGRESS]: {
    kicker: "STEP 2",
    title: "任务执行中",
    subtitle: "任务正在运行，请稍候。",
  },
  [STEP_RESULT]: {
    kicker: "STEP 3",
    title: "检测结果与问题选择",
    subtitle: "在缩略图中查看异常热区，勾选后可批量修复。",
  },
  [STEP_REPAIR]: {
    kicker: "STEP 4",
    title: "批量修复结果",
    subtitle: "查看已应用修复与跳过项明细。",
  },
};

const WHEEL_SPIN_DEG = 34;
const WHEEL_SPIN_DURATION_MS = 720;

const RUNNING_HINTS = {
  [INTENT_SCAN]: [
    "正在读取 CSV 并校验字段...",
    "正在扫描缺失值、离群值与低频类别...",
    "正在计算风险分数并生成列缩略图...",
    "正在整理检测结果...",
  ],
  [INTENT_REPAIR]: [
    "正在加载选中的异常问题...",
    "正在应用修复规则（填补/截断/类别替换）...",
    "正在验证修复效果并写入结果文件...",
    "正在生成修复摘要...",
  ],
  [INTENT_TRAIN]: [
    "正在读取训练数据...",
    "正在训练模型...",
    "正在评估并保存模型状态...",
    "正在整理训练结果...",
  ],
};

const STATUS_PROGRESS = {
  idle: 0,
  pending: 14,
  running: 70,
  succeeded: 100,
  failed: 100,
  canceled: 100,
  timed_out: 100,
};

const STATUS_ICON_SYMBOL = {
  idle: "•",
  pending: "…",
  running: "↻",
  succeeded: "✓",
  failed: "!",
  canceled: "×",
  timed_out: "⌛",
};

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled", "timed_out"]);

const state = {
  currentStep: STEP_CONFIG,
  currentTaskId: "",
  currentTask: null,
  pollingToken: 0,
  stepAnimating: false,
  queuedStep: "",
  isRunning: false,
  lastScanPayload: null,
  lastRepairPayload: null,
  lastRepairMode: "",
  lastRepairOverview: null,
  scanResult: null,
  scanViewResult: null,
  matrixDensity: 100,
  issueById: new Map(),
  issuesByColumn: new Map(),
  selectedIssueIds: new Set(),
  activeIssueId: "",
  activeIssueSegment: null,
  autoMode: false,
  availableColumns: [],
  runningIntent: "",
  taskStartAtMS: 0,
  lastRunningHint: "",
  advancedMode: false,
  responseExpanded: false,
  mockTasks: new Map(),
};

function hasBinding(methodName) {
  return Boolean(window?.go?.main?.App?.[methodName]);
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function toInt(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.trunc(n);
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function formatNumber(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value ?? "-");
  return n.toFixed(digits);
}

function formatPercent(ratio) {
  const n = Number(ratio);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(2)}%`;
}

function formatList(value, fallback = "-") {
  const list = asArray(value)
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
  return list.length > 0 ? list.join(", ") : fallback;
}

function escapeHtml(raw) {
  return String(raw ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderDescriptionList(target, rows) {
  if (!target) return;
  target.innerHTML = rows
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v ?? "-"))}</dd>`)
    .join("");
}

function renderCompactList(target, items, fallback = "暂无内容") {
  if (!target) return;
  const values = asArray(items)
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
  const lines = values.length > 0 ? values : [fallback];
  target.innerHTML = lines.map((line) => `<li>${escapeHtml(line)}</li>`).join("");
}

function shortPath(raw) {
  const text = String(raw || "").trim();
  if (!text) return "-";
  const normalized = text.replaceAll("\\", "/");
  const pieces = normalized.split("/");
  return pieces.length > 0 ? pieces[pieces.length - 1] : text;
}

function syncExportButtons(enabled) {
  const on = Boolean(enabled);
  if (exportJsonBtn) exportJsonBtn.disabled = !on;
  if (exportCsvBtn) exportCsvBtn.disabled = !on;
  if (exportJsonSideBtn) exportJsonSideBtn.disabled = !on;
  if (exportCsvSideBtn) exportCsvSideBtn.disabled = !on;
}

function renderConfigSidebar() {
  const modeText = state.autoMode ? "全自动（全列）" : "手动（可选目标列）";
  const targetText = state.autoMode ? "全部列" : getTargetColumn() || "全部列";
  const rows = [
    ["模式", modeText],
    ["输入", shortPath(csvPathInput?.value)],
    ["目标列", targetText],
    ["超时", `${toInt(timeoutInput?.value, 90000)} ms`],
  ];
  if (state.advancedMode) {
    rows.push(
      ["分桶", toInt(maxBinsInput?.value, 120)],
      ["密度", `${getMatrixDensityPercent()}%`],
      ["最多问题", toInt(maxIssuesInput?.value, 1200)]
    );
  } else {
    rows.push(["高级参数", "未展开（使用默认值）"]);
  }
  renderDescriptionList(configSummary, rows);

  const recommends = [];
  if (state.autoMode) {
    recommends.push("先用全自动模式跑首轮，快速定位异常列。");
    recommends.push("若需精修某列，再关闭全自动模式并手动指定目标列。");
  } else {
    recommends.push("优先选择业务标签列，可更快验证结果。");
    recommends.push("若不确定目标列，可先切全自动模式扫描。");
  }
  if (state.advancedMode) {
    recommends.push("大数据建议提高超时到 120000ms 以上。");
  }
  renderCompactList(configRecommend, recommends, "暂无推荐配置。");

  const last = asObject(state.lastScanPayload);
  if (Object.keys(last).length === 0) {
    renderDescriptionList(configLast, [["状态", "尚未执行检测"]]);
  } else {
    renderDescriptionList(configLast, [
      ["模式", last?.auto_mode ? "全自动" : "手动"],
      ["输入", shortPath(last?.csv_path)],
      ["目标列", String(last?.target_col || "全部列")],
      ["分桶", toInt(last?.max_bins, 120)],
      ["最多问题", toInt(last?.max_issues, 1200)],
    ]);
  }

  renderConfigFlowline();
}

function renderConfigFlowline() {
  if (!configFlowline) return;
  const rows = Array.from(configFlowline.querySelectorAll("li"));
  if (!rows.length) return;

  const hasInput = Boolean(String(csvPathInput?.value || "").trim());
  const hasScan = Object.keys(asObject(state.lastScanPayload)).length > 0;
  const hasRepair = Object.keys(asObject(state.lastRepairPayload)).length > 0;

  let activeStep = "upload";
  if (hasInput) activeStep = "detect";
  if (hasScan) activeStep = "repair";

  rows.forEach((row) => {
    const key = String(row.getAttribute("data-flow-step") || "").trim();
    row.classList.remove("done", "active", "pending");
    if (key === "upload") {
      if (!hasInput) row.classList.add("active");
      else row.classList.add("done");
      return;
    }
    if (key === "detect") {
      if (!hasInput) row.classList.add("pending");
      else if (!hasScan) row.classList.add("active");
      else row.classList.add("done");
      return;
    }
    if (key === "repair") {
      if (!hasScan) row.classList.add("pending");
      else if (!hasRepair) row.classList.add("active");
      else row.classList.add("done");
      return;
    }
    if (key === activeStep) row.classList.add("active");
    else row.classList.add("pending");
  });
}

function runningStageIndex(status, elapsedMS, stepCount) {
  if (stepCount <= 0) return 0;
  const normalized = String(status || "").toLowerCase();
  if (normalized === "pending") return 0;
  if (normalized === "running") {
    if (elapsedMS < 1200) return 0;
    if (elapsedMS < 3200) return Math.min(1, stepCount - 1);
    if (elapsedMS < 5200) return Math.min(2, stepCount - 1);
    return Math.min(3, stepCount - 1);
  }
  if (TERMINAL_STATUSES.has(normalized)) return stepCount - 1;
  return 0;
}

function renderProgressTimeline(task, intent) {
  if (!progressTimeline) return;
  const labels = RUNNING_HINTS[intent] || RUNNING_HINTS[INTENT_SCAN];
  const status = String(task?.status || "idle").toLowerCase();
  const elapsedMS = Math.max(0, Date.now() - toInt(state.taskStartAtMS, Date.now()));
  const activeIndex = runningStageIndex(status, elapsedMS, labels.length);

  progressTimeline.innerHTML = labels
    .map((label, idx) => {
      const cls = idx < activeIndex ? "done" : idx === activeIndex ? "active" : "";
      return `<li class="${cls}">${escapeHtml(label)}</li>`;
    })
    .join("");
}

function formatRemainingTime(task) {
  const timeoutMS = toInt(task?.timeout_ms, toInt(timeoutInput?.value, 90000));
  const elapsedMS = Math.max(0, Date.now() - toInt(state.taskStartAtMS, Date.now()));
  if (!state.isRunning) return "0s";
  if (timeoutMS <= 0) return "估算中";
  const remain = Math.max(0, timeoutMS - elapsedMS);
  return `${Math.ceil(remain / 1000)}s`;
}

function renderProgressMetrics(task, intent) {
  const request = asObject(task?.request);
  const response = asObject(task?.response);
  const result = asObject(response?.result);
  const profile = asObject(result?.data_profile);
  const scannedRows = toInt(profile?.rows, 0);
  const issueCount = toInt(result?.issue_count, 0);
  renderDescriptionList(progressMetrics, [
    ["任务类型", intent === INTENT_REPAIR ? "批量修复" : "全列检测"],
    ["已扫描行数", scannedRows > 0 ? scannedRows : "计算中"],
    ["发现问题", issueCount > 0 ? issueCount : "计算中"],
    ["超时设置", `${toInt(request?.payload?.timeout_ms, toInt(timeoutInput?.value, 90000))} ms`],
    ["预计剩余", formatRemainingTime(task)],
  ]);
}

function renderProgressSidebar(task, intent) {
  renderProgressTimeline(task, intent);
  renderProgressMetrics(task, intent);
}

function renderRepairOverviewSidebar(result) {
  const data = asObject(result);
  const before = Math.max(0, toInt(data?.scan_issue_count, 0));
  const applied = Math.max(0, toInt(data?.applied_issue_count, 0));
  const after = Math.max(0, before - applied);
  const maxValue = Math.max(1, before, after);
  const beforePct = (before / maxValue) * 100;
  const afterPct = (after / maxValue) * 100;

  if (repairBeforeBar) repairBeforeBar.style.width = `${beforePct.toFixed(2)}%`;
  if (repairAfterBar) repairAfterBar.style.width = `${afterPct.toFixed(2)}%`;
  if (repairBeforeValue) repairBeforeValue.textContent = String(before);
  if (repairAfterValue) repairAfterValue.textContent = String(after);

  const reduced = Math.max(0, before - after);
  const ratio = before > 0 ? ((reduced / before) * 100).toFixed(1) : "0.0";
  if (repairReductionText) {
    repairReductionText.textContent = `已减少 ${reduced} 个问题（改善 ${ratio}%）。`;
  }

  state.lastRepairOverview = { before, after, reduced, ratio };
}

function setRepairKpis(applied = 0, skipped = 0, cells = 0) {
  const values = [
    [kpiAppliedCount, applied],
    [kpiSkippedCount, skipped],
    [kpiCellsCount, cells],
  ];
  for (const [node, raw] of values) {
    if (!node) continue;
    const value = Math.max(0, toInt(raw, 0));
    node.textContent = String(value);
    node.classList.toggle("is-positive", value > 0);
  }
}

function resetRepairKpis() {
  setRepairKpis(0, 0, 0);
}

function resetRepairOverviewSidebar() {
  if (repairBeforeBar) repairBeforeBar.style.width = "0%";
  if (repairAfterBar) repairAfterBar.style.width = "0%";
  if (repairBeforeValue) repairBeforeValue.textContent = "0";
  if (repairAfterValue) repairAfterValue.textContent = "0";
  if (repairReductionText) repairReductionText.textContent = "等待修复任务完成后生成对比。";
  state.lastRepairOverview = null;
}

function normalizeColumnList(columns) {
  const values = asArray(columns)
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
  const uniq = [];
  const seen = new Set();
  for (const col of values) {
    if (seen.has(col)) continue;
    seen.add(col);
    uniq.push(col);
  }
  return uniq;
}

function getTargetColumn() {
  if (state.autoMode) return "";
  return String(targetColInput?.value || "").trim();
}

function getAutoRepairIssueIds() {
  const source = asArray(state.scanViewResult?.issues).length > 0 ? state.scanViewResult : state.scanResult;
  return asArray(source?.issues)
    .map((item) => String(item?.issue_id || "").trim())
    .filter(Boolean);
}

function getMatrixDensityPercent() {
  const fallback = toInt(state.matrixDensity, 100);
  const raw = matrixDensityInput?.value;
  return clamp(toInt(raw, fallback), 50, 180);
}

function updateMatrixDensityUi() {
  const percent = getMatrixDensityPercent();
  state.matrixDensity = percent;
  if (matrixDensityValue) {
    matrixDensityValue.textContent = `${percent}%`;
  }
}

function setTargetOptions(columns, preferred = "") {
  const uniq = normalizeColumnList(columns);
  state.availableColumns = uniq;

  if (!targetColInput) {
    return;
  }

  const current = String(preferred || targetColInput.value || "").trim();
  targetColInput.innerHTML = "";

  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = "全部列";
  targetColInput.appendChild(allOption);

  for (const col of uniq) {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    targetColInput.appendChild(option);
  }

  if (current && uniq.includes(current)) {
    targetColInput.value = current;
  } else if (uniq.includes("stroke")) {
    targetColInput.value = "stroke";
  } else {
    targetColInput.value = "";
  }
}

function buildScanView(scanResult) {
  const source = asObject(scanResult);
  const target = getTargetColumn();
  if (!target) {
    return source;
  }

  const issues = asArray(source?.issues).filter((item) => String(item?.column || "") === target);
  const thumbnails = asArray(source?.column_thumbnails).filter((item) => String(item?.column || "") === target);
  const summary = asObject(source?.scan_summary);

  return {
    ...source,
    issues,
    issue_count: issues.length,
    column_thumbnails: thumbnails,
    anomaly_columns: thumbnails.map((item) => String(item?.column || "")),
    scan_summary: {
      ...summary,
      anomaly_column_count: thumbnails.length,
      high_risk_columns: asArray(summary?.high_risk_columns).filter((item) => String(item || "") === target),
      medium_risk_columns: asArray(summary?.medium_risk_columns).filter((item) => String(item || "") === target),
      total_issues: issues.length,
    },
  };
}
function setTaskId(taskId) {
  if (taskIdLabel) taskIdLabel.textContent = `Task: ${taskId || "-"}`;
}

function setStatus(status, message) {
  const normalized = String(status || "idle").toLowerCase();
  const progress = STATUS_PROGRESS[normalized] ?? 0;

  if (statusIcon) {
    const symbol = STATUS_ICON_SYMBOL[normalized] || STATUS_ICON_SYMBOL.idle;
    statusIcon.className = `status-icon ${normalized}`;
    statusIcon.textContent = symbol;
  }
  if (statusPill) {
    statusPill.className = `status-pill ${normalized}`;
    statusPill.textContent = normalized;
  }
  if (statusMessage) statusMessage.textContent = message || normalized;
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
  if (!eventLog) return;
  const li = document.createElement("li");
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const tid = String(taskId || "").trim();
  const visible = tid ? `[${tid}] ${message}` : String(message);

  const timeSpan = document.createElement("span");
  timeSpan.className = "event-time";
  timeSpan.textContent = `${hh}:${mm}:${ss}`;

  const textSpan = document.createElement("span");
  textSpan.textContent = visible;

  li.append(timeSpan, textSpan);
  eventLog.prepend(li);
  emitFrontendLog(visible, tid);
}

function showError(message, hint = "请检查参数后重试。") {
  if (errorPanel) errorPanel.classList.remove("hidden");
  if (errorMessage) errorMessage.textContent = String(message || "-");
  if (errorHint) errorHint.textContent = String(hint || "");
}

function clearError() {
  if (errorPanel) errorPanel.classList.add("hidden");
  if (errorMessage) errorMessage.textContent = "-";
  if (errorHint) errorHint.textContent = "修正后可重试。";
}

function updateSelectedIssuePill() {
  const count = state.selectedIssueIds.size;
  const autoIssueCount = getAutoRepairIssueIds().length;
  if (selectedIssuePill) {
    selectedIssuePill.textContent = `已选 ${count} 项`;
  }
  if (runAutoRepairBtn) {
    runAutoRepairBtn.disabled = state.isRunning || !state.scanResult || autoIssueCount <= 0;
    runAutoRepairBtn.textContent = autoIssueCount > 0 ? `自动修复全部问题列 (${autoIssueCount})` : "自动修复全部问题列";
  }
  if (runBatchRepairBtn) {
    runBatchRepairBtn.textContent = count > 0 ? `执行修复 (${count})` : "执行修复";
    runBatchRepairBtn.disabled = state.isRunning || count <= 0 || !state.scanResult;
  }
}

function setRunningUi(isRunning) {
  state.isRunning = Boolean(isRunning);

  if (runDetectBtn) runDetectBtn.disabled = state.isRunning;
  if (retryDetectBtn) retryDetectBtn.disabled = state.isRunning || !state.lastScanPayload;
  if (cancelBtn) cancelBtn.disabled = !state.isRunning || !state.currentTaskId;
  if (chooseCsvBtn) chooseCsvBtn.disabled = state.isRunning;
  if (chooseOutputBtn) chooseOutputBtn.disabled = state.isRunning;

  if (csvPathInput) csvPathInput.disabled = state.isRunning;
  if (targetColInput) targetColInput.disabled = state.isRunning || state.autoMode;
  if (maxBinsInput) maxBinsInput.disabled = state.isRunning;
  if (matrixDensityInput) matrixDensityInput.disabled = state.isRunning;
  if (maxIssuesInput) maxIssuesInput.disabled = state.isRunning;
  if (timeoutInput) timeoutInput.disabled = state.isRunning;
  if (outputInput) outputInput.disabled = state.isRunning;

  if (newDetectBtn) newDetectBtn.disabled = state.isRunning;
  if (backResultBtn) backResultBtn.disabled = state.isRunning;
  if (newRepairDetectBtn) newRepairDetectBtn.disabled = state.isRunning;

  updateSelectedIssuePill();
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

  const showProgressAdvanced = state.advancedMode && step === STEP_PROGRESS;
  if (progressAdvanced) {
    progressAdvanced.classList.toggle("hidden", !showProgressAdvanced);
  }

  const showResultAdvanced = state.advancedMode && step === STEP_RESULT;
  if (resultAdvanced) {
    resultAdvanced.classList.toggle("hidden", !showResultAdvanced);
  }

  const showResponse = state.advancedMode && (step === STEP_RESULT || step === STEP_REPAIR);
  if (responseShell) {
    responseShell.classList.toggle("hidden", !showResponse);
  }
}

function setResponseExpanded(expanded) {
  state.responseExpanded = Boolean(expanded);
  if (responseShell) {
    responseShell.classList.toggle("collapsed", !state.responseExpanded);
  }
  if (toggleResponseBtn) {
    toggleResponseBtn.textContent = state.responseExpanded ? "收起原始响应" : "展开原始响应";
  }
}

function setConfigAdvancedVisibility(enabled) {
  const on = Boolean(enabled);
  if (configAdvanced) {
    configAdvanced.classList.toggle("hidden", !on);
  }
  if (configBasicNote) {
    configBasicNote.classList.toggle("hidden", on);
  }
  renderConfigSidebar();
}

function setAutoMode(enabled, options = {}) {
  state.autoMode = Boolean(enabled);
  if (autoModeInput) autoModeInput.checked = state.autoMode;

  if (targetColRow) {
    targetColRow.classList.toggle("hidden", state.autoMode);
  }
  if (targetColInput) {
    if (state.autoMode) {
      targetColInput.value = "";
    }
    targetColInput.disabled = state.isRunning || state.autoMode;
  }

  if (!options?.silent) {
    addEvent(state.autoMode ? "已开启全自动模式：将对全部列检测并可一键自动修复。" : "已关闭全自动模式。");
  }

  renderConfigSidebar();
  if (state.scanResult) {
    renderScanResult(state.scanResult);
  }
}

function setAdvancedMode(enabled) {
  state.advancedMode = Boolean(enabled);
  if (advancedToggleBtn) {
    advancedToggleBtn.textContent = state.advancedMode ? "高级模式: 开" : "高级模式: 关";
    advancedToggleBtn.classList.toggle("is-on", state.advancedMode);
  }
  if (!state.advancedMode) {
    if (progressAdvanced) progressAdvanced.open = false;
    if (resultAdvanced) resultAdvanced.open = false;
    setResponseExpanded(false);
  }
  setConfigAdvancedVisibility(state.advancedMode);
  setStepVisibility(state.currentStep || STEP_CONFIG);
}

function setStageProgress(step) {
  const idx = STEP_ORDER.indexOf(step);
  const ratio = idx <= 0 ? 0 : idx / (STEP_ORDER.length - 1);
  if (stageFill) stageFill.style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;

  if (!stageSteps) return;
  const items = Array.from(stageSteps.querySelectorAll("li"));
  items.forEach((item, i) => {
    item.classList.remove("active", "completed");
    if (i < idx) item.classList.add("completed");
    else if (i === idx) item.classList.add("active");
  });
}

function applyStepNow(step) {
  const normalized = STEP_META[step] ? step : STEP_CONFIG;
  state.currentStep = normalized;
  const meta = STEP_META[normalized];
  if (cardKicker) cardKicker.textContent = meta.kicker;
  if (cardTitle) cardTitle.textContent = meta.title;
  if (cardSubtitle) cardSubtitle.textContent = meta.subtitle;
  if (wheelFrame) wheelFrame.setAttribute("data-step", normalized);
  if (wizardCard) wizardCard.setAttribute("data-step", normalized);
  setStageProgress(normalized);
  setStepVisibility(normalized);
  if (normalized === STEP_CONFIG) renderConfigSidebar();
}

function clearWheelPose() {
  if (!wheelOrbit) return;
  wheelOrbit.style.transform = "rotate(0deg)";
  wheelOrbit.style.opacity = "1";
}

function cancelWheelAnimations() {
  if (!wheelOrbit || typeof wheelOrbit.getAnimations !== "function") return;
  for (const animation of wheelOrbit.getAnimations()) animation.cancel();
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

  return animation.finished.catch(() => {}).then(() => {
    if (!wheelOrbit) return;
    wheelOrbit.style.transform = `rotate(${toDeg}deg)`;
    wheelOrbit.style.opacity = String(toOpacity);
  });
}

function stepDirection(fromStep, toStep) {
  const fromIdx = Math.max(0, STEP_ORDER.indexOf(fromStep));
  const toIdx = Math.max(0, STEP_ORDER.indexOf(toStep));
  if (toIdx === fromIdx) return -1;
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

  try {
    const halfDuration = Math.round(WHEEL_SPIN_DURATION_MS / 2);
    const direction = stepDirection(state.currentStep, normalized);
    await animateOrbitTransform(0, direction * WHEEL_SPIN_DEG, 1, 0.03, halfDuration);
    applyStepNow(normalized);
    await animateOrbitTransform(-direction * WHEEL_SPIN_DEG, 0, 0.03, 1, halfDuration);
  } finally {
    wizardCard.classList.remove("is-animating");
    wizardCard.style.pointerEvents = "";
    state.stepAnimating = false;
    clearWheelPose();
  }

  if (state.queuedStep && state.queuedStep !== state.currentStep) {
    const queued = state.queuedStep;
    state.queuedStep = "";
    await transitionWizardStep(queued, false);
  }
}

function setWizardStep(step, options = {}) {
  void transitionWizardStep(step, Boolean(options?.immediate));
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

function csvEscape(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}
function clearIssueDetail(message = "将鼠标移到红色异常段查看详情。") {
  state.activeIssueId = "";
  state.activeIssueSegment = null;

  if (issueDetailEmpty) {
    issueDetailEmpty.textContent = message;
    issueDetailEmpty.classList.remove("hidden");
  }
  if (issueDetailContent) issueDetailContent.classList.add("hidden");
  if (issueDetailSummary) issueDetailSummary.innerHTML = "";
  if (issueDetailMessage) issueDetailMessage.textContent = "-";
  if (issueCompareBody) {
    issueCompareBody.innerHTML = "<tr><td>-</td><td>-</td><td>-</td></tr>";
  }
  if (issueCompareNote) {
    issueCompareNote.textContent = "预估值基于当前规则生成，最终结果以修复执行输出为准。";
  }

  if (issueDetailCheckbox) {
    issueDetailCheckbox.checked = false;
    issueDetailCheckbox.disabled = true;
    delete issueDetailCheckbox.dataset.issueId;
  }
}

function setScanRepairVisibility(hasIssues) {
  if (runAutoRepairBtn) runAutoRepairBtn.classList.toggle("hidden", !hasIssues || !state.autoMode);
  if (runBatchRepairBtn) runBatchRepairBtn.classList.toggle("hidden", !hasIssues);
  if (selectedIssuePill) selectedIssuePill.classList.toggle("hidden", !hasIssues);
}

function resetResultPanels() {
  if (detectionSummary) detectionSummary.innerHTML = "";
  if (detectionMessage) detectionMessage.textContent = "-";
  renderCompactList(repairableOverview, ["暂无可修复项"]);
  if (nextActionText) nextActionText.textContent = "先运行检测。";
  if (repairSummary) repairSummary.innerHTML = "";
  if (repairDetailList) repairDetailList.innerHTML = "";
  if (issueMapList) issueMapList.innerHTML = "";
  updateMapScaleLegend({});
  renderPhaseHints(INTENT_SCAN, "idle");
  renderProgressSidebar({ status: "idle" }, INTENT_SCAN);

  state.scanResult = null;
  state.scanViewResult = null;
  state.lastRepairMode = "";
  state.issueById = new Map();
  state.issuesByColumn = new Map();
  state.selectedIssueIds.clear();
  state.lastRunningHint = "";

  clearIssueDetail();
  setScanRepairVisibility(false);
  updateSelectedIssuePill();
  setResponseExpanded(false);
  resetRepairKpis();
  resetRepairOverviewSidebar();
  renderConfigSidebar();

  if (resultBox) resultBox.textContent = "{}";
  if (copyJsonBtn) copyJsonBtn.textContent = "复制";
  syncExportButtons(false);
}

function severityClass(level) {
  const normalized = String(level || "").toLowerCase();
  if (normalized === "high") return "high";
  if (normalized === "medium") return "medium";
  return "low";
}

function issueTypeLabel(type) {
  const normalized = String(type || "").toLowerCase();
  if (normalized === "missing_values") return "缺失值";
  if (normalized === "numeric_outlier") return "数值离群";
  if (normalized === "rare_category") return "低频类别";
  return normalized || "-";
}

function issueTypeReasonText(type) {
  const normalized = String(type || "").toLowerCase();
  if (normalized === "missing_values") return "该列存在缺失值，需要填补空单元格。";
  if (normalized === "numeric_outlier") return "该列存在超出合理范围的离群值，需要截断到稳定区间。";
  if (normalized === "rare_category") return "该列存在低频类别，影响分布稳定性，需要归并到常见类别。";
  return "检测到可修复异常，已按默认规则处理。";
}

function formatReplacementValue(value) {
  if (value == null) return "-";
  if (typeof value === "number") {
    return Number.isFinite(value) ? formatNumber(value, 6).replace(/\.?0+$/, "") : String(value);
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function issueTypeFixText(type, preview) {
  const normalized = String(type || "").toLowerCase();
  const replacement = formatReplacementValue(preview);
  if (normalized === "missing_values") {
    return `采用填补策略，将缺失值替换为 ${replacement}。`;
  }
  if (normalized === "numeric_outlier") {
    const range = asObject(preview);
    const lower = formatReplacementValue(range?.lower_bound);
    const upper = formatReplacementValue(range?.upper_bound);
    return `采用截断策略，将离群值限制在 [${lower}, ${upper}]。`;
  }
  if (normalized === "rare_category") {
    return `采用类别归并策略，将低频值替换为 ${replacement}。`;
  }
  return `已应用默认修复策略，替换预览: ${replacement}。`;
}

function skipReasonText(reason) {
  const normalized = String(reason || "").toLowerCase();
  if (normalized === "issue_not_found") return "未在当前扫描结果中找到该问题。";
  if (normalized === "column_not_found") return "对应列不存在（列名已变更或文件不一致）。";
  if (normalized === "unsupported_issue_type") return "当前版本暂不支持该异常类型的自动修复。";
  if (normalized === "no_rows_matched") return "未匹配到待修复行（可能已被手工修正）。";
  if (normalized === "strategy_disabled") return "该类型在当前修复策略中被禁用。";
  if (normalized === "dependency_unresolved") return "依赖列尚未完成修复，已按依赖规则跳过。";
  if (normalized === "dependency_column_not_found") return "依赖列不存在，当前问题已跳过。";
  if (normalized === "dependency_cycle") return "检测到列依赖环，当前问题已跳过。";
  if (normalized === "conflict_all_skipped") return "与其他修复项发生冲突，且按冲突策略被全部跳过。";
  if (!normalized) return "未提供跳过原因。";
  return normalized;
}

function getColumnProfile(column) {
  const profiles = asArray(state.scanResult?.data_profile?.column_profiles);
  for (const item of profiles) {
    if (String(item?.column || "") === String(column || "")) {
      return asObject(item);
    }
  }
  return {};
}

function normalizeSampleValue(value) {
  if (value == null || String(value).toLowerCase() === "nan") return "(缺失)";
  if (typeof value === "number") return formatReplacementValue(value);
  return String(value);
}

function estimateIssueAfterValue(issue, rawValue) {
  const issueType = String(issue?.issue_type || "").toLowerCase();
  const detail = asObject(issue?.detail);
  const columnProfile = getColumnProfile(issue?.column);

  if (issueType === "numeric_outlier") {
    const num = Number(rawValue);
    const lower = Number(detail?.lower_bound);
    const upper = Number(detail?.upper_bound);
    if (Number.isFinite(num) && Number.isFinite(lower) && Number.isFinite(upper)) {
      return formatReplacementValue(clamp(num, lower, upper));
    }
    return `截断到 [${formatReplacementValue(detail?.lower_bound)}, ${formatReplacementValue(detail?.upper_bound)}]`;
  }

  if (issueType === "rare_category") {
    return "替换为主类别（自动计算）";
  }

  if (issueType === "missing_values") {
    if (columnProfile?.is_numeric) {
      return "中位数填补（自动计算）";
    }
    return "众数/UNKNOWN 填补（自动计算）";
  }

  return "按默认修复策略处理";
}

function compareNoteText(issue) {
  const issueType = String(issue?.issue_type || "").toLowerCase();
  const detail = asObject(issue?.detail);
  if (issueType === "numeric_outlier") {
    return `数值离群将被限制在 [${formatReplacementValue(detail?.lower_bound)}, ${formatReplacementValue(
      detail?.upper_bound
    )}]。`;
  }
  if (issueType === "rare_category") {
    return "低频类别会归并到主类别，具体替换值由列分布自动确定。";
  }
  if (issueType === "missing_values") {
    return "缺失值会按列类型自动填补（数值列使用中位数，类别列使用众数或 UNKNOWN）。";
  }
  return "修复后预览基于当前策略估算，最终结果以实际修复输出为准。";
}

function renderIssueCompareTable(issue) {
  if (!issueCompareBody) return;
  const detail = asObject(issue?.detail);
  const preview = asArray(detail?.preview).slice(0, 5);
  const rows = [];

  for (const item of preview) {
    const rowLabel = item?.row ?? "-";
    const beforeValue = normalizeSampleValue(item?.value);
    const afterValue = estimateIssueAfterValue(issue, item?.value);
    rows.push([rowLabel, beforeValue, afterValue]);
  }

  if (!rows.length) {
    rows.push(["-", "无样例", estimateIssueAfterValue(issue, null)]);
  }

  issueCompareBody.innerHTML = rows
    .map(
      ([rowLabel, beforeValue, afterValue]) =>
        `<tr><td>${escapeHtml(rowLabel)}</td><td>${escapeHtml(beforeValue)}</td><td>${escapeHtml(afterValue)}</td></tr>`
    )
    .join("");

  if (issueCompareNote) {
    issueCompareNote.textContent = compareNoteText(issue);
  }
}

function updateColumnSelectionIndicators() {
  if (!issueMapList) return;
  const nodes = issueMapList.querySelectorAll(".column-selected-count");
  nodes.forEach((node) => {
    const column = String(node.getAttribute("data-column") || "");
    const issues = state.issuesByColumn.get(column) || [];
    const selected = issues.filter((item) => state.selectedIssueIds.has(String(item?.issue_id || ""))).length;
    node.textContent = selected > 0 ? `已选 ${selected}` : "未选";
    node.classList.toggle("has-selected", selected > 0);
  });
}

function refreshIssueSegmentStates() {
  if (!issueMapList) return;
  const segments = issueMapList.querySelectorAll(".thumb-segment, .matrix-cell[data-issue-id]");
  segments.forEach((node) => {
    const issueId = String(node.getAttribute("data-issue-id") || "");
    node.classList.toggle("is-selected", state.selectedIssueIds.has(issueId));
    node.classList.toggle("is-active", state.activeIssueId && state.activeIssueId === issueId);
  });
  if (issueDetailCheckbox && state.activeIssueId) {
    issueDetailCheckbox.checked = state.selectedIssueIds.has(state.activeIssueId);
    issueDetailCheckbox.dataset.issueId = state.activeIssueId;
  }
  updateColumnSelectionIndicators();
}

function setIssueSelection(issueId, selected) {
  const normalized = String(issueId || "").trim();
  if (!normalized || !state.issueById.has(normalized)) return;

  if (selected) state.selectedIssueIds.add(normalized);
  else state.selectedIssueIds.delete(normalized);

  updateSelectedIssuePill();
  refreshIssueSegmentStates();
}

function toggleIssueSelection(issueId) {
  const normalized = String(issueId || "").trim();
  if (!normalized) return;
  setIssueSelection(normalized, !state.selectedIssueIds.has(normalized));
}

function formatIssueDetailMessage(issue, segmentInfo) {
  const issueType = String(issue?.issue_type || "");
  const detail = asObject(issue?.detail);
  const preview = asArray(detail?.preview)
    .slice(0, 3)
    .map((item) => `row ${item?.row}: ${item?.value}`)
    .join(" | ");

  let text = "";
  if (issueType === "missing_values") {
    text = `缺失值 ${detail?.missing_count ?? issue?.count ?? "-"} 条`;
  } else if (issueType === "numeric_outlier") {
    text = `离群值 ${detail?.outlier_count ?? issue?.count ?? "-"} 条，建议范围 [${detail?.lower_bound ?? "-"}, ${
      detail?.upper_bound ?? "-"
    }]`;
  } else if (issueType === "rare_category") {
    text = `低频值 ${detail?.rare_count ?? issue?.count ?? "-"} 条，候选值 ${formatList(detail?.rare_values_preview, "-")}`;
  } else {
    text = "异常详情已载入。";
  }

  if (segmentInfo && Number.isFinite(Number(segmentInfo?.startRow)) && Number.isFinite(Number(segmentInfo?.endRow))) {
    text += `；当前热区约在行 ${segmentInfo.startRow} ~ ${segmentInfo.endRow}`;
  }
  if (preview) {
    text += `；样例 ${preview}`;
  }
  return text;
}

function setActiveIssue(issueId, segmentInfo = null) {
  const issue = state.issueById.get(String(issueId || ""));
  if (!issue) {
    clearIssueDetail();
    return;
  }

  state.activeIssueId = String(issue.issue_id || "");
  state.activeIssueSegment = segmentInfo || null;

  if (issueDetailEmpty) issueDetailEmpty.classList.add("hidden");
  if (issueDetailContent) issueDetailContent.classList.remove("hidden");

  const rows = [
    ["issue_id", issue?.issue_id ?? "-"],
    ["列名", issue?.column ?? "-"],
    ["类型", issueTypeLabel(issue?.issue_type)],
    ["严重度", issue?.severity ?? "-"],
    ["数量", issue?.count ?? "-"],
    ["占比", formatPercent(issue?.ratio)],
    ["风险", issue?.risk_level ?? "-"],
    ["评分", formatNumber(issue?.issue_score ?? 0, 4)],
  ];
  if (segmentInfo) {
    rows.push(["热区", `${segmentInfo.startRow ?? "-"} ~ ${segmentInfo.endRow ?? "-"}`]);
  }
  renderDescriptionList(issueDetailSummary, rows);

  if (issueDetailMessage) {
    issueDetailMessage.textContent = formatIssueDetailMessage(issue, segmentInfo);
  }
  renderIssueCompareTable(issue);
  if (issueDetailCheckbox) {
    issueDetailCheckbox.disabled = false;
    issueDetailCheckbox.dataset.issueId = String(issue?.issue_id || "");
    issueDetailCheckbox.checked = state.selectedIssueIds.has(String(issue?.issue_id || ""));
  }

  refreshIssueSegmentStates();
}

function binaryBinsToSegments(binaryBins) {
  const bins = asArray(binaryBins).map((v) => (Number(v) > 0 ? 1 : 0));
  const segments = [];
  let start = -1;
  for (let i = 0; i <= bins.length; i += 1) {
    const active = i < bins.length && bins[i] > 0;
    if (active && start < 0) {
      start = i;
      continue;
    }
    if (!active && start >= 0) {
      segments.push({ bin_start: start, bin_end: i - 1 });
      start = -1;
    }
  }
  return segments;
}

function normalizeIssueSegments(issue, totalBins) {
  const rawSegments = asArray(issue?.segments);
  const source = rawSegments.length > 0 ? rawSegments : binaryBinsToSegments(issue?.bins);

  if (source.length === 0) {
    return [
      {
        startBin: 0,
        endBin: 0,
        left: 0,
        width: 1,
        startRow: 0,
        endRow: 0,
      },
    ];
  }

  return source.map((rawSegment) => {
    const start = clamp(toInt(rawSegment?.bin_start, 0), 0, Math.max(totalBins - 1, 0));
    const end = clamp(toInt(rawSegment?.bin_end, start), start, Math.max(totalBins - 1, 0));
    return {
      startBin: start,
      endBin: end,
      left: (start / Math.max(totalBins, 1)) * 100,
      width: Math.max(((end - start + 1) / Math.max(totalBins, 1)) * 100, 0.8),
      startRow: toInt(rawSegment?.start_row, 0),
      endRow: toInt(rawSegment?.end_row, 0),
    };
  });
}

function severityWeight(level) {
  const normalized = severityClass(level);
  if (normalized === "high") return 3;
  if (normalized === "medium") return 2;
  return 1;
}

function chooseMatrixCols(totalBins) {
  const bins = Math.max(1, toInt(totalBins, 1));
  if (bins <= 12) {
    return bins;
  }
  const byRows = Math.ceil(bins / 6);
  const base = clamp(byRows, 16, 36);
  const density = getMatrixDensityPercent();
  const scaled = Math.round(base * (density / 100));
  return clamp(scaled, 10, Math.min(56, bins));
}

function buildBinIssueMap(issues, totalBins) {
  const map = new Array(Math.max(1, totalBins)).fill(null);
  const sorted = [...issues].sort((a, b) => Number(b?.issue_score ?? 0) - Number(a?.issue_score ?? 0));

  for (const issue of sorted) {
    const issueId = String(issue?.issue_id || "").trim();
    if (!issueId) continue;
    const rank = severityWeight(issue?.severity);
    const score = Number(issue?.issue_score ?? 0);
    const segments = normalizeIssueSegments(issue, totalBins);

    for (const segment of segments) {
      for (let bin = segment.startBin; bin <= segment.endBin; bin += 1) {
        const current = map[bin];
        if (!current || rank > current.rank || (rank === current.rank && score > current.score)) {
          map[bin] = {
            issueId,
            severity: String(issue?.severity || "low"),
            rank,
            score,
            segment: {
              startRow: segment.startRow,
              endRow: segment.endRow,
              startBin: segment.startBin,
              endBin: segment.endBin,
            },
          };
        }
      }
    }
  }

  return map;
}

function formatMatrixRowLabel(rowIndex, matrixCols, totalBins, rowCount) {
  const firstBin = rowIndex * matrixCols;
  const lastBin = Math.min(totalBins - 1, firstBin + matrixCols - 1);
  if (firstBin > lastBin) {
    return `R${rowIndex + 1}`;
  }
  const safeRowCount = Math.max(0, toInt(rowCount, 0));
  if (safeRowCount <= 0) {
    return `R${rowIndex + 1}`;
  }
  const binSize = Math.max(1, Math.ceil(safeRowCount / Math.max(totalBins, 1)));
  const start = Math.min(safeRowCount - 1, firstBin * binSize);
  const end = Math.min(safeRowCount - 1, (lastBin + 1) * binSize - 1);
  if (start < 0 || end < 0) {
    return `R${rowIndex + 1}`;
  }
  return `${start}-${end}`;
}

function renderScanIssueMap(scanResult) {
  if (!issueMapList) return;
  issueMapList.innerHTML = "";
  updateMapScaleLegend(scanResult);

  let thumbnails = asArray(scanResult?.column_thumbnails);
  if (state.autoMode) {
    thumbnails = thumbnails.filter((item) => toInt(item?.issue_count, 0) > 0 || toInt(item?.anomaly_points, 0) > 0);
  }
  if (thumbnails.length === 0) {
    issueMapList.innerHTML = state.autoMode
      ? '<div class="scan-placeholder">全自动模式下未发现异常列。</div>'
      : '<div class="scan-placeholder">暂无列缩略图。</div>';
    return;
  }

  const fragment = document.createDocumentFragment();
  const rowCount = toInt(scanResult?.data_profile?.rows, 0);

  for (const thumbnail of thumbnails) {
    const column = String(thumbnail?.column ?? "");
    const issues = state.issuesByColumn.get(column) || [];
    const totalBins = Math.max(1, toInt(thumbnail?.total_bins, asArray(thumbnail?.bins).length || 1));

    const card = document.createElement("article");
    card.className = "issue-column-card";
    card.dataset.column = column;

    const header = document.createElement("header");
    header.className = "issue-column-head";
    header.innerHTML = `
      <div>
        <p class="issue-column-title">${escapeHtml(column || "(unknown)")}</p>
        <div class="column-meta">
          <span>${escapeHtml(String(thumbnail?.dtype ?? "-"))}</span>
          <span>风险 ${escapeHtml(String(thumbnail?.risk_level ?? "-"))}</span>
          <span>问题 ${escapeHtml(String(thumbnail?.issue_count ?? 0))}</span>
        </div>
      </div>
      <span class="column-selected-count" data-column="${escapeHtml(column)}">未选</span>
    `;
    card.appendChild(header);

    const matrixWrap = document.createElement("div");
    matrixWrap.className = "thumbnail-matrix-wrap";

    const matrixCols = chooseMatrixCols(totalBins);
    const matrixRows = Math.max(1, Math.ceil(totalBins / matrixCols));
    const matrixSize = matrixCols * matrixRows;
    const binIssueMap = buildBinIssueMap(issues, totalBins);

    const grid = document.createElement("div");
    grid.className = `matrix-grid-table risk-${severityClass(thumbnail?.risk_level)}`;
    grid.style.gridTemplateColumns = `84px repeat(${matrixCols}, minmax(0, 1fr))`;

    const corner = document.createElement("div");
    corner.className = "matrix-head-corner";
    corner.textContent = "行/列";
    grid.appendChild(corner);

    for (let col = 0; col < matrixCols; col += 1) {
      const head = document.createElement("div");
      head.className = "matrix-head-col";
      head.textContent = `列${col + 1}`;
      head.title = `矩阵列 ${col + 1}`;
      grid.appendChild(head);
    }

    for (let row = 0; row < matrixRows; row += 1) {
      const rowHead = document.createElement("div");
      rowHead.className = "matrix-head-row";
      rowHead.textContent = formatMatrixRowLabel(row, matrixCols, totalBins, rowCount);
      rowHead.title = `矩阵行 ${row + 1}`;
      grid.appendChild(rowHead);

      for (let col = 0; col < matrixCols; col += 1) {
        const bin = row * matrixCols + col;
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "matrix-cell";

        if (bin >= totalBins) {
          cell.classList.add("filler");
          cell.disabled = true;
        } else {
          const entry = binIssueMap[bin];
          if (entry) {
            cell.classList.add("has-issue", `severity-${severityClass(entry.severity)}`);
            cell.setAttribute("data-issue-id", entry.issueId);
            cell.title = `${entry.issueId} | bin ${bin + 1}/${totalBins}`;

            cell.addEventListener("pointerenter", () => setActiveIssue(entry.issueId, entry.segment));
            cell.addEventListener("focus", () => setActiveIssue(entry.issueId, entry.segment));
            cell.addEventListener("click", (event) => {
              event.preventDefault();
              setActiveIssue(entry.issueId, entry.segment);
              if (event.ctrlKey || event.metaKey) {
                setIssueSelection(entry.issueId, false);
              } else {
                setIssueSelection(entry.issueId, true);
              }
            });
          } else {
            cell.classList.add("normal");
            cell.title = `normal | bin ${bin + 1}/${totalBins}`;
          }
        }

        grid.appendChild(cell);
      }
    }

    matrixWrap.appendChild(grid);

    if (issues.length === 0) {
      const emptyTag = document.createElement("span");
      emptyTag.className = "thumb-empty";
      emptyTag.textContent = "无异常";
      matrixWrap.appendChild(emptyTag);
    }

    card.appendChild(matrixWrap);
    fragment.appendChild(card);
  }

  issueMapList.appendChild(fragment);
  updateColumnSelectionIndicators();
  refreshIssueSegmentStates();
}

function indexScanIssues(scanResult) {
  state.issueById = new Map();
  state.issuesByColumn = new Map();

  const issues = asArray(scanResult?.issues);
  for (const rawIssue of issues) {
    const issue = asObject(rawIssue);
    const issueId = String(issue?.issue_id ?? "").trim();
    const column = String(issue?.column ?? "").trim();
    if (!issueId || !column) continue;

    state.issueById.set(issueId, issue);
    if (!state.issuesByColumn.has(column)) {
      state.issuesByColumn.set(column, []);
    }
    state.issuesByColumn.get(column).push(issue);
  }
}

function extractSuggestion(task) {
  const details = asObject(task?.response?.error?.details);
  const keys = ["suggestion", "hint", "next_step", "reason"];
  for (const key of keys) {
    const value = String(details?.[key] ?? "").trim();
    if (value) return value;
  }
  return "";
}

function normalizeReadableErrorText(raw) {
  let text = String(raw ?? "").trim();
  if (!text) {
    return "unknown error";
  }
  text = text.replace(/\s+/g, " ");
  text = text.replace(/[;,]?\s*raw=.*$/i, "");
  if (text.length > 420) {
    text = `${text.slice(0, 420)}...`;
  }
  return text;
}

function toReadableError(task, err) {
  if (err) return normalizeReadableErrorText(`request failed: ${String(err)}`);
  if (!task) return "task returned empty result";
  if (task?.response?.error?.message) {
    const code = String(task?.response?.error?.code || "UNKNOWN");
    return normalizeReadableErrorText(`Engine error [${code}] ${String(task.response.error.message)}`);
  }
  if (task?.error) return normalizeReadableErrorText(String(task.error));
  return `task ended with status: ${String(task?.status || "unknown")}`;
}

function buildIssueTypeSummary(issues) {
  const counts = new Map();
  for (const issue of asArray(issues)) {
    const key = String(issue?.issue_type || "unknown");
    counts.set(key, toInt(counts.get(key), 0) + 1);
  }
  const lines = [];
  for (const [issueType, count] of counts.entries()) {
    lines.push(`${issueTypeLabel(issueType)} ${count} 项`);
  }
  return lines;
}

function renderScanOverview(overview) {
  const issueCount = toInt(overview?.issueCount, 0);
  const targetColumn = String(overview?.targetColumn || "");
  const totalIssueCount = toInt(overview?.totalIssueCount, issueCount);
  const issueTypeLines = asArray(overview?.issueTypeLines);
  const highRiskColumns = formatList(overview?.highRiskColumns, "");
  const mediumRiskColumns = formatList(overview?.mediumRiskColumns, "");

  if (detectionMessage) {
    if (issueCount > 0) {
      detectionMessage.textContent = state.autoMode
        ? `全自动模式已完成：检测到 ${issueCount} 个问题（全局 ${totalIssueCount} 个），可直接一键自动修复全部问题列。`
        : targetColumn
        ? `目标列 ${targetColumn} 检测到 ${issueCount} 个问题（全局 ${totalIssueCount} 个），建议先勾选重点问题再修复。`
        : `检测到 ${issueCount} 个问题，请先浏览高风险列并勾选要修复的异常。`;
    } else {
      detectionMessage.textContent = targetColumn
        ? `目标列 ${targetColumn} 未检测到异常，暂不需要修复。`
        : "当前文件未检测到异常，暂不需要修复。";
    }
  }

  const repairableLines = [];
  if (issueTypeLines.length > 0) {
    repairableLines.push(...issueTypeLines);
  }
  if (highRiskColumns) {
    repairableLines.push(`高风险列：${highRiskColumns}`);
  }
  if (mediumRiskColumns) {
    repairableLines.push(`中风险列：${mediumRiskColumns}`);
  }
  if (!repairableLines.length) {
    repairableLines.push("未发现可修复问题。");
  }
  renderCompactList(repairableOverview, repairableLines, "未发现可修复问题。");

  if (nextActionText) {
    if (issueCount > 0) {
      nextActionText.textContent = state.autoMode
        ? "可直接点击“自动修复全部问题列”，或在矩阵中按需手动勾选后执行修复。"
        : "在下方矩阵中点击红色异常点进行勾选，然后执行批量修复。";
    } else {
      nextActionText.textContent = "可直接结束本次流程，或更换参数重新检测。";
    }
  }
}

function renderScanFailureOverview(reason, suggestion) {
  if (detectionMessage) {
    detectionMessage.textContent = normalizeReadableErrorText(reason || "检测失败：未知错误");
  }
  renderCompactList(repairableOverview, ["无法生成可修复项。", suggestion ? `建议：${suggestion}` : "建议检查输入参数与文件路径。"]);
  if (nextActionText) {
    nextActionText.textContent = "请返回上一步修正参数后重试。";
  }
}

function updateMapScaleLegend(scanResult) {
  if (!mapScaleNote) return;
  const rows = toInt(scanResult?.data_profile?.rows, 0);
  const thumbnails = asArray(scanResult?.column_thumbnails);
  const bins = thumbnails.length > 0 ? toInt(thumbnails[0]?.total_bins, 0) : 0;
  if (rows > 0 && bins > 0) {
    const approxRowsPerCell = Math.max(1, Math.ceil(rows / bins));
    mapScaleNote.textContent = `刻度说明：横轴为分桶列 (1-${bins})，纵轴标签为行范围；单格约代表 ${approxRowsPerCell} 行。`;
    return;
  }
  mapScaleNote.textContent = "刻度说明：横轴为分桶列，纵轴标签为行范围。";
}

function renderScanResult(result) {
  const scanResult = asObject(result);
  const scanView = buildScanView(scanResult);
  state.scanResult = scanResult;
  state.scanViewResult = scanView;
  state.selectedIssueIds.clear();
  state.activeIssueId = "";
  state.activeIssueSegment = null;

  indexScanIssues(scanView);

  const summary = asObject(scanView?.scan_summary);
  const profile = asObject(scanResult?.data_profile);
  const targetColumn = getTargetColumn();
  const issueCount = toInt(scanView?.issue_count, asArray(scanView?.issues).length);
  const totalIssueCount = toInt(scanResult?.issue_count, asArray(scanResult?.issues).length);

  const rows = [
    ["输入文件", scanResult?.csv_path ?? "-"],
    ["检测范围", state.autoMode ? "全自动（全部列）" : targetColumn || "全部列"],
    ["数据规模", `${profile?.rows ?? "-"} 行 × ${profile?.columns ?? "-"} 列`],
    ["发现问题", issueCount],
  ];
  if (targetColumn) {
    rows.push(["全局问题数", totalIssueCount]);
  }
  rows.push(["异常列数", summary?.anomaly_column_count ?? asArray(scanView?.anomaly_columns).length]);
  const highRiskColumns = formatList(summary?.high_risk_columns, "");
  if (highRiskColumns) rows.push(["高风险列", highRiskColumns]);
  const mediumRiskColumns = formatList(summary?.medium_risk_columns, "");
  if (mediumRiskColumns) rows.push(["中风险列", mediumRiskColumns]);

  renderDescriptionList(detectionSummary, rows);
  renderScanOverview({
    issueCount,
    targetColumn,
    totalIssueCount,
    issueTypeLines: buildIssueTypeSummary(scanView?.issues),
    highRiskColumns: summary?.high_risk_columns,
    mediumRiskColumns: summary?.medium_risk_columns,
  });

  if (issueCount > 0) {
    setScanRepairVisibility(true);
    clearIssueDetail();
  } else {
    setScanRepairVisibility(false);
    clearIssueDetail("当前文件未检测到异常，无需修复。");
  }

  renderScanIssueMap(scanView);
  updateSelectedIssuePill();
}

function renderScanFailure(reason, task = null, phase = "检测") {
  state.scanResult = null;
  state.scanViewResult = null;
  state.issueById = new Map();
  state.issuesByColumn = new Map();
  state.selectedIssueIds.clear();

  const taskId = String(task?.id || state.currentTaskId || "-");
  const status = String(task?.status || "failed").toLowerCase();
  const errorCode = String(task?.response?.error?.code || "-");
  const suggestion = extractSuggestion(task);

  renderDescriptionList(detectionSummary, [
    ["阶段", phase],
    ["任务ID", taskId],
    ["状态", status],
    ["错误码", errorCode],
    ["可修复", "否"],
  ]);

  renderScanFailureOverview(`${phase}失败：${normalizeReadableErrorText(String(reason || "未知错误"))}`, suggestion);

  if (issueMapList) {
    issueMapList.innerHTML = '<div class="scan-placeholder">未生成异常缩略图。请修正参数后重试。</div>';
  }
  updateMapScaleLegend({});
  clearIssueDetail("本次任务失败，暂无可查看的问题详情。");
  setScanRepairVisibility(false);
  updateSelectedIssuePill();
}

function describeIssueImpact(issue) {
  const count = toInt(issue?.count, 0);
  const ratio = Number(issue?.ratio);
  const issueType = issueTypeLabel(issue?.issue_type);
  const ratioText = Number.isFinite(ratio) ? formatPercent(ratio) : "-";
  if (count > 0) {
    return `检测阶段识别到 ${count} 条 ${issueType}（占比 ${ratioText}）。`;
  }
  return `检测阶段识别到 ${issueType}。`;
}

function renderRepairResult(result) {
  const repairResult = asObject(result);
  const comparison = asObject(repairResult?.comparison);
  const beforeIssueCount = toInt(comparison?.before_issue_count, toInt(repairResult?.scan_issue_count, 0));
  const afterIssueCount = toInt(comparison?.after_issue_count, Math.max(0, beforeIssueCount - toInt(repairResult?.applied_issue_count, 0)));
  const resolvedIssueCount = toInt(comparison?.resolved_issue_count, Math.max(0, beforeIssueCount - afterIssueCount));
  const appliedCount = toInt(repairResult?.applied_issue_count, 0);
  const skippedCount = asArray(repairResult?.skipped_issues).length;
  const modifiedCells = toInt(repairResult?.total_cells_modified, 0);
  const rollback = asObject(repairResult?.rollback);
  setRepairKpis(appliedCount, skippedCount, modifiedCells);
  renderRepairOverviewSidebar(repairResult);
  renderDescriptionList(repairSummary, [
    ["修复模式", state.lastRepairMode === "auto" ? "自动修复全部问题列" : "手动选择修复"],
    ["执行模式", String(repairResult?.execution_mode || "apply")],
    ["已选问题", repairResult?.selected_issue_count ?? 0],
    ["已修复问题", repairResult?.applied_issue_count ?? 0],
    ["跳过问题", skippedCount],
    ["修改单元格", repairResult?.total_cells_modified ?? 0],
    ["扫描问题总数", repairResult?.scan_issue_count ?? 0],
    ["修复前后", `${beforeIssueCount} -> ${afterIssueCount}（已减少 ${resolvedIssueCount}）`],
    ["输出文件", repairResult?.output_csv ?? "-"],
    ["已写出文件", String(repairResult?.write_output ?? false)],
    ["回滚清单", rollback?.manifest_path ? shortPath(rollback.manifest_path) : "-"],
  ]);

  if (!repairDetailList) return;
  repairDetailList.innerHTML = "";

  for (const item of asArray(repairResult?.applied_repairs)) {
    const issueID = String(item?.issue_id || "").trim();
    const issueType = issueTypeLabel(item?.issue_type);
    const column = String(item?.column || "-");
    const rowsTouched = toInt(item?.rows_touched, 0);
    const preview = item?.replacement_preview;
    const scanIssue = state.issueById.get(issueID) || null;

    const li = document.createElement("li");
    li.className = "batch-item-ok";
    const beforeCount = toInt(item?.before_count, toInt(scanIssue?.count, 0));
    const afterCount = toInt(item?.after_count, Math.max(0, beforeCount - rowsTouched));
    const resolvedCount = toInt(item?.resolved_count, Math.max(0, beforeCount - afterCount));
    const cellsPreview = asArray(item?.cells_preview)
      .slice(0, 2)
      .map((row) => `row ${row?.row}: ${row?.before} -> ${row?.after}`)
      .join(" | ");
    li.innerHTML = `
      <p class="batch-item-title">已修复 ${escapeHtml(column)} · ${escapeHtml(issueType)} · ${rowsTouched} 行</p>
      <p class="batch-item-body">原因：${escapeHtml(scanIssue ? describeIssueImpact(scanIssue) : issueTypeReasonText(item?.issue_type))}</p>
      <p class="batch-item-body">策略：${escapeHtml(issueTypeFixText(item?.issue_type, preview))}</p>
      <p class="batch-item-body">对比：${escapeHtml(`问题命中 ${beforeCount} -> ${afterCount}，已解决 ${resolvedCount}`)}</p>
      ${cellsPreview ? `<p class="batch-item-meta">样例: ${escapeHtml(cellsPreview)}</p>` : ""}
      <p class="batch-item-meta">问题ID: ${escapeHtml(issueID || "-")}</p>
    `;
    repairDetailList.appendChild(li);
  }
  for (const item of asArray(repairResult?.skipped_issues)) {
    const issueID = String(item?.issue_id || "").trim();
    const reason = skipReasonText(item?.reason);
    const issueType = issueTypeLabel(item?.issue_type);
    const li = document.createElement("li");
    li.className = "batch-item-skip";
    li.innerHTML = `
      <p class="batch-item-title">已跳过 ${escapeHtml(issueID || "-")}</p>
      <p class="batch-item-body">原因：${escapeHtml(reason)}</p>
      <p class="batch-item-meta">类型: ${escapeHtml(issueType)}</p>
    `;
    repairDetailList.appendChild(li);
  }
  if (!repairDetailList.childNodes.length) {
    const li = document.createElement("li");
    li.className = "batch-item-skip";
    li.textContent = "没有可展示的修复明细。";
    repairDetailList.appendChild(li);
  }
}

function renderRepairFailure(reason, task = null) {
  resetRepairKpis();
  resetRepairOverviewSidebar();
  renderDescriptionList(repairSummary, [
    ["阶段", "repair_batch"],
    ["任务ID", String(task?.id || state.currentTaskId || "-")],
    ["状态", String(task?.status || "failed")],
    ["错误码", String(task?.response?.error?.code || "-")],
  ]);

  if (!repairDetailList) return;
  repairDetailList.innerHTML = "";
  const li = document.createElement("li");
  li.className = "batch-item-skip";
  const suggestion = extractSuggestion(task);
  li.textContent = suggestion ? `修复失败：${reason}。建议：${suggestion}` : `修复失败：${String(reason || "未知错误")}`;
  repairDetailList.appendChild(li);
}

function collectScanPayload() {
  const payload = {
    action: "scan_file",
    csv_path: String(csvPathInput?.value || "").trim(),
    max_bins: toInt(maxBinsInput?.value, 120),
    max_issues: toInt(maxIssuesInput?.value, 1200),
    timeout_ms: toInt(timeoutInput?.value, 90000),
    auto_mode: state.autoMode,
  };
  const targetCol = getTargetColumn();
  if (targetCol) {
    payload.target_col = targetCol;
  }
  return payload;
}

function validateScanPayload(payload) {
  if (!String(payload?.csv_path || "").trim()) return "CSV 文件路径不能为空。";
  if (!Number.isInteger(payload.max_bins) || payload.max_bins < 20 || payload.max_bins > 360) {
    return "缩略图分桶必须在 20-360 之间。";
  }
  if (!Number.isInteger(payload.max_issues) || payload.max_issues < 10 || payload.max_issues > 5000) {
    return "最多问题数必须在 10-5000 之间。";
  }
  if (!Number.isInteger(payload.timeout_ms) || payload.timeout_ms < 1000) {
    return "超时必须是 >= 1000 的整数(ms)。";
  }
  return "";
}

function collectBatchRepairPayload() {
  const csvPath = String(state.scanResult?.csv_path || csvPathInput?.value || "").trim();
  const outputDir = String(outputInput?.value || "").trim();
  const payload = {
    action: "repair_batch",
    csv_path: csvPath,
    issue_ids: Array.from(state.selectedIssueIds),
    timeout_ms: toInt(timeoutInput?.value, 90000),
    write_output: true,
  };
  if (outputDir) payload.output_dir = outputDir;
  const scanConfig = asObject(state.scanResult?.scan_config);
  if (Object.keys(scanConfig).length > 0) payload.scan_config = scanConfig;
  return payload;
}

function validateBatchRepairPayload(payload) {
  if (!String(payload?.csv_path || "").trim()) return "CSV 文件路径不能为空。";
  if (!Array.isArray(payload?.issue_ids) || payload.issue_ids.length === 0) {
    return "请先在第3步勾选至少一个异常后再修复。";
  }
  if (!Number.isInteger(payload.timeout_ms) || payload.timeout_ms < 1000) {
    return "超时必须是 >= 1000 的整数(ms)。";
  }
  return "";
}
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function apiRunTask(payload) {
  if (hasBinding("RunTask")) return window.go.main.App.RunTask(payload);
  return mockRunTask(payload);
}

async function apiGetTaskStatus(taskId) {
  if (hasBinding("GetTaskStatus")) return window.go.main.App.GetTaskStatus(taskId);
  return mockGetTaskStatus(taskId);
}

async function apiCancelTask(taskId) {
  if (hasBinding("CancelTask")) return window.go.main.App.CancelTask(taskId);
  return mockCancelTask(taskId);
}

async function apiSelectCsv() {
  if (hasBinding("SelectCSV")) return window.go.main.App.SelectCSV();
  return "";
}

async function apiSelectOutputDir() {
  if (hasBinding("SelectOutputDir")) return window.go.main.App.SelectOutputDir();
  return "";
}

async function apiListTaskHistory(limit = 20) {
  if (hasBinding("ListTaskHistory")) return window.go.main.App.ListTaskHistory(limit);
  return [];
}

async function apiListCsvColumns(csvPath) {
  if (hasBinding("ListCSVColumns")) {
    return window.go.main.App.ListCSVColumns(String(csvPath || "").trim());
  }
  return mockListCsvColumns();
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

async function mockRunTask(payload) {
  const id = `mock-task-${Date.now()}`;
  state.mockTasks.set(id, {
    id,
    payload: { ...payload },
    action: String(payload?.action || "scan_file"),
    createdAt: Date.now(),
    canceled: false,
  });
  return {
    id,
    status: "pending",
    request: { action: String(payload?.action || "scan_file"), payload: { ...payload } },
    response: {},
    error: "",
  };
}

function buildMockScanResult(payload) {
  const totalBins = clamp(toInt(payload?.max_bins, 120), 20, 220);
  const totalRows = 1200;

  const issues = [
    {
      issue_id: "bmi::missing_values",
      column: "bmi",
      issue_type: "missing_values",
      severity: "high",
      issue_score: 68.4,
      count: 112,
      ratio: 112 / totalRows,
      risk_level: "high",
      segments: [
        { bin_start: 8, bin_end: 13, start_row: 80, end_row: 139 },
        { bin_start: 38, bin_end: 43, start_row: 380, end_row: 439 },
        { bin_start: 72, bin_end: 78, start_row: 720, end_row: 789 },
      ],
      detail: {
        missing_count: 112,
        preview: [
          { row: 12, value: null },
          { row: 438, value: null },
          { row: 780, value: null },
        ],
      },
    },
    {
      issue_id: "age::numeric_outlier",
      column: "age",
      issue_type: "numeric_outlier",
      severity: "medium",
      issue_score: 36.2,
      count: 38,
      ratio: 38 / totalRows,
      risk_level: "medium",
      segments: [
        { bin_start: 18, bin_end: 21, start_row: 180, end_row: 219 },
        { bin_start: 54, bin_end: 58, start_row: 540, end_row: 589 },
      ],
      detail: {
        outlier_count: 38,
        lower_bound: -3.5,
        upper_bound: 93.5,
        preview: [
          { row: 206, value: 103 },
          { row: 552, value: -8 },
        ],
      },
    },
    {
      issue_id: "smoking_status::rare_category",
      column: "smoking_status",
      issue_type: "rare_category",
      severity: "low",
      issue_score: 18.8,
      count: 21,
      ratio: 21 / totalRows,
      risk_level: "low",
      segments: [{ bin_start: 82, bin_end: 84, start_row: 820, end_row: 849 }],
      detail: {
        rare_count: 21,
        rare_values_preview: ["unknown", "never_worked"],
      },
    },
  ];

  const columnThumbnails = [
    { column: "bmi", dtype: "float64", issue_count: 1, risk_level: "high", risk_score: 68.4, total_bins: totalBins },
    { column: "age", dtype: "float64", issue_count: 1, risk_level: "medium", risk_score: 36.2, total_bins: totalBins },
    {
      column: "smoking_status",
      dtype: "object",
      issue_count: 1,
      risk_level: "low",
      risk_score: 18.8,
      total_bins: totalBins,
    },
    { column: "stroke", dtype: "int64", issue_count: 0, risk_level: "low", risk_score: 0, total_bins: totalBins },
  ];

  return {
    csv_path: String(payload?.csv_path || "data/raw/mock.csv"),
    scan_config: { max_bins: totalBins, max_issues: clamp(toInt(payload?.max_issues, 1200), 10, 5000) },
    data_profile: { rows: totalRows, columns: 4 },
    column_thumbnails: columnThumbnails,
    issues,
    issue_count: issues.length,
    anomaly_columns: ["bmi", "age", "smoking_status"],
    scan_summary: {
      anomaly_column_count: 3,
      high_risk_columns: ["bmi"],
      medium_risk_columns: ["age"],
      total_issues: issues.length,
    },
  };
}

async function mockGetTaskStatus(taskId) {
  const mock = state.mockTasks.get(taskId);
  if (!mock) throw new Error(`mock task not found: ${taskId}`);

  const elapsed = Date.now() - mock.createdAt;
  if (mock.canceled) {
    return {
      id: taskId,
      status: "canceled",
      request: { action: mock.action, payload: mock.payload },
      response: {},
      error: "canceled",
    };
  }

  if (elapsed < 700) {
    return {
      id: taskId,
      status: "pending",
      request: { action: mock.action, payload: mock.payload },
      response: {},
      error: "",
    };
  }

  if (elapsed < 2600) {
    return {
      id: taskId,
      status: "running",
      request: { action: mock.action, payload: mock.payload },
      response: {},
      error: "",
    };
  }

  if (mock.action === "scan_file") {
    return {
      id: taskId,
      status: "succeeded",
      request: { action: mock.action, payload: mock.payload },
      response: {
        task_id: taskId,
        status: "ok",
        result: buildMockScanResult(mock.payload),
        error: null,
        timestamp: new Date().toISOString(),
        duration_ms: 2500,
      },
      error: "",
    };
  }

  const selected = asArray(mock.payload?.issue_ids).map((item) => String(item || "").trim()).filter(Boolean);
  if (selected.length === 0) {
    return {
      id: taskId,
      status: "failed",
      request: { action: mock.action, payload: mock.payload },
      response: {
        task_id: taskId,
        status: "error",
        result: {},
        error: {
          code: "INVALID_INPUT",
          message: "issue_ids is empty",
          details: { suggestion: "Select at least one issue before repair." },
        },
      },
      error: "INVALID_INPUT: issue_ids is empty",
    };
  }

  const appliedRepairs = selected.map((issueId) => {
    const pieces = issueId.split("::");
    return {
      issue_id: issueId,
      column: pieces[0] || "unknown",
      issue_type: pieces[1] || "unknown",
      rows_touched: (issueId.length % 25) + 4,
      replacement_preview: "mock_fill",
    };
  });

  return {
    id: taskId,
    status: "succeeded",
    request: { action: mock.action, payload: mock.payload },
    response: {
      task_id: taskId,
      status: "ok",
      result: {
        csv_path: String(mock.payload?.csv_path || ""),
        output_csv: `${String(mock.payload?.output_dir || "outputs/results/mock_repair")}/mock.repaired.csv`,
        selected_issue_count: selected.length,
        applied_issue_count: appliedRepairs.length,
        total_cells_modified: appliedRepairs.reduce((sum, item) => sum + toInt(item?.rows_touched, 0), 0),
        selected_issue_ids: selected,
        applied_repairs: appliedRepairs,
        skipped_issues: [],
        scan_issue_count: selected.length,
      },
      error: null,
      timestamp: new Date().toISOString(),
      duration_ms: 2400,
    },
    error: "",
  };
}

async function mockCancelTask(taskId) {
  const mock = state.mockTasks.get(taskId);
  if (!mock) return false;
  mock.canceled = true;
  return true;
}

function actionToIntent(action) {
  const normalized = String(action || "").toLowerCase();
  if (normalized === "scan_file") return INTENT_SCAN;
  if (normalized === "repair_batch") return INTENT_REPAIR;
  if (normalized === "train") return INTENT_TRAIN;
  return "";
}

function intentFromTask(task) {
  const actionIntent = actionToIntent(task?.request?.action);
  if (actionIntent) return actionIntent;
  if (state.runningIntent) return state.runningIntent;
  return "";
}

function runningHintForIntent(intent, elapsedMS) {
  const hints = RUNNING_HINTS[intent] || RUNNING_HINTS[INTENT_SCAN];
  if (elapsedMS < 1200) return hints[0];
  if (elapsedMS < 3200) return hints[1];
  if (elapsedMS < 5200) return hints[2];
  return hints[3];
}

function completedHintForIntent(intent) {
  if (intent === INTENT_REPAIR) return "批量修复完成，正在整理输出结果。";
  if (intent === INTENT_TRAIN) return "训练完成，正在整理模型结果。";
  return "检测完成，正在整理异常摘要。";
}

function renderPhaseHints(intent, status) {
  if (!phaseHints) return;
  const normalizedStatus = String(status || "idle").toLowerCase();
  const hints = RUNNING_HINTS[intent] || RUNNING_HINTS[INTENT_SCAN];

  if (normalizedStatus === "idle" || hints.length === 0) {
    phaseHints.innerHTML = "<li>等待任务开始</li>";
    return;
  }

  let activeIndex = 0;
  if (normalizedStatus === "pending") {
    activeIndex = 0;
  } else if (normalizedStatus === "running") {
    const elapsedMS = Math.max(0, Date.now() - toInt(state.taskStartAtMS, Date.now()));
    if (elapsedMS < 1200) activeIndex = 0;
    else if (elapsedMS < 3200) activeIndex = 1;
    else if (elapsedMS < 5200) activeIndex = 2;
    else activeIndex = 3;
  } else {
    activeIndex = hints.length - 1;
  }

  const rows = [];
  for (let i = 0; i < hints.length; i += 1) {
    const cls = i < activeIndex ? "done" : i === activeIndex ? "active" : "";
    rows.push(`<li class="${cls}">${escapeHtml(hints[i])}</li>`);
  }
  phaseHints.innerHTML = rows.join("");
}

function buildTaskMessage(task, intent) {
  const status = String(task?.status || "idle").toLowerCase();
  const elapsedMS = Math.max(0, Date.now() - toInt(state.taskStartAtMS, Date.now()));

  if (status === "pending") return "任务已提交，等待执行槽位...";
  if (status === "running") return runningHintForIntent(intent, elapsedMS);
  if (status === "succeeded") return completedHintForIntent(intent);
  if (status === "canceled") return "任务已取消。";
  if (status === "timed_out") return "任务已超时，请适当增大超时参数。";
  if (status === "failed") {
    const readable = toReadableError(task);
    return `任务失败：${readable}`;
  }

  if (task?.error) {
    return normalizeReadableErrorText(String(task.error));
  }
  return `任务状态: ${status}`;
}

function renderTask(task, intent = "") {
  state.currentTask = task;
  setTaskId(task?.id || "");

  const status = String(task?.status || "idle").toLowerCase();
  const taskIntent = intent || intentFromTask(task);
  const message = buildTaskMessage(task, taskIntent);
  setStatus(status, message);
  renderPhaseHints(taskIntent, status);
  renderProgressSidebar(task, taskIntent);

  if (resultBox) resultBox.textContent = JSON.stringify(task || {}, null, 2);
  syncExportButtons(Boolean(task) && TERMINAL_STATUSES.has(status));
}

function handleTerminalTask(task, intent, options = {}) {
  const status = String(task?.status || "").toLowerCase();
  const taskId = String(task?.id || "");
  const fromHistory = Boolean(options?.fromHistory);
  state.runningIntent = "";
  state.taskStartAtMS = 0;
  state.lastRunningHint = "";

  if (intent === INTENT_SCAN) {
    if (status === "succeeded") {
      clearError();
      renderScanResult(task?.response?.result);
      setWizardStep(STEP_RESULT, { immediate: fromHistory });
      if (!fromHistory) addEvent("检测任务完成。", taskId);
    } else {
      const readable = toReadableError(task);
      showError(readable, "请调整参数后重试。");
      renderScanFailure(readable, task, "检测");
      setWizardStep(STEP_RESULT, { immediate: fromHistory });
      if (!fromHistory) addEvent(`检测任务结束(${status}): ${readable}`, taskId);
    }
    return;
  }

  if (intent === INTENT_REPAIR) {
    if (status === "succeeded") {
      clearError();
      renderRepairResult(task?.response?.result);
      setWizardStep(STEP_REPAIR, { immediate: fromHistory });
      if (!fromHistory) addEvent("批量修复任务完成。", taskId);
    } else {
      const readable = toReadableError(task);
      showError(readable, "请返回第3步调整勾选后重试。");
      renderRepairFailure(readable, task);
      setWizardStep(STEP_REPAIR, { immediate: fromHistory });
      if (!fromHistory) addEvent(`批量修复任务结束(${status}): ${readable}`, taskId);
    }
  }
}

async function pollTask(taskId, intent) {
  const token = Date.now();
  state.pollingToken = token;
  let lastStatus = "";
  let lastHint = "";

  while (state.pollingToken === token && state.currentTaskId === taskId) {
    let snapshot;
    try {
      snapshot = await apiGetTaskStatus(taskId);
    } catch (err) {
      setRunningUi(false);
      state.runningIntent = "";
      state.taskStartAtMS = 0;
      state.lastRunningHint = "";
      const message = `状态轮询失败: ${String(err)}`;
      showError(message, "请检查后端连接后重试。");
      if (intent === INTENT_SCAN) {
        renderScanFailure(message, null, "检测");
        setWizardStep(STEP_RESULT);
      } else {
        renderRepairFailure(message);
        setWizardStep(STEP_REPAIR);
      }
      addEvent(message, taskId);
      return null;
    }

    const resolvedIntent = intent || intentFromTask(snapshot);
    renderTask(snapshot, resolvedIntent);
    const status = String(snapshot?.status || "").toLowerCase();
    if (status !== lastStatus) {
      addEvent(`状态 -> ${status}`, taskId);
      lastStatus = status;
    }
    if (status === "running") {
      const hint = runningHintForIntent(resolvedIntent, Math.max(0, Date.now() - toInt(state.taskStartAtMS, Date.now())));
      if (hint !== lastHint) {
        addEvent(`阶段: ${hint}`, taskId);
        lastHint = hint;
      }
    }

    if (TERMINAL_STATUSES.has(status)) {
      setRunningUi(false);
      handleTerminalTask(snapshot, resolvedIntent);
      return snapshot;
    }

    await delay(450);
  }
  return null;
}

async function startTask(payload, intent) {
  clearError();
  setTaskId("");
  state.runningIntent = intent || "";
  state.taskStartAtMS = Date.now();
  state.lastRunningHint = "";
  setStatus("pending", "任务已提交，等待执行槽位...");
  setWizardStep(STEP_PROGRESS);

  let submitted;
  try {
    submitted = await apiRunTask(payload);
  } catch (err) {
    const message = `任务启动失败: ${String(err)}`;
    setRunningUi(false);
    state.runningIntent = "";
    state.taskStartAtMS = 0;
    state.lastRunningHint = "";
    showError(message, "请检查引擎路径与参数。");
    if (intent === INTENT_SCAN) {
      renderScanFailure(message, null, "检测");
      setWizardStep(STEP_RESULT);
    } else {
      renderRepairFailure(message);
      setWizardStep(STEP_REPAIR);
    }
    addEvent(message);
    return null;
  }

  state.currentTaskId = String(submitted?.id || "");
  setRunningUi(true);
  renderTask(submitted, intent);
  addEvent(`任务已提交(${intent}): ${state.currentTaskId}`, state.currentTaskId);
  return pollTask(state.currentTaskId, intent);
}

async function startScanWorkflow(payloadOverride = null) {
  const payload = payloadOverride ? { ...payloadOverride } : collectScanPayload();
  const invalid = validateScanPayload(payload);
  if (invalid) {
    showError(invalid, "请先修正参数。");
    renderScanFailure(invalid, null, "检测");
    setWizardStep(STEP_RESULT);
    addEvent(`参数校验失败: ${invalid}`);
    return;
  }

  state.lastScanPayload = { ...payload };
  renderConfigSidebar();
  state.currentTaskId = "";
  state.pollingToken = Date.now();
  resetResultPanels();
  setRunningUi(false);
  addEvent("开始执行全列异常检测。");
  await startTask(payload, INTENT_SCAN);
}

async function startBatchRepairWorkflow() {
  const payload = collectBatchRepairPayload();
  const invalid = validateBatchRepairPayload(payload);
  if (invalid) {
    showError(invalid, "请先完成第3步问题勾选。");
    addEvent(`修复参数校验失败: ${invalid}`);
    return;
  }

  state.lastRepairPayload = { ...payload };
  state.lastRepairMode = "manual";
  addEvent(`已选择 ${payload.issue_ids.length} 个问题，开始批量修复。`, state.currentTaskId);
  await startTask(payload, INTENT_REPAIR);
}

async function startAutoRepairWorkflow() {
  if (!state.scanResult) {
    showError("请先完成检测。", "完成第3步检测后再执行自动修复。");
    return;
  }

  const issueIds = getAutoRepairIssueIds();
  if (issueIds.length === 0) {
    showError("当前没有可自动修复的问题。", "若检测无异常则无需修复。");
    return;
  }

  state.selectedIssueIds = new Set(issueIds);
  updateSelectedIssuePill();

  const payload = collectBatchRepairPayload();
  payload.issue_ids = issueIds;
  const invalid = validateBatchRepairPayload(payload);
  if (invalid) {
    showError(invalid, "请检查参数后重试。");
    addEvent(`自动修复参数校验失败: ${invalid}`);
    return;
  }

  state.lastRepairPayload = { ...payload };
  state.lastRepairMode = "auto";
  addEvent(`全自动修复已选择 ${issueIds.length} 个问题，开始修复全部异常列。`, state.currentTaskId);
  await startTask(payload, INTENT_REPAIR);
}
function resetToDetectStep(message = "已返回参数配置，可开始新检测。") {
  state.currentTaskId = "";
  state.currentTask = null;
  state.pollingToken = Date.now();
  state.runningIntent = "";
  state.taskStartAtMS = 0;
  state.lastRunningHint = "";

  setTaskId("");
  setStatus("idle", "等待任务开始");
  clearError();
  setRunningUi(false);
  resetResultPanels();
  setWizardStep(STEP_CONFIG);
  addEvent(message);
}

async function chooseDirectory(targetInput, promptText, eventText) {
  const nativePicker = hasBinding("SelectOutputDir");
  try {
    const selected = await apiSelectOutputDir();
    if (selected) {
      if (targetInput) targetInput.value = selected;
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
    if (targetInput) targetInput.value = manual.trim();
    addEvent(`${eventText}(手动): ${manual.trim()}`);
  }
}

async function refreshColumnsForCsv(csvPath, source = "path change") {
  const path = String(csvPath || "").trim();
  if (!path) {
    setTargetOptions([]);
    renderConfigSidebar();
    return;
  }

  try {
    const before = getTargetColumn();
    const columns = await apiListCsvColumns(path);
    setTargetOptions(columns, before);
    addEvent(`已加载 ${state.availableColumns.length} 个可选目标列（${source}）。`);
    renderConfigSidebar();
  } catch (err) {
    setTargetOptions([]);
    addEvent(`目标列读取失败: ${normalizeReadableErrorText(String(err))}`);
    renderConfigSidebar();
  }
}

async function loadRecentHistory() {
  try {
    const tasks = await apiListTaskHistory(10);
    if (!Array.isArray(tasks) || tasks.length === 0) return;

    const latest = tasks[0];
    const intent = actionToIntent(latest?.request?.action);
    if (!intent) return;

    state.currentTaskId = String(latest?.id || "");
    state.runningIntent = intent;
    const startedAtMs = Date.parse(String(latest?.started_at || latest?.created_at || ""));
    state.taskStartAtMS = Number.isFinite(startedAtMs) ? startedAtMs : Date.now();
    renderTask(latest, intent);

    const status = String(latest?.status || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) {
      setRunningUi(false);
      handleTerminalTask(latest, intent, { fromHistory: true });
    } else if (latest?.id) {
      setRunningUi(true);
      setWizardStep(STEP_PROGRESS, { immediate: true });
      pollTask(latest.id, intent);
    }

    addEvent(`已恢复最近任务: ${latest?.id || "-"}`, latest?.id || "");
  } catch (err) {
    addEvent(`加载历史任务失败: ${String(err)}`);
  }
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
  if (!state.currentTask) return;
  saveFile(
    `${state.currentTask.id || "task"}-result.json`,
    `${JSON.stringify(state.currentTask, null, 2)}\n`,
    "application/json"
  );
}

function exportResultCsv() {
  const task = state.currentTask;
  if (!task) {
    addEvent("没有可导出的任务结果。");
    return;
  }

  const action = String(task?.request?.action || "").toLowerCase();
  const result = asObject(task?.response?.result);
  const rows = [];

  if (action === "scan_file") {
    rows.push(["issue_id", "column", "issue_type", "severity", "count", "ratio", "risk_level", "issue_score"]);
    const issues = asArray(result?.issues);
    if (issues.length === 0) rows.push(["-", "-", "-", "-", "0", "0", "-", "0"]);
    for (const issue of issues) {
      rows.push([
        issue?.issue_id ?? "",
        issue?.column ?? "",
        issue?.issue_type ?? "",
        issue?.severity ?? "",
        issue?.count ?? 0,
        issue?.ratio ?? 0,
        issue?.risk_level ?? "",
        issue?.issue_score ?? 0,
      ]);
    }
  } else {
    rows.push(["issue_id", "column", "issue_type", "rows_touched", "status"]);
    for (const item of asArray(result?.applied_repairs)) {
      rows.push([item?.issue_id ?? "", item?.column ?? "", item?.issue_type ?? "", item?.rows_touched ?? 0, "applied"]);
    }
    for (const item of asArray(result?.skipped_issues)) {
      rows.push([item?.issue_id ?? "", item?.column ?? "", item?.issue_type ?? "", 0, item?.reason ?? "skipped"]);
    }
    if (rows.length === 1) rows.push(["-", "-", "-", 0, "none"]);
  }

  const content = `${rows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`;
  saveFile(`${task?.id || "task"}-result.csv`, content, "text/csv;charset=utf-8");
}

if (detectForm) {
  detectForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await startScanWorkflow();
  });
}

if (retryDetectBtn) {
  retryDetectBtn.addEventListener("click", async () => {
    if (!state.lastScanPayload) return;
    addEvent("使用上次参数重试检测。");
    await startScanWorkflow(state.lastScanPayload);
  });
}

if (cancelBtn) {
  cancelBtn.addEventListener("click", async () => {
    if (!state.currentTaskId) return;
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

if (newDetectBtn) {
  newDetectBtn.addEventListener("click", () => {
    resetToDetectStep();
  });
}

if (newRepairDetectBtn) {
  newRepairDetectBtn.addEventListener("click", () => {
    resetToDetectStep("已返回第1步，可重新检测。");
  });
}

if (runBatchRepairBtn) {
  runBatchRepairBtn.addEventListener("click", async () => {
    await startBatchRepairWorkflow();
  });
}

if (runAutoRepairBtn) {
  runAutoRepairBtn.addEventListener("click", async () => {
    await startAutoRepairWorkflow();
  });
}

if (backResultBtn) {
  backResultBtn.addEventListener("click", () => {
    setWizardStep(STEP_RESULT);
  });
}

if (chooseCsvBtn) {
  chooseCsvBtn.addEventListener("click", async () => {
    const nativePicker = hasBinding("SelectCSV");
    try {
      const selected = await apiSelectCsv();
      if (selected) {
        if (csvPathInput) csvPathInput.value = selected;
        addEvent(`已选择 CSV 文件: ${selected}`);
        await refreshColumnsForCsv(selected, "文件选择");
        return;
      }
      if (nativePicker) {
        addEvent("CSV 选择已取消。");
        return;
      }
    } catch (err) {
      addEvent(`CSV 选择器不可用: ${String(err)}`);
    }

    if (csvFileInput) csvFileInput.click();
  });
}

if (csvFileInput) {
  csvFileInput.addEventListener("change", async () => {
    const file = csvFileInput.files?.[0];
    if (!file) return;
    if (csvPathInput) csvPathInput.value = file.name;
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

if (targetColInput) {
  targetColInput.addEventListener("change", () => {
    if (state.scanResult) {
      renderScanResult(state.scanResult);
      addEvent(`已切换目标列: ${getTargetColumn() || "全部列"}`);
    }
  });
}

if (autoModeInput) {
  autoModeInput.addEventListener("change", () => {
    setAutoMode(autoModeInput.checked);
  });
}

if (matrixDensityInput) {
  matrixDensityInput.addEventListener("input", () => {
    updateMatrixDensityUi();
    renderConfigSidebar();
    if (state.scanResult) {
      renderScanResult(state.scanResult);
    }
  });
}

const configWatchInputs = [csvPathInput, targetColInput, maxBinsInput, maxIssuesInput, timeoutInput, outputInput];
for (const input of configWatchInputs) {
  if (!input) continue;
  input.addEventListener("input", () => {
    renderConfigSidebar();
  });
  input.addEventListener("change", () => {
    renderConfigSidebar();
  });
}

if (chooseOutputBtn) {
  chooseOutputBtn.addEventListener("click", async () => {
    await chooseDirectory(outputInput, "输入输出目录", "已选择输出目录");
  });
}

if (issueDetailCheckbox) {
  issueDetailCheckbox.addEventListener("change", () => {
    const issueId = String(issueDetailCheckbox.dataset.issueId || "").trim();
    if (!issueId) return;
    setIssueSelection(issueId, issueDetailCheckbox.checked);
  });
}

if (advancedToggleBtn) {
  advancedToggleBtn.addEventListener("click", () => {
    setAdvancedMode(!state.advancedMode);
  });
}

if (toggleResponseBtn) {
  toggleResponseBtn.addEventListener("click", () => {
    setResponseExpanded(!state.responseExpanded);
  });
}

if (copyJsonBtn) copyJsonBtn.addEventListener("click", copyResultJson);
if (exportJsonBtn) exportJsonBtn.addEventListener("click", exportResultJson);
if (exportCsvBtn) exportCsvBtn.addEventListener("click", exportResultCsv);
if (exportJsonSideBtn) exportJsonSideBtn.addEventListener("click", exportResultJson);
if (exportCsvSideBtn) exportCsvSideBtn.addEventListener("click", exportResultCsv);

setWizardStep(STEP_CONFIG, { immediate: true });
setAdvancedMode(false);
setAutoMode(false, { silent: true });
setStatus("idle", "等待任务开始");
renderPhaseHints(INTENT_SCAN, "idle");
setTaskId("");
setRunningUi(false);
resetResultPanels();
clearError();
updateMatrixDensityUi();
renderConfigSidebar();
addEvent("前端已就绪，请先执行全列检测。");
refreshColumnsForCsv(csvPathInput?.value || "", "初始加载");
loadRecentHistory();


