function reportFrontendBootError(kind, reason) {
  const message = `${kind}: ${String(reason || "未知前端错误")}`;
  try {
    const title = document.getElementById("startup-title");
    const summary = document.getElementById("startup-summary");
    const pill = document.getElementById("startup-status-pill");
    const counts = document.getElementById("startup-counts");
    const list = document.getElementById("startup-check-list");
    if (title) title.textContent = "前端启动失败";
    if (summary) summary.textContent = message;
    if (pill) {
      pill.className = "startup-status-pill failed";
      pill.textContent = "前端错误";
    }
    if (counts) counts.textContent = "前端模块在启动自检前停止。";
    if (list) {
      list.innerHTML = `<li class="startup-check-item fail"><div><strong>前端运行时</strong><p>${message}</p></div></li>`;
    }
  } catch (_) {
    // Keep the error reporter side-effect-only and non-throwing.
  }
  try {
    if (window?.runtime?.LogError) window.runtime.LogError(`frontend_boot_error:${message}`);
  } catch (_) {
    // Runtime logging is optional while the Wails IPC bridge is still starting.
  }
  try {
    console.error("appshell 前端启动错误", message);
  } catch (_) {
    // Console logging is optional in packaged desktop runs.
  }
}

window.addEventListener("error", (event) => {
  reportFrontendBootError("错误", event?.message || event?.error || "未知错误");
});

window.addEventListener("unhandledrejection", (event) => {
  reportFrontendBootError("未处理的异步错误", event?.reason || "未知异步错误");
});

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

const startupGate = document.getElementById("startup-gate");
const startupTitle = document.getElementById("startup-title");
const startupSummary = document.getElementById("startup-summary");
const startupStatusPill = document.getElementById("startup-status-pill");
const startupCounts = document.getElementById("startup-counts");
const startupCheckList = document.getElementById("startup-check-list");
const startupDetails = document.getElementById("startup-details");
const startupRaw = document.getElementById("startup-raw");
const startupRetryBtn = document.getElementById("startup-retry-btn");
const startupCopyBtn = document.getElementById("startup-copy-btn");
const startupCloseBtn = document.getElementById("startup-close-btn");

const smartHomeView = document.getElementById("smart-home-view");
const smartRunView = document.getElementById("smart-run-view");
const smartResultView = document.getElementById("smart-result-view");
const advancedWorkspaceView = document.getElementById("advanced-workspace-view");
const continueLatestBtn = document.getElementById("continue-latest-btn");
const viewStartupBtn = document.getElementById("view-startup-btn");
const openAdvancedBtn = document.getElementById("open-advanced-btn");
const backSmartBtn = document.getElementById("back-smart-btn");
const smartDropzone = document.getElementById("smart-dropzone");
const smartChooseCsvBtn = document.getElementById("smart-choose-csv-btn");
const smartSelectedCsv = document.getElementById("smart-selected-csv");
const smartOutputDetails = document.getElementById("smart-output-details");
const smartOutputInput = document.getElementById("smart-output-dir");
const smartChooseOutputBtn = document.getElementById("smart-choose-output-btn");
const smartModeBanner = document.getElementById("smart-mode-banner");
const smartSummaryCard = document.getElementById("smart-summary-card");
const smartSummaryMessage = document.getElementById("smart-summary-message");
const smartSummaryList = document.getElementById("smart-summary-list");
const smartSummaryTrust = document.getElementById("smart-summary-trust");
const smartPreferenceWorkspace = document.getElementById("smart-preference-workspace");
const smartPreferenceBanner = document.getElementById("smart-preference-banner");
const smartPrefConservativeInput = document.getElementById("smart-pref-conservative");
const smartPrefAvoidTimeInput = document.getElementById("smart-pref-avoid-time");
const smartPrefRequireApprovalInput = document.getElementById("smart-pref-require-approval");
const smartPrefProtectedColumnsInput = document.getElementById("smart-pref-protected-columns");
const smartPrefSaveBtn = document.getElementById("smart-pref-save-btn");
const smartPrefResetBtn = document.getElementById("smart-pref-reset-btn");
const smartStartBtn = document.getElementById("smart-start-btn");
const smartRunStatusPill = document.getElementById("smart-run-status-pill");
const smartRunTaskId = document.getElementById("smart-run-task-id");
const smartRunTitle = document.getElementById("smart-run-title");
const smartRunMessage = document.getElementById("smart-run-message");
const smartRunProgressFill = document.getElementById("smart-run-progress-fill");
const smartRunStages = document.getElementById("smart-run-stages");
const smartRunMetrics = document.getElementById("smart-run-metrics");
const smartRunEvents = document.getElementById("smart-run-events");
const smartRunCancelBtn = document.getElementById("smart-run-cancel-btn");
const smartOpenAdvancedRunBtn = document.getElementById("smart-open-advanced-run-btn");
const smartSafetyBanner = document.getElementById("smart-safety-banner");
const smartResultPresentation = document.getElementById("smart-result-presentation");
const smartResultConclusion = document.getElementById("smart-result-conclusion");
const smartResultSummary = document.getElementById("smart-result-summary");
const smartApprovalCard = document.getElementById("smart-approval-card");
const smartApprovalMessage = document.getElementById("smart-approval-message");
const smartApprovalSummary = document.getElementById("smart-approval-summary");
const smartApprovalReasons = document.getElementById("smart-approval-reasons");
const smartApprovalPreferences = document.getElementById("smart-approval-preferences");
const smartApprovalContinueBtn = document.getElementById("smart-approval-continue-btn");
const smartApprovalRejectBtn = document.getElementById("smart-approval-reject-btn");
const smartArtifactList = document.getElementById("smart-artifact-list");
const smartNewRunBtn = document.getElementById("smart-new-run-btn");
const smartOpenAdvancedResultBtn = document.getElementById("smart-open-advanced-result-btn");
const smartExportJsonBtn = document.getElementById("smart-export-json-btn");
const smartExportCsvBtn = document.getElementById("smart-export-csv-btn");
const smartReasoningBody = document.getElementById("smart-reasoning-body");
const smartTraceSummary = document.getElementById("smart-trace-summary");
const smartTraceList = document.getElementById("smart-trace-list");
const smartResultRaw = document.getElementById("smart-result-raw");
const smartTrustChecks = document.getElementById("smart-trust-checks");

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
const runKpiProgress = document.getElementById("run-kpi-progress");
const runKpiStageCount = document.getElementById("run-kpi-stage-count");
const runKpiEventCount = document.getElementById("run-kpi-event-count");
const runKpiElapsed = document.getElementById("run-kpi-elapsed");
const progressFill = document.getElementById("progress-fill");
const phaseHints = document.getElementById("phase-hints");
const progressTimeline = document.getElementById("progress-timeline");
const progressMetrics = document.getElementById("progress-metrics");
const progressStageBars = document.getElementById("progress-stage-bars");
const progressFailure = document.getElementById("progress-failure");
const taskIdLabel = document.getElementById("task-id-label");
const eventLog = document.getElementById("event-log");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");
const errorHint = document.getElementById("error-hint");

const detectionSummary = document.getElementById("detection-summary");
const detectionMessage = document.getElementById("detection-message");
const repairableOverview = document.getElementById("repairable-overview");
const nextActionText = document.getElementById("next-action-text");
const resultObservability = document.getElementById("result-observability");
const resultPresentation = document.getElementById("result-presentation");
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
const repairObservability = document.getElementById("repair-observability");
const repairPresentation = document.getElementById("repair-presentation");

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

const VIEW_SMART_HOME = "smart_home";
const VIEW_SMART_RUN = "smart_run";
const VIEW_SMART_RESULT = "smart_result";
const VIEW_ADVANCED_WORKSPACE = "advanced_workspace";

const INTENT_SCAN = "scan";
const INTENT_REPAIR = "repair_batch";
const INTENT_AGENT_AUTO = "agent_auto";
const INTENT_TRAIN = "train";

const STEP_META = {
  [STEP_CONFIG]: {
    kicker: "第 1 步",
    title: "检测参数配置",
    subtitle: "选择 CSV 并设置扫描参数，对全列执行异常检测。",
  },
  [STEP_PROGRESS]: {
    kicker: "第 2 步",
    title: "任务执行中",
    subtitle: "任务正在运行，请稍候。",
  },
  [STEP_RESULT]: {
    kicker: "第 3 步",
    title: "检测结果与问题选择",
    subtitle: "在缩略图中查看异常热区，勾选后可批量修复。",
  },
  [STEP_REPAIR]: {
    kicker: "第 4 步",
    title: "批量修复结果",
    subtitle: "查看已应用修复与跳过项明细。",
  },
};

const WHEEL_SPIN_DEG = 34;
const WHEEL_SPIN_DURATION_MS = 720;
const STARTUP_BINDING_READY_TIMEOUT_MS = 4000;
const STARTUP_NATIVE_CALL_TIMEOUT_MS = 90000;
const STARTUP_NATIVE_CALL_GRACE_MS = 8000;

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
  [INTENT_AGENT_AUTO]: [
    "正在生成智能修复计划...",
    "正在预演候选方案并执行验证门禁...",
    "正在执行自动修复并复扫结果...",
    "正在整理解释、图表与审计轨迹...",
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

function taskStatusLabel(status) {
  const normalized = String(status || "idle").toLowerCase();
  if (normalized === "idle") return "待命";
  if (normalized === "pending") return "排队中";
  if (normalized === "running") return "运行中";
  if (normalized === "succeeded") return "已完成";
  if (normalized === "failed") return "失败";
  if (normalized === "canceled") return "已取消";
  if (normalized === "timed_out") return "已超时";
  return normalized || "未知";
}

const STAGE_LABEL_MAP = {
  validate_input: "参数校验",
  load_csv: "读取数据",
  load_model: "加载模型",
  preprocess: "预处理",
  scan_columns: "扫描列异常",
  train_model: "训练模型",
  evaluate_model: "评估模型",
  apply_repairs: "应用修复",
  write_output: "写出结果",
  complete: "完成",
  unknown: "未知阶段",
};

function defaultSmartPreferenceProfile() {
  return {
    conservative_mode: false,
    avoid_time_columns: true,
    protected_columns: [],
    require_approval_for_high_risk: true,
  };
}

function normalizeSmartPreferenceProfile(raw = {}) {
  const base = defaultSmartPreferenceProfile();
  const protectedColumns = Array.isArray(raw?.protected_columns)
    ? raw.protected_columns
    : String(raw?.protected_columns || "")
        .split(",")
        .map((item) => String(item || "").trim())
        .filter(Boolean);
  return {
    conservative_mode: Boolean(raw?.conservative_mode ?? base.conservative_mode),
    avoid_time_columns: Boolean(raw?.avoid_time_columns ?? base.avoid_time_columns),
    protected_columns: Array.from(new Set(protectedColumns)),
    require_approval_for_high_risk: Boolean(
      raw?.require_approval_for_high_risk ?? base.require_approval_for_high_risk
    ),
  };
}

function cloneSmartPreferenceProfile(raw = {}) {
  return normalizeSmartPreferenceProfile(raw);
}

const state = {
  currentStep: STEP_CONFIG,
  currentShellView: VIEW_SMART_HOME,
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
  seenProgressEventKeys: new Set(),
  advancedMode: false,
  responseExpanded: false,
  startupReport: null,
  startupRunning: false,
  appInitialized: false,
  recentTaskCandidate: null,
  smartDraft: {
    csvPath: "",
    outputDir: "",
  },
  smartPreferences: {
    workspaceID: "",
    draft: defaultSmartPreferenceProfile(),
    saved: defaultSmartPreferenceProfile(),
    loading: false,
    saving: false,
    loaded: false,
    message: "",
    tone: "info",
  },
  smartSessionSnapshot: null,
  smartTraceEvents: [],
  smartDetailsLoading: false,
  smartDetailsError: "",
  mockTasks: new Map(),
  mockAgentSessions: new Map(),
  mockAgentTrace: new Map(),
  mockPreferenceStore: new Map(),
};

function hasBinding(methodName) {
  return typeof window?.go?.main?.App?.[methodName] === "function";
}

function hasAnyAppBinding() {
  return Boolean(window?.go?.main?.App);
}

function hasWailsRuntime() {
  return Boolean(window?.runtime);
}

function isLikelyWailsHost() {
  const protocol = String(window?.location?.protocol || "").toLowerCase();
  const hostname = String(window?.location?.hostname || "").toLowerCase();
  return protocol === "wails:" || hostname.includes("wails");
}

function isPreviewMode() {
  return !hasAnyAppBinding() && !hasWailsRuntime() && !isLikelyWailsHost();
}

function isSmartAutofixAvailable() {
  return hasBinding("RunAgentAutofixSession") || isPreviewMode();
}

function logFrontendEvent(event, detail = {}) {
  const payload = { event, ...asObject(detail) };
  try {
    if (window?.runtime?.LogInfo) {
      window.runtime.LogInfo(`frontend:${JSON.stringify(payload)}`);
    }
  } catch (_) {
    // Ignore runtime logging failures; startup diagnostics should not depend on logging.
  }
  try {
    console.info("appshell", payload);
  } catch (_) {
    // Console logging is optional in packaged desktop runs.
  }
}

function setStartupProgressText(text) {
  if (startupCounts) startupCounts.textContent = String(text || "");
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function resolveSmartWorkspaceID(workspaceID = "", csvPath = "") {
  const explicit = String(workspaceID || "").trim();
  if (explicit) return explicit;
  const normalized = String(csvPath || "").trim().replace(/\\/g, "/");
  if (!normalized) return "";
  const segments = normalized.split("/").filter(Boolean);
  if (segments.length <= 1) return normalized;
  return segments.slice(0, -1).join("/");
}

function smartPreferenceColumnsText(columns) {
  return asArray(columns)
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .join(", ");
}

function setShellView(view) {
  state.currentShellView = view;
  if (smartHomeView) smartHomeView.classList.toggle("hidden", view !== VIEW_SMART_HOME);
  if (smartRunView) smartRunView.classList.toggle("hidden", view !== VIEW_SMART_RUN);
  if (smartResultView) smartResultView.classList.toggle("hidden", view !== VIEW_SMART_RESULT);
  if (advancedWorkspaceView) advancedWorkspaceView.classList.toggle("hidden", view !== VIEW_ADVANCED_WORKSPACE);
  if (openAdvancedBtn) openAdvancedBtn.classList.toggle("hidden", view === VIEW_ADVANCED_WORKSPACE);
  if (backSmartBtn) backSmartBtn.classList.toggle("hidden", view !== VIEW_ADVANCED_WORKSPACE);
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

function stageLabel(raw) {
  const value = String(raw || "").trim();
  if (!value) return STAGE_LABEL_MAP.unknown;
  const key = value.toLowerCase();
  if (STAGE_LABEL_MAP[key]) return STAGE_LABEL_MAP[key];
  if (value.includes("_")) {
    return value
      .split("_")
      .map((part) => STAGE_LABEL_MAP[part.toLowerCase()] || part)
      .join(" ");
  }
  return value;
}

function parseTimeMS(raw) {
  const text = String(raw || "").trim();
  if (!text) return 0;
  const value = Date.parse(text);
  return Number.isFinite(value) ? value : 0;
}

function getTaskProgress(task) {
  const direct = asObject(task?.progress);
  if (Object.keys(direct).length > 0) {
    return direct;
  }
  const obs = asObject(task?.response?.result?.observability);
  if (Object.keys(obs).length === 0) {
    return {};
  }
  return {
    current_stage: String(obs?.current_stage || ""),
    progress_percent: toInt(obs?.progress_percent, 0),
    last_message: String(obs?.last_message || ""),
    stage_durations_ms: asObject(obs?.stage_durations_ms),
    bottleneck_stage: String(obs?.bottleneck_stage || ""),
    bottleneck_ms: toInt(obs?.bottleneck_ms, 0),
    failure: asObject(obs?.failure),
    events: asArray(obs?.events),
  };
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

function padDate(value) {
  return String(value).padStart(2, "0");
}

function buildTimestampSlug(date = new Date()) {
  return `${date.getFullYear()}${padDate(date.getMonth() + 1)}${padDate(date.getDate())}_${padDate(date.getHours())}${padDate(date.getMinutes())}${padDate(date.getSeconds())}`;
}

function buildSmartDefaultOutputDir() {
  return `outputs/results/agent_auto_${buildTimestampSlug()}`;
}

function getSmartCsvPath() {
  const draft = String(state.smartDraft?.csvPath || "").trim();
  if (draft) return draft;
  return String(csvPathInput?.value || "").trim();
}

function getSmartOutputDir() {
  const preferred = String(smartOutputInput?.value || state.smartDraft?.outputDir || "").trim();
  if (preferred) return preferred;
  const generated = buildSmartDefaultOutputDir();
  state.smartDraft.outputDir = generated;
  if (smartOutputInput) smartOutputInput.value = generated;
  if (outputInput && !String(outputInput.value || "").trim()) outputInput.value = generated;
  return generated;
}

function syncSmartDraftToClassicInputs() {
  const csvPath = getSmartCsvPath();
  const outputDir = String(smartOutputInput?.value || state.smartDraft?.outputDir || "").trim();
  if (csvPathInput && csvPath) csvPathInput.value = csvPath;
  if (outputInput && outputDir) outputInput.value = outputDir;
}

function isAgentAutoAction(action) {
  const normalized = String(action || "").trim().toLowerCase();
  return normalized === "agent.session.auto" || normalized === "agent.session.approve";
}

function isTerminalTaskStatus(status) {
  return TERMINAL_STATUSES.has(String(status || "").toLowerCase());
}

function updateContinueLatestButton() {
  if (!continueLatestBtn) return;
  const task = state.recentTaskCandidate;
  if (!task || !task?.id) {
    continueLatestBtn.disabled = true;
    continueLatestBtn.textContent = "查看最近结果";
    return;
  }

  continueLatestBtn.disabled = false;
  const running = !isTerminalTaskStatus(task?.status);
  if (isAgentAutoAction(task?.request?.action) && running) {
    continueLatestBtn.textContent = "继续最近任务";
    return;
  }
  if (isAgentAutoAction(task?.request?.action)) {
    continueLatestBtn.textContent = "查看最近智能结果";
    return;
  }
  continueLatestBtn.textContent = "打开最近工作台任务";
}

function renderSmartModeBanner(message = "", tone = "info") {
  if (!smartModeBanner) return;
  const text = String(message || "").trim();
  smartModeBanner.textContent = text;
  smartModeBanner.className = "safety-banner";
  if (!text) {
    smartModeBanner.classList.add("hidden");
    return;
  }
  smartModeBanner.classList.add(`smart-tone-${tone}`);
  smartModeBanner.classList.remove("hidden");
}

function renderSmartPreferenceBanner(message = "", tone = "info") {
  if (!smartPreferenceBanner) return;
  const text = String(message || "").trim();
  smartPreferenceBanner.textContent = text;
  smartPreferenceBanner.className = "safety-banner";
  if (!text) {
    smartPreferenceBanner.classList.add("hidden");
    return;
  }
  smartPreferenceBanner.classList.add(`smart-tone-${tone}`);
  smartPreferenceBanner.classList.remove("hidden");
}

function syncSmartPreferenceInputsFromState() {
  const draft = cloneSmartPreferenceProfile(state.smartPreferences?.draft);
  if (smartPrefConservativeInput) smartPrefConservativeInput.checked = Boolean(draft.conservative_mode);
  if (smartPrefAvoidTimeInput) smartPrefAvoidTimeInput.checked = Boolean(draft.avoid_time_columns);
  if (smartPrefRequireApprovalInput) smartPrefRequireApprovalInput.checked = Boolean(draft.require_approval_for_high_risk);
  if (smartPrefProtectedColumnsInput) {
    smartPrefProtectedColumnsInput.value = smartPreferenceColumnsText(draft.protected_columns);
  }
}

function readSmartPreferenceDraftFromInputs() {
  return normalizeSmartPreferenceProfile({
    conservative_mode: Boolean(smartPrefConservativeInput?.checked),
    avoid_time_columns: Boolean(smartPrefAvoidTimeInput?.checked),
    require_approval_for_high_risk: Boolean(smartPrefRequireApprovalInput?.checked),
    protected_columns: String(smartPrefProtectedColumnsInput?.value || "")
      .split(",")
      .map((item) => String(item || "").trim())
      .filter(Boolean),
  });
}

function updateSmartPreferenceDraftFromInputs() {
  state.smartPreferences.draft = readSmartPreferenceDraftFromInputs();
  state.smartPreferences.message = "";
  state.smartPreferences.tone = "info";
  renderSmartPreferenceCard();
}

function buildSmartUserPreferencesPayload() {
  return cloneSmartPreferenceProfile(state.smartPreferences?.draft);
}

function renderSmartPreferenceCard() {
  const csvPath = getSmartCsvPath();
  const workspaceID =
    String(state.smartPreferences?.workspaceID || "").trim() ||
    resolveSmartWorkspaceID("", csvPath);
  const loading = Boolean(state.smartPreferences?.loading);
  const saving = Boolean(state.smartPreferences?.saving);
  const loaded = Boolean(state.smartPreferences?.loaded);

  syncSmartPreferenceInputsFromState();
  if (smartPreferenceWorkspace) {
    smartPreferenceWorkspace.textContent = workspaceID
      ? `工作区: ${workspaceID}`
      : "选择 CSV 后会加载当前工作区默认偏好。";
  }

  if (!csvPath) {
    renderSmartPreferenceBanner("当前还没有工作区上下文，先选择 CSV 再加载或保存默认偏好。", "info");
  } else if (loading) {
    renderSmartPreferenceBanner("正在加载当前工作区的默认偏好...", "info");
  } else if (saving) {
    renderSmartPreferenceBanner("正在保存当前偏好为工作区默认设置...", "info");
  } else if (String(state.smartPreferences?.message || "").trim()) {
    renderSmartPreferenceBanner(state.smartPreferences.message, state.smartPreferences.tone || "info");
  } else if (loaded) {
    renderSmartPreferenceBanner("当前草稿会随本次运行发送；只有点击“保存为工作区默认”才会写入持久化 profile。", "info");
  } else {
    renderSmartPreferenceBanner("", "info");
  }

  const disableInputs = !csvPath || loading || saving || state.isRunning;
  if (smartPrefConservativeInput) smartPrefConservativeInput.disabled = disableInputs;
  if (smartPrefAvoidTimeInput) smartPrefAvoidTimeInput.disabled = disableInputs;
  if (smartPrefRequireApprovalInput) smartPrefRequireApprovalInput.disabled = disableInputs;
  if (smartPrefProtectedColumnsInput) smartPrefProtectedColumnsInput.disabled = disableInputs;
  if (smartPrefSaveBtn) smartPrefSaveBtn.disabled = disableInputs;
  if (smartPrefResetBtn) smartPrefResetBtn.disabled = disableInputs || !loaded;
}

async function loadSmartPreferencesForCSV(csvPath, options = {}) {
  const path = String(csvPath || "").trim();
  const workspaceID = resolveSmartWorkspaceID("", path);
  if (!path) {
    state.smartPreferences = {
      workspaceID: "",
      draft: defaultSmartPreferenceProfile(),
      saved: defaultSmartPreferenceProfile(),
      loading: false,
      saving: false,
      loaded: false,
      message: "",
      tone: "info",
    };
    renderSmartPreferenceCard();
    return;
  }
  if (!options?.force && state.smartPreferences.loaded && state.smartPreferences.workspaceID === workspaceID) {
    renderSmartPreferenceCard();
    return;
  }

  state.smartPreferences.loading = true;
  state.smartPreferences.workspaceID = workspaceID;
  state.smartPreferences.message = "";
  state.smartPreferences.tone = "info";
  renderSmartPreferenceCard();
  try {
    const record = await apiGetAgentPreferences("", path);
    const profile = cloneSmartPreferenceProfile(asObject(record?.profile));
    state.smartPreferences = {
      workspaceID: String(record?.workspace_id || workspaceID).trim(),
      draft: cloneSmartPreferenceProfile(profile),
      saved: cloneSmartPreferenceProfile(profile),
      loading: false,
      saving: false,
      loaded: true,
      message: "已加载当前工作区默认偏好。",
      tone: "success",
    };
  } catch (err) {
    state.smartPreferences.loading = false;
    state.smartPreferences.loaded = false;
    state.smartPreferences.message = `加载偏好失败: ${normalizeReadableErrorText(String(err))}`;
    state.smartPreferences.tone = "warning";
  }
  renderSmartPreferenceCard();
}

async function saveSmartPreferencesForWorkspace() {
  const csvPath = getSmartCsvPath();
  if (!csvPath) {
    renderSmartPreferenceBanner("请先选择 CSV，再保存工作区默认偏好。", "warning");
    return null;
  }

  state.smartPreferences.saving = true;
  state.smartPreferences.message = "";
  renderSmartPreferenceCard();
  try {
    const record = await apiSaveAgentPreferences({
      csv_path: csvPath,
      workspace_id: state.smartPreferences.workspaceID || resolveSmartWorkspaceID("", csvPath),
      profile: buildSmartUserPreferencesPayload(),
    });
    const profile = cloneSmartPreferenceProfile(asObject(record?.profile));
    state.smartPreferences = {
      workspaceID: String(record?.workspace_id || resolveSmartWorkspaceID("", csvPath)).trim(),
      draft: cloneSmartPreferenceProfile(profile),
      saved: cloneSmartPreferenceProfile(profile),
      loading: false,
      saving: false,
      loaded: true,
      message: "已保存为当前工作区默认偏好。",
      tone: "success",
    };
    renderSmartPreferenceCard();
    return record;
  } catch (err) {
    state.smartPreferences.saving = false;
    state.smartPreferences.message = `保存偏好失败: ${normalizeReadableErrorText(String(err))}`;
    state.smartPreferences.tone = "warning";
    renderSmartPreferenceCard();
    return null;
  }
}

function renderSmartHome() {
  const csvPath = getSmartCsvPath();
  const outputDir = getSmartOutputDir();
  const fileSelected = Boolean(csvPath);
  const columnCount = state.availableColumns.length;
  const smartAvailable = isSmartAutofixAvailable();

  if (smartSelectedCsv) {
    smartSelectedCsv.textContent = fileSelected ? csvPath : "尚未选择文件";
  }
  if (smartDropzone) {
    smartDropzone.classList.toggle("is-ready", fileSelected);
    smartDropzone.classList.toggle("is-disabled", !smartAvailable);
  }
  if (smartSummaryCard) {
    smartSummaryCard.classList.toggle("hidden", !fileSelected);
  }
  if (smartSummaryMessage) {
    smartSummaryMessage.textContent = fileSelected
      ? "确认以下摘要后即可启动智能闭环。"
      : "选择文件后会自动生成本次任务摘要。";
  }
  if (smartSummaryList) {
    renderDescriptionList(smartSummaryList, [
      ["文件", fileSelected ? shortPath(csvPath) : "-"],
      ["完整路径", fileSelected ? csvPath : "-"],
      ["识别列数", fileSelected ? (columnCount > 0 ? columnCount : "读取中") : "-"],
      ["输出目录", outputDir || "-"],
      ["执行入口", "agent.session.auto"],
      ["安全策略", "自动复扫 + 自动验证 + 失败自动回滚"],
    ]);
  }
  if (smartSummaryTrust) {
    renderCompactList(smartSummaryTrust, [
      "自动复扫",
      "自动验证",
      "失败自动回滚",
      fileSelected && columnCount > 0 ? `已读取 ${columnCount} 个字段` : "等待列信息",
    ]);
  }

  if (!smartAvailable && hasAnyAppBinding()) {
    renderSmartModeBanner("当前环境未暴露 RunAgentAutofixSession，默认智能闭环不可用。可切换到高级工作台继续使用经典流程。", "warning");
  } else if (fileSelected) {
    renderSmartModeBanner("系统将先生成修复摘要，再在你点击“开始智能处理”后进入自动闭环。", "info");
  } else {
    renderSmartModeBanner("", "info");
  }

  if (smartStartBtn) {
    smartStartBtn.disabled = !smartAvailable || !fileSelected || state.isRunning;
  }
  if (smartChooseCsvBtn) smartChooseCsvBtn.disabled = state.isRunning;
  if (smartChooseOutputBtn) smartChooseOutputBtn.disabled = state.isRunning;
  if (smartOutputInput) smartOutputInput.disabled = state.isRunning;
  renderSmartPreferenceCard();
  updateContinueLatestButton();
}

function renderArtifactList(target, items, emptyText) {
  if (!target) return;
  const rows = asArray(items)
    .map((item) => asObject(item))
    .filter((item) => String(item?.path || "").trim());
  if (rows.length === 0) {
    target.innerHTML = `<p class="core-message">${escapeHtml(String(emptyText || "暂无产物。"))}</p>`;
    return;
  }
  target.innerHTML = rows
    .map(
      (item) => `
        <article class="artifact-item">
          <strong>${escapeHtml(String(item?.label || "产物"))}</strong>
          <span>${escapeHtml(String(item?.path || "-"))}</span>
        </article>
      `
    )
    .join("");
}

function smartVerdictTone(verdict) {
  const normalized = String(verdict || "").trim().toLowerCase();
  if (normalized === "accepted") return "success";
  if (normalized === "approval_required" || normalized === "validation_rejected" || normalized === "rolled_back") return "warning";
  if (normalized === "approval_rejected") return "neutral";
  if (normalized === "rollback_failed") return "danger";
  return "neutral";
}

function renderSmartSafetyBanner(verdict, text) {
  if (!smartSafetyBanner) return;
  const normalized = String(verdict || "").trim().toLowerCase();
  const message = String(text || "").trim();
  if (!normalized && !message) {
    smartSafetyBanner.className = "smart-safety-banner hidden";
    smartSafetyBanner.innerHTML = "";
    return;
  }
  smartSafetyBanner.className = `smart-safety-banner ${smartVerdictTone(normalized)}`;
  smartSafetyBanner.innerHTML = `
    <strong>${escapeHtml(normalized || "result")}</strong>
    <p>${escapeHtml(message || "任务已完成。")}</p>
  `;
}

function renderSmartTracePlaceholder(message) {
  if (smartTraceSummary) smartTraceSummary.textContent = String(message || "等待会话轨迹。");
  if (smartTraceList) smartTraceList.innerHTML = "";
}

function syncExportButtons(enabled) {
  const on = Boolean(enabled);
  if (copyJsonBtn) copyJsonBtn.disabled = !on;
  if (exportJsonBtn) exportJsonBtn.disabled = !on;
  if (exportCsvBtn) exportCsvBtn.disabled = !on;
  if (exportJsonSideBtn) exportJsonSideBtn.disabled = !on;
  if (exportCsvSideBtn) exportCsvSideBtn.disabled = !on;
  if (smartExportJsonBtn) smartExportJsonBtn.disabled = !on;
  if (smartExportCsvBtn) smartExportCsvBtn.disabled = !on;
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

function normalizeStageLabel(raw) {
  const value = String(raw || "").trim();
  if (!value) return "未知阶段";
  return stageLabel(value);
}

function fallbackTimelineLabels(intent) {
  return RUNNING_HINTS[intent] || RUNNING_HINTS[INTENT_SCAN];
}

function extractProgressStages(task, intent) {
  const progress = getTaskProgress(task);
  const events = asArray(progress?.events).map((item) => asObject(item));
  const stages = [];
  const seen = new Set();

  for (const event of events) {
    const stage = normalizeStageLabel(event?.stage);
    if (seen.has(stage)) continue;
    seen.add(stage);
    stages.push(stage);
  }

  if (stages.length > 0) return stages;
  return fallbackTimelineLabels(intent);
}

function stageStateForTimeline(stage, stageIndex, currentStage, currentIndex, status) {
  const normalizedStatus = String(status || "idle").toLowerCase();
  if (TERMINAL_STATUSES.has(normalizedStatus)) return "done";
  if (normalizedStatus === "pending" || normalizedStatus === "running") {
    if (currentIndex >= 0) {
      if (stageIndex < currentIndex) return "done";
      if (stageIndex === currentIndex) return "active";
      return "";
    }
    if (stageIndex === 0) return "active";
    return "";
  }
  return stage === currentStage ? "active" : "";
}

function renderProgressTimeline(task, intent) {
  if (!progressTimeline) return;
  const status = String(task?.status || "idle").toLowerCase();
  const progress = getTaskProgress(task);
  const stages = extractProgressStages(task, intent);
  const currentStage = normalizeStageLabel(progress?.current_stage || stages[0] || "");
  const currentIndex = stages.findIndex((stage) => stage === currentStage);

  progressTimeline.innerHTML = stages
    .map((stage, idx) => {
      const cls = stageStateForTimeline(stage, idx, currentStage, currentIndex, status);
      return `<li class="${cls}">${escapeHtml(stage)}</li>`;
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

function formatDurationMS(value) {
  const ms = toInt(value, 0);
  if (ms <= 0) return "-";
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatFailureLocation(failure) {
  const f = asObject(failure);
  const file = String(f?.file || "").trim();
  const column = String(f?.column || "").trim();
  const rule = String(f?.rule || "").trim();
  const parts = [];
  if (file) parts.push(`文件: ${shortPath(file)}`);
  if (column) parts.push(`列: ${column}`);
  if (rule) parts.push(`规则: ${rule}`);
  if (parts.length > 0) return parts.join(" | ");
  return "-";
}

function taskElapsedMS(task) {
  const now = Date.now();
  const fallbackStart = toInt(state.taskStartAtMS, 0);
  const started =
    parseTimeMS(task?.started_at) || parseTimeMS(task?.created_at) || (fallbackStart > 0 ? fallbackStart : now);
  const ended = parseTimeMS(task?.ended_at);
  const end = ended > 0 ? ended : now;
  return Math.max(0, end - started);
}

function formatElapsedTime(ms) {
  const value = Math.max(0, toInt(ms, 0));
  if (value < 1000) return `${value}ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(1)}s`;
  const mins = Math.floor(value / 60_000);
  const secs = Math.floor((value % 60_000) / 1000);
  return `${mins}m ${secs}s`;
}

function formatTaskType(intent) {
  if (intent === INTENT_AGENT_AUTO) return "智能闭环";
  if (intent === INTENT_REPAIR) return "批量修复";
  if (intent === INTENT_TRAIN) return "模型训练";
  return "全列检测";
}

function buildStageDurationEntries(task, intent) {
  const progress = getTaskProgress(task);
  const durations = asObject(progress?.stage_durations_ms);
  const list = [];

  for (const [stageRaw, rawMS] of Object.entries(durations)) {
    const ms = Math.max(0, toInt(rawMS, 0));
    if (ms <= 0) continue;
    list.push({
      stageRaw,
      stage: normalizeStageLabel(stageRaw),
      ms,
    });
  }
  if (list.length > 0) {
    return list;
  }

  const starts = new Map();
  const derived = new Map();
  const events = asArray(progress?.events).map((item) => asObject(item));
  for (const event of events) {
    const stage = normalizeStageLabel(event?.stage);
    const phase = String(event?.phase || "").toLowerCase();
    const atMS = toInt(event?.at_ms, 0);
    if (!stage || atMS <= 0) continue;
    if (phase === "start") {
      starts.set(stage, atMS);
      continue;
    }
    if (phase === "complete" || phase === "done" || phase === "success" || phase === "error") {
      if (!starts.has(stage)) continue;
      const duration = Math.max(0, atMS - toInt(starts.get(stage), atMS));
      starts.delete(stage);
      if (duration <= 0) continue;
      const prev = toInt(derived.get(stage), 0);
      if (duration > prev) {
        derived.set(stage, duration);
      }
    }
  }

  return Array.from(derived.entries()).map(([stage, ms]) => ({
    stageRaw: stage,
    stage,
    ms: toInt(ms, 0),
  }));
}

function renderProgressStageBars(task, intent) {
  if (!progressStageBars) return;
  const progress = getTaskProgress(task);
  const entries = buildStageDurationEntries(task, intent);
  if (entries.length === 0) {
    progressStageBars.innerHTML = '<p class="stage-bar-empty">等待任务运行后生成阶段耗时。</p>';
    return;
  }

  const maxMS = Math.max(1, ...entries.map((entry) => entry.ms));
  const bottleneck = normalizeStageLabel(progress?.bottleneck_stage || "");

  progressStageBars.innerHTML = entries
    .sort((a, b) => b.ms - a.ms)
    .slice(0, 8)
    .map((entry) => {
      const width = Math.max(8, Math.round((entry.ms / maxMS) * 100));
      const isBottleneck = bottleneck && entry.stage === bottleneck;
      return `
        <div class="stage-bar-row ${isBottleneck ? "is-bottleneck" : ""}">
          <div class="stage-bar-head">
            <span>${escapeHtml(entry.stage)}</span>
            <strong>${escapeHtml(formatDurationMS(entry.ms))}</strong>
          </div>
          <div class="stage-bar-track">
            <i class="stage-bar-fill" style="width:${width}%"></i>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderProgressFailure(task) {
  const status = String(task?.status || "idle").toLowerCase();
  const progress = getTaskProgress(task);
  const failure = asObject(progress?.failure);

  let message = String(failure?.message || "").trim();
  if (!message && status === "failed") {
    message = toReadableError(task);
  }
  if (!message && status === "succeeded") {
    message = "无失败信息";
  }
  if (!message && (status === "running" || status === "pending")) {
    message = "任务运行中，暂未检测到失败。";
  }
  if (!message) {
    message = "-";
  }

  renderDescriptionList(progressFailure, [
    ["阶段", String(failure?.stage || progress?.current_stage || "-")],
    ["错误码", String(failure?.error_code || "-")],
    ["文件", failure?.file ? shortPath(failure.file) : "-"],
    ["列", String(failure?.column || "-")],
    ["规则", String(failure?.rule || "-")],
    ["原因", normalizeReadableErrorText(message)],
  ]);
}

function renderRunKpis(task, intent) {
  const status = String(task?.status || "idle").toLowerCase();
  const progress = getTaskProgress(task);
  const fallbackPercent = STATUS_PROGRESS[status] ?? 0;
  const percent = clamp(toInt(progress?.progress_percent, fallbackPercent), 0, 100);
  const stages = extractProgressStages(task, intent);
  const events = asArray(progress?.events);
  const elapsed = taskElapsedMS(task);

  if (runKpiProgress) runKpiProgress.textContent = `${percent}%`;
  if (runKpiStageCount) runKpiStageCount.textContent = String(stages.length);
  if (runKpiEventCount) runKpiEventCount.textContent = String(events.length);
  if (runKpiElapsed) runKpiElapsed.textContent = formatElapsedTime(elapsed);
}

function renderProgressMetrics(task, intent) {
  const request = asObject(task?.request);
  const response = asObject(task?.response);
  const result = asObject(response?.result);
  const profile = asObject(result?.data_profile);
  const progress = getTaskProgress(task);
  const scannedRows = toInt(profile?.rows, 0);
  const issueCount = toInt(result?.issue_count, 0);
  const bottleneckStage = normalizeStageLabel(progress?.bottleneck_stage || "");
  const bottleneckMS = toInt(progress?.bottleneck_ms, 0);

  renderDescriptionList(progressMetrics, [
    ["任务类型", formatTaskType(intent)],
    ["当前阶段", String(progress?.current_stage || "等待开始")],
    ["当前进度", `${clamp(toInt(progress?.progress_percent, 0), 0, 100)}%`],
    ["已扫描行数", scannedRows > 0 ? scannedRows : "计算中"],
    ["发现问题", issueCount > 0 ? issueCount : "计算中"],
    ["瓶颈阶段", bottleneckStage ? `${bottleneckStage} (${formatDurationMS(bottleneckMS)})` : "计算中"],
    ["超时设置", `${toInt(request?.payload?.timeout_ms, toInt(timeoutInput?.value, 90000))} ms`],
    ["预计剩余", formatRemainingTime(task)],
  ]);
}

function renderProgressSidebar(task, intent) {
  renderRunKpis(task, intent);
  renderProgressTimeline(task, intent);
  renderProgressMetrics(task, intent);
  renderProgressStageBars(task, intent);
  renderProgressFailure(task);
}

function renderTaskObservability(target, task) {
  if (!target) return;
  if (!task) {
    renderDescriptionList(target, [
      ["状态", "暂无任务"],
      ["来源", "-"],
      ["阶段", "-"],
      ["进度", "0%"],
      ["总耗时", "-"],
      ["瓶颈", "-"],
      ["定位", "-"],
      ["原因", "-"],
    ]);
    return;
  }
  const progress = getTaskProgress(task);
  const durations = buildStageDurationEntries(task, intentFromTask(task));
  const durationMS =
    toInt(task?.response?.duration_ms, 0) ||
    durations.reduce((sum, item) => sum + Math.max(0, toInt(item?.ms, 0)), 0);
  const source = asArray(progress?.events).length > 0 ? "实时事件流" : "终态摘要";
  const failure = asObject(progress?.failure);
  const failureLocation = formatFailureLocation(failure);
  const failureMessage = String(failure?.message || "").trim();

  renderDescriptionList(target, [
    ["来源", source],
    ["阶段", String(progress?.current_stage || "-")],
    ["进度", `${clamp(toInt(progress?.progress_percent, 0), 0, 100)}%`],
    ["总耗时", formatDurationMS(durationMS)],
    ["瓶颈", progress?.bottleneck_stage ? `${normalizeStageLabel(progress.bottleneck_stage)} (${formatDurationMS(progress?.bottleneck_ms)})` : "-"],
    ["定位", failureLocation],
    ["原因", failureMessage ? normalizeReadableErrorText(failureMessage) : "-"],
  ]);
}

function presentationToneClass(tone) {
  const normalized = String(tone || "neutral")
    .trim()
    .toLowerCase()
    .replaceAll(/\s+/g, "-");
  return normalized || "neutral";
}

function getPresentationBundle(result) {
  const root = asObject(result);
  const nestedAgent = asObject(root?.agent);
  const nestedPresentation = asObject(nestedAgent?.presentation);
  if (Object.keys(nestedPresentation).length > 0) {
    return nestedPresentation;
  }
  return asObject(root?.presentation);
}

function renderPresentationVerdict(verdict) {
  const text = String(verdict || "").trim();
  if (!text) return "";
  return `<span class="presentation-verdict ${presentationToneClass(text)}">${escapeHtml(text)}</span>`;
}

function renderPresentationHighlights(highlights) {
  const cards = asArray(highlights)
    .map((item) => asObject(item))
    .filter((item) => Object.keys(item).length > 0);
  if (cards.length === 0) return "";
  return `
    <section class="metric-strip">
      ${cards
        .map((item) => {
          const tone = presentationToneClass(item?.tone);
          return `
            <article class="metric-card ${tone}">
              <span class="metric-card-label">${escapeHtml(String(item?.label || item?.id || "-"))}</span>
              <strong class="metric-card-value">${escapeHtml(String(item?.value ?? "-"))}</strong>
              ${item?.hint ? `<span class="metric-card-hint">${escapeHtml(String(item.hint))}</span>` : ""}
            </article>
          `;
        })
        .join("")}
    </section>
  `;
}

function renderEvidenceRefs(refs) {
  const items = asArray(refs)
    .map((item) => String(item || "").trim())
    .filter(Boolean);
  if (items.length === 0) return "";
  return `
    <div class="evidence-list">
      ${items.map((item) => `<span class="evidence-tag">${escapeHtml(item)}</span>`).join("")}
    </div>
  `;
}

function renderRankedBarChart(series) {
  const rows = asArray(series).map((item) => asObject(item)).filter((item) => Object.keys(item).length > 0);
  if (rows.length === 0) return "";
  const maxValue = Math.max(1, ...rows.map((item) => Math.max(0, Number(item?.value) || 0)));
  return `
    <div class="presentation-bar-list">
      ${rows
        .map((item) => {
          const value = Math.max(0, Number(item?.value) || 0);
          const width = `${((value / maxValue) * 100).toFixed(2)}%`;
          return `
            <div class="presentation-bar-row">
              <div class="presentation-bar-head">
                <span>${escapeHtml(String(item?.label || "-"))}</span>
                <strong>${escapeHtml(String(item?.value ?? 0))}</strong>
              </div>
              <div class="presentation-bar-track">
                <i class="presentation-bar-fill ${presentationToneClass(item?.tone)}" style="width:${width}"></i>
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderStackedBarChart(series) {
  const rows = asArray(series).map((item) => asObject(item)).filter((item) => Object.keys(item).length > 0);
  if (rows.length === 0) return "";
  const total = Math.max(1, rows.reduce((sum, item) => sum + Math.max(0, Number(item?.value) || 0), 0));
  return `
    <div class="presentation-segment-strip">
      ${rows
        .map((item) => {
          const value = Math.max(0, Number(item?.value) || 0);
          const width = `${Math.max(3, (value / total) * 100).toFixed(2)}%`;
          return `<i class="presentation-segment ${presentationToneClass(item?.tone)}" style="width:${width}"></i>`;
        })
        .join("")}
    </div>
    <div class="presentation-segment-legend">
      ${rows
        .map(
          (item) =>
            `<span class="presentation-segment-pill">${escapeHtml(String(item?.label || "-"))}: ${escapeHtml(String(item?.value ?? 0))}</span>`
        )
        .join("")}
    </div>
  `;
}

function renderComparisonBarChart(series, delta) {
  const rows = asArray(series).map((item) => asObject(item)).filter((item) => Object.keys(item).length > 0);
  if (rows.length === 0) return "";
  const paired = rows.some(
    (item) =>
      Object.prototype.hasOwnProperty.call(item, "before") || Object.prototype.hasOwnProperty.call(item, "after")
  );
  if (!paired) {
    const deltaText = Number.isFinite(Number(delta)) ? `<p>净变化: ${escapeHtml(String(delta))}</p>` : "";
    return `${renderRankedBarChart(rows)}${deltaText}`;
  }
  const maxValue = Math.max(
    1,
    ...rows.map((item) => Math.max(0, Number(item?.before) || 0, Number(item?.after) || 0))
  );
  return `
    <div class="presentation-comparison-list">
      ${rows
        .map((item) => {
          const before = Math.max(0, Number(item?.before) || 0);
          const after = Math.max(0, Number(item?.after) || 0);
          const beforeWidth = `${((before / maxValue) * 100).toFixed(2)}%`;
          const afterWidth = `${((after / maxValue) * 100).toFixed(2)}%`;
          const deltaValue = Number(item?.delta);
          const deltaText = Number.isFinite(deltaValue) ? `变化 ${deltaValue > 0 ? "+" : ""}${deltaValue}` : "";
          return `
            <div class="presentation-comparison-row">
              <div class="presentation-comparison-head">
                <span>${escapeHtml(String(item?.label || "-"))}</span>
                <strong>${deltaText ? escapeHtml(deltaText) : ""}</strong>
              </div>
              <div class="compare-row">
                <span>Before</span>
                <div class="presentation-mini-track"><i class="presentation-mini-fill before" style="width:${beforeWidth}"></i></div>
                <strong>${escapeHtml(String(item?.before ?? 0))}</strong>
              </div>
              <div class="compare-row">
                <span>After</span>
                <div class="presentation-mini-track"><i class="presentation-mini-fill after" style="width:${afterWidth}"></i></div>
                <strong>${escapeHtml(String(item?.after ?? 0))}</strong>
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderTimelineChart(events) {
  const rows = asArray(events).map((item) => asObject(item)).filter((item) => Object.keys(item).length > 0);
  if (rows.length === 0) return "";
  return `
    <div class="presentation-timeline">
      ${rows
        .map((item) => {
          const hint = String(item?.hint || "").trim();
          return `
            <article class="presentation-timeline-item ${presentationToneClass(item?.tone)}">
              <strong>${escapeHtml(String(item?.label || "-"))} · ${escapeHtml(String(item?.value || "-"))}</strong>
              ${hint ? `<p>${escapeHtml(hint)}</p>` : ""}
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderSpotlightCard(data) {
  const payload = asObject(data);
  const entries = Object.entries(payload).filter(([, value]) => String(value ?? "").trim() !== "");
  if (entries.length === 0) return "";
  return `
    <div class="presentation-spotlight">
      <dl>
        ${entries
          .map(([key, value]) => `<dt>${escapeHtml(String(key))}</dt><dd>${escapeHtml(String(value))}</dd>`)
          .join("")}
      </dl>
    </div>
  `;
}

function renderHeatmapHintChart(data) {
  const payload = asObject(data);
  const thumbnails = asArray(payload?.column_thumbnails);
  if (thumbnails.length === 0) return "";
  const topColumns = thumbnails
    .slice()
    .sort((left, right) => (Number(right?.risk_score) || 0) - (Number(left?.risk_score) || 0))
    .slice(0, 3)
    .map((item) => `${String(item?.column || "-")}: ${Math.round(Number(item?.risk_score) || 0)}`);
  return `
    <div class="presentation-insights">
      <div class="safety-banner">异常热力图继续复用下方列缩略图组件，便于和原始 issue 详情联动查看。</div>
      <dl>
        <dt>热区列数</dt>
        <dd>${escapeHtml(String(thumbnails.length))}</dd>
        <dt>重点列</dt>
        <dd>${escapeHtml(topColumns.join(" | ") || "-")}</dd>
      </dl>
    </div>
  `;
}

function renderPresentationChart(chart) {
  const spec = asObject(chart);
  const data = asObject(spec?.data);
  let body = "";
  switch (String(spec?.kind || "").trim()) {
    case "ranked_bar":
      body = renderRankedBarChart(data?.series);
      break;
    case "stacked_bar":
      body = renderStackedBarChart(data?.series);
      break;
    case "comparison_bar":
      body = renderComparisonBarChart(data?.series, data?.delta);
      break;
    case "timeline":
      body = renderTimelineChart(data?.events);
      break;
    case "spotlight_card":
      body = renderSpotlightCard(data);
      break;
    case "heatmap_grid":
      body = renderHeatmapHintChart(data);
      break;
    default:
      body = "";
  }

  if (!body) {
    body = `<div class="presentation-empty">${escapeHtml(String(spec?.empty_state || "No chart data available."))}</div>`;
  }

  return `
    <article class="chart-card" data-chart-kind="${escapeHtml(String(spec?.kind || ""))}">
      <h4>${escapeHtml(String(spec?.title || spec?.id || "Chart"))}</h4>
      ${spec?.subtitle ? `<p>${escapeHtml(String(spec.subtitle))}</p>` : ""}
      ${body}
    </article>
  `;
}

function renderPresentationBundle(target, bundle) {
  if (!target) return;
  const payload = asObject(bundle);
  if (Object.keys(payload).length === 0) {
    target.innerHTML = "";
    target.classList.add("hidden");
    return;
  }

  const sections = asArray(payload?.sections).map((item) => asObject(item)).filter((item) => Object.keys(item).length > 0);
  const charts = asArray(payload?.charts).map((item) => asObject(item)).filter((item) => Object.keys(item).length > 0);
  const headerTitle = String(payload?.headline || payload?.kind || "Presentation");
  const summary = String(payload?.summary || "").trim();
  const riskAndSafety = sections.find((item) => String(item?.id || "").trim() === "risk_and_safety");

  target.classList.remove("hidden");
  target.innerHTML = `
    <div class="presentation-header">
      ${renderPresentationVerdict(payload?.verdict)}
      <h2>${escapeHtml(headerTitle)}</h2>
      ${summary ? `<p>${escapeHtml(summary)}</p>` : ""}
    </div>
    ${renderPresentationHighlights(payload?.highlights)}
    ${riskAndSafety?.body ? `<div class="safety-banner">${escapeHtml(String(riskAndSafety.body))}</div>` : ""}
    ${charts.length > 0 ? `<section class="chart-grid">${charts.map((item) => renderPresentationChart(item)).join("")}</section>` : ""}
    ${
      sections.length > 0
        ? `
      <section class="presentation-insights">
        ${sections
          .map((section) => {
            const bullets = asArray(section?.bullets).map((item) => String(item || "").trim()).filter(Boolean);
            return `
              <article class="insight-section" data-section-id="${escapeHtml(String(section?.id || ""))}">
                <h4>${escapeHtml(String(section?.title || section?.id || "Section"))}</h4>
                ${section?.body ? `<p>${escapeHtml(String(section.body))}</p>` : ""}
                ${bullets.length > 0 ? `<ul>${bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
                ${renderEvidenceRefs(section?.evidence_refs)}
              </article>
            `;
          })
          .join("")}
      </section>
    `
        : ""
    }
  `;
}

function buildRepairView(result) {
  const root = asObject(result);
  const agentBlock = asObject(root?.agent);
  const safety = asObject(root?.safety);
  const presentation = getPresentationBundle(root);
  if (Object.keys(agentBlock).length === 0) {
    return {
      repairResult: root,
      presentation,
      isAgent: false,
      safety,
      validation: {},
      agentBlock: {},
    };
  }

  const execution = asObject(agentBlock?.execution);
  const validation = asObject(agentBlock?.validation);
  const previewValidation = asObject(validation?.preview);
  const comparison = asObject(execution?.comparison || previewValidation || validation);
  const plan = asObject(agentBlock?.plan);
  const explanationBlock = asObject(agentBlock?.explanation);
  const repairResult = {
    ...root,
    ...execution,
  };

  if (Object.keys(comparison).length > 0) {
    repairResult.comparison = comparison;
  }
  if (!repairResult.selected_issue_count) {
    repairResult.selected_issue_count = asArray(plan?.selected_issue_ids).length;
  }
  if (!repairResult.applied_issue_count) {
    repairResult.applied_issue_count = Math.max(0, toInt(comparison?.resolved_issue_count, 0));
  }
  if (!repairResult.total_cells_modified) {
    repairResult.total_cells_modified = Math.max(0, toInt(comparison?.changed_cell_count, 0));
  }
  if (!repairResult.scan_issue_count) {
    repairResult.scan_issue_count = Math.max(0, toInt(comparison?.before_issue_count, 0));
  }
  if (!repairResult.selected_source) {
    repairResult.selected_source = String(plan?.selected_source || "");
  }
  if (Object.keys(asObject(repairResult?.issue_source_map)).length === 0) {
    repairResult.issue_source_map = asObject(plan?.issue_source_map);
  }
  if (asArray(repairResult?.skipped_issues).length === 0) {
    repairResult.skipped_issues = asArray(plan?.skipped_issues);
  }
  if (!repairResult.execution_mode) {
    repairResult.execution_mode = String(agentBlock?.run_mode || "agent");
  }
  if (!Object.keys(asObject(repairResult?.rollback)).length) {
    repairResult.rollback = asObject(execution?.rollback);
  }
  if (!Object.keys(asObject(repairResult?.rollback)).length) {
    repairResult.rollback = asObject(safety?.rollback_execution);
  }
  if (!Object.prototype.hasOwnProperty.call(repairResult, "write_output")) {
    repairResult.write_output = Boolean(execution?.output_csv);
  }
  return {
    repairResult,
    presentation,
    isAgent: true,
    safety,
    validation,
    agentBlock,
  };
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
  if (taskIdLabel) taskIdLabel.textContent = `任务: ${taskId || "-"}`;
}

function setStatus(status, message, progressOverride = null) {
  const normalized = String(status || "idle").toLowerCase();
  const fallback = STATUS_PROGRESS[normalized] ?? 0;
  const customProgress = Number(progressOverride);
  const progress = Number.isFinite(customProgress) ? clamp(Math.trunc(customProgress), 0, 100) : fallback;

  if (statusIcon) {
    const symbol = STATUS_ICON_SYMBOL[normalized] || STATUS_ICON_SYMBOL.idle;
    statusIcon.className = `status-icon ${normalized}`;
    statusIcon.textContent = symbol;
  }
  if (statusPill) {
    statusPill.className = `status-pill ${normalized}`;
    statusPill.textContent = taskStatusLabel(normalized);
  }
  if (statusMessage) statusMessage.textContent = message || taskStatusLabel(normalized);
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
  if (smartRunCancelBtn) smartRunCancelBtn.disabled = !state.isRunning || !state.currentTaskId;
  if (smartChooseCsvBtn) smartChooseCsvBtn.disabled = state.isRunning;
  if (smartChooseOutputBtn) smartChooseOutputBtn.disabled = state.isRunning;
  if (smartOutputInput) smartOutputInput.disabled = state.isRunning;
  if (smartStartBtn) smartStartBtn.disabled = state.isRunning || !getSmartCsvPath() || !isSmartAutofixAvailable();

  updateSelectedIssuePill();
  renderSmartHome();
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
  if (!String(smartOutputInput?.value || "").trim()) {
    smartOutputInput.value = buildSmartDefaultOutputDir();
    state.smartDraft.outputDir = String(smartOutputInput.value || "");
  }
  renderSmartHome();
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

function startupStatusLabel(status) {
  const normalized = String(status || "checking").trim().toLowerCase();
  if (normalized === "ok") return "通过";
  if (normalized === "warning") return "有警告";
  if (normalized === "failed") return "阻塞失败";
  return "检查中";
}

function startupSummaryText(report, loading = false) {
  if (loading || !report) {
    return "正在检查 Python 引擎、运行时依赖、输出目录和任务历史数据库，请稍候。";
  }

  const status = String(report?.overall_status || "failed").toLowerCase();
  if (status === "ok") {
    return "运行环境准备就绪，正在进入应用。";
  }
  if (status === "warning") {
    return "启动自检已完成，应用可以进入，但仍有非阻塞警告需要留意。";
  }
  return "发现阻塞性问题，应用不会继续进入主流程。请根据下列提示修复后重新自检。";
}

function startupCountsText(report, loading = false) {
  if (loading || !report) return "等待检查结果...";
  const summary = asObject(report?.summary);
  return `通过 ${toInt(summary?.passed, 0)} 项 · 警告 ${toInt(summary?.warnings, 0)} 项 · 失败 ${toInt(summary?.failed, 0)} 项`;
}

function startupItemDetailText(item) {
  const detail = asObject(item?.detail);
  const parts = [];
  if (detail?.provider) parts.push(`提供方: ${String(detail.provider)}`);
  if (detail?.cognition_status) parts.push(`认知状态: ${String(detail.cognition_status)}`);
  if (detail?.planner_mode) parts.push(`规划模式: ${String(detail.planner_mode)}`);
  if (detail?.llm_mode) parts.push(`LLM 模式: ${String(detail.llm_mode)}`);
  if (detail?.graph_id || detail?.version) {
    const graphLabel = [String(detail?.graph_id || "").trim(), String(detail?.version || "").trim()]
      .filter(Boolean)
      .join(" @ ");
    if (graphLabel) parts.push(`图谱: ${graphLabel}`);
  }
  if (detail?.model) parts.push(`模型: ${String(detail.model)}`);
  if (detail?.fallback_reason_code) parts.push(`降级原因: ${String(detail.fallback_reason_code)}`);
  if (detail?.fallback_message) parts.push(String(detail.fallback_message));
  if (detail?.fallback_active === true) parts.push("降级路径已启用");
  if (item?.path) parts.push(`路径: ${String(item.path)}`);
  if (detail?.db_path) parts.push(`数据库: ${String(detail.db_path)}`);
  if (detail?.selected_path) parts.push(`模型目录: ${String(detail.selected_path)}`);
  if (detail?.reason) parts.push(`原因: ${normalizeReadableErrorText(String(detail.reason))}`);
  if (detail?.dependency) parts.push(`依赖: ${String(detail.dependency)}`);
  if (Array.isArray(detail?.missing_files) && detail.missing_files.length > 0) {
    parts.push(`缺失文件: ${detail.missing_files.join(", ")}`);
  }
  if (item?.auto_fixed) parts.push("已自动创建缺失目录");
  return parts.join(" | ");
}

function setStartupGateVisible(visible) {
  if (!startupGate) return;
  startupGate.classList.toggle("hidden", !visible);
}

function renderStartupGate(report = null, options = {}) {
  const loading = Boolean(options?.loading);
  const data = loading ? null : asObject(report);
  const status = loading ? "checking" : String(data?.overall_status || "failed").toLowerCase();

  setStartupGateVisible(true);

  if (startupTitle) {
    startupTitle.textContent =
      status === "ok"
        ? "启动自检通过"
        : status === "warning"
        ? "启动自检完成（有警告）"
        : status === "failed"
        ? "启动自检未通过"
        : "正在执行启动自检";
  }
  if (startupSummary) startupSummary.textContent = startupSummaryText(data, loading);
  if (startupStatusPill) {
    startupStatusPill.className = `startup-status-pill ${status}`;
    startupStatusPill.textContent = loading ? "检查中" : startupStatusLabel(status);
  }
  if (startupCounts) startupCounts.textContent = startupCountsText(data, loading);

  if (startupCheckList) {
    if (loading) {
      startupCheckList.innerHTML = `
        <li class="startup-check-item checking">
          <div>
            <strong>启动自检进行中</strong>
            <p>正在准备应用运行环境。</p>
          </div>
        </li>
      `;
    } else {
      const items = asArray(data?.items);
      startupCheckList.innerHTML = items
        .map((item) => {
          const detailText = startupItemDetailText(item);
          return `
            <li class="startup-check-item ${escapeHtml(String(item?.status || "fail"))}">
              <div>
                <strong>${escapeHtml(String(item?.label || item?.key || "未知检查项"))}</strong>
                <p>${escapeHtml(String(item?.message || "-"))}</p>
                ${detailText ? `<small>${escapeHtml(detailText)}</small>` : ""}
              </div>
            </li>
          `;
        })
        .join("");
    }
  }

  if (startupDetails) {
    startupDetails.classList.toggle("hidden", loading);
    startupDetails.open = !loading && status === "failed";
  }
  if (startupRaw) {
    startupRaw.textContent = loading ? "{}" : `${JSON.stringify(data || {}, null, 2)}\n`;
  }
  if (startupRetryBtn) {
    startupRetryBtn.disabled = loading || Boolean(data?.can_enter);
  }
  if (startupCopyBtn) {
    startupCopyBtn.disabled = loading || !data;
  }
  if (startupCloseBtn) {
    startupCloseBtn.disabled = loading || !Boolean(data?.can_enter);
  }
}

async function copyStartupDiagnostics() {
  if (!state.startupReport) return;

  const text = `${JSON.stringify(state.startupReport, null, 2)}\n`;
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
    if (startupCopyBtn) {
      startupCopyBtn.textContent = "已复制";
      setTimeout(() => {
        startupCopyBtn.textContent = "复制诊断信息";
      }, 1200);
    }
  } catch (err) {
    addEvent(`复制启动诊断失败: ${String(err)}`);
  }
}

function openStartupDiagnostics() {
  renderStartupGate(state.startupReport || null);
  setStartupGateVisible(true);
}

function closeStartupDiagnostics() {
  if (state.startupReport?.can_enter) {
    setStartupGateVisible(false);
  }
}

function openAdvancedWorkspace() {
  const currentAction = String(state.currentTask?.request?.action || "").toLowerCase();
  if (isAgentAutoAction(currentAction)) {
    const status = String(state.currentTask?.status || "").toLowerCase();
    if (isTerminalTaskStatus(status)) {
      renderRepairResult(state.currentTask?.response?.result, state.currentTask);
      setWizardStep(STEP_REPAIR, { immediate: true });
    } else if (state.currentTask?.id) {
      setWizardStep(STEP_PROGRESS, { immediate: true });
    }
  }
  setShellView(VIEW_ADVANCED_WORKSPACE);
}

function returnToSmartSurface() {
  const currentAction = String(state.currentTask?.request?.action || "").toLowerCase();
  const currentStatus = String(state.currentTask?.status || "").toLowerCase();
  if (isAgentAutoAction(currentAction) && state.currentTask?.id) {
    if (isTerminalTaskStatus(currentStatus)) {
      renderSmartResult(state.currentTask, { skipHydrate: false });
      return;
    }
    renderSmartRun(state.currentTask);
    return;
  }
  setShellView(VIEW_SMART_HOME);
  renderSmartHome();
}

function pickRecentHistoryCandidate(tasks) {
  const items = asArray(tasks).map((item) => asObject(item));
  const runningAuto = items.find((item) => isAgentAutoAction(item?.request?.action) && !isTerminalTaskStatus(item?.status));
  if (runningAuto) return runningAuto;
  const latestAuto = items.find((item) => isAgentAutoAction(item?.request?.action));
  if (latestAuto) return latestAuto;
  return items.find((item) => actionToIntent(item?.request?.action)) || null;
}

function restoreRecentTask(task, options = {}) {
  if (!task || !task?.id) return;
  const intent = actionToIntent(task?.request?.action);
  if (!intent) return;

  state.currentTaskId = String(task?.id || "");
  state.currentTask = task;
  state.runningIntent = intent;
  const startedAtMs = Date.parse(String(task?.started_at || task?.created_at || ""));
  state.taskStartAtMS = Number.isFinite(startedAtMs) ? startedAtMs : Date.now();
  renderTask(task, intent);

  const fromHistory = Boolean(options?.fromHistory);
  const status = String(task?.status || "").toLowerCase();
  if (!isTerminalTaskStatus(status)) {
    setRunningUi(true);
    if (intent === INTENT_AGENT_AUTO) {
      renderSmartRun(task);
    } else {
      setWizardStep(STEP_PROGRESS, { immediate: fromHistory });
    }
    void pollTask(String(task.id), intent);
    return;
  }

  setRunningUi(false);
  if (intent === INTENT_AGENT_AUTO) {
    renderSmartResult(task, { skipHydrate: false });
    return;
  }
  handleTerminalTask(task, intent, { fromHistory });
}

async function initializeAppShell() {
  if (state.appInitialized) return;
  setWizardStep(STEP_CONFIG, { immediate: true });
  setShellView(VIEW_SMART_HOME);
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
  await refreshColumnsForCsv(csvPathInput?.value || "", "初始加载");
  await loadRecentHistory();
  state.appInitialized = true;
}

async function runStartupChecksFlow(source = "启动") {
  if (state.startupRunning) return;
  state.startupRunning = true;
  logFrontendEvent("startup_checks_flow_started", {
    source,
    has_app_binding: hasAnyAppBinding(),
    has_startup_binding: hasBinding("RunStartupChecks"),
    has_wails_runtime: hasWailsRuntime(),
    location: String(window?.location?.href || ""),
    preview_mode: isPreviewMode(),
  });
  renderStartupGate(null, { loading: true });

  try {
    const report = await apiRunStartupChecks();
    state.startupReport = report;
    renderStartupGate(report);

    const overallStatus = String(report?.overall_status || "failed").toLowerCase();
    addEvent(`启动自检(${source})完成: ${startupStatusLabel(overallStatus)}。`);
    logFrontendEvent("startup_checks_flow_completed", {
      source,
      overall_status: overallStatus,
      can_enter: Boolean(report?.can_enter),
    });

    if (report?.can_enter) {
      await initializeAppShell();
      await delay(hasBinding("RunStartupChecks") ? 280 : 520);
      setStartupGateVisible(false);
    }
  } catch (err) {
    const reason = normalizeReadableErrorText(String(err));
    const fallbackReport = {
      overall_status: "failed",
      can_enter: false,
      checked_at: new Date().toISOString(),
      items: [
        {
          key: "startup_gate",
          label: "启动自检",
          status: "fail",
          blocking: true,
          message: "启动自检执行失败。",
          detail: { reason },
        },
      ],
      summary: { passed: 0, warnings: 0, failed: 1 },
      raw: { reason },
    };
    state.startupReport = fallbackReport;
    renderStartupGate(fallbackReport);
    addEvent(`启动自检失败: ${reason}`);
    logFrontendEvent("startup_checks_flow_failed", { source, reason });
  } finally {
    state.startupRunning = false;
  }
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
  renderTaskObservability(resultObservability, null);
  renderTaskObservability(repairObservability, null);
  renderPresentationBundle(resultPresentation, null);
  renderPresentationBundle(repairPresentation, null);
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
    return "未知错误";
  }
  text = text.replace(/\s+/g, " ");
  text = text.replace(/[;,]?\s*raw=.*$/i, "");
  if (text.length > 420) {
    text = `${text.slice(0, 420)}...`;
  }
  return text;
}

function toReadableError(task, err) {
  if (err) return normalizeReadableErrorText(`请求失败: ${String(err)}`);
  if (!task) return "任务返回为空。";
  if (task?.response?.error?.message) {
    const code = String(task?.response?.error?.code || "UNKNOWN");
    return normalizeReadableErrorText(`引擎错误 [${code}] ${String(task.response.error.message)}`);
  }
  if (task?.error) return normalizeReadableErrorText(String(task.error));
  return `任务结束状态: ${taskStatusLabel(task?.status || "unknown")}`;
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

function renderScanResult(result, taskSnapshot = state.currentTask) {
  const scanResult = asObject(result);
  const scanView = buildScanView(scanResult);
  const presentation = getPresentationBundle(scanResult);
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
  renderTaskObservability(resultObservability, taskSnapshot);
  renderPresentationBundle(resultPresentation, presentation);
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
  renderTaskObservability(resultObservability, task || state.currentTask);
  renderPresentationBundle(resultPresentation, null);

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

function renderRepairResult(result, taskSnapshot = state.currentTask) {
  const view = buildRepairView(result);
  const repairResult = asObject(view?.repairResult);
  const comparison = asObject(repairResult?.comparison);
  const beforeIssueCount = toInt(comparison?.before_issue_count, toInt(repairResult?.scan_issue_count, 0));
  const afterIssueCount = toInt(comparison?.after_issue_count, Math.max(0, beforeIssueCount - toInt(repairResult?.applied_issue_count, 0)));
  const resolvedIssueCount = toInt(comparison?.resolved_issue_count, Math.max(0, beforeIssueCount - afterIssueCount));
  const appliedCount = toInt(repairResult?.applied_issue_count, 0);
  const skippedCount = asArray(repairResult?.skipped_issues).length;
  const modifiedCells = toInt(repairResult?.total_cells_modified, 0);
  const rollback = asObject(repairResult?.rollback);
  const repairModeLabel = view?.isAgent
    ? "智能自动修复"
    : state.lastRepairMode === "auto"
    ? "自动修复全部问题列"
    : "手动选择修复";
  setRepairKpis(appliedCount, skippedCount, modifiedCells);
  renderRepairOverviewSidebar(repairResult);
  renderPresentationBundle(repairPresentation, view?.presentation);
  renderDescriptionList(repairSummary, [
    ["修复模式", repairModeLabel],
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
    ["来源", repairResult?.selected_source ? String(repairResult.selected_source) : "-"],
  ]);
  renderTaskObservability(repairObservability, taskSnapshot);

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
  renderPresentationBundle(repairPresentation, null);
  renderDescriptionList(repairSummary, [
    ["阶段", String(task?.request?.action || "repair_batch")],
    ["任务ID", String(task?.id || state.currentTaskId || "-")],
    ["状态", String(task?.status || "failed")],
    ["错误码", String(task?.response?.error?.code || "-")],
  ]);
  renderTaskObservability(repairObservability, task || state.currentTask);

  if (!repairDetailList) return;
  repairDetailList.innerHTML = "";
  const li = document.createElement("li");
  li.className = "batch-item-skip";
  const suggestion = extractSuggestion(task);
  li.textContent = suggestion ? `修复失败：${reason}。建议：${suggestion}` : `修复失败：${String(reason || "未知错误")}`;
  repairDetailList.appendChild(li);
}

function approvalReasonLabel(code) {
  switch (String(code || "").trim()) {
    case "high_risk_columns_selected":
      return "命中了高风险列";
    case "time_like_columns_selected":
      return "命中了时间/日期列";
    case "protected_columns_selected":
      return "命中了保护列";
    case "planner_requested_approval":
      return "Planner 请求在写入前人工确认";
    default:
      return String(code || "-");
  }
}

function firstNonEmptyText(...values) {
  for (const value of values) {
    const text = String(value || "").trim();
    if (text) return text;
  }
  return "";
}

function buildSmartCognition(view, session = null) {
  const agentBlock = asObject(view?.agentBlock);
  const explanation = asObject(agentBlock?.explanation);
  const plan = asObject(agentBlock?.plan || asObject(session?.latest_plan));
  const traceSummary = asObject(session?.trace_summary || agentBlock?.trace_summary);
  const traceCognition = asObject(traceSummary?.cognition);
  const sessionContext = asObject(session?.context);
  const contextCognition = asObject(sessionContext?.cognition_state);
  const explanationCognition = asObject(explanation?.cognition);
  const planCognition = asObject(plan?.cognition);
  const cognition =
    Object.keys(explanationCognition).length > 0
      ? explanationCognition
      : Object.keys(planCognition).length > 0
      ? planCognition
      : Object.keys(contextCognition).length > 0
      ? contextCognition
      : traceCognition;
  return {
    mode: String(explanation?.mode || "").trim(),
    provider: String(cognition?.provider || traceCognition?.provider || "").trim(),
    status: String(cognition?.status || traceCognition?.status || "").trim(),
    plannerMode: String(cognition?.planner_mode || traceCognition?.planner_mode || "").trim(),
    llmMode: String(cognition?.llm_mode || traceCognition?.llm_mode || "").trim(),
    graphID: String(cognition?.graph_id || "").trim(),
    version: String(cognition?.version || "").trim(),
    summary: firstNonEmptyText(cognition?.summary, traceCognition?.last_summary),
    fallbackReasonCode: String(cognition?.fallback_reason_code || traceCognition?.fallback_reason_code || "").trim(),
    fallbackMessage: String(cognition?.fallback_message || "").trim(),
    reasonCodes: asArray(cognition?.reason_codes || traceCognition?.reason_codes)
      .map((item) => String(item || "").trim())
      .filter(Boolean),
    selectedCandidateID: String(cognition?.selected_candidate_id || traceCognition?.selected_candidate_id || "").trim(),
    eventCount: toInt(traceCognition?.event_count, 0),
  };
}

function smartPlanIssueIDs(plan, key) {
  return asArray(asObject(plan)?.[key])
    .map((item) => String(item || "").trim())
    .filter(Boolean);
}

function smartIssueBucketEntries(plan) {
  return [
    ["自动修复", smartPlanIssueIDs(plan, "auto_repair_issue_ids")],
    ["谨慎复核", smartPlanIssueIDs(plan, "cautious_issue_ids")],
    ["人工复核", smartPlanIssueIDs(plan, "manual_review_issue_ids")],
    ["阻塞", smartPlanIssueIDs(plan, "blocked_issue_ids")],
  ];
}

function smartBucketCountText(plan) {
  const entries = smartIssueBucketEntries(plan);
  if (entries.every(([, ids]) => ids.length === 0)) {
    const selected = smartPlanIssueIDs(plan, "selected_issue_ids");
    return selected.length > 0 ? `自动修复: ${selected.length}` : "";
  }
  return entries.map(([label, ids]) => `${label}: ${ids.length}`).join(" / ");
}

function smartBucketDetailBullets(plan) {
  const entries = smartIssueBucketEntries(plan).filter(([, ids]) => ids.length > 0);
  if (entries.length === 0) return [];
  return entries.map(([label, ids]) => {
    const preview = ids.slice(0, 3).join(", ");
    const suffix = ids.length > 3 ? ` +${ids.length - 3}` : "";
    return `${label}: ${ids.length}${preview ? ` (${preview}${suffix})` : ""}`;
  });
}

function smartPostValidation(view, session = null) {
  const validation = asObject(view?.validation);
  const postExecute = asObject(validation?.post_execute);
  if (Object.keys(postExecute).length > 0) return postExecute;
  return asObject(asObject(session?.context)?.post_validation);
}

function smartPreviewValidation(view, session = null) {
  const validation = asObject(view?.validation);
  const preview = asObject(validation?.preview);
  if (Object.keys(preview).length > 0) return preview;
  return asObject(asObject(session?.context)?.preview_validation);
}

function smartValidationVerdict(postExecute) {
  return String(postExecute?.verdict || postExecute?.status || "").trim();
}

function smartValidationRiskNotes(postExecute, safety = null) {
  return asArray(postExecute?.risk_notes || postExecute?.risk_flags || asObject(safety)?.risk_flags)
    .map((item) => String(item || "").trim())
    .filter(Boolean);
}

function smartCognitionLabel(cognition) {
  const provider = String(cognition?.provider || "").trim();
  const status = String(cognition?.status || "").trim();
  if (provider && status) return `${provider}/${status}`;
  return provider || status || "";
}

function smartCognitionSummaryText(cognition) {
  return firstNonEmptyText(cognition?.summary, cognition?.fallbackMessage);
}

function smartCognitionFallbackText(cognition) {
  const reason = String(cognition?.fallbackReasonCode || "").trim();
  const message = String(cognition?.fallbackMessage || "").trim();
  if (!reason && !message) return "";
  if (!reason) return message;
  if (!message) return `降级原因: ${reason}`;
  return `降级原因: ${reason} (${message})`;
}

function buildSmartApprovalDetails(view, session = null) {
  const agentBlock = asObject(view?.agentBlock);
  const approval = asObject(agentBlock?.approval);
  const sessionContext = asObject(session?.context);
  const riskAssessment = asObject(sessionContext?.risk_assessment);
  const approvalState = asObject(sessionContext?.approval_state);
  const required = Boolean(approval?.required || approvalState?.required || riskAssessment?.required);
  const reasonSource =
    asArray(approval?.reason_codes).length > 0
      ? approval.reason_codes
      : asArray(approvalState?.reason_codes).length > 0
      ? approvalState.reason_codes
      : riskAssessment?.reason_codes;
  return {
    status: String(approval?.status || approvalState?.status || (required ? "required" : "not_required")).trim(),
    required,
    reasonCodes: asArray(reasonSource)
      .map((item) => String(item || "").trim())
      .filter(Boolean),
    candidateColumns: asArray(riskAssessment?.candidate_columns || sessionContext?.candidate_columns)
      .map((item) => String(item || "").trim())
      .filter(Boolean),
    riskColumns: asArray(approval?.risk_columns || riskAssessment?.risk_columns),
    protectedColumns: asArray(approval?.protected_columns || riskAssessment?.protected_columns),
    timeLikeColumns: asArray(approval?.time_like_columns || riskAssessment?.time_like_columns || sessionContext?.time_like_columns),
    preferenceSnapshot: cloneSmartPreferenceProfile(asObject(sessionContext?.preference_snapshot)),
    message: String(approval?.message || approvalState?.message || riskAssessment?.message || "").trim(),
    selectedSource: String(riskAssessment?.selected_source || asObject(agentBlock?.plan)?.selected_source || asObject(session?.latest_plan)?.selected_source || "").trim(),
    workspaceID: String(sessionContext?.workspace_id || state.smartPreferences?.workspaceID || "").trim(),
  };
}

function renderSmartApprovalCard(view, task, session = null) {
  if (!smartApprovalCard) return;
  const details = buildSmartApprovalDetails(view, session);
  const showCard = details.required || ["required", "rejected", "approved"].includes(details.status);
  smartApprovalCard.classList.toggle("hidden", !showCard);
  if (!showCard) return;

  const columns = Array.from(
    new Set(
      [...details.candidateColumns, ...details.riskColumns, ...details.protectedColumns, ...details.timeLikeColumns]
        .map((item) => String(item || "").trim())
        .filter(Boolean)
    )
  );
  const message =
    details.status === "required"
      ? details.message || "当前任务已经通过 preview validation，但在真正写文件前需要你确认。"
      : details.status === "approved"
      ? "本次执行已获批准，并从审批点继续完成写入与后验验证。"
      : "本次执行已在写文件前被取消，没有输出文件被写入，也没有触发回滚。";
  if (smartApprovalMessage) smartApprovalMessage.textContent = message;
  if (smartApprovalSummary) {
    renderDescriptionList(smartApprovalSummary, [
      ["审批状态", details.status || "-"],
      ["候选来源", details.selectedSource || "-"],
      ["影响列", columns.length > 0 ? columns.join(", ") : "-"],
      ["工作区", details.workspaceID || "-"],
    ]);
  }
  if (smartApprovalReasons) {
    renderCompactList(
      smartApprovalReasons,
      details.reasonCodes.map((code) => approvalReasonLabel(code)),
      "当前没有额外触发原因。"
    );
  }
  if (smartApprovalPreferences) {
    renderDescriptionList(smartApprovalPreferences, [
      ["保守模式", details.preferenceSnapshot.conservative_mode ? "是" : "否"],
      ["避开时间列", details.preferenceSnapshot.avoid_time_columns ? "是" : "否"],
      ["高风险需审批", details.preferenceSnapshot.require_approval_for_high_risk ? "是" : "否"],
      ["保护列", smartPreferenceColumnsText(details.preferenceSnapshot.protected_columns) || "-"],
    ]);
  }
  const actionable = details.status === "required" && !state.isRunning;
  if (smartApprovalContinueBtn) {
    smartApprovalContinueBtn.disabled = !actionable;
    smartApprovalContinueBtn.classList.toggle("hidden", details.status !== "required");
  }
  if (smartApprovalRejectBtn) {
    smartApprovalRejectBtn.disabled = !actionable;
    smartApprovalRejectBtn.classList.toggle("hidden", details.status !== "required");
  }
}

function buildSmartTrustList(view, task, verdict) {
  const safety = asObject(view?.safety);
  const session = state.smartSessionSnapshot;
  const preview = smartPreviewValidation(view, session);
  const postExecute = smartPostValidation(view, session);
  const approval = buildSmartApprovalDetails(view, session);
  const cognition = buildSmartCognition(view, session);
  const plan = asObject(view?.agentBlock?.plan || asObject(session?.latest_plan));
  const bucketSummary = smartBucketCountText(plan);
  const postVerdict = smartValidationVerdict(postExecute);
  const list = [
    `最终结论: ${verdict || "未知"}`,
    preview?.message ? `预演: ${preview.message}` : "预演: -",
  ];
  if (bucketSummary) {
    list.push(`方案分桶: ${bucketSummary}`);
  }
  if (approval.required || approval.status === "approved" || approval.status === "rejected") {
    list.push(`审批: ${approval.status || "required"}`);
  }
  if (smartCognitionLabel(cognition)) {
    list.push(`认知状态: ${smartCognitionLabel(cognition)}`);
  }
  if (cognition?.fallbackReasonCode) {
    list.push(`降级原因: ${cognition.fallbackReasonCode}`);
  }
  if (postVerdict) {
    list.push(`验证门禁: ${postVerdict}`);
  }
  if (postExecute?.message) {
    list.push(`后验验证: ${postExecute.message}`);
  }
  if (Array.isArray(safety?.risk_flags) && safety.risk_flags.length > 0) {
    list.push(`风险标记: ${safety.risk_flags.join(", ")}`);
  }
  if (task?.response?.error?.code) {
    list.push(`错误码: ${String(task.response.error.code)}`);
  }
  return list;
}

function renderSmartReasoning(view, task, session = null) {
  if (!smartReasoningBody) return;
  const agentBlock = asObject(view?.agentBlock);
  const plan = asObject(agentBlock?.plan || asObject(session?.latest_plan));
  const explanationBlock = asObject(agentBlock?.explanation);
  const safety = asObject(view?.safety);
  const preview = smartPreviewValidation(view, session);
  const postExecute = smartPostValidation(view, session);
  const sessionContext = asObject(session?.context);
  const approval = buildSmartApprovalDetails(view, session);
  const cognition = buildSmartCognition(view, session);
  const bucketBullets = smartBucketDetailBullets(plan);
  const postVerdict = smartValidationVerdict(postExecute);
  const riskNotes = smartValidationRiskNotes(postExecute, safety);
  const reasoningSummary = String(explanationBlock?.summary || plan?.reasoning_summary || "").trim();
  const userExplanation = String(explanationBlock?.final_message || plan?.user_explanation || "").trim();
  const shortBullets = asArray(explanationBlock?.short_bullets)
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .slice(0, 3);
  const bullets = [
    reasoningSummary || "系统已根据当前数据与安全策略选择执行路径。",
    userExplanation || "你可以在轨迹与验证中继续查看更细的审计信息。",
    ...shortBullets,
    smartCognitionLabel(cognition) ? `认知状态: ${smartCognitionLabel(cognition)}` : "",
    smartCognitionSummaryText(cognition) ? `认知摘要: ${smartCognitionSummaryText(cognition)}` : "",
    smartCognitionFallbackText(cognition),
    explanationBlock?.risk_note ? `风险说明: ${String(explanationBlock.risk_note)}` : "",
    approval?.status && approval.status !== "not_required" ? `审批: ${approval.status}` : "",
    preview?.message ? `预演: ${preview.message}` : "",
    ...bucketBullets.map((item) => `方案分桶: ${item}`),
    postVerdict ? `验证门禁: ${postVerdict}` : "",
    riskNotes.length > 0 ? `风险说明: ${riskNotes.join(", ")}` : "",
    postExecute?.message ? `后验验证: ${postExecute.message}` : "",
    postExecute?.explanation ? `验证解释: ${String(postExecute.explanation)}` : "",
    sessionContext?.final_verdict ? `会话结论: ${String(sessionContext.final_verdict)}` : "",
  ].filter(Boolean);
  smartReasoningBody.innerHTML = `
    <article class="insight-section">
      <h4>为什么这样做</h4>
      <ul>
        ${bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </article>
  `;
}

function renderSmartTrace(view, task, session = null, traceEvents = []) {
  const validation = asObject(view?.validation);
  const traceSummary = asObject(session?.trace_summary || asObject(view?.agentBlock?.trace_summary));
  const traces = asArray(traceEvents).map((item) => asObject(item));
  const cognition = buildSmartCognition(view, session);
  if (smartTraceSummary) {
    const traceTypeCounts = asObject(traceSummary?.trace_type_counts);
    smartTraceSummary.textContent =
      traces.length > 0
        ? `已加载 ${traces.length} 条会话轨迹。tool calls: ${toInt(traceSummary?.tool_call_count, toInt(traceSummary?.tool_calls, 0))}，validation events: ${toInt(traceTypeCounts?.validation, toInt(traceSummary?.validation_events, 0))}。`
        : validation?.message
        ? validation.message
        : "暂无额外轨迹，已回退到任务级可观测信息。";
  }
  if (smartTraceSummary) {
    const traceTypeCounts = asObject(traceSummary?.trace_type_counts);
    const summaryParts = [];
    if (traces.length > 0) {
      summaryParts.push(`已加载 ${traces.length} 条轨迹事件。`);
      summaryParts.push(`工具调用: ${toInt(traceSummary?.tool_call_count, toInt(traceSummary?.tool_calls, 0))}`);
      summaryParts.push(`验证事件: ${toInt(traceTypeCounts?.validation, toInt(traceSummary?.validation_events, 0))}`);
    } else if (validation?.message) {
      summaryParts.push(validation.message);
    } else {
      summaryParts.push("暂无额外会话轨迹，当前使用任务级诊断信息。");
    }
    if (smartCognitionLabel(cognition)) {
      summaryParts.push(`LangGraph 状态: ${smartCognitionLabel(cognition)}`);
    }
    if (smartCognitionSummaryText(cognition)) {
      summaryParts.push(smartCognitionSummaryText(cognition));
    }
    if (cognition?.fallbackReasonCode) {
      summaryParts.push(`降级原因: ${cognition.fallbackReasonCode}`);
    }
    smartTraceSummary.textContent = summaryParts.join(" ");
  }
  if (!smartTraceList) return;
  if (traces.length === 0) {
    const progress = getTaskProgress(task);
    const fallbackStages = extractProgressStages(task, INTENT_AGENT_AUTO);
    smartTraceList.innerHTML = fallbackStages
      .map((stage) => `<li>${escapeHtml(stage)}</li>`)
      .join("");
    if (fallbackStages.length === 0 && progress?.last_message) {
      smartTraceList.innerHTML = `<li>${escapeHtml(String(progress.last_message))}</li>`;
    }
    return;
  }
  smartTraceList.innerHTML = traces
    .map(
      (item) => `
        <li>
          <strong>${escapeHtml(String(item?.agent_name || "agent"))}</strong>
          <span>${escapeHtml(String(item?.trace_type || "-"))}</span>
          <p>${escapeHtml(String(item?.summary || "-"))}</p>
        </li>
      `
    )
    .join("");
}

function renderSmartRun(task) {
  const progress = getTaskProgress(task);
  const status = String(task?.status || "idle").toLowerCase();
  const intent = INTENT_AGENT_AUTO;
  if (smartRunStatusPill) {
    smartRunStatusPill.className = `status-pill ${status || "idle"}`;
    smartRunStatusPill.textContent = taskStatusLabel(status || "idle");
  }
  if (smartRunTaskId) smartRunTaskId.textContent = `任务: ${String(task?.id || "-")}`;
  if (smartRunTitle) smartRunTitle.textContent = "智能处理进行中";
  if (smartRunMessage) smartRunMessage.textContent = buildTaskMessage(task, intent);
  if (smartRunProgressFill) {
    smartRunProgressFill.className = `progress-fill ${status || "idle"}`;
    smartRunProgressFill.style.width = `${clamp(toInt(progress?.progress_percent, STATUS_PROGRESS[status] ?? 0), 0, 100)}%`;
  }
  if (smartRunStages) {
    const stages = extractProgressStages(task, intent);
    const currentStage = normalizeStageLabel(progress?.current_stage || stages[0] || "");
    const currentIndex = stages.findIndex((item) => item === currentStage);
    smartRunStages.innerHTML = stages
      .map((stage, idx) => `<li class="${stageStateForTimeline(stage, idx, currentStage, currentIndex, status)}">${escapeHtml(stage)}</li>`)
      .join("");
  }
  if (smartRunMetrics) {
    const durations = buildStageDurationEntries(task, intent);
    const durationMS =
      toInt(task?.response?.duration_ms, 0) ||
      durations.reduce((sum, item) => sum + Math.max(0, toInt(item?.ms, 0)), 0);
    renderDescriptionList(smartRunMetrics, [
      ["当前阶段", String(progress?.current_stage || "-")],
      ["当前进度", `${clamp(toInt(progress?.progress_percent, 0), 0, 100)}%`],
      ["已用时", formatDurationMS(durationMS)],
      ["预计剩余", formatRemainingTime(task)],
    ]);
  }
  if (smartRunEvents) {
    const events = asArray(progress?.events).slice(-8);
    smartRunEvents.innerHTML =
      events.length > 0
        ? events
            .map(
              (event) =>
                `<li>${escapeHtml(String(event?.stage || "-"))}: ${escapeHtml(String(event?.message || event?.phase || "-"))}</li>`
            )
            .join("")
        : "<li>等待实时事件...</li>";
  }
  if (smartRunCancelBtn) smartRunCancelBtn.disabled = !state.isRunning || !state.currentTaskId;
  setShellView(VIEW_SMART_RUN);
}

async function hydrateSmartAgentDetails(task) {
  const sessionID = String(asObject(task?.response?.result?.agent)?.session_id || "").trim();
  state.smartSessionSnapshot = null;
  state.smartTraceEvents = [];
  state.smartDetailsLoading = Boolean(sessionID);
  state.smartDetailsError = "";
  if (!sessionID) {
    renderSmartTracePlaceholder("当前结果未携带 session_id，已回退到任务级信息。");
    return;
  }
  renderSmartTracePlaceholder("正在加载会话轨迹与验证详情...");
  try {
    const [session, trace] = await Promise.all([apiGetAgentSession(sessionID), apiListAgentTrace(sessionID)]);
    if (String(state.currentTask?.id || "") !== String(task?.id || "")) return;
    state.smartSessionSnapshot = asObject(session);
    state.smartTraceEvents = asArray(trace);
    renderSmartResult(task, { skipHydrate: true });
  } catch (err) {
    state.smartDetailsError = String(err);
    renderSmartTracePlaceholder(`会话详情加载失败，已回退到任务级信息: ${normalizeReadableErrorText(String(err))}`);
  } finally {
    state.smartDetailsLoading = false;
  }
}

function renderSmartResult(task, options = {}) {
  const result = asObject(task?.response?.result);
  const view = buildRepairView(result);
  const repairResult = asObject(view?.repairResult);
  const safety = asObject(view?.safety);
  const validation = asObject(view?.validation);
  const session = asObject(state.smartSessionSnapshot);
  const traceEvents = asArray(state.smartTraceEvents);
  const comparison = asObject(repairResult?.comparison);
  const agentPlan = asObject(view?.agentBlock?.plan || asObject(session?.latest_plan));
  const postExecute = smartPostValidation(view, session);
  const postVerdict = smartValidationVerdict(postExecute);
  const riskNotes = smartValidationRiskNotes(postExecute, safety);
  const bucketSummary = smartBucketCountText(agentPlan);
  const verdict = String(safety?.final_verdict || (String(task?.status || "").toLowerCase() === "succeeded" ? "accepted" : "failed")).trim();
  const outputCSV = String(repairResult?.output_csv || "").trim();
  const presentationArtifact = String(session?.presentation_artifact || "").trim();
  const rollback = asObject(repairResult?.rollback || safety?.rollback_execution);
  const rollbackManifest = String(rollback?.manifest_path || "").trim();
  const rejectedSnapshot = String(safety?.rejected_output_snapshot || "").trim();
  const cognition = buildSmartCognition(view, session);
  const beforeIssueCount = toInt(comparison?.before_issue_count, toInt(postExecute?.before_issue_count, toInt(safety?.baseline_scan_summary?.issue_count, 0)));
  const afterIssueCount = toInt(comparison?.after_issue_count, toInt(postExecute?.after_issue_count, toInt(safety?.post_scan_summary?.issue_count, 0)));
  const resolvedIssueCount = toInt(postExecute?.resolved_issue_count, toInt(comparison?.resolved_issue_count, Math.max(0, beforeIssueCount - afterIssueCount)));
  const totalCellsModified = toInt(postExecute?.total_cells_modified, toInt(repairResult?.total_cells_modified, 0));
  const selectedSource = String(repairResult?.selected_source || agentPlan?.selected_source || "-").trim();
  const conclusion =
    verdict === "accepted"
      ? "系统已完成自动扫描、修复、复扫与验证，本次输出被正式接纳。"
      : verdict === "validation_rejected"
      ? "系统在 preview gate 阶段拒绝了自动执行，因此没有写出不可信结果。"
      : verdict === "rolled_back"
      ? "系统执行后发现风险未下降，已自动回滚输出产物。"
      : "系统尝试自动恢复，但仍需要人工介入复核。";

  let finalConclusion = conclusion;
  if (verdict === "approval_required") {
    finalConclusion = "系统已经完成 preview validation 和风险评估，但在真正写文件前暂停，等待你确认。";
  } else if (verdict === "approval_rejected") {
    finalConclusion = "本次执行已在写文件前取消，没有写出输出文件，也没有触发回滚。";
  }
  renderSmartSafetyBanner(verdict, finalConclusion);
  renderPresentationBundle(smartResultPresentation, view?.presentation);
  if (smartResultConclusion) smartResultConclusion.textContent = finalConclusion;
  if (smartResultSummary) {
    renderDescriptionList(smartResultSummary, [
      ["最终结论", verdict || "-"],
      ["采用来源", selectedSource || "-"],
      ["问题数变化", beforeIssueCount > 0 || afterIssueCount > 0 ? `${beforeIssueCount} -> ${afterIssueCount}` : "-"],
      ["已解决问题", resolvedIssueCount || 0],
      ["修改单元格", totalCellsModified],
      ["验证门禁", postVerdict || "-"],
      ["Plan 分桶", bucketSummary || "-"],
      ["风险提示", riskNotes.length > 0 ? riskNotes.join(", ") : "-"],
      ["输出文件", outputCSV || "-"],
    ]);
    if (smartCognitionLabel(cognition)) {
      smartResultSummary.innerHTML += `<dt>${escapeHtml("认知状态")}</dt><dd>${escapeHtml(smartCognitionLabel(cognition))}</dd>`;
    }
  }
  renderArtifactList(
    smartArtifactList,
    [
      outputCSV ? { label: "输出 CSV", path: outputCSV } : null,
      presentationArtifact ? { label: "presentation.json", path: presentationArtifact } : null,
      rollbackManifest ? { label: "回滚清单", path: rollbackManifest } : null,
      rejectedSnapshot ? { label: "被拒绝快照", path: rejectedSnapshot } : null,
    ],
    "当前结果没有额外产物。"
  );
  renderCompactList(smartTrustChecks, buildSmartTrustList(view, task, verdict), "等待任务结果。");
  renderSmartApprovalCard(view, task, session);
  renderSmartReasoning(view, task, session);
  renderSmartTrace(view, task, session, traceEvents);
  if (smartResultRaw) smartResultRaw.textContent = `${JSON.stringify(task || {}, null, 2)}\n`;
  syncExportButtons(Boolean(task));
  setShellView(VIEW_SMART_RESULT);
  if (!options?.skipHydrate) {
    void hydrateSmartAgentDetails(task);
  }
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

async function waitForBinding(methodName, timeoutMS = STARTUP_BINDING_READY_TIMEOUT_MS) {
  if (hasBinding(methodName)) return true;
  const deadline = Date.now() + timeoutMS;
  while (Date.now() < deadline) {
    await delay(80);
    if (hasBinding(methodName)) return true;
  }
  return hasBinding(methodName);
}

async function withNativeCallTimeout(value, label, timeoutMS = STARTUP_NATIVE_CALL_TIMEOUT_MS) {
  let timeoutID = null;
  const timeout = new Promise((_, reject) => {
    timeoutID = setTimeout(() => {
      reject(new Error(`${label} 在 ${timeoutMS} ms 内未返回`));
    }, timeoutMS);
  });
  try {
    return await Promise.race([Promise.resolve(value), timeout]);
  } finally {
    if (timeoutID !== null) clearTimeout(timeoutID);
  }
}

async function apiRunStartupChecks() {
  setStartupProgressText("正在等待 Wails 绑定...");
  const bindingReady = await waitForBinding("RunStartupChecks");
  logFrontendEvent("startup_checks_binding_probe", {
    ready: bindingReady,
    has_app_binding: hasAnyAppBinding(),
    has_wails_runtime: hasWailsRuntime(),
    location: String(window?.location?.href || ""),
    preview_mode: isPreviewMode(),
  });
  if (bindingReady) {
    const runStartupChecks = window.go.main.App.RunStartupChecks;
    setStartupProgressText("Wails 绑定已就绪，正在连接 IPC...");
    await delay(STARTUP_NATIVE_CALL_GRACE_MS);
    if (typeof runStartupChecks?.setTimeout === "function") {
      runStartupChecks.setTimeout(STARTUP_NATIVE_CALL_TIMEOUT_MS);
    }
    setStartupProgressText("正在执行后端启动自检...");
    return withNativeCallTimeout(runStartupChecks(), "RunStartupChecks");
  }
  if (!isPreviewMode()) {
    throw new Error("Wails 绑定 RunStartupChecks 尚未就绪");
  }
  return mockRunStartupChecks();
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

async function apiRunAgentAutofixSession(payload) {
  if (hasBinding("RunAgentAutofixSession")) return window.go.main.App.RunAgentAutofixSession(payload);
  return mockRunAgentAutofixSession(payload);
}

async function apiApproveAgentSession(payload) {
  if (hasBinding("ApproveAgentSession")) return window.go.main.App.ApproveAgentSession(payload);
  return mockApproveAgentSession(payload);
}

async function apiGetAgentSession(sessionID) {
  if (hasBinding("GetAgentSession")) return window.go.main.App.GetAgentSession(sessionID);
  return mockGetAgentSession(sessionID);
}

async function apiGetAgentPreferences(workspaceID, csvPath) {
  if (hasBinding("GetAgentPreferences")) return window.go.main.App.GetAgentPreferences(workspaceID, csvPath);
  return mockGetAgentPreferences(workspaceID, csvPath);
}

async function apiSaveAgentPreferences(payload) {
  if (hasBinding("SaveAgentPreferences")) return window.go.main.App.SaveAgentPreferences(payload);
  return mockSaveAgentPreferences(payload);
}

async function apiListAgentTrace(sessionID) {
  if (hasBinding("ListAgentTrace")) return window.go.main.App.ListAgentTrace(sessionID);
  return mockListAgentTrace(sessionID);
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

async function mockRunStartupChecks() {
  return {
    overall_status: "warning",
    can_enter: true,
    checked_at: new Date().toISOString(),
    summary: { passed: 4, warnings: 1, failed: 0 },
    items: [
      {
        key: "engine_script",
        label: "Python 引擎脚本",
        status: "pass",
        blocking: true,
        message: "静态预览模式未检查真实引擎脚本，使用模拟通过结果。",
      },
      {
        key: "engine_health",
        label: "Python 引擎健康检查",
        status: "warning",
        blocking: true,
        message: "当前是静态预览模式，未执行真实 Python 引擎健康检查。",
      },
      {
        key: "runtime_dependencies",
        label: "运行时依赖",
        status: "pass",
        blocking: true,
        message: "静态预览模式下使用模拟依赖信息。",
      },
      {
        key: "task_history_sqlite",
        label: "SQLite 任务历史",
        status: "pass",
        blocking: true,
        message: "静态预览模式未连接真实任务历史数据库。",
      },
      {
        key: "results_output_root",
        label: "结果输出目录",
        status: "pass",
        blocking: true,
        message: "静态预览模式未验证真实输出目录。",
      },
    ],
    raw: {
      preview_mode: true,
      note: "当前是静态预览模式，未执行真实引擎/SQLite 检查。",
    },
  };
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

function mockVerdictFromPayload(payload) {
  const forced = String(payload?.__mock_verdict || "").trim();
  if (forced) return forced;
  const text = String(payload?.csv_path || "").toLowerCase();
  if (text.includes("rollbackfail")) return "rollback_failed";
  if (text.includes("rollback")) return "rolled_back";
  if (text.includes("reject")) return "validation_rejected";
  return "accepted";
}

function buildMockPresentationBundle(kind, verdict, title, summary, highlights, sections, charts) {
  return {
    version: "1.0",
    kind,
    headline: title,
    summary,
    verdict,
    highlights,
    sections,
    charts,
    artifacts: [],
  };
}

function buildMockApprovalProfile(payload) {
  const workspaceID = resolveSmartWorkspaceID(payload?.workspace_id, payload?.csv_path);
  const stored = state.mockPreferenceStore.get(workspaceID);
  const saved = stored ? cloneSmartPreferenceProfile(stored.profile) : defaultSmartPreferenceProfile();
  return {
    workspaceID,
    snapshot: normalizeSmartPreferenceProfile({
      ...saved,
      ...asObject(payload?.user_preferences),
    }),
  };
}

function buildMockApprovalContext(payload, scan) {
  const info = buildMockApprovalProfile(payload);
  const csvPath = String(payload?.csv_path || "").toLowerCase();
  const selectedSource = "hybrid";
  let candidateColumns = ["bmi"];
  let timeLikeColumns = [];
  let selectedIssueCatalog = [
    {
      issue_id: "bmi::missing_values",
      column: "bmi",
      issue_type: "missing_values",
      risk_level: "high",
    },
  ];
  const reasonCodes = [];
  if (csvPath.includes("time") && info.snapshot.avoid_time_columns) {
    candidateColumns = ["event_time"];
    timeLikeColumns = ["event_time"];
    selectedIssueCatalog = [
      {
        issue_id: "event_time::missing_values",
        column: "event_time",
        issue_type: "missing_values",
        risk_level: "high",
      },
    ];
    reasonCodes.push("time_like_columns_selected");
  }
  if (asArray(info.snapshot.protected_columns).map((item) => String(item || "").trim()).includes("bmi")) {
    reasonCodes.push("protected_columns_selected");
  }
  if (csvPath.includes("approval") || csvPath.includes("interrupt")) {
    reasonCodes.push("planner_requested_approval");
  }
  return {
    ...info,
    scan,
    required: reasonCodes.length > 0,
    reasonCodes: Array.from(new Set(reasonCodes)),
    candidateColumns,
    timeLikeColumns,
    riskColumns: candidateColumns.includes("bmi") ? ["bmi"] : [],
    protectedColumns: asArray(info.snapshot.protected_columns)
      .map((item) => String(item || "").trim())
      .filter((item) => candidateColumns.includes(item)),
    selectedIssueCatalog,
    selectedSource,
    message:
      reasonCodes.length > 0
        ? "本次运行触及受保护列或规划器要求的风险门禁，写出结果前需要审批。"
        : "确定性预演已通过，当前不需要额外审批。",
  };
}

function buildMockApprovalResult(context, status) {
  return {
    status,
    required: status === "required" || status === "approved" || status === "rejected",
    reason_codes: [...context.reasonCodes],
    risk_columns: [...context.riskColumns],
    protected_columns: [...context.protectedColumns],
    time_like_columns: [...context.timeLikeColumns],
    message: context.message,
  };
}

function buildMockExplanationMode(cognition) {
  const status = String(cognition?.status || "").trim();
  const provider = String(cognition?.provider || "").trim();
  if (status === "engaged") return "langgraph_llm";
  if (status === "degraded") return "langgraph_degraded";
  if (["fallback", "disabled", "unavailable"].includes(status)) {
    return provider === "langgraph" || String(cognition?.fallback_reason_code || "").trim()
      ? "langgraph_fallback"
      : "deterministic";
  }
  return "deterministic";
}

function buildMockCognitionState(payload, selectedCandidateID, summary, riskNote) {
  const csvPath = String(payload?.csv_path || "").trim().toLowerCase();
  const cognition = {
    provider: "langgraph",
    status: "engaged",
    planner_mode: "llm",
    llm_mode: "configured",
    graph_id: "phase_e_cognition_graph",
    version: "phase_e",
    selected_candidate_id: String(selectedCandidateID || "").trim(),
    reason_codes: ["langgraph_summary_available"],
    risk_note: String(riskNote || "").trim(),
    summary: String(summary || "LangGraph 已选择候选方案并生成简要解释。").trim(),
    fallback_reason_code: "",
    fallback_message: "",
  };

  if (csvPath.includes("degraded")) {
    cognition.status = "degraded";
    cognition.reason_codes = ["explain_request_failed"];
    cognition.fallback_reason_code = "explain_request_failed";
    cognition.fallback_message = "LangGraph 已选择候选方案，但 Go 侧保留了确定性降级解释。";
    cognition.summary =
      String(summary || "").trim() || "LangGraph 已选择候选方案，但解释渲染已降级为 Go 侧确定性说明。";
    return cognition;
  }

  if (csvPath.includes("disabled")) {
    cognition.provider = "deterministic";
    cognition.status = "disabled";
    cognition.planner_mode = "fallback";
    cognition.llm_mode = "unavailable";
    cognition.reason_codes = ["deterministic_fallback"];
    cognition.fallback_reason_code = "disabled";
    cognition.fallback_message = "LangGraph sidecar 已禁用，当前继续使用确定性规划。";
    cognition.summary =
      String(summary || "").trim() || "由于 LangGraph 已禁用，当前继续使用确定性规划。";
    return cognition;
  }

  if (csvPath.includes("fallback") || csvPath.includes("offline") || csvPath.includes("no-llm")) {
    cognition.provider = "deterministic";
    cognition.status = "fallback";
    cognition.planner_mode = "fallback";
    cognition.llm_mode = "unavailable";
    cognition.reason_codes = ["deterministic_fallback"];
    cognition.fallback_reason_code = "planner_mode_fallback";
    cognition.fallback_message = "LangGraph 降级路径已启用，当前由确定性规划接管。";
    cognition.summary =
      String(summary || "").trim() || "由于 LangGraph 不可用，当前继续使用确定性规划。";
  }

  return cognition;
}

function buildMockCognitionTraceEvent(taskID, sessionID, seq, cognition, createdAt) {
  const summary =
    String(cognition?.summary || "").trim() ||
    (String(cognition?.fallback_reason_code || "").trim()
      ? `降级原因: ${String(cognition.fallback_reason_code).trim()}`
      : "认知状态已记录。");
  return {
    id: seq,
    session_id: sessionID,
    task_id: taskID,
    seq,
    agent_name: "repair_planner",
    trace_type: "cognition_trace",
    summary,
    payload: {
      phase: "plan_complete",
      provider: String(cognition?.provider || "").trim(),
      status: String(cognition?.status || "").trim(),
      planner_mode: String(cognition?.planner_mode || "").trim(),
      llm_mode: String(cognition?.llm_mode || "").trim(),
      graph_id: String(cognition?.graph_id || "").trim(),
      version: String(cognition?.version || "").trim(),
      selected_candidate_id: String(cognition?.selected_candidate_id || "").trim(),
      reason_codes: asArray(cognition?.reason_codes)
        .map((item) => String(item || "").trim())
        .filter(Boolean),
      fallback_reason_code: String(cognition?.fallback_reason_code || "").trim(),
      summary,
    },
    created_at: createdAt,
  };
}

function buildMockApprovalTraceSummary(trace) {
  const traceTypeCounts = {};
  const agentNames = [];
  for (const item of asArray(trace)) {
    const traceType = String(item?.trace_type || "").trim();
    if (traceType) traceTypeCounts[traceType] = toInt(traceTypeCounts[traceType], 0) + 1;
    const agentName = String(item?.agent_name || "").trim();
    if (agentName && !agentNames.includes(agentName)) agentNames.push(agentName);
  }
  const last = asArray(trace).slice(-1)[0] || {};
  const cognitionEvents = asArray(trace).filter((item) => String(item?.trace_type || "").trim() === "cognition_trace");
  const lastCognition = asObject(cognitionEvents.slice(-1)[0]?.payload);
  return {
    event_count: asArray(trace).length,
    tool_call_count: 2,
    agent_names: agentNames,
    trace_type_counts: traceTypeCounts,
    cognition:
      cognitionEvents.length > 0
        ? {
            event_count: cognitionEvents.length,
            provider: String(lastCognition?.provider || "").trim(),
            status: String(lastCognition?.status || "").trim(),
            last_phase: String(lastCognition?.phase || "").trim(),
            last_summary: String(lastCognition?.summary || cognitionEvents.slice(-1)[0]?.summary || "").trim(),
            planner_mode: String(lastCognition?.planner_mode || "").trim(),
            llm_mode: String(lastCognition?.llm_mode || "").trim(),
            fallback_reason_code: String(lastCognition?.fallback_reason_code || "").trim(),
            reason_codes: asArray(lastCognition?.reason_codes)
              .map((item) => String(item || "").trim())
              .filter(Boolean),
            selected_candidate_id: String(lastCognition?.selected_candidate_id || "").trim(),
          }
        : {},
    last_trace_type: String(last?.trace_type || ""),
    last_trace_summary: String(last?.summary || ""),
  };
}

function buildMockApprovalRequiredResult(taskID, payload) {
  const base = buildMockAgentAutofixResult(taskID, { ...payload, __mock_verdict: "accepted" });
  const context = buildMockApprovalContext(payload, buildMockScanResult(payload));
  const approval = buildMockApprovalResult(context, "required");
  const now = new Date().toISOString();
  const trace = [
    {
      id: 1,
      session_id: base.session.session_id,
      task_id: taskID,
      seq: 1,
      agent_name: "supervisor",
      trace_type: "session_started",
      summary: "智能自动会话已启动",
      payload: { goal: base.session.user_goal },
      created_at: now,
    },
    {
      id: 2,
      session_id: base.session.session_id,
      task_id: taskID,
      seq: 2,
      agent_name: "profile_memory",
      trace_type: "memory_updated",
      summary: "已保存工作区偏好快照与审批上下文",
      payload: {
        workspace_id: context.workspaceID,
        preference_snapshot: cloneSmartPreferenceProfile(context.snapshot),
      },
      created_at: now,
    },
    {
      id: 3,
      session_id: base.session.session_id,
      task_id: taskID,
      seq: 3,
      agent_name: "repair_planner",
      trace_type: "agent_decision",
      summary: "规划器已选择候选方案，并要求写出前审批",
      payload: { selected_source: context.selectedSource, plan_id: base.result.agent.plan_id },
      created_at: now,
    },
    {
      id: 4,
      session_id: base.session.session_id,
      task_id: taskID,
      seq: 4,
      agent_name: "validator",
      trace_type: "validation",
      summary: "预演验证已通过，但执行被审批门禁暂停。",
      payload: {
        status: "accepted",
        phase: "preview",
        before_issue_count: 3,
        after_issue_count: 1,
        resolved_issue_count: 2,
        changed_cell_count: 42,
      },
      created_at: now,
    },
    {
      id: 5,
      session_id: base.session.session_id,
      task_id: taskID,
      seq: 5,
      agent_name: "supervisor",
      trace_type: "approval_requested",
      summary: "写出结果前需要审批，执行已暂停",
      payload: {
        approval_state: {
          status: "required",
          required: true,
          reason_codes: [...context.reasonCodes],
          message: context.message,
          requested_at: now,
          task_id: taskID,
        },
      },
      created_at: now,
    },
  ];
  trace.splice(
    3,
    0,
    buildMockCognitionTraceEvent(taskID, base.session.session_id, 4, asObject(base.result.agent.plan?.cognition), now)
  );
  trace.forEach((item, idx) => {
    item.id = idx + 1;
    item.seq = idx + 1;
  });
  const traceSummary = buildMockApprovalTraceSummary(trace);

  base.status = "succeeded";
  base.result.safety = {
    final_verdict: "approval_required",
    risk_flags: ["approval_required"],
    baseline_scan_summary: {
      issue_count: asArray(base.result.agent.plan?.selected_issue_ids).length || 3,
      high_risk_issue_count: context.riskColumns.length,
      total_issue_score: 123.4,
    },
    post_scan_summary: {},
    rollback_recommendation: {
      action: "wait_for_approval",
      reason: "approval_required",
    },
    rollback_execution: { status: "not_run" },
    rejected_output_snapshot: "",
  };
  base.result.agent.validation = {
    status: "accepted",
    message: context.message,
    can_execute: true,
    preview: {
      status: "accepted",
      message: "预演验证已通过，但写入文件前仍需要审批。",
      before_issue_count: 3,
      after_issue_count: 1,
      resolved_issue_count: 2,
      changed_cell_count: 42,
    },
    post_execute: {},
  };
  base.result.agent.execution = {
    status: "paused",
    auto_mode: true,
    selected_source: context.selectedSource,
    rollback_applied: false,
    output_csv: "",
    post_scan_output_csv: "",
    comparison: {
      before_issue_count: 3,
      after_issue_count: 1,
      resolved_issue_count: 2,
      changed_cell_count: 42,
    },
  };
  base.result.agent.approval = approval;
  base.result.agent.explanation = {
    mode: buildMockExplanationMode(asObject(base.result.agent.plan?.cognition)),
    summary: context.message,
    final_message: "本次运行已暂停在审批门禁。继续将写出结果；取消则不会产生文件改动。",
    short_bullets: [
      `候选来源: ${context.selectedSource}`,
      `原因码: ${context.reasonCodes.join(", ") || "-"}`,
    ],
    reason_codes: [...asArray(base.result.agent.plan?.reason_codes)],
    risk_note: base.result.agent.plan?.risk_note || "",
    cognition: { ...asObject(base.result.agent.plan?.cognition) },
  };
  base.result.agent.trace_summary = traceSummary;
  base.session.status = "awaiting_approval";
  base.session.current_task_id = taskID;
  base.session.context = {
    csv_path: String(payload?.csv_path || ""),
    baseline_scan: base.result.safety.baseline_scan_summary,
    preview_validation: base.result.agent.validation.preview,
    post_scan: {},
    post_validation: {},
    rollback_summary: {},
    final_verdict: "approval_required",
    rejected_output_snapshot: "",
    workspace_id: context.workspaceID,
    preference_snapshot: cloneSmartPreferenceProfile(context.snapshot),
    cognition_state: { ...asObject(base.result.agent.plan?.cognition) },
    approval_state: {
      status: "required",
      required: true,
      reason_codes: [...context.reasonCodes],
      message: context.message,
      requested_at: now,
      task_id: taskID,
    },
    risk_assessment: {
      required: true,
      reason_codes: [...context.reasonCodes],
      candidate_columns: [...context.candidateColumns],
      risk_columns: [...context.riskColumns],
      protected_columns: [...context.protectedColumns],
      time_like_columns: [...context.timeLikeColumns],
      selected_source: context.selectedSource,
      selected_issue_catalog: [...context.selectedIssueCatalog],
      message: context.message,
    },
    selected_issue_catalog: [...context.selectedIssueCatalog],
    candidate_columns: [...context.candidateColumns],
    time_like_columns: [...context.timeLikeColumns],
  };
  base.session.trace_summary = traceSummary;
  base.trace = trace;
  return base;
}

function buildMockApprovalRejectedResult(taskID, payload, existingSession, existingTrace = []) {
  const context = buildMockApprovalContext(payload, buildMockScanResult(payload));
  const approval = buildMockApprovalResult(context, "rejected");
  const trace = [...asArray(existingTrace)];
  trace.push({
    id: trace.length + 1,
    session_id: existingSession.session_id,
    task_id: taskID,
    seq: trace.length + 1,
    agent_name: "supervisor",
    trace_type: "approval_rejected",
    summary: "执行已在审批门禁阶段取消",
    payload: {
      approval_state: {
        status: "rejected",
        required: true,
        decision: "rejected",
        reason_codes: [...context.reasonCodes],
        message: context.message,
        task_id: taskID,
      },
    },
    created_at: new Date().toISOString(),
  });
  const traceSummary = buildMockApprovalTraceSummary(trace);
  const result = {
    csv_path: String(payload?.csv_path || ""),
    safety: {
      final_verdict: "approval_rejected",
      risk_flags: ["approval_rejected"],
      baseline_scan_summary: asObject(existingSession?.context?.baseline_scan),
      post_scan_summary: {},
      rollback_recommendation: {
        action: "keep_source_only",
        reason: "approval_rejected",
      },
      rollback_execution: { status: "not_run" },
      rejected_output_snapshot: "",
    },
    agent: {
      session_id: existingSession.session_id,
      plan_id: existingSession.latest_plan?.plan_id,
      run_mode: "auto",
      goal: String(payload?.user_goal || "scan_and_auto_repair"),
      plan: asObject(existingSession.latest_plan),
      approval,
      explanation: {
        mode: buildMockExplanationMode(asObject(existingSession?.latest_plan?.cognition)),
        summary: "本次运行在写入任何文件前已取消。",
        final_message: "没有写出输出 CSV，也不需要回滚。",
        short_bullets: Array.from(asArray(existingSession?.latest_plan?.explanation_bullets)),
        reason_codes: Array.from(asArray(existingSession?.latest_plan?.reason_codes)),
        risk_note: String(existingSession?.latest_plan?.risk_note || "").trim(),
        cognition: { ...asObject(existingSession?.latest_plan?.cognition) },
      },
      validation: {
        status: "accepted",
        message: context.message,
        can_execute: true,
        preview: asObject(existingSession?.context?.preview_validation),
        post_execute: {},
      },
      execution: {
        status: "skipped",
        auto_mode: true,
        rollback_applied: false,
        output_csv: "",
        post_scan_output_csv: "",
      },
      trace_summary: traceSummary,
      presentation: {},
    },
  };
  return {
    status: "succeeded",
    result,
    session: {
      ...existingSession,
      current_task_id: taskID,
      status: "approval_rejected",
      context: {
        ...asObject(existingSession?.context),
        final_verdict: "approval_rejected",
        approval_state: {
          status: "rejected",
          required: true,
          decision: "rejected",
          reason_codes: [...context.reasonCodes],
          message: context.message,
          task_id: taskID,
        },
      },
      trace_summary: traceSummary,
    },
    trace,
  };
}

function buildMockApprovalApprovedResult(taskID, payload, existingSession, existingTrace = []) {
  const base = buildMockAgentAutofixResult(taskID, { ...payload, __mock_verdict: "accepted" });
  const context = buildMockApprovalContext(payload, buildMockScanResult(payload));
  const approval = buildMockApprovalResult(context, "approved");
  const trace = [...asArray(existingTrace)];
  trace.push({
    id: trace.length + 1,
    session_id: existingSession.session_id,
    task_id: taskID,
    seq: trace.length + 1,
    agent_name: "supervisor",
    trace_type: "approval_granted",
    summary: "审批已通过，会话已恢复执行",
    payload: {
      approval_state: {
        status: "approved",
        required: true,
        decision: "approved",
        reason_codes: [...context.reasonCodes],
        message: context.message,
        task_id: taskID,
      },
    },
    created_at: new Date().toISOString(),
  });
  const traceSummary = buildMockApprovalTraceSummary(trace);
  base.result.agent.session_id = existingSession.session_id;
  base.result.agent.plan_id = existingSession.latest_plan?.plan_id;
  base.result.agent.plan.plan_id = existingSession.latest_plan?.plan_id;
  base.result.agent.approval = approval;
  base.result.agent.trace_summary = traceSummary;
  base.result.agent.explanation = {
    ...asObject(base.result.agent.explanation),
    final_message: "审批已通过，会话已从保存的预演检查点恢复并完成执行。",
  };
  base.session.session_id = existingSession.session_id;
  base.session.root_task_id = existingSession.root_task_id;
  base.session.current_task_id = taskID;
  base.session.status = "completed";
  base.session.latest_plan = {
    ...asObject(existingSession.latest_plan),
    ...asObject(base.session.latest_plan),
    plan_id: existingSession.latest_plan?.plan_id,
  };
  base.session.context = {
    ...asObject(base.session.context),
    ...asObject(existingSession?.context),
    final_verdict: "accepted",
    approval_state: {
      status: "approved",
      required: true,
      decision: "approved",
      reason_codes: [...context.reasonCodes],
      message: context.message,
      task_id: taskID,
    },
    risk_assessment: {
      ...asObject(existingSession?.context?.risk_assessment),
      message: context.message,
    },
  };
  base.session.trace_summary = traceSummary;
  base.trace = trace;
  return base;
}

function buildMockAgentAutofixResult(taskID, payload) {
  const scan = buildMockScanResult(payload);
  const verdict = mockVerdictFromPayload(payload);
  const outputDir = String(payload?.output_dir || buildSmartDefaultOutputDir());
  const sessionID = `mock-session-${taskID}`;
  const planID = `mock-plan-${taskID}`;
  const beforeIssueCount = asArray(scan?.issues).length;
  const afterIssueCount =
    verdict === "accepted" ? 1 : verdict === "validation_rejected" ? beforeIssueCount : beforeIssueCount + 1;
  const postRiskScore =
    verdict === "accepted" ? 32.4 : verdict === "validation_rejected" ? 54.2 : 66.8;
  const scanIssues = asArray(scan?.issues).map((item) => asObject(item));
  const autoRepairIssueIDs = scanIssues
    .filter((item) => ["missing_values", "rare_category"].includes(String(item?.issue_type || "")))
    .map((item) => String(item?.issue_id || "").trim())
    .filter(Boolean);
  const cautiousIssueIDs = scanIssues
    .filter((item) => String(item?.issue_type || "") === "numeric_outlier")
    .map((item) => String(item?.issue_id || "").trim())
    .filter(Boolean);
  const manualReviewIssueIDs = scanIssues
    .filter((item) => ["duplicate_record", "cross_column_consistency"].includes(String(item?.issue_type || "")))
    .map((item) => String(item?.issue_id || "").trim())
    .filter(Boolean);
  const blockedIssueIDs = scanIssues
    .filter((item) => {
      const issueType = String(item?.issue_type || "");
      return !["missing_values", "rare_category", "numeric_outlier", "duplicate_record", "cross_column_consistency"].includes(issueType);
    })
    .map((item) => String(item?.issue_id || "").trim())
    .filter(Boolean);
  const selectedSource = verdict === "accepted" ? "hybrid" : verdict === "validation_rejected" ? "rule" : "gower";
  const outputCSV = `${outputDir}/mock.smart.repaired.csv`;
  const presentationArtifact = `${outputDir}/presentation.json`;
  const rollbackManifest = `${outputDir}/rollback/manifest.v2.json`;
  const totalCellsModified = verdict === "accepted" ? 42 : 48;
  const rejectedSnapshot =
    verdict === "rolled_back" || verdict === "rollback_failed" ? `${outputDir}/rollback/${taskID}.rejected.csv` : "";
  const rollbackExecution =
    verdict === "rolled_back"
      ? { status: "executed", manifest_path: rollbackManifest, restore_target: outputCSV }
      : verdict === "rollback_failed"
      ? { status: "failed", manifest_path: rollbackManifest, reason: "模拟回滚失败" }
      : {};
  const safety = {
    final_verdict: verdict,
    risk_flags:
      verdict === "accepted"
        ? ["validated"]
        : verdict === "validation_rejected"
        ? ["preview_rejected"]
        : ["post_validation_failed", verdict],
    baseline_scan_summary: {
      issue_count: beforeIssueCount,
      high_risk_issue_count: 1,
      total_issue_score: 123.4,
    },
    post_scan_summary: {
      issue_count: afterIssueCount,
      high_risk_issue_count: verdict === "accepted" ? 0 : 1,
      total_issue_score: postRiskScore,
    },
    rollback_recommendation: {
      action: verdict === "accepted" ? "keep_output" : "restore_output_csv",
      reason: verdict === "accepted" ? "validated" : "risk_not_reduced",
    },
    rollback_execution: rollbackExecution,
    rejected_output_snapshot: rejectedSnapshot,
  };
  const validation = {
    status: verdict === "validation_rejected" ? "rejected" : "accepted",
    message:
      verdict === "accepted"
        ? "预演验证已通过，执行后验证也已接纳结果。"
        : verdict === "validation_rejected"
        ? "预演验证拒绝了自动执行。"
        : "执行后验证检测到风险升高，并触发回滚。",
    can_execute: verdict !== "validation_rejected",
    preview: {
      status: verdict === "validation_rejected" ? "rejected" : "accepted",
      message: verdict === "validation_rejected" ? "已解决问题数未优于基线。" : "候选方案可安全执行。",
      before_issue_count: beforeIssueCount,
      after_issue_count: verdict === "validation_rejected" ? beforeIssueCount : Math.max(1, beforeIssueCount - 1),
      resolved_issue_count: verdict === "validation_rejected" ? 0 : beforeIssueCount - 1,
      changed_cell_count: verdict === "validation_rejected" ? 0 : 42,
    },
    post_execute:
      verdict === "validation_rejected"
        ? {}
        : {
            status: verdict === "accepted" ? "accepted" : "rejected",
            accepted: verdict === "accepted",
            verdict: verdict === "accepted" ? "accept" : "reject",
            phase: "post_execute",
            message:
              verdict === "accepted"
                ? "复扫确认问题数与总问题分均已下降。"
                : "复扫未达到接纳阈值。",
            before_issue_count: beforeIssueCount,
            after_issue_count: afterIssueCount,
            resolved_issue_count: Math.max(0, beforeIssueCount - afterIssueCount),
            total_cells_modified: totalCellsModified,
            changed_cell_count: totalCellsModified,
            before_total_issue_score: 123.4,
            after_total_issue_score: postRiskScore,
            risk_notes: verdict === "accepted" ? [] : ["issue_count_increased"],
            risk_flags: verdict === "accepted" ? [] : ["issue_count_increased"],
            rollback_recommended: verdict !== "accepted",
            explanation:
              verdict === "accepted"
                ? `问题数从 ${beforeIssueCount} 变为 ${afterIssueCount}，验证门禁未发现风险。`
                : `验证门禁因 issue_count_increased 判定拒绝；问题数从 ${beforeIssueCount} 变为 ${afterIssueCount}。`,
          },
  };
  const execution =
    verdict === "validation_rejected"
      ? {
          status: "skipped",
          auto_mode: true,
          rollback_applied: false,
          output_csv: "",
          post_scan_output_csv: "",
        }
      : {
          status: "executed",
          auto_mode: true,
          selected_source: selectedSource,
          output_csv: outputCSV,
          post_scan_output_csv: outputCSV,
          rollback_applied: verdict === "rolled_back",
          applied_issue_count: verdict === "accepted" ? 3 : 2,
          total_cells_modified: totalCellsModified,
          rollback: rollbackExecution,
          comparison: {
            before_issue_count: beforeIssueCount,
            after_issue_count: afterIssueCount,
            resolved_issue_count: Math.max(0, beforeIssueCount - afterIssueCount),
            changed_cell_count: totalCellsModified,
          },
        };
  const plan = {
    plan_id: planID,
    selected_candidate_id: `${selectedSource}-candidate`,
    selected_source: selectedSource,
    selected_issue_ids: autoRepairIssueIDs,
    auto_repair_issue_ids: autoRepairIssueIDs,
    cautious_issue_ids: cautiousIssueIDs,
    manual_review_issue_ids: manualReviewIssueIDs,
    blocked_issue_ids: blockedIssueIDs,
    skipped_issues: [
      ...cautiousIssueIDs.map((issueID) => ({ issue_id: issueID, issue_type: "numeric_outlier", reason: "cautious_review_required" })),
      ...manualReviewIssueIDs.map((issueID) => ({ issue_id: issueID, reason: "manual_review_required" })),
      ...blockedIssueIDs.map((issueID) => ({ issue_id: issueID, reason: "blocked_by_deterministic_planner" })),
    ],
    reasoning_summary:
      verdict === "accepted"
        ? "Hybrid 候选在保持改动量可接受的同时最大幅降低了问题数。"
        : "安全策略保留了确定性证据，并阻止或回滚了可信度较低的路径。",
    user_explanation: "系统优先比较 rule、gower 与 hybrid 三个候选，再根据验证结果决定执行与否。",
  };
  const cognition = buildMockCognitionState(
    payload,
    plan.selected_candidate_id,
    plan.reasoning_summary,
    "写入前仍以 Go 侧验证为最终准入依据。"
  );
  plan.reason_codes = [...asArray(cognition.reason_codes)];
  plan.risk_note = String(cognition.risk_note || "").trim();
  plan.explanation_bullets = [
    `认知状态: ${smartCognitionLabel(cognition) || "deterministic"}`,
    smartCognitionSummaryText(cognition),
    smartCognitionFallbackText(cognition),
  ].filter(Boolean);
  plan.cognition = cognition;
  let traceSummary = {
    total_events: verdict === "validation_rejected" ? 8 : 12,
    tool_calls: verdict === "validation_rejected" ? 2 : 4,
    validation_events: verdict === "validation_rejected" ? 1 : 2,
    rollback_events: verdict === "rolled_back" || verdict === "rollback_failed" ? 2 : 0,
  };
  const presentation = buildMockPresentationBundle(
    "agent",
    verdict,
    verdict === "accepted" ? "智能闭环已完成" : "智能闭环已结束",
    verdict === "accepted"
      ? "系统已自动完成扫描、修复、复扫与验证，结果满足接纳条件。"
      : verdict === "validation_rejected"
      ? "系统在预演阶段拒绝了自动执行，因此没有写出不可信结果。"
      : verdict === "rolled_back"
      ? "系统执行后检测到风险未下降，已自动回滚输出产物。"
      : "系统尝试自动回滚，但回滚过程失败，需要人工介入。",
    [
      { id: "issues", label: "问题数变化", value: `${beforeIssueCount} -> ${afterIssueCount}`, tone: verdict === "accepted" ? "success" : "warning" },
      { id: "source", label: "采用来源", value: selectedSource, tone: "neutral" },
      { id: "verdict", label: "最终结论", value: verdict, tone: smartVerdictTone(verdict) },
    ],
    [
      {
        id: "overview",
        title: "总体结论",
        body:
          verdict === "accepted"
            ? "本次任务通过自动闭环完成了扫描、修复和后验验证。"
            : "本次任务保留了验证优先和回滚优先的安全边界。",
        bullets: [
          "默认入口为 agent.session.auto",
          "任务包含自动验证与回滚保护",
          `最终采用 ${selectedSource} 候选`,
        ],
        evidence_refs: ["validation", "trace_summary"],
      },
      {
        id: "risk_and_safety",
        title: "风险与安全",
        body:
          verdict === "accepted"
            ? "复扫显示问题数与风险分数均下降。"
            : verdict === "validation_rejected"
            ? "系统在 preview gate 阶段阻止了不可信执行。"
            : verdict === "rolled_back"
            ? "系统在 post-validation 阶段判定结果不安全，并已回滚。"
            : "系统已尝试回滚，但回滚执行失败，需要人工复核。",
        bullets: [`风险标记: ${safety.risk_flags.join(", ")}`],
        evidence_refs: ["safety", "rollback"],
      },
    ],
    [
      {
        id: "before_after_issue_comparison",
        kind: "comparison_bar",
        title: "问题数前后对比",
        subtitle: "自动模式的核心结果口径",
        empty_state: "暂无比较数据",
        data: {
          series: [{ label: "issues", before: beforeIssueCount, after: afterIssueCount, delta: afterIssueCount - beforeIssueCount }],
        },
      },
      {
        id: "validation_verdict_timeline",
        kind: "timeline",
        title: "验证时间线",
        subtitle: "preview 与 post-execute 的关键判断",
        empty_state: "暂无验证事件",
        data: {
          events:
            verdict === "validation_rejected"
              ? [{ label: "preview", value: "rejected", tone: "warning", hint: "自动执行被安全门禁阻止" }]
              : [
                  { label: "preview", value: "accepted", tone: "success", hint: "候选方案允许执行" },
                  { label: "post_execute", value: verdict === "accepted" ? "accepted" : "rejected", tone: verdict === "accepted" ? "success" : "warning", hint: verdict === "accepted" ? "结果被接纳" : "结果被回滚" },
                ],
        },
      },
    ]
  );

  const result = {
    csv_path: String(payload?.csv_path || ""),
    safety,
    agent: {
      session_id: sessionID,
      plan_id: planID,
      run_mode: "auto",
      goal: String(payload?.user_goal || "扫描并自动修复"),
      plan,
      explanation: {
        mode: buildMockExplanationMode(cognition),
        summary: plan.reasoning_summary,
        final_message:
          verdict === "accepted"
            ? "系统已自动交付可信结果。"
            : "系统保留了安全边界，并给出了可审计的结束原因。",
      },
      explanation: {
        ...asObject({
          mode: buildMockExplanationMode(cognition),
          summary: plan.reasoning_summary,
          /*
          final_message: asObject({
            accepted: "绯荤粺宸茶嚜鍔ㄤ氦浠樺彲淇＄粨鏋溿€?,
            fallback: "绯荤粺淇濈暀浜嗗畨鍏ㄨ竟鐣岋紝骞剁粰鍑轰簡鍙璁＄殑缁撴潫鍘熷洜銆?,
          })[verdict === "accepted" ? "accepted" : "fallback"],
          */
          short_bullets: Array.from(asArray(plan.explanation_bullets)),
          final_message:
            verdict === "accepted"
              ? "本次运行已完成，并产出被接纳的结果。"
              : "本次运行保留了安全边界，并给出了可审计的结束原因。",
          reason_codes: Array.from(asArray(plan.reason_codes)),
          risk_note: plan.risk_note,
          cognition,
        }),
      },
      approval: {
        status: "not_required",
        required: false,
        reason_codes: [],
        risk_columns: [],
        protected_columns: [],
        time_like_columns: [],
        message: "当前没有启用审批门禁。",
      },
      validation,
      execution,
      trace_summary: traceSummary,
      presentation,
    },
  };

  const session = {
    session_id: sessionID,
    root_task_id: taskID,
    current_task_id: taskID,
    status: verdict === "accepted" ? "completed" : verdict === "rolled_back" ? "rolled_back" : verdict === "rollback_failed" ? "rollback_failed" : "validation_rejected",
    mode: "auto",
    user_goal: String(payload?.user_goal || "扫描并自动修复"),
    context: {
      csv_path: String(payload?.csv_path || ""),
      baseline_scan: safety.baseline_scan_summary,
      preview_validation: validation.preview,
      post_scan: safety.post_scan_summary,
      post_validation: validation.post_execute,
      rollback_summary: rollbackExecution,
      final_verdict: verdict,
      rejected_output_snapshot: rejectedSnapshot,
      workspace_id: resolveSmartWorkspaceID(payload?.workspace_id, payload?.csv_path),
      preference_snapshot: cloneSmartPreferenceProfile(asObject(payload?.user_preferences)),
      cognition_state: { ...cognition },
    },
    latest_plan: plan,
    presentation,
    presentation_artifact: presentationArtifact,
    trace_summary: traceSummary,
  };

  const trace = [
    {
      id: 1,
      session_id: sessionID,
      task_id: taskID,
      seq: 1,
      agent_name: "supervisor",
      trace_type: "session_started",
      summary: "智能自动会话已启动",
      payload: { goal: session.user_goal },
      created_at: new Date().toISOString(),
    },
    {
      id: 2,
      session_id: sessionID,
      task_id: taskID,
      seq: 2,
      agent_name: "repair_planner",
      trace_type: "agent_decision",
      summary: "规划器已选择最佳候选方案",
      payload: { selected_source: selectedSource, plan_id: planID },
      created_at: new Date().toISOString(),
    },
    {
      id: 3,
      session_id: sessionID,
      task_id: taskID,
      seq: 3,
      agent_name: "validator",
      trace_type: "validation",
      summary: validation.preview.message,
      payload: { ...validation.preview, phase: "preview" },
      created_at: new Date().toISOString(),
    },
  ];
  trace.splice(2, 0, buildMockCognitionTraceEvent(taskID, sessionID, 3, cognition, new Date().toISOString()));
  trace.forEach((item, idx) => {
    item.id = idx + 1;
    item.seq = idx + 1;
  });
  if (verdict !== "validation_rejected") {
    trace.push({
      id: 4,
      session_id: sessionID,
      task_id: taskID,
      seq: 4,
      agent_name: "validator",
      trace_type: "validation",
      summary: validation.post_execute.message,
      payload: { ...validation.post_execute, phase: "post_execute" },
      created_at: new Date().toISOString(),
    });
  }
  if (verdict === "rolled_back" || verdict === "rollback_failed") {
    trace.push({
      id: 5,
      session_id: sessionID,
      task_id: taskID,
      seq: 5,
      agent_name: "validator",
      trace_type: "rollback_decision",
      summary: "不安全输出需要回滚",
      payload: { verdict },
      created_at: new Date().toISOString(),
    });
    trace.push({
      id: 6,
      session_id: sessionID,
      task_id: taskID,
      seq: 6,
      agent_name: "validator",
      trace_type: "rollback_executed",
      summary: verdict === "rolled_back" ? "回滚已恢复修复产物" : "已尝试回滚但执行失败",
      payload: rollbackExecution,
      created_at: new Date().toISOString(),
    });
  }
  traceSummary = buildMockApprovalTraceSummary(trace);
  result.agent.trace_summary = traceSummary;
  session.trace_summary = traceSummary;

  return {
    status: verdict === "accepted" ? "succeeded" : "failed",
    result,
    session,
    trace,
  };
}

async function mockRunAgentAutofixSession(payload) {
  const id = `mock-agent-task-${Date.now()}`;
  const approvalContext = buildMockApprovalContext(payload, buildMockScanResult(payload));
  const built = approvalContext.required ? buildMockApprovalRequiredResult(id, payload) : buildMockAgentAutofixResult(id, payload);
  state.mockTasks.set(id, {
    id,
    payload: { ...payload },
    action: "agent.session.auto",
    createdAt: Date.now(),
    canceled: false,
    finalStatus: built.status,
    finalResult: built.result,
    sessionID: built.session.session_id,
  });
  state.mockAgentSessions.set(built.session.session_id, built.session);
  state.mockAgentTrace.set(built.session.session_id, built.trace);
  return {
    id,
    status: "pending",
    request: { action: "agent.session.auto", payload: { ...payload } },
    response: {},
    error: "",
  };
}

async function mockApproveAgentSession(payload) {
  const sessionID = String(payload?.session_id || "").trim();
  const decision = String(payload?.decision || "").trim().toLowerCase();
  const existingSession = asObject(state.mockAgentSessions.get(sessionID));
  if (!sessionID || Object.keys(existingSession).length === 0) {
    throw new Error(`未找到模拟会话: ${sessionID}`);
  }
  const existingTrace = asArray(state.mockAgentTrace.get(sessionID));
  const originalTask = state.mockTasks.get(existingSession.root_task_id);
  const effectivePayload = {
    ...asObject(originalTask?.payload),
    ...asObject(payload),
    csv_path: String(asObject(originalTask?.payload)?.csv_path || existingSession?.context?.csv_path || ""),
    user_preferences:
      asObject(payload?.user_preferences) && Object.keys(asObject(payload?.user_preferences)).length > 0
        ? asObject(payload?.user_preferences)
        : asObject(existingSession?.context?.preference_snapshot),
    workspace_id: String(payload?.workspace_id || existingSession?.context?.workspace_id || "").trim(),
  };
  const taskID = `mock-agent-approval-${Date.now()}`;
  const built =
    decision === "reject"
      ? buildMockApprovalRejectedResult(taskID, effectivePayload, existingSession, existingTrace)
      : buildMockApprovalApprovedResult(taskID, effectivePayload, existingSession, existingTrace);

  state.mockTasks.set(taskID, {
    id: taskID,
    payload: { ...effectivePayload, session_id: sessionID, plan_id: payload?.plan_id, decision },
    action: "agent.session.approve",
    createdAt: Date.now(),
    canceled: false,
    finalStatus: built.status,
    finalResult: built.result,
    sessionID,
  });
  state.mockAgentSessions.set(sessionID, built.session);
  state.mockAgentTrace.set(sessionID, built.trace);
  return {
    id: taskID,
    status: "pending",
    request: {
      action: "agent.session.approve",
      payload: { ...payload, session_id: sessionID, decision },
    },
    response: {},
    error: "",
  };
}

async function mockGetAgentPreferences(workspaceID, csvPath) {
  const resolved = resolveSmartWorkspaceID(workspaceID, csvPath);
  const existing = state.mockPreferenceStore.get(resolved);
  if (existing) {
    return {
      workspace_id: resolved,
      profile: cloneSmartPreferenceProfile(existing.profile),
      updated_at: existing.updated_at,
    };
  }
  return {
    workspace_id: resolved,
    profile: defaultSmartPreferenceProfile(),
    updated_at: "",
  };
}

async function mockSaveAgentPreferences(payload) {
  const resolved = resolveSmartWorkspaceID(payload?.workspace_id, payload?.csv_path);
  const profileSource = Object.keys(asObject(payload?.profile)).length > 0 ? payload.profile : payload;
  const profile = cloneSmartPreferenceProfile(profileSource);
  const record = {
    workspace_id: resolved,
    profile,
    updated_at: new Date().toISOString(),
  };
  state.mockPreferenceStore.set(resolved, record);
  return {
    workspace_id: resolved,
    profile: cloneSmartPreferenceProfile(profile),
    updated_at: record.updated_at,
  };
}

async function mockGetAgentSession(sessionID) {
  return state.mockAgentSessions.get(sessionID) || {};
}

async function mockListAgentTrace(sessionID) {
  return state.mockAgentTrace.get(sessionID) || [];
}

async function mockGetTaskStatus(taskId) {
  const mock = state.mockTasks.get(taskId);
  if (!mock) throw new Error(`未找到模拟任务: ${taskId}`);

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

  if (mock.action === "agent.session.auto" || mock.action === "agent.session.approve") {
    return {
      id: taskId,
      status: mock.finalStatus,
      request: { action: mock.action, payload: mock.payload },
      response: {
        task_id: taskId,
        status: mock.finalStatus === "succeeded" ? "ok" : "error",
        result: mock.finalResult,
        error:
          mock.finalStatus === "succeeded"
            ? null
            : {
                code: String(asObject(mock.finalResult?.safety)?.final_verdict || "AGENT_VALIDATION_REJECTED").toUpperCase(),
                message: "模拟智能修复已在安全门禁保护下完成。",
              },
        timestamp: new Date().toISOString(),
        duration_ms: 3200,
      },
      error: mock.finalStatus === "succeeded" ? "" : "AGENT_AUTOFIX_GUARDED_RESULT",
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
          message: "issue_ids 为空",
          details: { suggestion: "修复前请至少选择一个问题。" },
        },
      },
      error: "INVALID_INPUT: issue_ids 为空",
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
  if (normalized === "repair_batch" || normalized === "repair_with_gower") return INTENT_REPAIR;
  if (normalized === "agent.session.auto" || normalized === "agent.session.approve") {
    return INTENT_AGENT_AUTO;
  }
  if (normalized === "agent.session.plan" || normalized === "agent.session.execute") {
    return INTENT_REPAIR;
  }
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
  if (intent === INTENT_AGENT_AUTO) return "智能闭环完成，正在整理验证结果与审计轨迹。";
  if (intent === INTENT_REPAIR) return "批量修复完成，正在整理输出结果。";
  if (intent === INTENT_TRAIN) return "训练完成，正在整理模型结果。";
  return "检测完成，正在整理异常摘要。";
}

function renderPhaseHints(intent, status, task = null) {
  if (!phaseHints) return;
  const normalizedStatus = String(status || "idle").toLowerCase();
  const hints = extractProgressStages(task, intent);
  const progress = getTaskProgress(task);
  const currentStage = normalizeStageLabel(progress?.current_stage || hints[0] || "");
  const currentIndex = hints.findIndex((item) => item === currentStage);

  if (normalizedStatus === "idle" || hints.length === 0) {
    phaseHints.innerHTML = "<li>等待任务开始</li>";
    return;
  }

  const rows = [];
  for (let i = 0; i < hints.length; i += 1) {
    const cls = stageStateForTimeline(hints[i], i, currentStage, currentIndex, normalizedStatus);
    rows.push(`<li class="${cls}">${escapeHtml(hints[i])}</li>`);
  }
  phaseHints.innerHTML = rows.join("");
}

function buildTaskMessage(task, intent) {
  const status = String(task?.status || "idle").toLowerCase();
  const progress = getTaskProgress(task);
  const lastMessage = String(progress?.last_message || "").trim();

  if (status === "pending") return "任务已提交，等待执行槽位...";
  if (status === "running") return lastMessage || runningHintForIntent(intent, 0);
  if (status === "succeeded") return completedHintForIntent(intent);
  if (status === "canceled") return "任务已取消。";
  if (status === "timed_out") return "任务已超时，请适当增大超时参数。";
  if (status === "failed") {
    const failureMessage = String(progress?.failure?.message || "").trim();
    if (failureMessage) return `任务失败：${normalizeReadableErrorText(failureMessage)}`;
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
  const progress = getTaskProgress(task);
  const progressPercent = status === "succeeded" ? 100 : toInt(progress?.progress_percent, STATUS_PROGRESS[status] ?? 0);
  const message = buildTaskMessage(task, taskIntent);
  setStatus(status, message, progressPercent);
  renderPhaseHints(taskIntent, status, task);
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
      renderScanResult(task?.response?.result, task);
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
      renderRepairResult(task?.response?.result, task);
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

function appendProgressEvents(task, taskId) {
  const progress = getTaskProgress(task);
  const events = asArray(progress?.events).map((item) => asObject(item));
  if (events.length === 0) return;

  for (const event of events) {
    const key = `${taskId}|${toInt(event?.at_ms, 0)}|${String(event?.stage || "")}|${String(event?.phase || "")}|${String(event?.message || "")}`;
    if (state.seenProgressEventKeys.has(key)) continue;
    state.seenProgressEventKeys.add(key);

    const stage = String(event?.stage || "未知阶段");
    const phase = String(event?.phase || "info").toLowerCase();
    const msg = String(event?.message || "").trim();
    const file = String(event?.file || "").trim();
    const column = String(event?.column || "").trim();
    const rule = String(event?.rule || "").trim();
    const errorCode = String(event?.error_code || "").trim();
    const scopeParts = [];
    if (file) scopeParts.push(`文件:${shortPath(file)}`);
    if (column) scopeParts.push(`列:${column}`);
    if (rule) scopeParts.push(`规则:${rule}`);
    if (errorCode) scopeParts.push(`错误码:${errorCode}`);

    const phaseText =
      phase === "start"
        ? "开始"
        : phase === "complete" || phase === "done" || phase === "success"
        ? "完成"
        : phase === "error"
        ? "失败"
        : "进行中";
    const content = msg || `${stage}${phaseText}`;
    const scopeText = scopeParts.length > 0 ? ` (${scopeParts.join(" | ")})` : "";
    addEvent(`阶段[${stage}] ${phaseText}: ${content}${scopeText}`, taskId);
  }
}

async function pollTask(taskId, intent) {
  const token = Date.now();
  state.pollingToken = token;
  let lastStatus = "";

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
    appendProgressEvents(snapshot, taskId);

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
  state.seenProgressEventKeys = new Set();
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

async function refreshColumnsForCsv(csvPath, source = "路径变更") {
  const path = String(csvPath || "").trim();
  if (!path) {
    setTargetOptions([]);
    renderConfigSidebar();
    await loadSmartPreferencesForCSV("");
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
  await loadSmartPreferencesForCSV(path);
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

function updateContinueLatestButton() {
  if (!continueLatestBtn) return;
  const task = state.recentTaskCandidate;
  if (!task || !task?.id) {
    continueLatestBtn.disabled = true;
    continueLatestBtn.textContent = "查看最近结果";
    return;
  }

  continueLatestBtn.disabled = false;
  const running = !isTerminalTaskStatus(task?.status);
  if (isAgentAutoAction(task?.request?.action) && running) {
    continueLatestBtn.textContent = "继续最近任务";
    return;
  }
  if (isAgentAutoAction(task?.request?.action)) {
    continueLatestBtn.textContent = "查看最近智能结果";
    return;
  }
  continueLatestBtn.textContent = "打开最近工作台任务";
}

async function syncSmartFileSelection(csvPath, source = "smart-home") {
  const path = String(csvPath || "").trim();
  state.smartDraft.csvPath = path;
  if (csvPathInput) csvPathInput.value = path;
  syncSmartDraftToClassicInputs();
  if (!path) {
    setTargetOptions([]);
    await loadSmartPreferencesForCSV("");
    renderSmartHome();
    return;
  }

  try {
    const before = getTargetColumn();
    const columns = await apiListCsvColumns(path);
    setTargetOptions(columns, before);
    addEvent(`智能模式已读取列信息 (${source})`, state.currentTaskId);
  } catch (err) {
    setTargetOptions([]);
    addEvent(`智能模式读取列失败: ${normalizeReadableErrorText(String(err))}`);
  }
  renderConfigSidebar();
  await loadSmartPreferencesForCSV(path);
  renderSmartHome();
}

function collectSmartAutofixPayload() {
  const csvPath = getSmartCsvPath();
  const outputDir = getSmartOutputDir();
  const workspaceID =
    String(state.smartPreferences?.workspaceID || "").trim() ||
    resolveSmartWorkspaceID("", csvPath);
  return {
    csv_path: csvPath,
    user_goal: "扫描并自动修复",
    output_dir: outputDir,
    workspace_id: workspaceID,
    user_preferences: buildSmartUserPreferencesPayload(),
    timeout_ms: toInt(timeoutInput?.value, 90000),
  };
}

function validateSmartAutofixPayload(payload) {
  if (!String(payload?.csv_path || "").trim()) return "CSV 文件路径不能为空。";
  if (!String(payload?.output_dir || "").trim()) return "输出目录不能为空。";
  if (!Number.isInteger(payload?.timeout_ms) || payload.timeout_ms < 1000) {
    return "超时必须是 >= 1000 的整数(ms)。";
  }
  return "";
}

async function pollSmartTask(taskId) {
  const token = Date.now();
  state.pollingToken = token;

  while (state.pollingToken === token && state.currentTaskId === taskId) {
    let snapshot;
    try {
      snapshot = await apiGetTaskStatus(taskId);
    } catch (err) {
      setRunningUi(false);
      const message = `状态轮询失败: ${String(err)}`;
      showError(message, "请检查后端连接后重试。");
      renderSmartSafetyBanner("rollback_failed", message);
      if (smartResultConclusion) smartResultConclusion.textContent = message;
      renderSmartTracePlaceholder(message);
      setShellView(VIEW_SMART_RESULT);
      addEvent(message, taskId);
      return null;
    }

    state.currentTask = snapshot;
    state.runningIntent = INTENT_AGENT_AUTO;
    setTaskId(snapshot?.id || "");
    renderTask(snapshot, INTENT_AGENT_AUTO);
    const status = String(snapshot?.status || "").toLowerCase();
    appendProgressEvents(snapshot, taskId);

    if (isTerminalTaskStatus(status)) {
      setRunningUi(false);
      renderSmartResult(snapshot, { skipHydrate: false });
      return snapshot;
    }

    renderSmartRun(snapshot);
    await delay(450);
  }
  return null;
}

async function startAgentAutofixTask(payload) {
  clearError();
  state.currentTaskId = "";
  state.currentTask = null;
  state.runningIntent = INTENT_AGENT_AUTO;
  state.taskStartAtMS = Date.now();
  state.lastRunningHint = "";
  state.seenProgressEventKeys = new Set();
  state.smartSessionSnapshot = null;
  state.smartTraceEvents = [];
  setRunningUi(true);
  setShellView(VIEW_SMART_RUN);
  renderSmartTracePlaceholder("等待任务轨迹。");

  let submitted;
  try {
    submitted = await apiRunAgentAutofixSession(payload);
  } catch (err) {
    setRunningUi(false);
    const message = `智能任务启动失败: ${String(err)}`;
    showError(message, "可切换到高级工作台继续使用经典流程。");
    renderSmartModeBanner(message, "warning");
    setShellView(VIEW_SMART_HOME);
    addEvent(message);
    return null;
  }

  state.currentTaskId = String(submitted?.id || "");
  state.currentTask = submitted;
  state.recentTaskCandidate = submitted;
  updateContinueLatestButton();
  renderTask(submitted, INTENT_AGENT_AUTO);
  renderSmartRun(submitted);
  addEvent(`智能任务已提交: ${state.currentTaskId}`, state.currentTaskId);
  return pollSmartTask(state.currentTaskId);
}

async function startSmartApprovalDecision(decision) {
  const task = state.currentTask;
  const result = asObject(task?.response?.result);
  const agentBlock = asObject(result?.agent);
  const session = asObject(state.smartSessionSnapshot);
  const sessionID = String(agentBlock?.session_id || session?.session_id || "").trim();
  const planID = String(agentBlock?.plan_id || asObject(agentBlock?.plan)?.plan_id || asObject(session?.latest_plan)?.plan_id || "").trim();
  if (!sessionID || !planID) {
    showError("审批恢复失败: 缺少 session_id 或 plan_id");
    return null;
  }

  clearError();
  state.currentTaskId = "";
  state.currentTask = null;
  state.runningIntent = INTENT_AGENT_AUTO;
  state.taskStartAtMS = Date.now();
  state.lastRunningHint = "";
  state.seenProgressEventKeys = new Set();
  state.smartSessionSnapshot = null;
  state.smartTraceEvents = [];
  setRunningUi(true);
  setShellView(VIEW_SMART_RUN);
  renderSmartTracePlaceholder("正在提交审批决定并恢复任务...");

  let submitted;
  try {
    submitted = await apiApproveAgentSession({
      session_id: sessionID,
      plan_id: planID,
      decision,
      timeout_ms: toInt(timeoutInput?.value, 90000),
    });
  } catch (err) {
    setRunningUi(false);
    const message = `审批操作失败: ${normalizeReadableErrorText(String(err))}`;
    showError(message);
    renderSmartTracePlaceholder(message);
    setShellView(VIEW_SMART_RESULT);
    return null;
  }

  state.currentTaskId = String(submitted?.id || "");
  state.currentTask = submitted;
  state.recentTaskCandidate = submitted;
  updateContinueLatestButton();
  renderTask(submitted, INTENT_AGENT_AUTO);
  renderSmartRun(submitted);
  addEvent(`审批决定已提交: ${decision}`, state.currentTaskId);
  return pollSmartTask(state.currentTaskId);
}

async function startSmartAutofixWorkflow() {
  if (!isSmartAutofixAvailable()) {
    renderSmartModeBanner("当前环境无法直接调用 RunAgentAutofixSession，已为你保留高级工作台入口。", "warning");
    openAdvancedWorkspace();
    return;
  }

  const payload = collectSmartAutofixPayload();
  const invalid = validateSmartAutofixPayload(payload);
  if (invalid) {
    renderSmartModeBanner(invalid, "warning");
    return;
  }

  state.smartDraft.csvPath = String(payload.csv_path || "");
  state.smartDraft.outputDir = String(payload.output_dir || "");
  syncSmartDraftToClassicInputs();
  await startAgentAutofixTask(payload);
}

async function resumeLatestTaskFromNav() {
  if (!state.recentTaskCandidate) {
    await loadRecentHistory();
  }
  const task = state.recentTaskCandidate;
  if (!task || !task?.id) return;
  if (isAgentAutoAction(task?.request?.action)) {
    restoreRecentTask(task, { fromHistory: true });
    return;
  }
  restoreRecentTask(task, { fromHistory: true });
  openAdvancedWorkspace();
}

async function loadRecentHistory() {
  try {
    const tasks = await apiListTaskHistory(10);
    const candidate = pickRecentHistoryCandidate(tasks);
    state.recentTaskCandidate = candidate;
    updateContinueLatestButton();
    if (!candidate || !candidate?.id) return;
    addEvent(`已识别最近任务: ${candidate?.id || "-"}`, candidate?.id || "");
    if (isAgentAutoAction(candidate?.request?.action)) {
      restoreRecentTask(candidate, { fromHistory: true });
    }
  } catch (err) {
    addEvent(`加载历史任务失败: ${String(err)}`);
  }
}

function resetSmartSurface() {
  state.currentTaskId = "";
  state.currentTask = null;
  state.pollingToken = Date.now();
  state.runningIntent = "";
  state.taskStartAtMS = 0;
  state.lastRunningHint = "";
  state.smartSessionSnapshot = null;
  state.smartTraceEvents = [];
  setRunningUi(false);
  clearError();
  renderSmartSafetyBanner("", "");
  if (smartResultConclusion) smartResultConclusion.textContent = "系统完成后会在这里解释做了什么以及结果是否可信。";
  if (smartResultSummary) renderDescriptionList(smartResultSummary, []);
  renderArtifactList(smartArtifactList, [], "任务完成后会在这里展示输出文件、presentation.json 和 rollback manifest。");
  if (smartApprovalCard) smartApprovalCard.classList.add("hidden");
  if (smartApprovalMessage) smartApprovalMessage.textContent = "当前结果未触发额外审批。";
  if (smartApprovalSummary) renderDescriptionList(smartApprovalSummary, []);
  if (smartApprovalReasons) renderCompactList(smartApprovalReasons, ["暂无。"]);
  if (smartApprovalPreferences) renderDescriptionList(smartApprovalPreferences, []);
  if (smartReasoningBody) smartReasoningBody.innerHTML = "";
  renderSmartTracePlaceholder("等待会话轨迹。");
  if (smartResultRaw) smartResultRaw.textContent = "{}\n";
  renderPresentationBundle(smartResultPresentation, null);
  renderCompactList(smartTrustChecks, ["等待任务结果。"]);
  setShellView(VIEW_SMART_HOME);
  renderSmartHome();
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
    const selected = String(file?.path || file?.name || "").trim();
    if (!selected) return;
    if (csvPathInput) csvPathInput.value = selected;
    addEvent(`浏览器模式选择文件: ${selected}`);
    await syncSmartFileSelection(selected, "浏览器文件选择");
  });
}

if (csvPathInput) {
  const syncCsvPathFromInput = async (source) => {
    const path = String(csvPathInput.value || "").trim();
    state.smartDraft.csvPath = path;
    await refreshColumnsForCsv(path, source);
    renderSmartHome();
  };
  csvPathInput.addEventListener("change", async () => {
    await syncCsvPathFromInput("路径变更");
  });
  csvPathInput.addEventListener("blur", async () => {
    await syncCsvPathFromInput("路径失焦");
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
if (smartExportJsonBtn) smartExportJsonBtn.addEventListener("click", exportResultJson);
if (smartExportCsvBtn) smartExportCsvBtn.addEventListener("click", exportResultCsv);

async function openSmartCsvPicker(source = "smart-home") {
  const nativePicker = hasBinding("SelectCSV");
  try {
    const selected = await apiSelectCsv();
    if (selected) {
      addEvent(`智能模式已选择 CSV: ${selected}`);
      await syncSmartFileSelection(selected, source);
      return;
    }
    if (nativePicker) {
      addEvent("智能模式 CSV 选择已取消。");
      return;
    }
  } catch (err) {
    addEvent(`智能模式 CSV 选择器不可用: ${String(err)}`);
  }
  if (csvFileInput) csvFileInput.click();
}

if (continueLatestBtn) {
  continueLatestBtn.addEventListener("click", async () => {
    await resumeLatestTaskFromNav();
  });
}

if (viewStartupBtn) {
  viewStartupBtn.addEventListener("click", () => {
    openStartupDiagnostics();
  });
}

if (openAdvancedBtn) {
  openAdvancedBtn.addEventListener("click", () => {
    openAdvancedWorkspace();
  });
}

if (backSmartBtn) {
  backSmartBtn.addEventListener("click", () => {
    returnToSmartSurface();
  });
}

if (smartChooseCsvBtn) {
  smartChooseCsvBtn.addEventListener("click", async () => {
    await openSmartCsvPicker("智能首页按钮");
  });
}

if (smartDropzone) {
  smartDropzone.addEventListener("click", async () => {
    await openSmartCsvPicker("智能首页拖拽区");
  });
  smartDropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    smartDropzone.classList.add("is-hover");
  });
  smartDropzone.addEventListener("dragleave", () => {
    smartDropzone.classList.remove("is-hover");
  });
  smartDropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    smartDropzone.classList.remove("is-hover");
    const file = event.dataTransfer?.files?.[0];
    const selected = String(file?.path || file?.name || "").trim();
    if (!selected) return;
    addEvent(`智能模式拖拽载入文件: ${selected}`);
    await syncSmartFileSelection(selected, "拖拽上传");
  });
}

if (smartChooseOutputBtn) {
  smartChooseOutputBtn.addEventListener("click", async () => {
    await chooseDirectory(smartOutputInput, "输入输出目录", "已选择智能模式输出目录");
    state.smartDraft.outputDir = String(smartOutputInput?.value || "").trim();
    syncSmartDraftToClassicInputs();
    renderConfigSidebar();
    renderSmartHome();
  });
}

if (smartOutputInput) {
  const syncSmartOutput = () => {
    state.smartDraft.outputDir = String(smartOutputInput.value || "").trim();
    syncSmartDraftToClassicInputs();
    renderConfigSidebar();
    renderSmartHome();
  };
  smartOutputInput.addEventListener("input", syncSmartOutput);
  smartOutputInput.addEventListener("change", syncSmartOutput);
}

for (const input of [smartPrefConservativeInput, smartPrefAvoidTimeInput, smartPrefRequireApprovalInput]) {
  if (!input) continue;
  input.addEventListener("change", () => {
    updateSmartPreferenceDraftFromInputs();
  });
}

if (smartPrefProtectedColumnsInput) {
  smartPrefProtectedColumnsInput.addEventListener("input", () => {
    updateSmartPreferenceDraftFromInputs();
  });
  smartPrefProtectedColumnsInput.addEventListener("change", () => {
    updateSmartPreferenceDraftFromInputs();
  });
}

if (smartPrefSaveBtn) {
  smartPrefSaveBtn.addEventListener("click", async () => {
    await saveSmartPreferencesForWorkspace();
  });
}

if (smartPrefResetBtn) {
  smartPrefResetBtn.addEventListener("click", () => {
    state.smartPreferences.draft = cloneSmartPreferenceProfile(state.smartPreferences.saved);
    state.smartPreferences.message = "已恢复到最近一次加载或保存的工作区默认偏好。";
    state.smartPreferences.tone = "info";
    renderSmartPreferenceCard();
  });
}

if (outputInput) {
  const syncClassicOutputToSmart = () => {
    state.smartDraft.outputDir = String(outputInput.value || "").trim();
    if (smartOutputInput) smartOutputInput.value = state.smartDraft.outputDir;
    renderSmartHome();
  };
  outputInput.addEventListener("input", syncClassicOutputToSmart);
  outputInput.addEventListener("change", syncClassicOutputToSmart);
}

if (smartStartBtn) {
  smartStartBtn.addEventListener("click", async () => {
    await startSmartAutofixWorkflow();
  });
}

if (smartApprovalContinueBtn) {
  smartApprovalContinueBtn.addEventListener("click", async () => {
    await startSmartApprovalDecision("approve");
  });
}

if (smartApprovalRejectBtn) {
  smartApprovalRejectBtn.addEventListener("click", async () => {
    await startSmartApprovalDecision("reject");
  });
}

if (smartRunCancelBtn) {
  smartRunCancelBtn.addEventListener("click", async () => {
    if (!state.currentTaskId) return;
    smartRunCancelBtn.disabled = true;
    try {
      const ok = await apiCancelTask(state.currentTaskId);
      addEvent(ok ? "智能任务取消请求已发送。" : "智能任务取消无效（任务可能已结束）。", state.currentTaskId);
    } catch (err) {
      showError(`取消失败: ${String(err)}`);
      addEvent(`智能任务取消失败: ${String(err)}`, state.currentTaskId);
    }
  });
}

if (smartOpenAdvancedRunBtn) {
  smartOpenAdvancedRunBtn.addEventListener("click", () => {
    openAdvancedWorkspace();
  });
}

if (smartOpenAdvancedResultBtn) {
  smartOpenAdvancedResultBtn.addEventListener("click", () => {
    openAdvancedWorkspace();
  });
}

if (smartNewRunBtn) {
  smartNewRunBtn.addEventListener("click", () => {
    resetSmartSurface();
  });
}

if (startupRetryBtn) {
  startupRetryBtn.addEventListener("click", async () => {
    await runStartupChecksFlow("手动重试");
  });
}

if (startupCopyBtn) {
  startupCopyBtn.addEventListener("click", copyStartupDiagnostics);
}

if (startupCloseBtn) {
  startupCloseBtn.addEventListener("click", () => {
    closeStartupDiagnostics();
  });
}

renderStartupGate(null, { loading: true });
void runStartupChecksFlow("初始加载");


