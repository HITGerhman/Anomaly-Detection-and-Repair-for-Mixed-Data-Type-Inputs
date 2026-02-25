# MEMO

Last updated: 2026-02-25 19:36:25

## 最终目标

将当前“混合类型异常检测与修复”项目产品化为可交付桌面应用：
- Python 负责稳定可复用的算法内核。
- Go 负责任务生命周期与进程编排。
- Wails 前端负责可用、可观测、可导出的用户流程。
- 全链路具备可追溯日志与历史任务持久化能力。

## 当前总体方法

1. 分层解耦：算法层（Python）与服务/编排层（Go）和展示层（Wails）彻底分离。
2. 协议优先：统一 `input.json -> output.json` 协议、错误码与结构化日志。
3. 任务化运行：前端不直接连算法脚本，通过 Go 的任务服务调度 Python 引擎。
4. 可观测性先行：`task_id` 贯穿前端事件、Go 日志、Python 日志，并持久化历史。

## 新方向实施计划（全列扫描 + 批量修复）

- 最终形态：
  - 不再先选单个目标列进行检测，而是对整份 CSV 的每一列做异常扫描。
  - 第三页展示“列缩略图 + 异常热区 + 悬停详情 + 勾选修复”。
  - 支持一次勾选多个异常并批量修复，修复按钮显示当前勾选数量。
- 分步路线：
  1. 第一步（已完成）：协议与引擎入口改造。
  2. 第二步（已完成）：扫描算法增强（更稳定的异常评分与分桶摘要）。
  3. 第三步（待开始）：前端第三页重构为缩略图交互（hover 详情 + 勾选）。
  4. 第四步（待开始）：批量修复策略增强（更多规则与冲突处理）。
  5. 第五步（待开始）：端到端联调、性能优化与可观测性补充。

## 已完成阶段

### 阶段1：Python 内核产品化（已完成）

- 核心训练逻辑从脚本抽离到 `src/training_core.py`。
- 保留兼容导出 `src/utils.py`，避免影响 `app.py` 现有调用。
- 完成 Python 引擎三层拆分：
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/engine_main.py`
- 完成统一协议与错误码：`appshell/core/python_engine/engine_protocol.py`。
- 完成结构化日志：`appshell/core/python_engine/engine_logging.py`。
- 测试覆盖核心验收：同输入 10 次一致、异常输入返回明确错误码、核心算法测试通过。

### 阶段2：Go 后端编排层（已完成）

- 在 `appshell/backend/internal/task/service.go` 实现：
  - `RunTask` / `CancelTask` / `GetTaskStatus`
  - 并发队列与 worker 池（默认并发 3）
  - 超时、取消、状态流转与兼容旧接口
- `appshell/backend/internal/engine/runner.go` 基于 `exec.CommandContext` 调 Python。
- 验收达成：至少 3 并发、取消 2 秒内生效、超时后无僵尸进程。

### 阶段3：Wails 前端 MVP（已完成）

- 完成 UI 流程：参数配置 -> 启动任务 -> 进度展示 -> 结果查看 -> 导出。
- 新增 Wails 绑定入口与调用方法（`RunTask`/`GetTaskStatus`/`CancelTask` 等）。
- 完成 macOS 风格视觉重构与 3D Wizard 卡片翻转交互。
- 新增 CSV 列自动识别：前端从后端读取可用列，避免手输目标列。
- 新增 Train/Repair 双模式任务表单，可直接发起修复任务并查看修复摘要。

### 阶段4：数据与可观测性（已完成）

- Go 结构化日志落盘：`outputs/appshell/go_backend.log`（可环境变量覆盖）。
- Python `stderr` 结构化日志转发到 Go 日志流。
- 任务历史 sqlite 持久化：`outputs/appshell/task_history.sqlite`。
- 重启后可查询最近历史；支持仅保留最近 N 条任务记录。

## 本次改动记录（阶段2：算法覆盖面扩展）

- 改动日期：2026-02-25 19:36:25
- 改动内容简述：
  - 目标：提升算法覆盖面，支持连续目标训练并扩展扫描异常类型，同时增强结果可审计性。
  - 动机：当前 `bmi/age` 这类连续目标无法顺滑进入训练；扫描结果对“为什么判异常”解释不足。
  - 方法：在 Python 引擎中补齐 `task_type` 分流与回归指标，扩展扫描规则，并把置信度/解释特征加入标准输出。
- 相关模块/文件：
  - `src/training_core.py`
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/README.md`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `tests/python_engine/test_training_core.py`
  - `tests/python_engine/test_engine_cli.py`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 训练支持 `task_type=auto/classification/regression`，连续目标在 `auto` 下可直接进入回归训练。
  - 回归输出新增：`mae/rmse/r2/mape` 与 `prediction_confidence_mean/p10/p90`。
  - 扫描新增异常类型：
    - `time_series_shift`（数值序列突变）
    - `cross_column_consistency`（跨列一致性约束）
    - `duplicate_record`（重复记录）
  - 每条扫描 issue 新增 `confidence` 和 `explain_features`，`scan_summary` 新增 `issue_type_counts`。
  - Go 请求归一化补齐新字段透传（`task_type` 与扩展扫描配置项）。
  - `repair` 动作增加模型类型保护：非分类模型会明确报错原因与建议路径。
  - 测试通过：
    - `python -m pytest -q tests/python_engine`（21 passed）
    - `go test ./...`（`appshell/backend`）
- 待处理事项：
  - 回归场景下仍未提供“模型驱动的单样本修复”，当前仅支持规则型批量修复路径。
  - 跨列一致性目前以规则约束为主，后续可补充规则管理界面与行业模板库。
  - 时序突变当前默认按行序检测，后续建议支持显式时间列与窗口策略配置。

## 本次改动记录（算法优化）

- 改动日期：2026-02-21 21:03:40
- 改动内容简述：
  - 目标：提升异常检测在中等难度数据上的有效性与可解释性，减少“只看准确率”导致的漏检风险。
  - 动机：此前策略在 `simple_medium_anomaly.csv` 上出现高精度但召回偏低（漏检偏多）。
  - 方法：引入“类不平衡处理 + 阈值调优 + 指标体系升级”的组合优化。
- 相关模块/文件：
  - `src/training_core.py`
  - `src/utils.py`
  - `app.py`
  - `appshell/core/python_engine/engine_core.py`
  - `tests/python_engine/test_training_core.py`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增类不平衡训练策略：`class_weight`（默认 `balanced`）。
  - 新增阈值调优流程：
    - 在训练集内再划分验证集。
    - 以 `F-beta`（默认 `beta=2`）为主目标，在最小精度约束下选阈值。
    - 选出的阈值写入模型属性 `decision_threshold`。
  - 新增阈值推理接口：`predict_with_threshold()` 与 `get_decision_threshold()`。
  - `app.py` 与引擎汇总改为阈值感知预测，不再固定 `0.5`。
  - 指标体系升级：
    - 二分类输出 anomaly-focused 指标（`precision/recall/f1` 指向异常类）
    - 保留 `*_weighted` 指标
    - 输出 `decision_threshold` 与 `threshold_optimization` 元信息
  - 新增/更新测试：验证阈值字段、异常指标字段、阈值预测一致性与确定性。
- 待处理事项：
  - 连续目标训练能力已在 2026-02-25 补齐；后续重点是“回归场景修复策略”而非训练支持本身。
  - 后续可增加自动任务类型判断（分类/回归）并扩展回归型异常评分路径。
  - 可加入阈值校准模式开关（偏召回 / 偏精度 / 平衡）供前端选择。

## 本次改动记录（修复 MVP）

- 改动日期：2026-02-21 21:48:37
- 改动内容简述：
  - 目标：先打通“检测后修复”最小可用闭环，支持从已训练模型产物中选择样本并自动给出修复方案。
  - 动机：项目最终目标是“Detection + Repair”，此前编排链路以训练为主，修复能力尚未产品化。
  - 方法：新增 Python `repair` 动作（约束修复搜索）+ Go 请求编排透传 + 前端 Train/Repair 双模式交互。
- 相关模块/文件：
  - `src/repair_core.py`
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/engine_protocol.py`
  - `appshell/core/python_engine/README.md`
  - `tests/python_engine/test_repair_core.py`
  - `tests/python_engine/test_engine_cli.py`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/backend/cmd/demo/main.go`
  - `appshell/backend/README.md`
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增 Python 引擎动作：`repair`。
  - 新增修复内核：健康邻域候选生成 + 约束下贪心最小改动搜索（最多改 N 个字段）。
  - 修复输出包含：`repair_summary`、`repair_changes`、`original_sample`、`repaired_sample`。
  - 支持可选修复报告落盘（CSV/JSON）。
  - Go `normalizeRequest` 支持 `repair` 载荷（`model_dir/sample_index/max_changes/k_neighbors`）。
  - Demo 支持 `-action repair`。
  - 前端支持 Train/Repair 双模式，修复模式可配置模型目录、样本索引、改动上限、邻居数，并展示修复结果。
- 待处理事项：
  - 当前修复样本来源于 `test_data.pkl`（编码后特征），尚未打通“原始 CSV 任意行”直接修复。
  - 仍缺少业务规则约束（跨字段逻辑、不可改字段模板）管理页面。
  - 批量修复（仅对异常样本自动修复并导出全量报告）尚未实现。

## 本次改动记录（修复交互重构：4卡轮盘）

- 改动日期：2026-02-22 08:50:59
- 改动内容简述：
  - 目标：按用户体验要求重构前端流程，避免“过早出现修复入口”，改为“先检测、后条件修复”。
  - 动机：原先修复能力在入口层暴露过早，与“先判定是否异常再决定修复”的自然流程不一致。
  - 方法：引入 `dry_run` 检测任务 + 四阶段卡片流程 + 轮盘式卡片切换动画 + 阶段进度条。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `src/repair_core.py`
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 前端从 3 阶段改为 4 阶段：
    - 1) 参数配置
    - 2) 运行中
    - 3) 检测结果
    - 4) 修复界面
  - 第三张卡片仅展示检测结果：
    - 若 `before_pred=0`（无异常），隐藏修复入口。
    - 若 `before_pred=1`（有异常），显示“进入修复界面”按钮。
  - 修复按钮点击后才进入第四张卡片并执行修复。
  - 动画由 3D 翻转改为“下方不可见圆心”的轮盘旋转（卡片沿圆弧切换，保持阅读方向）。
  - 新增全局阶段进度条（4段），实时高亮用户所处阶段。
  - 后端修复动作支持 `dry_run`（仅检测不改值），用于第三阶段判定异常与否。
- 待处理事项：
  - 目前轮盘角度与动画时长为经验参数，后续可按真机体验继续微调（角度、阻尼、位移）。
  - 第四阶段尚未加入“逐字段手动确认/拒绝”能力，当前为参数驱动的自动修复执行。

## 本次改动记录（轮盘动画故障修复）

- 改动日期：2026-02-22 09:06:26
- 改动内容简述：
  - 目标：修复“切换后只抖动、半屏裁切、按钮不可点”的严重交互故障。
  - 动机：当前实现把旋转状态持久化到了卡片本体，导致界面停留在错误姿态且被容器裁切。
  - 方法：改为“瞬时轮盘动画 + 切换后复位”，并调整容器裁切与滚动策略，保证每一步都回到可交互正位。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 动画改为整卡轮盘轨迹（圆心位于页面下方视野外），出场向左旋出、入场从右旋入。
  - 不再保留错误旋转终态，切换完成后强制复位到 `rotate(0deg)`，避免卡片半屏和错位。
  - 切换期间统一禁用指针，结束后恢复，修复“按钮按不到/状态卡死”。
  - 去除关键裁切：`wheel-frame` 改为可见溢出；卡片内部改为可滚动，防止内容被截断。
  - 增加动画态阴影抬升，切换过程更接近“轮盘抬起”空间感。
- 待处理事项：
  - 需在你机器上复测 3 个路径：`参数->运行`、`运行->检测`、`检测->修复`，确认观感符合预期。
  - 若希望“消失得更彻底/更慢回弹”，可继续微调 `WHEEL_SPIN_DEG` 与 `WHEEL_SPIN_DURATION_MS`。

## 本次改动记录（目标列恢复 + 紧凑布局）

- 改动日期：2026-02-22 09:23:59
- 改动内容简述：
  - 目标：解决“目标列下拉消失”和“界面过于空旷”的两个可用性问题。
  - 动机：用户无法直接选择检测目标列（如 `stroke`），且当前卡片与窗口边界留白过大，信息密度偏低。
  - 方法：恢复 CSV 列读取链路（前端 -> Wails `ListCSVColumns`）并增加检测前重训开关；同时整体压缩布局尺寸与间距。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 参数卡片恢复 `CSV 文件` 输入与 `目标列` 下拉，支持自动读取并显示可用列（默认优先 `stroke`）。
  - 新增“检测前重训”开关（默认开启）：
    - 开启时：先按 `CSV + 目标列` 训练模型，再执行检测。
    - 关闭时：直接基于 `model_dir` 执行检测（适合复用已有模型）。
  - 补回 CSV 文件选择器调用与浏览器回退逻辑。
  - 布局紧凑化：缩小外边距、卡片尺寸、表单间距、输入与按钮高度，减少大面积空白。
- 待处理事项：
  - 需要在真实 Wails 窗口中确认不同分辨率下的紧凑度是否合适。
  - 若要进一步提高信息密度，可考虑把“样本索引/超时/重训开关”做成单行栅格布局。

## 本次改动记录（失败流回到第3卡并展示原因）

- 改动日期：2026-02-22 09:37:43
- 改动内容简述：
  - 目标：修复“检测失败后直接回到第1卡且没有可见提示”的流程问题。
  - 动机：失败信息只在运行卡错误面板出现，切回参数卡后用户看不到失败原因，也无法判断是否可进入修复。
  - 方法：统一检测链路失败分支（训练失败、检测失败、任务启动失败、轮询失败）到第3卡，并在检测结果区渲染失败摘要。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增 `renderDetectionFailure()`，在第3卡显示：
    - 失败阶段（训练/检测）
    - 任务 ID
    - 状态
    - 错误码
    - `can_repair=false`
  - 失败文案会明确说明“当前无法判断是否可修复”，并附带后端建议（若有）。
  - 检测链路的提交失败/校验失败/运行失败/轮询失败统一跳转到第3卡，不再默认回第1卡。
  - 失败时强制隐藏并禁用“进入修复界面”按钮，避免误操作。
- 待处理事项：
  - 可进一步把失败态做成专用视觉块（图标 + 高亮错误码），提升可读性。
  - 可增加“一键复制错误详情”按钮，便于问题上报与排查。

## 本次改动记录（轮盘方向改左 + 窗口自适应）

- 改动日期：2026-02-22 10:01:53
- 改动内容简述：
  - 目标：将卡片切换方向从“向右”改为“向左”，并缓解窗口过大时界面空隙过多的问题。
  - 动机：当前动画方向与预期不一致；大窗口场景下 UI 未充分利用可用空间。
  - 方法：调整轮盘出入场角度符号；在 Wails 启动阶段按屏幕尺寸动态设置窗口大小；放宽前端舞台宽高的响应式上限。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/main.go`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 卡片切换方向已改为向左（左旋出场 + 右侧入场）。
  - Wails 启动时自动读取当前/主屏幕尺寸并设置窗口宽高、最小尺寸与居中位置。
  - 默认窗口尺寸改为更紧凑的初始值，减少启动时空白感。
  - 前端主舞台宽度与卡片高度改为更自适应的 `clamp`/`viewport` 方案，大窗口下可展示更多内容并减少留白。
- 待处理事项：
  - 可继续微调窗口占屏比例（当前宽 82%、高 86%）以匹配你的显示器习惯。
  - 若需要“窗口随内容动态收缩”而不是“按屏幕比例”，可在下一步增加内容测量策略。

## 本次改动记录（逆向切换向右 + 进一步收窄）

- 改动日期：2026-02-22 10:32:55
- 改动内容简述：
  - 目标：修复“返回上一步仍向左切换”的方向问题，并把界面进一步收窄。
  - 动机：用户从第3卡点击“重新检测”回第1卡时应表现为逆向（向右）切换；当前窗口与卡片宽度仍偏大。
  - 方法：将轮盘方向改为“基于步骤索引自动判断”；同时下调窗口宽度比例与前端舞台宽度上限。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/main.go`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增 `stepDirection(fromStep, toStep)`：
    - 前进（步骤索引增大）保持向左切换
    - 后退（步骤索引减小）改为向右切换
  - 因此“重新检测（3 -> 1）”现在会向右切换，符合逆向动效。
  - 启动窗口尺寸进一步收窄：屏幕宽度占比从 `0.82` 调整到 `0.72`，并收窄默认窗口宽度。
  - 前端主舞台上限从超宽布局收敛到更窄宽度，减少横向铺张。
- 待处理事项：
  - 若仍觉得偏宽，可继续下调舞台宽度（例如 `980px`）或将窗口占屏宽降到 `0.68`。
  - 若希望“返回动画更明显”，可增加逆向时的角度系数（如 `1.1x`）。

## 本次改动记录（全列扫描/批量修复：第1步 协议与引擎入口）

- 改动日期：2026-02-22 10:58:42
- 改动内容简述：
  - 目标：启动“全列检测 + 批量修复”新方向，先完成协议层和引擎入口扩展。
  - 动机：后续 UI 缩略图和多点勾选修复依赖新的后端动作协议，需先把 action 与结果结构稳定下来。
  - 方法：新增 Python engine action（`scan_file`/`repair_batch`），扩展 Go `normalizeRequest`，并补齐自动化测试。
- 相关模块/文件：
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/engine_protocol.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/README.md`
  - `tests/python_engine/test_engine_cli.py`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增动作 `scan_file`：
    - 输入：`csv_path`（可选 `max_bins`/`max_issues`）
    - 输出：`column_profiles`、`column_thumbnails`、`issues`、`anomaly_columns`
    - 检测规则（第一版）：缺失值、数值离群（IQR/MAD）、类别低频值。
  - 新增动作 `repair_batch`：
    - 输入：`csv_path`、`issue_ids`（可选 `write_output`/`output_dir`/`output_csv`）
    - 输出：`applied_repairs`、`skipped_issues`、`total_cells_modified`、`output_csv`
    - 修复规则（第一版）：缺失填补、数值离群裁剪、低频类别替换。
  - 新增错误码：`SCAN_FAILED`、`REPAIR_BATCH_FAILED`。
  - Go 层 `normalizeRequest` 增加 `scan_file` 与 `repair_batch` 载荷归一化。
  - 新增/更新测试：
    - Python 引擎 CLI 用例覆盖 `scan_file` 与 `repair_batch`。
    - Go `normalizeRequest` 用例覆盖新 action。
- 待处理事项：
  - 第二步将增强扫描精度与摘要结构（适配前端“缩略图热区”渲染）。
  - 第三步将把前端第三页改为可交互缩略图并接入批量勾选修复。

## 本次改动记录（全列扫描/批量修复：第2步 扫描算法增强）

- 改动日期：2026-02-22 11:28:01
- 改动内容简述：
  - 目标：提升 `scan_file` 输出稳定性与可视化友好度，直接支撑后续前端缩略图交互。
  - 动机：第一步已打通 action，但扫描结果仍偏“原始”，缺少可排序风险分数和列级热区摘要。
  - 方法：引入可配置扫描参数、统一 issue 打分排序、列级风险评估、热度分桶与合并段摘要。
- 相关模块/文件：
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/README.md`
  - `tests/python_engine/test_engine_cli.py`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增扫描参数模型（支持 `scan_config` 或顶层字段覆盖）：
    - `max_bins` / `max_issues`
    - `numeric_iqr_factor` / `robust_z_threshold`
    - `rare_ratio_threshold` / `rare_count_floor`
    - `min_numeric_samples` / `min_categorical_samples`
    - `preview_limit`
  - `scan_file` 增强：
    - issue 输出新增 `issue_score`、`risk_level`、`heat_bins`、合并后的 `segments`
    - 每个 issue 的 `detail` 增加 `preview`（行号+值）用于 hover 详情
    - 列级输出新增 `risk_score`、`risk_level`、`anomaly_ratio`、`hot_segments`
    - 新增 `scan_summary`（高/中风险列汇总）
    - issue 列表按 `issue_score` 稳定降序排序
  - `repair_batch` 增强：
    - 支持复用同一 `scan_config`，确保“扫描-修复” issue 识别一致
    - 响应返回 `scan_config` 以便前端追踪本次修复依据
  - Go 归一化扩展：
    - `scan_file` / `repair_batch` 透传高级扫描参数与 `scan_config`
  - 测试增强：
    - Python 新增扫描阈值覆盖场景与新字段断言
    - Go 新增高级参数透传断言
- 待处理事项：
  - 第三步进入前端重构：第三页实现“列缩略图 + 热区 hover + 勾选计数 + 批量修复提交”。
  - 需要在大数据量下评估 `max_bins` 与响应体大小的平衡。

## 算法优化思路重点说明

1. 不再只追求 Accuracy
- 异常检测通常类不平衡，准确率高不代表漏检少。
- 本轮将“异常类召回”提升到与精度同等优先级。

2. 训练阶段先处理类不平衡
- `class_weight=balanced` 让模型对少数类（异常）更敏感。
- 目的不是盲目提高召回，而是降低模型对多数类的偏置。

3. 推理阶段做阈值优化而不是固定 0.5
- 概率阈值直接影响 FP/FN 平衡。
- 使用验证集搜索阈值：在满足最小精度前提下，最大化 `F-beta`（默认偏向召回）。
- 这样可以把“业务偏好”显式参数化，而不是隐藏在模型默认行为里。

4. 指标分层输出，便于决策
- anomaly-focused 指标用于回答“异常抓得怎样”。
- weighted 指标用于回答“整体预测稳定性怎样”。
- 两组同时看，避免单指标误导。

## 本轮实测结果（中等难度样本）

数据：`data/raw/simple_medium_anomaly.csv`（目标列 `stroke`）

- 优化前（历史记录）：
  - `accuracy=0.875`
  - `precision=1.0000`
  - `recall=0.3750`
  - `confusion_matrix=[[64,0],[10,6]]`
- 优化后（当前实现）：
  - `decision_threshold=0.4898679324`
  - `accuracy=0.8375`
  - `precision=0.6364`
  - `recall=0.4375`
  - `f1=0.5185`
  - `auc=0.7852`
  - `confusion_matrix=[[60,4],[9,7]]`
  - `threshold_optimization.selected_fbeta=0.6098`

结论：
- 召回从 `0.3750` 提升到 `0.4375`（漏检减少）。
- 精度与准确率有所下降，属于“更偏召回”的可控权衡。
- 该权衡符合异常检测“宁可多报少漏”的常见目标，可继续按业务需求调节阈值策略。

## 已验证结果（本轮）

- `python -m pytest -q tests/python_engine` -> `16 passed`
- `go test ./...`（`appshell/backend`）通过
- `node --check appshell/frontend/src/main.js` 通过
- Python 引擎端到端验证：
  - `train` 成功产出模型目录
  - `repair` 在异常样本上可完成状态翻转（`before_pred=1 -> after_pred=0`，实测 `score_reduction=0.4359`）
- 中等样本重复 10 次训练/评估签名一致（阈值与核心指标稳定）。

## 当前正在解决的问题

1. 连续目标列（如 `bmi`）不适用当前分类训练路径。
2. 修复 MVP 目前基于编码特征空间，需补充“原始值级别”的可解释映射与修复约束模板。
3. 需要把“阈值策略 + 修复策略”进一步产品化为可配置选项，并在 UI 中可解释展示。
4. 需要在后续阶段补充打包与部署（阶段5）链路，完成交付闭环。

## 下一步建议（按优先级）

1. 阶段5：Windows 打包落地（Python 引擎可执行化 + Wails 打包 + 安装器）。
2. 修复增强：支持从原始 CSV 行直接修复（含编码映射回写）并输出批量修复报告。
3. 算法扩展：补充回归目标的异常评分方案（如残差/分位数/孤立森林混合路径）。
4. 体验增强：前端增加阈值策略选择与“精度-召回”说明，支持重跑对比。

## 本次改动记录（全列扫描/批量修复：第3步 前端交互落地）

- 改动日期：2026-02-22 15:32:53
- 改动内容简述：
  - 目标：完成第3步前端重构，把第3卡从“静态结果”升级为“可交互异常地图”，并接入批量修复任务提交流程。
  - 动机：当前产品目标是“检测 + 修复”闭环，用户需要在检测后直接看到各列异常位置并按需多选修复。
  - 方法：重建前端任务流（`scan_file -> result -> repair_batch`），在第3卡渲染列缩略图热区，支持 hover 详情与多选计数，修复后进入第4卡展示应用明细。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `appshell/frontend/index.html`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 第3卡新增“列缩略图异常热区”交互：
    - 每列显示淡绿色基底 + 红/橙热区段（粗粒度，性能友好）
    - 鼠标悬停热区后显示 `issue_id/列名/类型/严重度/占比/评分/行区间` 等详情
  - 新增异常多选修复机制：
    - 可点击热区或详情勾选框进行选择
    - “执行修复”按钮旁实时显示已选数量（并同步按钮文案）
    - 调用 Go `RunTask(action=repair_batch)` 执行批量修复
  - 修复第3步失败可见性：
    - `scan_file` 失败时固定落在第3卡展示错误原因、错误码与建议，不再无提示回第1卡
  - 第4卡新增修复明细列表：
    - 区分 `Applied` 与 `Skipped`，展示每个 issue 的处理结果
  - 兼容浏览器预览模式：
    - 保留 mock 任务流（无 Wails 绑定时仍可演示完整交互）
- 待处理事项：
  - 进一步提升缩略图精度（例如按风险等级动态聚合 lane，降低重叠）。
  - 在真实数据大文件下做响应体与渲染性能压测（重点关注 `max_bins` 和 issue 数量上限）。
  - 下一步进入“第4步实现”：将“勾选对象”从 issue 进一步细化到更小粒度片段（可选），并优化修复策略说明文案。

## 当前总目标与进展对齐（更新）

- 最终目标：把项目从“单点检测脚本”推进到“可视化检测 + 人机协同修复”的桌面应用流程。
- 当前方法：
  - 后端统一任务编排（Go）
  - 引擎统一协议（Python `scan_file/repair_batch`）
  - 前端向导化分步（Wails）
- 已完成步骤：
  - 第1步：协议与引擎入口扩展（`scan_file`/`repair_batch`）
  - 第2步：扫描算法增强（评分、热区、摘要）
  - 第3步：前端交互落地（缩略图、hover 详情、多选修复）
- 当前正在解决的问题：
  - 多异常重叠场景下的交互可读性与批量修复可解释性仍可继续提升。

## 本次改动记录（第3步测试验证）

- 改动日期：2026-02-22 16:07:52
- 改动内容简述：
  - 目标：验证第3步前端重构后的主流程是否可用（检测 -> 结果交互 -> 批量修复）。
  - 动机：确保本轮功能不是仅“可编译”，而是端到端链路可运行。
  - 方法：执行前端语法检查、Go 后端单测、Python 引擎单测，并补充 `scan_file/repair_batch` 冒烟测试。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/core/python_engine/engine_core.py`
  - `tests/python_engine/test_engine_cli.py`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 测试结果：
    - `node --check appshell/frontend/src/main.js` -> 通过
    - `go test ./...`（`appshell/backend`）-> 通过
    - `python -m pytest -q tests/python_engine` -> `16 passed`
  - 引擎冒烟（真实数据）：
    - `scan_file` 成功：`rows=5110, cols=12, issues=5`
    - 基于扫描结果选择 2 个 `issue_id` 执行 `repair_batch` 成功：`applied=2, modified=828`
  - 前端 DOM 一致性检查：`main.js` 引用的 `id` 均在 `index.html` 存在。
- 待处理事项：
  - 需要你在 Wails UI 中手工走一遍第3卡 hover 与勾选交互，确认视觉与操作体验满足预期。
  - 后续可增加前端 E2E 自动化（例如 Playwright）覆盖“检测失败/成功/多选修复”路径。

## 本次改动记录（UI 可点击性 + 目标列下拉恢复 + JSON 序列化修复）

- 改动日期：2026-02-22 16:41:55
- 改动内容简述：
  - 目标：修复“界面尺寸导致按钮难以点击”与“卡片1无法选择目标列”两个回归问题，同时处理导致第3卡超长报错的根因。
  - 动机：当前体验中失败信息过长会撑爆第3卡，且目标列下拉在重构后被移除，影响可用性。
  - 方法：
    - 前端恢复目标列选择（自动读取 CSV 列）；
    - 压缩错误文案与卡片布局（限制错误文本高度，收紧卡片高度，配置页操作区吸底）；
    - Python 引擎修复 `NaN/Inf` 输出为标准 JSON，避免 Go 侧解析失败。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `appshell/core/python_engine/engine_core.py`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 恢复卡片1 `目标列（可选）` 下拉，并在以下场景自动刷新列：
    - 初始加载
    - 选择 CSV 文件
    - 手工改路径（change/blur）
  - 第3卡失败信息不再塞入整段原始 JSON：
    - 前端 `toReadableError` 增加文本规范化与长度截断
    - 去除 `raw=...` 长尾内容，避免页面被长文本撑爆
  - 轮盘切换增加 `try/finally` 兜底，避免异常情况下 `pointer-events` 卡死导致“按钮点不到”。
  - UI 尺寸紧凑化：
    - `wheel-frame` 高度收敛
    - 配置页操作按钮区吸底（sticky）
    - 结果文案与响应区高度限制，避免挤占交互区
  - 引擎 JSON 严格化：
    - `_to_builtin` 对 `NaN/Inf` 转换为 `None`
    - `np.ndarray` 转换后递归清洗，避免残留非标准数值
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过
  - `go test ./...`（`appshell/backend`）通过
  - `python -m pytest -q tests/python_engine` -> `16 passed`
  - 严格 JSON 冒烟（Node `JSON.parse`）通过：`scan_file` 返回可被严格解析
- 待处理事项：
  - 需要在你本地 Wails 窗口手工确认“极小窗口/高 DPI”下按钮可达性是否满足预期。
  - 若你希望“目标列”真正参与后端扫描逻辑（而非前端视图过滤），下一步可在 Python `scan_file` 增加 `target_col/focus_columns` 支持。

## 本次改动记录（新增修复功能测试样本 CSV）

- 改动日期：2026-02-22 16:59:46
- 改动内容简述：
  - 目标：提供一个可直接用于“检测+修复”联调的最小测试样本文件。
  - 动机：需要一个同时包含多类可识别异常的数据集，便于在 UI 中验证批量修复流程。
  - 方法：新增 `repair_test_case.csv`，刻意包含缺失值、数值离群值、低频类别三类异常，并用引擎执行 `scan_file` / `repair_batch` 冒烟验证。
- 相关模块/文件：
  - `data/raw/repair_test_case.csv`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增测试样本：`data/raw/repair_test_case.csv`（30 行，5 列）。
  - 可检出的异常（实测）：
    - `bmi::missing_values`（缺失 3 条）
    - `age::numeric_outlier`（离群 2 条）
    - `smoking_status::rare_category`（低频 2 条）
  - 修复链路实测通过：
    - `repair_batch` 选中上述 3 个 issue -> `applied=3`, `total_cells_modified=7`, `skipped=0`。
- 待处理事项：
  - `repair_batch` 对整数列裁剪时出现 pandas FutureWarning（dtype 兼容写入），后续可在修复写回时增加显式 dtype 处理。

## 本次改动记录（缩略图宽度/失败态尺寸/多选交互修正）

- 改动日期：2026-02-22 17:11:02
- 改动内容简述：
  - 目标：修复你反馈的三个体验问题：
    1) 缩略图太小；
    2) 失败态框体尺寸与正常态不一致；
    3) 选择第二个异常时第一个像被重置。
  - 方法：调整缩略图布局为全宽单列、统一失败态与正常态占位高度、修改热区点击语义为“默认加入选择（不反选）”。
- 相关模块/文件：
  - `appshell/frontend/src/style.css`
  - `appshell/frontend/src/main.js`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 缩略图区域改为单列全宽展示（每行一列），视觉上占满卡片宽度。
  - 失败态占位（`scan-placeholder`）与详情空态（`issue-detail-empty`）统一最小高度，避免“错误框忽大忽小”。
  - 热区选择逻辑优化：
    - 普通点击：只会“选中该异常”，不会把已选项反选掉。
    - `Ctrl/Cmd + 点击`：可快速取消某个已选项。
  - 选中态描边强化，已选更容易识别。
- 验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 若你希望支持“按片段（segment）而不是按 issue”进行多选修复，需要后端协议从 `issue_ids` 扩展为 `segment_ids` 或 `row_range`；当前仍是 issue 粒度。

## 本次改动记录（缩略图改为矩阵热图 + 方形错误标记）

- 改动日期：2026-02-22 17:41:31
- 改动内容简述：
  - 目标：按你的反馈把“长条缩略图”改为“矩阵热图”，并将异常标记改成方形，避免异常点挤在一条线上。
  - 动机：长条模式在异常聚集时可读性差，不符合你希望的“接近表格长宽比”的视觉表达。
  - 方法：前端渲染从 `thumb-segment` 横条改为 `matrix-cell` 网格；按 bin 映射 issue，按严重度着色，并保留 hover 详情 + 多选修复交互。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 缩略图改为“矩阵”表现：
    - 每列使用二维网格显示（自动计算列数与行数）
    - 错误标记为方形色块（高/中/低风险不同颜色）
  - 异常不再挤在一条横线上：
    - 通过网格换行分布显示，热点更易区分
  - 交互保持不变并增强：
    - hover/focus 仍可显示问题详情
    - 点击默认加入选择，`Ctrl/Cmd + 点击` 取消该项
    - 已选/当前聚焦有明显描边高亮
  - 可视区加大：
    - 缩略图容器高度上调，便于展示更完整矩阵。
- 验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 若后续要实现“严格还原原表二维结构（行x列）”的总览热图，需要在协议层提供跨列统一坐标映射（当前仍是按列分别渲染）。

## 本次改动记录（矩阵密度滑杆 + 行列标注表格化）

- 改动日期：2026-02-22 18:01:52
- 改动内容简述：
  - 目标：按需求增加“密度滑杆”，并将列缩略图升级为带行列名称的表格化矩阵，提升可读性。
  - 动机：此前矩阵虽已替代长条，但缺少坐标语义；用户难以快速定位“哪一行段/哪一列位”的异常分布。
  - 方法：在参数区新增 `matrix-density` 滑杆，前端根据密度动态计算矩阵列数；渲染层改为“表头 + 行头 + 方格单元”结构。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增“矩阵密度”滑杆：
    - 范围 `50% ~ 180%`，实时显示数值；
    - 调整后可在不重跑任务的情况下即时重绘矩阵（如果当前有检测结果）。
  - 缩略图改为表格化矩阵：
    - 顶部显示每列名称（`列1, 列2, ...`）；
    - 左侧显示每行名称（按行段映射的行范围，如 `0-99`）；
    - 中间为方形单元格异常标记（红/橙/黄）。
  - 异常标记保持可交互：
    - hover 查看详情；
    - 点击加入选择；
    - `Ctrl/Cmd + 点击` 取消该项。
- 验证结果：
  - `node --check appshell/frontend/src/main.js` 通过；
  - DOM 绑定校验通过（JS 中引用的 `id` 均存在于 HTML）。
- 待处理事项：
  - 当前“行名称”是行段范围映射（非原始逐行名称）。若要显示真实行号/索引标签，需要在 UI 侧增加分页或虚拟滚动策略，避免大数据量标签过载。

## 本次改动记录（运行信息降噪 + 修复解释增强）

- 改动日期：2026-02-22 20:09:49
- 改动内容简述：
  - 目标：解决“信息杂乱、关键信息缺失”的体验问题，让用户在运行与修复阶段看到可行动的信息，而非大量技术噪声。
  - 动机：当前界面虽然功能完整，但状态文案偏“机器日志”，修复明细缺少“修复了什么/为什么修复/如何修复”的解释。
  - 方法：前端重构状态文案与修复明细展示，增加运行阶段提示与原始响应折叠开关，默认展示业务信息，调试信息按需展开。
- 相关模块/文件：
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 运行过程文案改为“阶段提示”：
    - `scan_file` 显示如“读取 CSV -> 扫描异常 -> 生成缩略图 -> 整理结果”；
    - `repair_batch` 显示如“加载问题 -> 应用修复规则 -> 写出结果 -> 整理摘要”；
    - 预留 `train` 文案（含“正在训练模型”）。
  - 修复明细可解释化：
    - 每条 `applied_repair` 显示“修复对象/影响行数/修复原因/采用策略/问题ID”；
    - 每条 `skipped_issue` 显示中文跳过原因（如 `issue_not_found/no_rows_matched` 等）。
  - 信息降噪：
    - 错误文本进一步去除 `raw=...` 长尾，避免把超长 JSON 注入主视图；
    - “原始响应”改为调试面板，默认折叠，用户可手动展开查看。
  - 摘要项重命名为中文业务字段，减少技术字段直出。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 运行阶段提示目前为前端时间片推断；若要更准确，需要后端/引擎回传细粒度进度事件。
  - 修复说明当前基于 issue 类型模板化生成；后续可接入更细的策略参数（例如中位数/众数来源、裁剪前后统计对比）。

## 本次改动记录（UI 第1/2项：信息分层 + 结果可视化升级）

- 改动日期：2026-02-22 21:12:16
- 改动内容简述：
  - 目标：按规划先落地两项美化优化：`1) 信息分层`、`2) 结果可视化升级`。
  - 动机：当前功能完整但信息密度偏高，用户需要先看到“结论/可修复项/下一步”，调试细节应按需查看。
  - 方法：重构结果页信息层级，新增“高级模式”开关与抽屉，同时增强异常矩阵说明与问题详情可解释性。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 信息分层（第1项）：
    - 新增“高级模式”开关（默认关闭），调试信息进入抽屉：
      - 运行页：`Task ID + 事件日志` 收纳到 `高级模式（运行日志）`。
      - 结果页：`detection-summary` 收纳到 `高级模式（原始检测摘要）`。
      - 原始 JSON 响应面板仅在高级模式下显示。
    - 结果页重构为 3 个核心信息块：
      - `结论`
      - `可修复项`
      - `下一步`
    - 默认视图优先业务信息，减少技术字段直接暴露。
  - 结果可视化升级（第2项）：
    - 缩略图区域新增图例与刻度说明：
      - 正常/低风险/中风险/高风险颜色说明；
      - 动态刻度文案（含分桶范围与每格近似代表行数）。
    - 问题详情新增“修复前/修复后（预估）”对比小表：
      - 展示样例行 `before/after`；
      - 按异常类型给出估算修复值（缺失填补、离群截断、低频归并）；
      - 附带说明“预估值仅作参考，最终以修复输出为准”。
  - 逻辑联动：
    - `renderScanResult/renderScanFailure` 同步驱动三块核心信息内容；
    - 高级模式切换可即时控制抽屉与原始响应显示。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 当前“修复后（预估）”为规则估算；若需真实对比，需要在后端返回修复前后样本切片。
  - 可继续补充第4步修复页的同类信息分层（业务摘要 vs 调试明细）以保持全流程一致。

## 本次改动记录（第2页显示补全 + 第1页高级模式生效）

- 改动日期：2026-02-22 21:26:50
- 改动内容简述：
  - 目标：修复“第2页显示不全/信息不足”的体验问题，并让第1页高级模式真正影响参数区展示。
  - 动机：第2页在基础模式下信息过少；第1页高级模式开关此前对参数区没有明显效果。
  - 方法：恢复关键运行信息常显、增加运行阶段列表；将第1页参数改为“基础/高级”分层可切换。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 第2页显示补全：
    - 恢复 `Task ID` 常显（状态条右侧）；
    - 新增“运行阶段”可视列表（pending/running/完成状态对应高亮与完成态）；
    - 高级模式下仍可展开完整运行日志。
  - 第1页高级模式生效：
    - 基础模式仅显示核心参数（文件、目标列）；
    - 高级模式展开显示进阶参数（分桶、矩阵密度、最多问题数、超时、输出目录）；
    - 增加基础模式说明文案，减少“参数不全”的困惑。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 若后续希望“第2页阶段列表”与后端真实执行步骤完全一致，需要后端提供明确进度阶段事件（当前为前端推断）。

## 本次改动记录（UI 第一步美化：结构收敛与空白压缩）

- 改动日期：2026-02-23 08:12:20
- 改动内容简述：
  - 目标：落实“先做结构收敛”这一步，降低卡片1/2/4空旷感，同时保持卡片3检测矩阵的可视空间。
  - 动机：当前主要问题不是功能缺失，而是容器高度与留白过大导致信息密度偏低。
  - 方法：给向导容器绑定当前步骤状态，按步骤动态调整 frame 高度；同步压缩页边距、卡片内边距与标题区间距。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 动态高度分配：
    - 为 `wheel-frame` 增加 `data-step` 驱动的高度策略：
      - `config/progress/repair` 使用更紧凑高度；
      - `result` 保持较大高度以容纳矩阵与详情。
  - 卡片布局收敛：
    - 压缩全局与卡片留白（`wizard-app`、`stage-header`、`wizard-card`、`card-step`、`form`）。
    - 非结果页（1/2/4）强制内容从上对齐，避免“底部大片空白”。
  - 移动端同步：
    - 在 `@media` 下同样按步骤分配 `wheel-frame` 高度，减少小屏空洞。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 下一步可继续做第2步美化（在卡片1/2/4增加有意义的右侧信息区块），进一步提升信息密度与视觉重心。

## 本次改动记录（全自动模式 + 一键自动修复闭环）

- 改动日期：2026-02-23 08:54:26
- 改动内容简述：
  - 目标：新增“全自动模式”流程：第1页无需选目标列，第3页可一键修复全部问题列，第4页输出修复报告。
  - 动机：降低操作门槛，让用户可直接完成“自动检测 -> 自动修复 -> 查看报告”最短路径。
  - 方法：在前端引入 `autoMode` 状态，控制检测范围、结果页按钮和修复任务 issue 选择策略。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 第1页新增“全自动模式”开关：
    - 开启后隐藏目标列选择，检测范围固定为全部列。
  - 第3页新增“一键自动修复全部问题列”按钮：
    - 自动收集当前扫描结果中的全部 `issue_id`；
    - 直接触发 `repair_batch`，无需手动逐项勾选。
  - 自动模式下结果展示优化：
    - 缩略图仅展示有异常的列（减少干扰）；
    - 结论与下一步文案提示“可直接一键自动修复”。
  - 第4页修复报告增强：
    - 修复摘要新增“修复模式”（自动/手动）标识。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 若后续希望自动模式支持“策略模板”（例如仅修复高风险列），可在第1页增加预设策略选项并将筛选逻辑映射到 `issue_ids` 生成阶段。

## 本次改动记录（美化工程第2步：卡片1/2/4右侧信息区块）

- 改动日期：2026-02-23 09:13:20
- 改动内容简述：
  - 目标：实现第2步美化要求，为卡片1/2/4增加“有意义的右侧区块”，提升信息密度和决策效率。
  - 动机：当前主流程可用，但关键上下文分散在主区和日志中，缺少稳定可扫读的侧边信息面板。
  - 方法：将第1/2/4页重构为 `左主区 + 右侧信息区` 网格，并新增对应数据渲染函数与导出状态同步逻辑。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 卡片1（参数配置）右侧新增：
    - `参数摘要`（模式/输入/目标列/超时/高级参数）；
    - `推荐配置`（按自动/手动模式动态给建议）；
    - `最近一次配置`（回显最近检测参数）。
  - 卡片2（运行中）右侧新增：
    - `阶段时间线`（按当前状态动态高亮）；
    - `任务关键指标`（任务类型、已扫描行数、问题数、超时、预计剩余）。
  - 卡片4（修复结果）右侧新增：
    - `修复结果总览图`（修复前后问题数对比条）；
    - `导出入口`（JSON/CSV 快捷导出按钮）。
  - 交互联动补充：
    - 导出按钮主区与侧边区统一启停；
    - 修复结果完成后自动刷新“前后对比”图；
    - 参数变化时右侧摘要实时更新。
  - 样式层新增：
    - `step-grid/step-side/side-card` 布局系统；
    - 时间线样式、对比图条形样式；
    - 移动端下自动回落单列布局。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 运行中“已扫描行数/预计剩余”目前部分场景仍为估算或“计算中”；若需精确值，需要后端提供实时进度指标。

## 本次改动记录（卡片4重叠修复）

- 改动日期：2026-02-23 09:27:31
- 改动内容简述：
  - 目标：修复第4页“修复摘要/修复明细”视觉重叠问题，避免长路径与长文本挤压到相邻卡片。
  - 动机：修复结果中 `输出文件` 路径较长时，摘要卡片文本发生溢出并影响右侧明细区可读性。
  - 方法：优化第4页网格宽度分配，并为描述列表值字段补齐可收缩与强制换行规则。
- 相关模块/文件：
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 第4页布局空间优化：
    - `repair-grid` 右侧栏宽度由 `320px` 调整为 `292px`，释放主内容区宽度。
  - 防溢出样式修复：
    - `result-grid` 子项新增 `min-width: 0`；
    - `result-card` 增加 `overflow: hidden`；
    - `dl > *` 与 `dd` 增加可收缩与换行规则（`min-width: 0`、`overflow-wrap: anywhere`、`word-break: break-word`）。
  - 修复摘要可读性优化：
    - `#repair-summary` 列宽调整为 `100px + 1fr`；
    - `#repair-summary dd` 字号和行高微调，减少拥挤感。
- 待处理事项：
  - 若后续在更窄窗口仍出现拥挤，可进一步在中等断点下将第4页主区改为单列堆叠（摘要在上，明细在下）。

## 本次改动记录（美化工程第3步：视觉锚点强化）

- 改动日期：2026-02-23 10:17:24
- 改动内容简述：
  - 目标：执行美化工程第3阶段，增强卡片1/2/4的视觉锚点，减少“纯文字信息流”。
  - 动机：当前流程可用，但关键视觉焦点不够突出，用户难以一眼定位“当前阶段”和“修复结果核心指标”。
  - 方法：在结构层增加流程线/KPI 区块，在交互层绑定状态，在样式层引入节点高亮和状态图标。
- 相关模块/文件：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 卡片1（参数配置）：
    - 新增轻量流程线（上传 -> 检测 -> 修复）；
    - 流程线状态与当前会话联动（`done/active/pending`）。
  - 卡片2（运行中）：
    - 新增状态图标（idle/pending/running/succeeded/failed/canceled/timed_out）；
    - 阶段列表（主区与右侧时间线）增加节点图标与连线高亮，不再只有文字。
  - 卡片4（修复结果）：
    - 首屏新增 3 个 KPI badge：`修复数量`、`跳过数量`、`修改单元格`；
    - KPI 与修复结果实时联动，失败或重置时自动归零。
- 已验证结果：
  - `node --check appshell/frontend/src/main.js` 通过。
- 待处理事项：
  - 若后续需要更强品牌化视觉，可在状态图标替换为统一 SVG icon 集并补充动效节奏（例如节点高亮过渡延迟）。

## 本次改动记录（修复能力升级：策略配置/冲突处理/真实对比/回滚）

- 改动日期：2026-02-25 19:01:07
- 改动内容简述：
  - 目标：推进“修复能力从 MVP 到可控生产”的第一步，实现策略可配置、冲突可处理、真实前后对比、回滚与仅计划模式。
  - 动机：原 `repair_batch` 仅按固定规则直接改值，缺少策略治理、依赖控制与可回退保障，难以作为生产修复链路使用。
  - 方法：重构 Python `repair_batch` 执行流程为“计划 -> 冲突决策 -> 应用 -> 对比 -> 可回滚输出”，并补充 Go 透传与测试覆盖。
- 相关模块/文件：
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/engine_protocol.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/README.md`
  - `tests/python_engine/test_engine_cli.py`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/frontend/src/main.js`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 策略配置：
    - 新增 `repair_strategy`：支持 `conflict_policy`、`missing_numeric`、`missing_categorical`、`outlier`、`rare_category`、`issue_priority`、`preview_limit`。
  - 冲突处理：
    - 对同一单元格修复冲突引入策略：`first_wins` / `last_wins` / `skip_conflict`；
    - 响应新增 `conflict_summary` 与冲突样例。
  - 列间依赖：
    - 新增 `column_dependencies`（对象或列表形式）；
    - 支持依赖排序、依赖未满足跳过、依赖环检测跳过。
  - 真实前后对比：
    - 响应新增 `comparison`（修复前后问题总数、类型统计、列统计、真实变更单元格预览）；
    - `applied_repairs` 新增 `before_count/after_count/resolved_count/cells_preview`。
  - 仅计划模式：
    - 新增 `plan_only`，只生成修复计划与真实对比，不写文件；
    - 响应新增 `execution_mode` 与 `write_output_requested`。
  - 回滚机制：
    - 修复写出时可生成回滚备份与 manifest；
    - 新增 action：`rollback_repair_batch`（按 manifest 恢复到 source/output/custom 目标）；
    - 新增错误码：`ROLLBACK_FAILED`。
  - Go 层透传扩展：
    - `repair_batch` 新增透传：`plan_only`、`enable_rollback`、`rollback_dir`、`repair_strategy`、`column_dependencies`；
    - 新增 `rollback_repair_batch` 请求归一化。
  - 前端可读性补充：
    - 第4卡修复明细显示真实对比结果与样例变更；
    - 新增依赖/冲突/策略相关跳过原因中文映射。
- 已验证结果：
  - `python -m pytest -q tests/python_engine` -> `18 passed`
  - `python -m pytest -q tests/python_engine/test_engine_cli.py` -> `12 passed`
  - `go test ./...`（`appshell/backend`）通过
  - `node --check appshell/frontend/src/main.js` 通过
- 待处理事项：
  - 当前列间依赖是规则级排序与跳过机制；后续可扩展为“依赖列值条件触发”的细粒度策略（如条件表达式）。
  - 回滚目前基于 manifest + 文件备份；下一步可增加前端“一键回滚最近一次修复”的可视化入口。
  - 冲突处理已支持通用机制，但真实冲突在当前三类问题中较少；可后续补充更易冲突的新异常类型以强化策略价值验证。
