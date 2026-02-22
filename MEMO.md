# MEMO

Last updated: 2026-02-22 10:32:55

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
  - 当前仍以“离散标签（分类）”为主，连续目标（如 `bmi`）暂不支持直接训练。
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

- `python -m pytest -q tests/python_engine` -> `13 passed`
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
