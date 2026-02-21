# MEMO

Last updated: 2026-02-21 17:57:58

## 1. 已完成内容

- 已新增 `appshell/`，用于并行演进新架构（不影响现有 `app.py`/Streamlit）。
- 已完成阶段化模板与验收文档：
  - `appshell/PHASES_ACCEPTANCE.md`
  - `appshell/README.md`
- 已完成运行环境依赖安装与校验：
  - Python 训练依赖（`appshell/core/python_engine/requirements.txt`）
  - 项目主依赖（`requirements.txt`）
  - Node.js LTS（`v24.13.1`）
  - Wails CLI（`v2.11.0`）
- 已完成阶段1：Python 内核产品化（算法层/服务层/CLI 分层）
  - 算法核心从脚本抽离到 `src/training_core.py`
  - `src/utils.py` 保留兼容导出，避免影响 `app.py`
  - Python 引擎分层：
    - `appshell/core/python_engine/engine_core.py`（算法动作）
    - `appshell/core/python_engine/engine_service.py`（action 路由）
    - `appshell/core/python_engine/engine_main.py`（CLI I/O 与统一异常出口）
  - 协议与错误码统一：`appshell/core/python_engine/engine_protocol.py`
  - 结构化日志：`appshell/core/python_engine/engine_logging.py`（JSON 日志到 stderr）
  - 协议样例文件：
    - `appshell/core/python_engine/input.json`
    - `appshell/core/python_engine/output.json`
  - 引擎文档更新：`appshell/core/python_engine/README.md`
  - 阶段1测试补齐：
    - `tests/python_engine/test_training_core.py`
    - `tests/python_engine/test_engine_cli.py`
    - `pytest.ini`
- 已完成阶段2：Go 后端编排层（任务生命周期下沉 Go）
  - 任务编排服务升级：`appshell/backend/internal/task/service.go`
    - 新增核心接口：`RunTask` / `CancelTask` / `GetTaskStatus`
    - 引入并发队列与 worker 池（默认并发 `3`）
    - 支持状态：`pending/running/succeeded/failed/canceled/timed_out`
    - 支持超时、取消、任务 ID 防冲突、旧接口兼容（`Start/Get/Cancel`）
  - Demo 升级：`appshell/backend/cmd/demo/main.go`
    - 支持 `-parallel` 并发提交
    - 支持 `-cancel-after` 定时取消
    - 支持 `-timeout` 任务超时控制
  - 后端文档更新：`appshell/backend/README.md`
  - 阶段2测试补齐：
    - `appshell/backend/internal/task/service_test.go`
    - `appshell/backend/internal/engine/runner_test.go`
- 已完成阶段3：Wails 前端 MVP（最小可用流程）
  - 新增 Wails 绑定入口：
    - `appshell/backend/cmd/wails/main.go`
    - `appshell/backend/cmd/wails/app.go`
  - Go 侧暴露绑定方法：
    - `RunTask(payload)` / `GetTaskStatus(taskID)` / `CancelTask(taskID)`
    - `SelectCSV()` / `SelectOutputDir()`
    - 兼容旧调用：`RunTrainTask(payload)`
  - 前端页面升级：
    - `appshell/frontend/index.html`
    - `appshell/frontend/src/main.js`
    - `appshell/frontend/src/style.css`
  - 前端能力覆盖：参数配置、文件选择、任务启动、轮询进度、结果展示、错误提示、重试、JSON/CSV 导出
  - 新增 Wails 绑定参数测试：
    - `appshell/backend/cmd/wails/app_test.go`
- 已完成阶段4：数据与可观测性（可排查、可追溯）
  - Go 结构化日志：
    - 新增 `appshell/backend/internal/observability/logger.go`
    - `service.go` / `runner.go` 输出 JSON 日志并贯穿 `task_id`
    - `runner.go` 转发 Python `stderr` JSON 日志到 Go 日志流（保留 `task_id`）
    - `cmd/demo` / `cmd/wails` 默认落盘到 `outputs/appshell/go_backend.log`（可用环境变量覆盖）
  - 任务历史本地持久化（sqlite）：
    - 新增 `appshell/backend/internal/task/history_store.go`
    - 新增 `appshell/backend/internal/task/history_sqlite.go`
    - `service.go` 接入历史存储：任务提交/状态变更/结束都会持久化
    - `service.go` 新增历史回查与列表能力：`GetTaskStatus` 回退历史、`ListRecentTasks`
    - 支持仅保留最近 N 条（`APPSHELL_TASK_HISTORY_KEEP`）
  - Wails 侧历史能力：
    - `cmd/wails/app.go` 初始化 sqlite 历史库（`APPSHELL_TASK_DB`）
    - 新增绑定方法：`ListTaskHistory(limit)`
    - 前端 `main.js` 启动时加载最近历史，应用重启可直接看到最近结果
    - 前端事件日志增加 task_id 标记，并输出前端结构化日志到浏览器控制台
  - Demo 与文档同步：
    - `cmd/demo/main.go` 新增 `-history-db` / `-history-keep`
    - `appshell/backend/README.md` 新增可观测性与历史持久化说明
  - 阶段4测试补齐：
    - `service_test.go` 新增：
      - 重启后历史可回查
      - 最近 N 条裁剪
    - `app_test.go` 新增环境配置解析测试

## 2. 已验证结果

- `Python engine health` 成功，返回结构化 JSON。
- `Python engine train` 成功，返回 `status: ok`，并落盘模型产物。
- `Go -> Python` 闭环成功：
  - `go run ./cmd/demo -action health` 可达 `succeeded`
  - `go run ./cmd/demo -action train ...` 可达 `succeeded`
- 阶段1验收已达成：
  - 同一输入 10 次输出一致（`test_train_model_is_deterministic_across_10_runs` 通过）
  - 异常输入返回明确错误码（`INVALID_JSON`/`INVALID_INPUT`/`UNKNOWN_ACTION`/`INVALID_TARGET_COLUMN`）
  - 核心算法覆盖率：`src/training_core.py` 为 `95%`（>= 80%）
- 当前新增测试总计 `7 passed`。
- 阶段2验收已达成：
  - 可同时跑至少 3 个任务（`TestRunTaskSupportsAtLeastThreeConcurrentTasks` 通过，demo `-parallel 3` 实测 3 个任务同时 `running`）
  - 取消后 2 秒内状态变更（`TestCancelTaskTransitionsToCanceledWithinTwoSeconds` 通过，实测约 0.04s）
  - 超时任务可回收且无僵尸（`TestRunnerTimeoutDoesNotLeaveZombieProcess` 通过）
- Go 测试结果：`go test ./...` 通过。
- 阶段3验收已达成：
  - UI 支持“配置(含文件选择) -> 运行 -> 查看结果 -> 导出(JSON/CSV)”全流程
  - 运行期间通过异步轮询更新状态（前端不阻塞）
  - 错误可读且可重试（错误面板 + `重试上次参数`）
  - Wails 入口编译通过：`go run ./cmd/wails -h`
- 阶段4验收已达成：
  - 给定任务 ID 可联查三层日志：
    - 前端：事件行含 `[task_id]`，并在 console 输出 JSON（`layer=frontend`）
    - Go：`service/runner` JSON 日志均包含 `task_id`（默认落盘 `outputs/appshell/go_backend.log`）
    - Python：`engine_main.py` 结构化日志含 `task_id`，且被 Go 转发
  - 应用重启后可查看历史任务结果：
    - sqlite 落盘：`outputs/appshell/task_history.sqlite`
    - `GetTaskStatus` 可回退查询历史
    - 前端启动自动加载最近历史任务
- 当前后端测试结果：`go test ./...` 通过（含新增 sqlite 历史测试）。

## 3. 当前环境与限制

- 当前机器有 Go（`go1.25.6`）。
- 当前机器已有 Node.js（`v24.13.1`）与 Wails（`v2.11.0`）。
- 当前 Python 训练依赖已安装（`pandas/numpy/lightgbm/scikit-learn/joblib` 可导入）。
- 个别终端会话可能未刷新 PATH；如遇 `node` 不识别，可执行：`$env:Path += ';D:\NOde'` 或重开终端。
- 在受限环境执行 Go 仍建议设置可写缓存（如：`GOCACHE=./.gocache`）。
- 现有 Python 环境存在 Anaconda 与用户 site-packages 混用，后续建议切到项目独立 venv 以降低环境漂移风险。

## 4. 快速复现命令

```bash
# Python engine health
echo '{"task_id":"health-1","action":"health","payload":{}}' | python appshell/core/python_engine/engine_main.py

# Python engine train
python appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/sample_train_request.json

# 错误码与协议测试
python -m pytest -q tests/python_engine/test_engine_cli.py

# 核心算法测试（含 10 次一致性）
python -m pytest -q tests/python_engine/test_training_core.py

# 核心算法覆盖率（推荐此方式）
python -m coverage run -m pytest -q tests/python_engine/test_training_core.py
python -m coverage report -m src/training_core.py

# Go demo health/train
cd appshell/backend
GOCACHE=./.gocache go run ./cmd/demo -action health
GOCACHE=./.gocache go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke

# Go 编排层验收（并发/取消/超时）
GOCACHE=./.gocache go test ./...
GOCACHE=./.gocache go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -parallel 3 -output ../../outputs/results/parallel_phase2
GOCACHE=./.gocache go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -cancel-after 1s -timeout 90s
GOCACHE=./.gocache go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke -timeout 400ms

# Wails MVP
go run ./cmd/wails -engine ../core/python_engine/engine_main.py
# (快速编译校验)
go run ./cmd/wails -h

# 阶段4：可观测性与历史持久化
go run ./cmd/demo -action health -history-db ../../outputs/appshell/task_history.sqlite -history-keep 100
go run ./cmd/wails -engine ../core/python_engine/engine_main.py
# (可选) 指定历史与日志路径
$env:APPSHELL_TASK_DB="D:\\code\\pythoncode\\Anomaly Detection and Repair for Mixed Data Type Inputs\\outputs\\appshell\\task_history.sqlite"
$env:APPSHELL_TASK_HISTORY_KEEP="100"
$env:APPSHELL_GO_LOG_FILE="D:\\code\\pythoncode\\Anomaly Detection and Repair for Mixed Data Type Inputs\\outputs\\appshell\\go_backend.log"
```

## 5. 下一步建议（优先顺序）

1. 阶段5：Windows 打包落地（Python 引擎可执行化 + Wails 资源打包 + 安装包出品）。
2. 阶段5补充：补齐安装器行为（桌面快捷方式、卸载入口、版本号与升级策略）。
3. 质量收尾：补充 UI E2E 与历史任务恢复场景测试（含重启回放）。
4. 工程化收尾：统一 Python 依赖锁定与虚拟环境策略，避免跨解释器污染。

## 6. 新对话续接提示词（可直接粘贴）

```text
请基于项目根目录的 MEMO.md 继续推进任务，开始阶段5：
1) 完善 appshell/build/windows/build.ps1，串联 Python 引擎打包 + Wails 打包 + 安装器产物输出。
2) 明确安装目录结构与运行时资源查找规则（engine 可执行文件、模型输出目录）。
3) 给出可执行打包命令、验收标准和常见失败排查清单。
```

## 7. 本次改动记录（阶段4）

- 改动日期：2026-02-21 17:57:58
- 改动内容简述：
  - 最终目标：实现“问题可排查、结果可追溯”的阶段4能力，确保任务 ID 可贯穿三层日志，且任务历史可在重启后回查。
  - 采取方法：在 Go 后端新增统一结构化日志与 sqlite 历史存储，把任务生命周期快照持续落盘；在 Wails/前端补齐历史读取与 task_id 事件标记。
  - 当前已完成步骤：Go/Python/前端 task_id 联查链路已打通；sqlite 历史持久化已接入服务层；重启回查与最近 N 条策略已通过测试。
- 相关模块/文件：
  - `appshell/backend/internal/observability/logger.go`
  - `appshell/backend/internal/task/history_store.go`
  - `appshell/backend/internal/task/history_sqlite.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/backend/internal/engine/runner.go`
  - `appshell/backend/cmd/wails/app.go`
  - `appshell/backend/cmd/demo/main.go`
  - `appshell/backend/internal/task/service_test.go`
  - `appshell/backend/cmd/wails/app_test.go`
  - `appshell/frontend/src/main.js`
  - `appshell/backend/README.md`
  - `appshell/backend/go.mod`
  - `appshell/backend/go.sum`
- 已解决的问题/新增功能：
  - 新增 Go 结构化日志（JSON）并在任务关键节点打点，日志含 `task_id`。
  - 新增 Python `stderr` 日志转发到 Go 日志流，方便按 `task_id` 跨层联查。
  - 新增任务历史 sqlite 持久化，支持重启后按任务 ID 回查。
  - 新增“保留最近 N 条历史”策略，防止历史无限增长。
  - 新增 Wails 绑定 `ListTaskHistory(limit)`，前端启动时可加载最近历史任务。
- 待处理事项：
  - 增加历史任务列表筛选（按状态/时间/task_id）与点击查看详情能力（当前默认加载最近记录）。
  - 将 Go 结构化日志默认落盘策略进一步产品化（例如统一日志目录与轮转策略）。
  - 在阶段5打包中纳入 sqlite 与日志目录初始化逻辑，并补充安装后排障手册。
