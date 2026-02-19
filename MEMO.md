# MEMO

Last updated: 2026-02-17

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
```

## 5. 下一步建议（优先顺序）

1. 阶段4：Windows 打包落地（Python 引擎可执行化 + Wails 资源打包 + 安装包出品）。
2. 阶段4补充：补齐安装器行为（桌面快捷方式、卸载入口、版本号与升级策略）。
3. 工程化收尾：统一 Python 依赖锁定与虚拟环境策略，避免跨解释器污染。
4. 质量收尾：补充 UI E2E 用例与关键路径冒烟脚本。

## 6. 新对话续接提示词（可直接粘贴）

```text
请基于项目根目录的 MEMO.md 继续推进任务，开始阶段4：
1) 完善 appshell/build/windows/build.ps1，串联 Python 引擎打包 + Wails 打包 + 安装器产物输出。
2) 明确安装目录结构与运行时资源查找规则（engine 可执行文件、模型输出目录）。
3) 给出可执行打包命令、验收标准和常见失败排查清单。
```
