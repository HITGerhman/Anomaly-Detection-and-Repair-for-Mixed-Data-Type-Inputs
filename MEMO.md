# MEMO

Last updated: 2026-02-17

## 1. 已完成内容

- 已新增 `appshell/`，用于并行演进新架构（不影响现有 `app.py`/Streamlit）。
- 已完成阶段化模板与验收文档：
  - `appshell/PHASES_ACCEPTANCE.md`
  - `appshell/README.md`
- 已完成 Python 引擎模板（`stdin/stdout JSON` 协议）：
  - `appshell/core/python_engine/engine_main.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/engine_protocol.py`
  - `appshell/core/python_engine/sample_train_request.json`
  - `appshell/core/python_engine/requirements.txt`
- 已完成 Go 后端模板（Runner + Task 生命周期）：
  - `appshell/backend/internal/engine/protocol.go`
  - `appshell/backend/internal/engine/runner.go`
  - `appshell/backend/internal/task/service.go`
  - `appshell/backend/cmd/demo/main.go`
- 已完成前端/Wails 对接模板：
  - `appshell/frontend/index.html`
  - `appshell/frontend/src/main.js`
  - `appshell/frontend/src/style.css`
- 已完成 Windows 打包占位模板：
  - `appshell/build/windows/build.ps1`
  - `appshell/build/windows/installer.iss`
- 已完成运行环境依赖安装与校验：
  - Python 训练依赖（`appshell/core/python_engine/requirements.txt`）
  - 项目主依赖（`requirements.txt`）
  - Node.js LTS（`v24.13.1`）
  - Wails CLI（`v2.11.0`）

## 2. 已验证结果

- `Python engine health` 成功，返回结构化 JSON。
- `Go -> Python health` 闭环成功，任务状态可从 `running` 进入 `succeeded`。
- `Python engine train` 已成功，返回 `status: ok`（不再出现 `MISSING_DEPENDENCY`）。
- `Go demo train` 已成功，产物落盘到 `data/processed/`。
- 已修正 Go demo 默认引擎路径为 `../core/python_engine/engine_main.py`。

## 3. 当前环境与限制

- 当前机器有 Go（`go1.25.6`）。
- 当前机器已有 Node.js（`v24.13.1`）与 Wails（`v2.11.0`）。
- 当前 Python 训练依赖已安装（`pandas/numpy/lightgbm/scikit-learn/joblib` 可导入）。
- 个别终端会话可能未刷新 PATH；如遇 `node` 不识别，可执行：`$env:Path += ';D:\\NOde'` 或重开终端。
- 在受限环境执行 Go 仍建议设置可写缓存（如：`GOCACHE=./.gocache`）。

## 4. 快速复现命令

```bash
# Python health
echo '{"task_id":"health-1","action":"health","payload":{}}' | python appshell/core/python_engine/engine_main.py

# 安装 Python 引擎训练依赖
pip install -r appshell/core/python_engine/requirements.txt

# 安装项目主依赖
pip install -r requirements.txt

# Python train
python appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/sample_train_request.json

# 检查 Node/Wails
node -v
npm -v
wails version

# Go demo health
cd appshell/backend
GOCACHE=./.gocache go run ./cmd/demo -action health

# Go demo train
GOCACHE=./.gocache go run ./cmd/demo -action train -csv ../../data/raw/healthcare-dataset-stroke-data.csv -target stroke
```

## 5. 下一步建议（优先顺序）

1. 后端接 Wails：在 Go 里实现 `RunTrainTask`/`GetTask`/`CancelTask` 并暴露绑定。
2. 前端联调：把 `appshell/frontend/src/main.js` 从 mock 调用切换为真实 Wails 绑定。
3. 打包落地：按 `appshell/build/windows/build.ps1` 补全实际命令，完成安装包出品。
4. （可选）统一 Python 依赖版本锁定策略，减少 Anaconda/用户 site-packages 混用带来的环境漂移。

## 6. 新对话续接提示词（可直接粘贴）

```text
请基于项目根目录的 MEMO.md 继续推进任务。优先处理第5节中的第1和第2步：
1) 在 Go 侧实现并暴露 Wails 绑定方法（RunTrainTask/GetTask/CancelTask）。
2) 修改 frontend/src/main.js 使用真实绑定并完成最小可用联调。
完成后请给出可执行命令和验收标准。
```
