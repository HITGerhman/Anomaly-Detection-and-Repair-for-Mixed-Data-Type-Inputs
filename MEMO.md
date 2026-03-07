# MEMO

Last updated: 2026-03-07 16:40:10

## 改动日期

2026-03-07 16:40:10

## 改动内容简述

- 目标：在保留最新项目状态快照的基础上，补充一份面向编程新手的系统学习路径文档。
- 动机：项目同时包含 Python 算法、Go 编排、Wails 前端和答辩材料，新手容易不知道从哪开始学、先跑什么命令、按什么顺序读代码。
- 方法：
  - 新建私有学习文档 `LEARNING_PATH_PRIVATE.md`；
  - 在 `.gitignore` 中忽略该文件，避免把个人学习笔记纳入版本控制；
  - 文档中按“旧版 Streamlit 路径 -> Python 核心 -> Engine 协议 -> Go 后端 -> 前端/Wails”的顺序组织学习路线，并补充 PowerShell 下可直接运行的命令。

## 最终目标

将“混合类型异常检测与修复”项目整理为一条可交付、可演示、可追踪的完整链路：

- 保留 Python 算法能力，支持训练、扫描、修复与结果导出。
- 以 `appshell/` 为产品化主路径，形成 `Python engine + Go 编排 + Wails 前端` 的桌面应用架构。
- 保留 `app.py` 作为旧版 Streamlit 演示入口，便于算法验证和对照。
- 同步维护答辩材料，保证研究结果、流程图和对比图可直接复用。

## 当前采取的方法

1. 双轨并行：
   - 旧路径：`app.py` + `src/`，继续承担算法演示和兼容入口。
   - 新路径：`appshell/`，承担协议化、任务化、桌面化交付。
2. 协议优先：
   - Python 引擎统一使用 JSON 请求/响应协议。
   - 已形成 `health / train / scan_file / repair / repair_batch / rollback_repair_batch` 动作集。
3. 任务化运行：
   - 前端不直接调 Python，统一通过 Go 后端启动、轮询、取消任务。
4. 可观测性优先：
   - 通过阶段事件、结构化日志、sqlite 任务历史保留运行痕迹。
5. 研究与交付并行：
   - `outputs/` 保存结果与日志。
   - `thesis-defense/` 保存答辩图表与海报素材。

## 当前项目状态

### 1. Python 算法与协议层

- `src/training_core.py` 已承担训练核心逻辑，支持 `task_type=auto/classification/regression`。
- `src/repair_core.py` 已承担单样本修复搜索逻辑。
- `appshell/core/python_engine/` 已形成稳定引擎边界：
  - `engine_main.py`：CLI/输入输出入口
  - `engine_service.py`：动作路由
  - `engine_protocol.py`：协议与错误结构
  - `engine_logging.py`：结构化日志
  - `engine_core.py`：训练、扫描、修复、批量修复、回滚主逻辑
- `scan_file` 已支持整表列级扫描，并输出：
  - `column_thumbnails`
  - `hot_segments`
  - `issue_type_counts`
  - `confidence`
  - `explain_features`
- 当前扫描已覆盖的异常类型包括：
  - 缺失值/异常值/稀有类别等常规问题
  - `time_series_shift`
  - `cross_column_consistency`
  - `duplicate_record`
- `repair_batch` 已支持按选中 `issue_ids` 执行批量修复，并可生成回滚清单。
- `rollback_repair_batch` 已在引擎层可用。

### 2. Go 编排层

- `appshell/backend/internal/engine/runner.go` 已负责调用 Python 引擎并流式接收 `stderr` 事件。
- `appshell/backend/internal/task/service.go` 已实现：
  - `RunTask`
  - `GetTaskStatus`
  - `CancelTask`
  - 并发任务调度
  - 超时与取消
  - 实时进度聚合
- `outputs/appshell/` 下已存在：
  - `go_backend.log`
  - `task_history.sqlite`
- 当前任务状态链路已经覆盖：
  - `pending`
  - `running`
  - `succeeded`
  - `failed`
  - `canceled`
  - `timed_out`

### 3. Wails 前端层

- `appshell/frontend/src/main.js` 已接入真实任务调用，而不只是静态页面。
- 前端当前已具备：
  - CSV 列读取
  - 扫描任务发起
  - 批量修复任务发起
  - 真实进度展示
  - 失败定位展示
  - 结果页任务诊断信息展示
  - 回滚清单结果展示
- 当前前端已经读取并渲染：
  - `task.progress`
  - `response.result.observability`
  - `column_thumbnails`
  - `selected issue_ids`

### 4. 数据、测试与输出资产

- `data/raw/` 当前可见主要数据集：
  - `healthcare-dataset-stroke-data.csv`
  - `creditcard.csv`
  - 若干异常修复测试用 CSV
- `tests/python_engine/` 当前包含：
  - `test_training_core.py`
  - `test_repair_core.py`
  - `test_engine_cli.py`
- `outputs/results/` 已保留多轮训练、检测、修复与基准结果目录。

### 5. 答辩材料目录

- `thesis-defense/` 当前已包含：
  - 流程图 `Figure_A_End_to_End_Workflow*`
  - 中风数据集模型对比图
  - 严格版模型对比图
  - 综合评分图
  - `Defense_Poster_Draft_2026-03.pptx`
- 这些文件属于本仓库内的答辩输出物，应保留为项目相关资产。

## 目前已完成的关键步骤

1. 已完成算法核心从 UI 逻辑中抽离，形成可调用引擎边界。
2. 已完成 Python 引擎协议、错误码和结构化日志。
3. 已完成 Go 任务编排、超时取消、状态轮询和任务历史持久化。
4. 已完成前端对真实进度和真实结果的接入，不再只是估算状态。
5. 已完成整表扫描、问题勾选、批量修复和回滚清单生成的主流程能力。
6. 已完成研究输出物整理，仓库内保留了答辩图表和海报草稿。

## 当前问题

1. 文档存在漂移：
   - 根目录 `README.md` 仍以旧版 Streamlit 说明为主。
   - `appshell/backend/README.md` 仍只写到 `train/repair`，没有同步 `scan_file / repair_batch / rollback_repair_batch`。
2. Windows 打包链路只有脚手架：
   - `appshell/build/windows/build.ps1`
   - `appshell/build/windows/installer.iss`
   但 memo 中没有可靠证据表明“干净机器安装验证”已经完成。
3. 回滚能力目前主要停留在引擎和结果展示层，前端未形成单独的“执行回滚”交互闭环。
4. 单样本模型驱动修复目前仍以分类模型为主；回归场景更适合走规则型批量修复路径。
5. 本次仅更新 memo，没有重新执行 Python/Go 端测试，因此当前 memo 不把历史测试通过记录当作本次验证结论。

## 相关模块/文件

- `.gitignore`
- `LEARNING_PATH_PRIVATE.md`
- `MEMO.md`
- `README.md`
- `app.py`
- `src/training_core.py`
- `src/repair_core.py`
- `src/utils.py`
- `appshell/core/python_engine/engine_core.py`
- `appshell/core/python_engine/engine_main.py`
- `appshell/backend/internal/task/service.go`
- `appshell/backend/internal/engine/runner.go`
- `appshell/frontend/src/main.js`
- `tests/python_engine/test_training_core.py`
- `tests/python_engine/test_repair_core.py`
- `tests/python_engine/test_engine_cli.py`
- `outputs/appshell/go_backend.log`
- `outputs/appshell/task_history.sqlite`
- `thesis-defense/`

## 已解决的问题/新增功能

- 已新增面向新手的私有学习路径文档，内容覆盖：
  - 推荐平台与工具；
  - PowerShell 常用命令；
  - 项目分层理解方式；
  - 旧版 Streamlit 入口的学习顺序；
  - Python engine、Go backend、Wails 前端的阅读顺序；
  - 建议练习和常见踩坑点。
- 已将 `LEARNING_PATH_PRIVATE.md` 加入 `.gitignore`，避免提交到仓库。
- 当前 memo 继续保留为“项目状态快照”，学习路线转移到独立文件维护，职责更清晰。

## 待处理事项

1. 同步更新 `README.md` 和 `appshell/backend/README.md`，使文档与当前代码能力一致。
2. 补做或补记 Windows 打包与安装链路的实际验收结果。
3. 若继续推进桌面闭环，补上前端“执行回滚”和历史任务回放入口。
4. 后续可把 `LEARNING_PATH_PRIVATE.md` 继续细化为“第一周任务版”或“逐文件精读版”。
5. 下次涉及代码改动后，重新记录对应测试或手工验证结果。
6. 后续每次项目更新继续在本文件维持“当前状态快照”，避免再次堆积无关历史信息。
