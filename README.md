# Mixed-Type Data Anomaly Detection & Repair System 🔍

混合数据类型异常检测与修复系统 - 毕业设计项目

本系统集成 **LightGBM** 进行异常检测，**SHAP** 提供可解释性，并使用 **Gower 距离 + KNN** 实现智能修复建议。

## 🏗️ 项目架构

本项目采用**双轨并行**架构：

- **旧版入口 (app.py):** Streamlit 演示界面，用于算法验证和快速原型
- **新版入口 (appshell/):** 产品化桌面应用，采用 `Python Engine + Go Backend + Wails Frontend` 架构

推荐使用 **appshell** 作为主要交付路径，它提供更完整的任务管理、进度追踪和可观测性。

## 🚀 核心功能

- **混合类型支持:** 原生处理数值型和分类型混合数据，无需复杂预处理
- **可解释 AI:** 使用 SHAP 值提供全局和局部解释
- **智能修复:** 基于 Gower 距离的 KNN 修复建议 + 修复验证
- **整表扫描:** 支持列级异常检测，输出缩略图、热点片段和问题统计
- **批量修复:** 支持按选中问题 ID 批量修复，并生成回滚清单
- **启动自检:** 应用启动前检查 Python 引擎、运行时依赖、结果目录、SQLite 和默认模型状态
- **可视化分析:** ROC 曲线、混淆矩阵、特征重要性图表
- **桌面应用:** 使用 Wails 构建的跨平台桌面界面

## ✨ 功能特性

### 🎯 Python 引擎支持的操作

- **health:** 健康检查
- **train:** 模型训练（支持 auto/classification/regression 任务类型）
- **scan_file:** 整表列级扫描，输出列缩略图、热点片段、问题类型统计、置信度和特征解释
- **repair:** 单样本修复（基于模型驱动的搜索）
- **repair_batch:** 批量修复（按选中的 issue_ids 执行）
- **rollback_repair_batch:** 回滚批量修复

### Stage 0 Foundation

- 当前 `health / train / repair / scan_file / repair_batch / rollback_repair_batch` 六个 engine actions 已冻结为未来 Tool Layer 的稳定基础。
- `MULTI_AGENT_BLUEPRINT.md` 定义长期智能化升级方向，`TOOL_LAYER_FOUNDATION.md` 定义 Stage 0 的 action/tool/algorithm asset 映射。
- Stage 0 只做“保留与包装现有资产”，不引入用户可见的智能化入口变化，也不改变现有线协议与主流程。

### 🔍 支持的异常类型

- 缺失值 (missing values)
- 异常值 (outliers)
- 稀有类别 (rare categories)
- 时间序列偏移 (time_series_shift)
- 跨列一致性问题 (cross_column_consistency)
- 重复记录 (duplicate_record)

### 📊 Streamlit 演示界面 (app.py)

- 数据统计概览与类型分布
- 模型训练与性能可视化（ROC、混淆矩阵、特征重要性）
- 单条样本检测与 SHAP 解释
- 修复建议与验证
- 批量检测与 CSV 导出

### 🖥️ Wails 桌面应用 (appshell/)

- 启动自检与诊断信息复制
- CSV 列读取与目标列选择
- 训练任务发起与实时进度展示
- 整表扫描任务与问题勾选
- 批量修复任务与回滚清单展示
- 任务历史查询与诊断信息

## 📁 项目结构

```
├── app.py                      # Streamlit 演示入口（旧版）
├── config.py                   # 配置文件（路径、参数）
├── requirements.txt            # Python 依赖包
├── README.md
│
├── src/                        # 核心算法模块
│   ├── training_core.py         # 训练核心逻辑
│   ├── repair_core.py           # 单样本修复搜索逻辑
│   ├── data_loader.py           # 数据加载与预处理
│   └── utils.py                 # 工具函数
│
├── appshell/                   # 桌面应用主路径（新版）
│   ├── core/python_engine/      # Python 引擎层
│   │   ├── engine_main.py       # CLI 入口
│   │   ├── engine_service.py    # 动作路由
│   │   ├── engine_protocol.py   # 协议与错误结构
│   │   ├── engine_logging.py    # 结构化日志
│   │   └── engine_core.py       # 核心业务逻辑
│   ├── backend/                 # Go 编排层
│   │   ├── internal/engine/     # Python 进程管理
│   │   ├── internal/task/       # 任务编排与历史
│   │   └── cmd/wails/           # Wails 应用入口
│   ├── frontend/                # Wails 前端
│   └── build/windows/           # Windows 打包脚本
│
├── data/                       # 数据目录
│   ├── raw/                     # 原始数据集
│   └── processed/               # 模型和处理后的数据
│
├── outputs/                    # 输出文件
│   ├── results/                 # 训练与检测结果
│   └── appshell/                # Go 日志与任务历史
│       ├── go_backend.log
│       └── task_history.sqlite
│
├── tests/                      # 测试用例
│   └── python_engine/
│
├── thesis-defense/             # 答辩材料
│   ├── 流程图
│   ├── 对比图
│   └── Defense_Poster_Draft_2026-03.pptx
│
└── scripts/                    # 独立脚本工具
```

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.9+, Go 1.21+ |
| **机器学习** | LightGBM, Scikit-learn |
| **可解释性** | SHAP |
| **距离度量** | Gower (支持混合类型) |
| **可视化** | Matplotlib, Streamlit Charts |
| **前端界面** | Streamlit (演示), Wails (桌面应用) |
| **任务编排** | Go (并发调度、超时控制) |
| **持久化** | SQLite (任务历史) |

## 📦 快速开始

### 方式 1: Streamlit 演示界面（快速验证）

#### 1. 克隆仓库

```bash
git clone <repository-url>
cd "Anomaly Detection and Repair for Mixed Data Type Inputs"
```

#### 1.5 推荐独立环境（Windows）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows_env.ps1
```

这会在仓库内创建 `.venv-win`，并按 `requirements.lock.txt` 安装已经验证过的依赖集合。更完整的环境说明见 `ENVIRONMENT.md`。
#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 运行应用

```bash
streamlit run app.py
```

#### 4. 使用流程

1. **页面 1 - 数据与模型训练**
   - 上传 CSV 数据集
   - 查看数据统计分析
   - 选择目标列（标签列）
   - 点击训练模型
   - 查看性能指标和可视化图表

2. **页面 2 - 检测与修复**
   - **单条检测 Tab:** 选择样本 → 运行检测 → 查看修复建议 → 验证修复效果
   - **批量检测 Tab:** 选择范围 → 批量扫描 → 筛选结果 → 导出 CSV

### 方式 2: Wails 桌面应用（产品化路径）

#### 1. 环境准备

- Python 3.11（推荐使用仓库内 `.venv-win` 独立环境）
- Go 1.21+
- Node.js (用于前端构建)
- Wails CLI: `go install github.com/wailsapp/wails/v2/cmd/wails@latest`

推荐先执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows_env.ps1
```

#### 2. 运行开发模式

```bash
cd appshell
wails dev
```

#### 3. 构建桌面应用

```bash
cd appshell
wails build
```

Windows 打包脚本位于 `appshell/build/windows/`。

#### 4. 启动自检

桌面端在进入现有四步向导前会先执行一次阻塞式启动自检：

- 检查 `engine_main.py` 是否存在且可执行
- 调用 Python `health` 动作验证 `pandas`、`numpy`、`lightgbm`、`scikit-learn`、`joblib`
- 自动创建并检查 `outputs/results/` 是否可写
- 自动创建并检查 `APPSHELL_TASK_DB` 对应的 SQLite 目录与库文件
- 检查默认模型候选目录 `outputs/results/wails_repair/` 和 `data/processed/`

判定规则：

- Python 引擎、运行时依赖、输出目录、SQLite 失败时会阻塞进入应用
- 默认模型缺失只会显示 warning，表示“尚未训练”而不是环境损坏
- 浏览器静态预览模式会使用 mock 自检结果，因此不会执行真实的 Python/SQLite 检查

## 📊 示例数据集

项目使用 [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) 作为演示数据。

## 📝 模块说明

### Python 引擎层 (appshell/core/python_engine/)

| 模块 | 功能 |
|------|------|
| `engine_main.py` | CLI 入口，处理 JSON 请求/响应 |
| `engine_service.py` | 动作路由（health/train/scan_file/repair/repair_batch/rollback_repair_batch） |
| `engine_protocol.py` | 协议定义与错误码 |
| `engine_logging.py` | 结构化日志输出 |
| `engine_core.py` | 训练、扫描、修复、批量修复、回滚主逻辑 |

### 核心算法层 (src/)

| 模块 | 功能 |
|------|------|
| `training_core.py` | 训练核心逻辑，支持 auto/classification/regression |
| `repair_core.py` | 单样本修复搜索逻辑 |
| `data_loader.py` | 数据加载、清洗、类型转换 |
| `utils.py` | 模型训练、评估、可视化、状态保存/加载 |

### Go 编排层 (appshell/backend/)

| 模块 | 功能 |
|------|------|
| `internal/engine/runner.go` | Python 进程管理与事件流接收 |
| `internal/task/service.go` | 任务编排、并发调度、超时取消、状态轮询 |
| `internal/observability/` | 结构化日志 |
| `cmd/wails/` | Wails 应用入口与 Go 绑定 |

## 🔧 配置

所有路径和参数集中在 `config.py` 中管理，无需硬编码：

```python
from config import FILES, PATHS

# 获取模型路径
model_path = FILES["model"]

# 获取数据目录
data_dir = PATHS["data_processed"]
```

## 🎯 性能优化

- **SHAP Explainer 缓存:** 使用 `@st.cache_resource` 缓存，首次加载后毫秒级响应
- **模型状态缓存:** 避免重复加载模型文件
- **批量预测:** 向量化操作，高效处理大量样本

## 📄 License

MIT License

## Cross-Dataset Validation Pipeline

The thesis experiment workflow now includes a reproducible cross-dataset
validation pipeline. It keeps the existing M1/M2/M3 stroke artifacts intact and
writes new paper-level outputs under:

```text
artifacts/experiments/cross_dataset/
```

Run the complete validation:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --all
```

Run individual stages:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --generate
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --detect
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --repair
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --threshold-sensitivity
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --scale
```

The pipeline covers `stroke`, `orders_transactions`, and `user_device_logs`.
It generates or reuses clean CSVs, injects five issue types, writes ground truth,
runs the existing Python Engine scan and `repair_batch` paths, and emits summary
CSV files for detection, repair, numeric outlier threshold sensitivity, and
scale testing.

See `docs/cross_dataset_experiments.md` for the dataset fields, injection rules,
metric definitions, output files, and thesis-writing guidance.
