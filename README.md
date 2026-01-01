# Mixed-Type Data Anomaly Detection & Repair System 🔍

混合数据类型异常检测与修复系统 - 毕业设计项目

本系统集成 **LightGBM** 进行异常检测，**SHAP** 提供可解释性，并使用 **Gower 距离 + KNN** 实现智能修复建议。

## 🚀 核心功能

- **混合类型支持:** 原生处理数值型和分类型混合数据，无需复杂预处理
- **可解释 AI:** 使用 SHAP 值提供全局和局部解释
- **智能修复:** 基于 Gower 距离的 KNN 修复建议 + 修复验证
- **批量检测:** 支持批量检测并导出 CSV 报告
- **可视化分析:** ROC 曲线、混淆矩阵、特征重要性图表
- **交互式界面:** 使用 Streamlit 构建的现代化 UI

## ✨ 功能特性

### 📊 数据分析面板
- 数据统计概览（行数、列数、缺失值、内存占用）
- 数据类型分布（数值型 vs 分类型）
- 缺失值分析报告
- 数值特征描述性统计
- 分类特征分布可视化
- 目标列类别分布 & 不平衡比率

### 📈 模型性能可视化
- **5 大核心指标:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **ROC 曲线:** 带 AUC 面积填充的交互式图表
- **混淆矩阵:** 热力图 + TN/FP/FN/TP 详细解读
- **特征重要性:** 条形图 + Top 5 排名表

### 🔬 单条检测 & 修复
- 样本详情展示
- 实时异常检测
- SHAP 可解释性分析
- 智能修复建议
- **修复验证功能:** 应用修复后查看预测变化

### 📤 批量检测 & 导出
- 全量/自定义范围批量扫描
- 检测结果汇总统计
- 结果筛选（全部/仅异常/仅正常）
- CSV 导出（全部结果/仅异常）

## 📁 项目结构

```
├── app.py                      # Streamlit 主应用入口
├── config.py                   # 配置文件（路径、参数）
├── requirements.txt            # 依赖包
├── README.md
│
├── src/                        # 核心源码模块
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载与预处理
│   ├── repair_module.py        # 异常修复模块（Gower + KNN）
│   └── utils.py                # 工具函数（训练、评估、保存）
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据集
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/              # 模型和处理后的数据
│       ├── model_lgb.pkl
│       ├── test_data.pkl
│       ├── normal_data.pkl
│       └── config.pkl
│
├── outputs/                    # 输出文件
│   ├── figures/                # 生成的图表
│   └── results/                # 结果数据
│
├── scripts/                    # 独立脚本工具
│   ├── benchmark.py            # 基准测试
│   ├── find_anomalies.py       # 查找异常样本
│   └── plot_results.py         # 绘图脚本
│
└── archive/                    # 归档（旧版本代码）
```

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.9+ |
| **机器学习** | LightGBM, Scikit-learn |
| **可解释性** | SHAP |
| **距离度量** | Gower (支持混合类型) |
| **可视化** | Matplotlib, Streamlit Charts |
| **前端界面** | Streamlit |

## 📦 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/Mixed-Type-Anomaly-Detection.git
cd Mixed-Type-Anomaly-Detection
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行应用

```bash
streamlit run app.py
```

### 4. 使用流程

1. **页面 1 - 数据与模型训练**
   - 上传 CSV 数据集
   - 查看数据统计分析
   - 选择目标列（标签列）
   - 点击训练模型
   - 查看性能指标和可视化图表

2. **页面 2 - 检测与修复**
   - **单条检测 Tab:** 选择样本 → 运行检测 → 查看修复建议 → 验证修复效果
   - **批量检测 Tab:** 选择范围 → 批量扫描 → 筛选结果 → 导出 CSV

## 📊 示例数据集

项目使用 [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) 作为演示数据。

## 📝 模块说明

| 模块 | 功能 |
|------|------|
| `src/data_loader.py` | 数据加载、清洗、类型转换 |
| `src/repair_module.py` | 基于 Gower 距离的 KNN 修复建议 |
| `src/utils.py` | 模型训练、评估、可视化数据生成、状态保存/加载 |
| `config.py` | 集中管理路径和配置参数 |

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
