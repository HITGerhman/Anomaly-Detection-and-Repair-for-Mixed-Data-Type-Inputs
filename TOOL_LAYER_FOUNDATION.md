# Stage 0 Tool Layer Foundation

本文档是 `MULTI_AGENT_BLUEPRINT.md` 在 Stage 0 的落地定义文件，用于把当前稳定能力正式映射为未来 Tool Layer 的基础资产。

## 术语约定

- `action`：当前对外稳定调用名，由 Python engine 通过 JSON 协议暴露。
- `tool`：未来 Agent Runtime 调用的能力单元，是对 action 或算法资产的稳定包装。
- `algorithm asset`：被 tool 封装的底层算法、规则、模型逻辑或状态读写能力。
- `artifact`：模型文件、结果文件、回滚清单、任务历史和衍生数据产物。

## 当前 Action 到未来 Tool 的映射

| 当前 action | future tool id | 核心输入 | 关键输出 | 副作用 | 负责层 |
| --- | --- | --- | --- | --- | --- |
| `health` | `engine.health` | 无 | 引擎元数据、依赖快照、稳定 action 列表 | 无 | Deterministic Tool Layer |
| `train` | `engine.train_model` | `csv_path`, `target_col` | `artifacts`, `data_profile`, `metrics` | 写入模型与状态产物 | Deterministic Tool Layer |
| `scan_file` | `engine.scan_table` | `csv_path` | `issues`, `column_thumbnails`, `scan_summary`, `data_profile` | 只读扫描 | Deterministic Tool Layer |
| `repair` | `engine.repair_sample` | `model_dir` | `repair_summary`, `repair_changes`, `repaired_sample` | 可选写出单样本修复结果 | Deterministic Tool Layer |
| `repair_batch` | `engine.repair_batch` | `csv_path` | `comparison`, `applied_repairs`, `skipped_issues`, `rollback` | 可写出修复结果与回滚清单 | Deterministic Tool Layer |
| `repair_with_gower` | `engine.repair_with_gower` | `csv_path` | `comparison`, `applied_repairs`, `neighbor_evidence`, `rollback` | 可写出 Gower 邻居修复结果与回滚清单 | Deterministic Tool Layer |
| `rollback_repair_batch` | `engine.rollback_batch` | `manifest_path` | `restored_to`, `backup_csv` | 覆写目标文件为回滚版本 | Deterministic Tool Layer |

### Stage 0 冻结结论

- 上表六个 action 是当前唯一稳定对外执行入口。
- Stage 0 不新增对外 action，不改变名称，不改变请求/响应协议。
- 未来 Agent Runtime 应优先调用 `future tool id`，而不是重新定义同类执行能力。

## 当前算法资产到未来包装方式的映射

| 算法资产 | 当前职责 | 当前位置 | 未来包装方式 |
| --- | --- | --- | --- |
| `LightGBM` 训练与评分 | 训练模型、阈值控制、特征重要性、候选排序 | `src/training_core.py` | 继续作为 `engine.train_model` 与未来评分类 tool 的底层资产 |
| 规则扫描 | 缺失值、离群点、稀有类别、一致性、重复、时序偏移检测 | `appshell/core/python_engine/engine_core.py` | 继续作为 `engine.scan_table` 与未来验证类 tool 的底层资产 |
| `repair_core.py` | 单样本确定性修复搜索 | `src/repair_core.py` | 继续作为 `engine.repair_sample` 的主修复器 |
| `repair_module.py` / Gower | 混合类型近邻检索与修复建议 | `src/repair_module.py` | Stage 2 已包装为 `repair_with_gower` 正式工具，并作为多候选修复来源之一 |
| 模型状态读写 | 保存/加载 `model_lgb.pkl`、`test_data.pkl` 等 | `src/training_core.py`, `src/utils.py` | 继续作为训练/修复工具的状态资产读写基础 |

### Stage 0 资产结论

- `LightGBM` 不是待替换模块，而是未来 tool 体系中的评分与排序基础。
- 规则扫描是标准检测基础，不允许在 agent 层重复实现。
- `repair_core.py` 继续承担当前主修复器职责。
- `repair_module.py` 已在 Stage 2 中正式接入为 `repair_with_gower`，并继续作为邻居检索 tool 的底层资产。

## 当前产物与状态文件映射

| 产物 / 状态文件 | 主要来源 action | 当前用途 | 后续去向 |
| --- | --- | --- | --- |
| `model_lgb.pkl` | `train` | LightGBM 模型文件 | Artifact Layer 持续保留 |
| `test_data.pkl` | `train` | 测试集状态 | Artifact Layer 持续保留 |
| `normal_data.pkl` | `train` | 正常样本库 | Artifact Layer 持续保留，并供未来邻居类 tool 使用 |
| `config.pkl` | `train` | 特征配置与状态 | Artifact Layer 持续保留 |
| 扫描问题清单 | `scan_file` | 异常目录、缩略图、解释信息 | 未来进入 Task Trace 与结果图表层 |
| 单样本修复 CSV / JSON | `repair` | 修复结果与报告 | 未来进入 Artifact Layer |
| 批量修复输出 CSV | `repair_batch` | 修复后的表格结果 | 未来进入 Artifact Layer |
| rollback manifest / backup | `repair_batch`, `rollback_repair_batch` | 回滚保护与恢复依据 | 未来进入 Task Trace 与 Artifact Layer |
| 任务历史 SQLite | Go task service | 任务生命周期持久化 | Stage 1 起扩展为 Task Trace 基础 |

## Stage 0 非目标

- 不引入 Agent Runtime。
- 不在前端暴露新工具概念。
- 不把 `repair_module.py` 直接接成新的用户可调用 action。
- 不重写训练、扫描、修复和回滚逻辑。

## 下一阶段入口

完成 Stage 0 后，Stage 1 应基于本文件的 `future tool id`、术语和资产映射，引入最小化的 Go 侧 Agent Runtime 与 Tool Registry，而不是重新定义当前 action 边界。
