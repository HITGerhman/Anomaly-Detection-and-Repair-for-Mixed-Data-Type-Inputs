# PRESENTATION_CATALOG

## 定位

`PRESENTATION_CATALOG.md` 是 Stage 4 的表达层目录文档。

它冻结三类内容：

- 统一解释模板 ID
- 统一图表 ID 与输入字段
- 默认显示条件、用户价值和文案口径

本文件与 [MULTI_AGENT_BLUEPRINT.md](/D:/code/pythoncode/Anomaly Detection and Repair for Mixed Data Type Inputs/MULTI_AGENT_BLUEPRINT.md) 配合使用：

- `MULTI_AGENT_BLUEPRINT.md` 说明为什么要做 Stage 4
- 本文件说明 Stage 4 具体展示什么、如何命名、何时显示

## 表达模型

统一表达资产为 `presentation.json`，核心结构如下：

- `version`
- `kind`
- `headline`
- `summary`
- `verdict`
- `highlights`
- `sections`
- `charts`
- `artifacts`

适用来源：

- `scan_file`
- `repair_batch`
- `repair_with_gower`
- `agent.session.plan`
- `agent.session.execute`
- `agent.session.auto`
- `GetAgentSession`

## 解释模板

### `overview`

- 目标：让用户先在 5 秒内知道“这次结果到底是什么”
- 默认输入：
  - `scan_summary`
  - `data_profile`
  - `comparison`
  - `execution`
  - `validation`
  - `safety`
- 默认文案口径：
  - 扫描结果用“发现了多少问题、数据规模如何”
  - 修复结果用“解决了多少问题、写出了什么产物”
  - agent 结果用“最终裁决是什么、是否执行/回滚”

### `anomaly_findings`

- 目标：解释异常分布，而不是只报一个总数
- 默认输入：
  - `issues`
  - `column_thumbnails`
  - `scan_summary.high_risk_columns`
- 默认文案口径：
  - 问题类型分布
  - 高风险列
  - 当前可修复问题数

### `repair_strategy`

- 目标：说明系统为什么选规则修复、Gower 修复或 hybrid
- 默认输入：
  - `plan`
  - `selected_source`
  - `applied_repairs`
  - `neighbor_evidence`
  - `skipped_issues`
- 默认文案口径：
  - 来源选择理由
  - 候选方案数
  - 跳过原因

### `repair_impact`

- 目标：解释修复带来的净收益
- 默认输入：
  - `comparison`
  - `applied_repairs`
  - `changed_cell_count`
  - `validation`
- 默认文案口径：
  - 修复前后问题数变化
  - 实际修改的单元格数量
  - 若为 agent，会补充 preview/post-execute 结论

### `risk_and_safety`

- 目标：明确告诉用户“这次输出是否安全”
- 默认输入：
  - `rollback`
  - `post_validation`
  - `safety`
  - `rejected_output_snapshot`
- 默认文案口径：
  - 是否通过验证
  - 是否触发回滚
  - 是否保留失败快照

### `next_steps`

- 目标：给用户一个明确的后续动作
- 默认输入：
  - `verdict`
  - `artifacts`
  - `rollback`
- 默认文案口径：
  - 通过时建议导出和复扫
  - 回滚时建议查看轨迹和失败快照
  - 计划阶段建议比较候选再执行

## 图表目录

### 异常理解类

#### `column_risk_ranking`

- kind：`ranked_bar`
- 默认输入：
  - `column_thumbnails[].column`
  - `column_thumbnails[].risk_score`
  - `column_thumbnails[].risk_level`
- 显示条件：
  - 存在 `column_thumbnails`
- 用户价值：
  - 快速知道先看哪几列
- 默认文案口径：
  - “按列风险排序查看最值得优先处理的列”

#### `issue_type_distribution`

- kind：`stacked_bar`
- 默认输入：
  - `issues[].issue_type`
- 显示条件：
  - 存在 `issues`
- 用户价值：
  - 先知道问题集中在哪种异常，而不是只看总数
- 默认文案口径：
  - “查看缺失值、离群值和类别问题的占比”

#### `anomaly_density_heatmap`

- kind：`heatmap_grid`
- 默认输入：
  - `column_thumbnails`
  - `data_profile.rows`
- 显示条件：
  - 存在 `column_thumbnails`
- 用户价值：
  - 快速锁定热区
- 默认文案口径：
  - “异常热力图继续复用列缩略图组件，与 issue 详情联动”

#### `high_risk_column_spotlight`

- kind：`spotlight_card`
- 默认输入：
  - 风险最高的列对象
- 显示条件：
  - 存在至少一列风险数据
- 用户价值：
  - 一眼看到最关键的一列
- 默认文案口径：
  - “给出当前最值得优先排查的一列”

### 修复影响类

#### `before_after_issue_comparison`

- kind：`comparison_bar`
- 默认输入：
  - `comparison.before_issue_count`
  - `comparison.after_issue_count`
  - `comparison.resolved_issue_count`
- 显示条件：
  - 存在 `comparison`
- 用户价值：
  - 直接判断修复有没有收益
- 默认文案口径：
  - “直接查看修复是否有效减少问题”

#### `repaired_vs_skipped_breakdown`

- kind：`stacked_bar`
- 默认输入：
  - `applied_issue_count`
  - `skipped_issues`
- 显示条件：
  - 修复或执行结果存在
- 用户价值：
  - 了解系统实际做了多少、保留了多少
- 默认文案口径：
  - “区分已应用修复和保留问题”

#### `column_issue_delta`

- kind：`comparison_bar`
- 默认输入：
  - `comparison.before_column_issue_counts`
  - `comparison.after_column_issue_counts`
- 显示条件：
  - 两组列级对比同时存在
- 用户价值：
  - 判断哪些列真正改善，哪些列没有改善
- 默认文案口径：
  - “按列查看修复前后的问题变化”

#### `repair_source_breakdown`

- kind：`stacked_bar`
- 默认输入：
  - `selected_source`
  - `issue_source_map`
  - `neighbor_evidence`
  - `plan.candidates`
- 显示条件：
  - `repair_with_gower` 或 `hybrid` 或 agent 多候选结果存在
- 用户价值：
  - 解释“这次主要靠规则还是靠近邻”
- 默认文案口径：
  - “区分规则、Gower 或混合路径的贡献”

### 执行与可观测类

#### `validation_verdict_timeline`

- kind：`timeline`
- 默认输入：
  - `validation.preview`
  - `validation.post_execute`
  - `safety.final_verdict`
- 显示条件：
  - agent 结果存在 `validation`
- 用户价值：
  - 看清楚是在哪个验证阶段被接受或拒绝
- 默认文案口径：
  - “查看 preview 与 post-execute 两段验证结果”

#### `trace_stage_timeline`

- kind：`timeline`
- 默认输入：
  - `observability.stage_durations_ms`
  - `trace_summary`
- 显示条件：
  - agent 或带观测信息的任务存在
- 用户价值：
  - 快速定位哪个阶段最慢、谁参与了执行
- 默认文案口径：
  - “结合观测信息和 trace 摘要查看执行轨迹”

#### `rollback_summary`

- kind：`spotlight_card`
- 默认输入：
  - `safety.rollback_execution`
  - `safety.rejected_output_snapshot`
  - `safety.final_verdict`
- 显示条件：
  - `final_verdict` 为 `rolled_back` 或 `rollback_failed`
- 用户价值：
  - 一眼知道是否已回滚、失败快照在哪
- 默认文案口径：
  - “仅在回滚或回滚失败路径显示”

#### `safety_risk_delta`

- kind：`comparison_bar`
- 默认输入：
  - `safety.baseline_scan_summary`
  - `safety.post_scan_summary`
- 显示条件：
  - 同时存在 baseline/post-scan 安全摘要
- 用户价值：
  - 判断自动闭环后风险是否真的下降
- 默认文案口径：
  - “对比基线与后验扫描的风险变化”

## 显示规则

- 扫描结果优先显示：
  - `overview`
  - `anomaly_findings`
  - 异常理解类图表

- 普通修复结果优先显示：
  - `overview`
  - `repair_strategy`
  - `repair_impact`
  - 修复影响类图表

- agent 自动闭环优先显示：
  - `overview`
  - `repair_strategy`
  - `repair_impact`
  - `risk_and_safety`
  - 执行与可观测类图表

- 旧任务没有 `presentation` 时：
  - 继续走原视图
  - 前端不能假设所有任务都已经生成 Stage 4 表达层

## 资产约束

- `presentation.json` 是 Stage 4 唯一共享表达资产
- Wails 结果页直接渲染它
- Streamlit 答辩端优先读取它
- 后续导出报告、截图、答辩页和演示页都应围绕它扩展
- 表达层只负责解释、归纳、聚合和图表组织，不新增业务决策逻辑
