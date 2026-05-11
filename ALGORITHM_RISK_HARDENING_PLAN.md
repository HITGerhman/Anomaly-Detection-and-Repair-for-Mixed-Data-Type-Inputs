# Algorithm Risk Hardening Plan

## 1. 文档定位

本文档对应 `feat/algorithm-risk-hardening` 分支的 R1 阶段，目标是在已有 Auto Agent 成果之上，冻结“算法风险控制增强”的问题定义、设计边界和后续实施计划。

本阶段只做设计，不修改业务代码、不修改实验数据、不修改已有评估结果、不修改核心 action 协议，也不引入新依赖。

本计划的重点不是替换模型或追求更复杂的算法，而是让自动修复在面对不确定异常，尤其是 `numeric_outlier` 时更加保守、可解释、可回滚。

## 2. 当前算法主链路

### 2.1 `scan_file`

`scan_file` 是当前异常发现入口。它读取 CSV 后生成 `issues`、`scan_summary`、`column_profiles` 和列级风险信息。

当前主要检测类型包括：

- `missing_values`
- `numeric_outlier`
- `rare_category`
- `duplicate_record`
- `cross_column_consistency`
- 可选的 `time_series_shift`

其中 `numeric_outlier` 主要由 IQR 边界和 robust z-score 共同触发，并在 issue 中携带 `lower_bound`、`upper_bound`、`iqr_hits`、`robust_hits`、`preview` 和默认 `repair_rule.strategy=clip`。

这说明当前 scan 层能很好地发现统计离群点，但 scan 结果本身还不能证明这些离群值一定是数据错误。

### 2.2 `repair_batch`

`repair_batch` 是规则型批量修复入口。它接收 `issue_ids`，默认支持：

- `missing_values`：数值列用 median/mean/constant，类别列用 mode/constant；
- `numeric_outlier`：默认按 `repair_strategy.outlier=clip` 裁剪到扫描边界；
- `rare_category`：默认替换为非 rare 值中的 mode。

`repair_batch` 会输出 `applied_repairs`、`skipped_issues`、`comparison`、`total_cells_modified` 和 rollback 信息。它可以 `plan_only`，也可以写出修复后 CSV。

当前风险点是：只要 `numeric_outlier` issue 被传入并且策略不是 `skip`，规则层就会执行 clip。若 scan 层存在误报，clip 会把真实极端值改成边界值。

### 2.3 `repair_with_gower`

`repair_with_gower` 是混合类型 Gower/KNN 修复入口。它同样支持：

- `missing_values`
- `numeric_outlier`
- `rare_category`

该链路会为异常行检索健康邻居，并根据邻居的目标列取 median 或 mode 作为候选修复值，同时输出 `neighbor_evidence`、`candidate_confidence`、距离摘要和 rollback 信息。

Gower 方案比直接 clip 更有上下文，但它依然不能单独证明 `numeric_outlier` 一定是错误。对于真实高血糖、高 BMI 或其他真实极端值，邻居共识可能会把真实值拉回人群常态，从而造成语义损伤。

### 2.4 `repair_core` 单样本模型修复

`src/repair_core.py` 面向单样本异常修复。它使用已训练模型的预测结果和 feature importance，基于健康样本邻居生成候选值，并以最多 `max_changes` 个字段的贪心搜索降低异常分数。

该模块的特点是：

- 依赖 LightGBM 等已训练模型的预测和 feature importance；
- 倾向选择对异常分数影响更大的字段；
- 对数值候选执行 bounds 裁剪；
- 输出修复后的单行样本、summary 和 changes。

它适合作为模型解释和单样本候选生成能力，但不应被扩展成无约束批量自动改数值的入口。后续风险增强应继续尊重 `max_changes`、immutable columns、numeric bounds 等约束。

### 2.5 Auto Agent planner

当前 Auto Agent 采用“工具在下，planner 在上”的结构。Go runtime 负责扫描、预览、执行、复扫、验证和回滚；planner 只消费 `PlanningInput` 并返回 `AgentPlan`，不能直接写文件或调用工具。

当前 deterministic planner 的默认分流是：

- `missing_values`、`rare_category` 进入自动修复候选；
- `numeric_outlier` 进入 cautious，默认排除在自动写出 payload 外；
- `duplicate_record`、`cross_column_consistency` 进入 manual review；
- unknown issue 进入 blocked。

planner 会比较 rule、Gower 和 hybrid 候选，并选择全局比较更优的方案。这个设计已经比早期直接把所有 repairable issue 交给 `repair_batch` 更安全，但后续仍需要更细粒度的 `numeric_outlier` 风险分层，避免把所有数值离群都混成同一种 cautious。

### 2.6 Validation gate

当前 validation gate 在自动执行后比较 baseline scan、repair result、post scan 和 plan，输出：

- `accept`
- `warn`
- `reject`
- `rollback_recommended`

它已经覆盖若干关键风险：

- 修复后 issue count 上升则 reject；
- high-risk issue count 上升则 reject；
- 自动修复 manual review issue 则 reject；
- 自动修复 high-risk issue 则 reject；
- 写出结果但缺少 rollback manifest 则建议 rollback；
- 修改单元格数量异常高时 warn。

当前修改量阈值是 `max(50, beforeIssueCount * 20)`。这个规则能捕捉非常粗暴的批量修改，但对 M3 暴露出的“issue count 下降但误修了大量正常单元格”仍不够敏感，需要在 R4 增强为更细的副作用约束。

## 3. 当前实验暴露的主要问题

### 3.1 M2：`numeric_outlier` 召回高但误报多

`data/experiments/m2_stroke_detection/detection_metrics.json` 显示：

- overall recall = `1.0`
- overall precision = `0.45045`
- `numeric_outlier` recall = `1.0`
- `numeric_outlier` precision = `0.164384`
- `numeric_outlier` tp = `24`
- `numeric_outlier` fp = `122`
- `numeric_outlier` predicted_count = `146`

这说明当前检测器没有漏掉 M1 注入的数值离群，但把大量真实分布中的极端值也标成了 outlier。尤其是 `avg_glucose_level` 中既有注入的 `420.0`，也有许多 150-170 左右的真实高值被 IQR 边界捕获。

### 3.2 M3：批量修复产生明显副作用

`data/experiments/m3_stroke_repair/repair_metrics.json` 显示：

- `repair_batch.total_cells_modified = 194`
- `metrics.overall.non_ground_truth_cells_modified = 122`
- repairable ground truth count = `72`
- repairable changed count = `72`
- `numeric_outlier.changed_count = 24`
- `numeric_outlier.improved_or_exact_rate = 1.0`

这组结果有两层含义：

- 对被注入的 `numeric_outlier`，当前 clip 能让数值更接近原始值，说明修复工具有可用性；
- 但 122 个非 ground-truth 单元格也被修改，说明检测误报会直接传导成误修。

因此，当前最大风险不是“不会修”，而是“修得太积极”。

### 3.3 数值离群误报会进一步导致误修

当前 `scan_file` 会把统计离群转成带 clip 规则的 issue；`repair_batch` 在收到该 issue 后会按边界裁剪；若 Auto Agent 或用户把误报 issue 纳入自动执行，误报就会变成真实数据损伤。

这个链条可以概括为：

```text
统计离群检测偏敏感
-> numeric_outlier false positive 增多
-> repair_batch / repair_with_gower 接收误报 issue
-> clip 或邻居替换修改真实极端值
-> non-ground-truth cells modified 增加
```

R2-R4 的核心目标就是切断这个传导链。

## 4. 为什么不能无脑自动修复 `numeric_outlier`

### 4.1 统计离群不等于数据错误

IQR 和 robust z-score 本质上衡量的是“相对当前分布是否罕见”，不是“业务上是否错误”。在偏态分布、长尾分布或人群差异明显的数据集中，极端值可能是真实观测。

### 4.2 医疗类极端值可能具有真实语义

在 stroke 数据集中，高血糖、高 BMI、高年龄等数值可能代表真实高风险人群。它们可能是模型最需要保留的信号，而不是应该被自动拉回中位数或 IQR 上界的脏数据。

例如：

- `avg_glucose_level` 高于普通人群上界，可能是真实高血糖；
- `bmi` 较高可能是真实肥胖样本；
- 年龄极高需要结合业务上限判断，不能只看统计边界。

### 4.3 自动 clip 会损害真实数据

clip 的优点是简单、确定、可回滚；缺点是会把所有边界外值压到同一个阈值附近。对真实极端值来说，这会造成：

- 数值分布被截断；
- 下游模型风险信号被削弱；
- 个体样本语义被改变；
- 修复报告看似 issue 下降，但真实数据质量下降。

因此，`numeric_outlier` 应默认从“可自动修复项”降级为“需要风险判断的候选项”。

## 5. 目标优化方向

### 5.1 `numeric_outlier` 风险分层

后续 R2 应将 `numeric_outlier` 至少分为三档：

| 分层 | 含义 | 默认动作 |
|---|---|---|
| `mild` | 只略微越过统计边界，可能是真实长尾值 | 只提示，不自动修 |
| `strong` | 明显越界，且 IQR / robust z 等信号较强 | 可进入自动修复候选，但需要 planner 标注风险 |
| `extreme` | 极端不合理或接近注入式异常，例如明显超过业务合理范围 | 可进入自动修复候选，优先要求 validation 强约束和 rollback |

分层依据可以逐步引入：

- 距离 IQR 边界的相对幅度；
- robust z-score 强度；
- 是否同时命中 IQR 和 robust z；
- 列级业务边界或经验边界；
- 同列 outlier ratio；
- 是否集中在明显异常值，例如重复的极端注入值。

R2 不应改变 M1-M3 已有结果，只在新分支的新实验或新输出中新增风险字段。

### 5.2 mild 默认只提示，不自动修

`mild numeric_outlier` 的默认策略应是：

- 在 scan / plan / report 中提示；
- 保留 preview；
- 不进入 `repair_batch` 写出 payload；
- 不进入 `repair_with_gower` 写出 payload；
- 可由用户显式审批后执行。

这能直接减少 `avg_glucose_level` 中真实高值被 clip 的风险。

### 5.3 strong / extreme 才允许进入自动修复候选

`strong` 和 `extreme` 可以进入自动修复候选，但必须满足额外条件：

- planner 明确记录风险层级和原因；
- repair plan 中区分 rule / gower / hybrid 的选择依据；
- 写出结果必须有 rollback manifest；
- post validation 必须检查修改量和修复收益；
- 对敏感列或 protected columns 仍需审批。

这不是把 `numeric_outlier` 全部放开，而是给极端异常一个可控入口。

### 5.4 Auto Agent 对 `numeric_outlier` 默认更保守

R3 应在当前 cautious 策略基础上进一步收紧：

- 默认只自动处理 `missing_values` 和低风险 `rare_category`；
- `numeric_outlier.mild` 固定 plan-only；
- `numeric_outlier.strong` 默认 cautious，可在策略允许时进入候选；
- `numeric_outlier.extreme` 可进入候选，但必须写入风险说明；
- LLM planner 不得绕过 deterministic risk policy；
- fallback planner 与 LLM planner 使用同一套风险底线。

简言之，LLM 可以解释和排序，但不能把风险项升级成无条件自动修。

### 5.5 Validation gate 增强副作用约束

R4 应让 validation gate 不只看 issue count 是否下降，还要看修复副作用。

建议新增或强化的规则：

- 修改单元格数相对 selected issue 命中行数过高时 warn/reject；
- `numeric_outlier` 修改占比过高时 warn；
- 修复后 issue count 下降但 changed_cell_count 很高时至少 warn；
- 若实验评估场景有 ground truth，则统计 `non_ground_truth_cells_modified` 并写入报告；
- 对 `mild numeric_outlier` 被自动修改直接 reject；
- 对 manual review / blocked issue 被修改继续 reject；
- rollback manifest 缺失继续 rollback_recommended。

目标是防止“指标表面变好，但正常单元格被大量改写”。

## 6. 不做的事情

本分支风险增强明确不做以下事情：

- 不替换 LightGBM；
- 不重构 Python Engine 协议；
- 不重写 `repair_batch`；
- 不修改已有 M1-M3 实验结果；
- 不引入复杂深度学习模型；
- 不把 LLM 变成修复执行器；
- 不让 agent 直接修改 CSV；
- 不为了降低误报而直接覆盖冻结的 M2/M3 结论。

现有结果应作为问题证据保留，而不是被回改。

## 7. 后续任务阶段

### R1：设计文档

产出本文档，明确当前短板、风险来源、优化方向和阶段边界。

验收标准：

- 只新增 `ALGORITHM_RISK_HARDENING_PLAN.md`；
- 不写代码；
- 不修改实验数据；
- 不修改已有评估结果；
- 不修改核心 action 协议；
- 不新增依赖。

### R2：`numeric_outlier` 风险分层

目标是在不破坏旧 action 协议的前提下，为新扫描/规划结果增加 `mild / strong / extreme` 风险语义。

建议产出：

- 风险分层函数；
- 单元测试；
- 新字段或 explain features；
- 对既有 scan 输出保持兼容。

### R3：Auto Agent planner 风险策略收紧

目标是让 planner 使用 R2 的分层结果，进一步控制自动写出 payload。

建议策略：

- mild 只提示；
- strong 默认 cautious；
- extreme 可作为候选但必须带风险说明；
- LLM planner 输出必须经过 deterministic policy clamp；
- fallback 路径保持同样保守。

### R4：validation gate 副作用约束增强

目标是让 validation gate 对“误修正常单元格”更敏感。

建议方向：

- type-aware changed cell 阈值；
- selected issue 命中行数与实际修改量对比；
- mild outlier 自动修改直接 reject；
- changed_cell_count 高但 issue_score 只小幅改善时 warn/reject；
- benchmark 模式下输出 side-effect 指标。

### R5：rule / Gower / hybrid 对比实验

目标是比较三种候选在不同 outlier 风险分层下的收益和副作用。

建议指标：

- resolved issue items；
- changed cell count；
- non-ground-truth modified cells；
- numeric outlier precision-sensitive repair rate；
- rollback manifest 是否生成；
- validation verdict 分布。

### R6：K 敏感性实验

目标是评估 `repair_with_gower` 中 `k_neighbors` 对修复质量和副作用的影响。

建议设置：

- 固定数据集和 issue 选择策略；
- 比较多个 K 值；
- 分别记录 missing、rare、numeric outlier 的表现；
- 重点观察 numeric outlier 是否随 K 改变而更容易过度平滑。

### R7：文档与简历口径更新

目标是把风险控制增强转化为论文后续工作、答辩补充和简历表达。

建议口径：

- 强调不是盲目提高自动化，而是通过风险分层、planner policy 和 validation gate 降低误修；
- 明确保留 deterministic tools、rollback manifest 和 fallback；
- 将 M2/M3 暴露的问题转化为工程迭代证据。

## 8. 阶段完成定义

R1-R7 完成后，本分支应能证明：

- 系统承认 `numeric_outlier` 的统计不确定性；
- 自动修复不再把所有数值离群视作同等风险；
- Auto Agent planner 对数值离群默认保守；
- validation gate 能识别“修复收益”和“正常数据损伤”之间的权衡；
- rule、Gower、hybrid 的选择有实验依据；
- 既有 M1-M3 结果不被篡改，而是作为风险增强的动机和对照基线。

一句话目标：

> 在不替换现有算法主链路的前提下，把自动修复从“能改”推进到“知道什么时候不该改”。
