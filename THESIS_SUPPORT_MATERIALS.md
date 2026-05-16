# 论文支撑材料：混合类型数据输入的异常检测与修复系统

本文件对应毕业设计路线图 M6“论文支撑材料整理”。它不是最终论文定稿，而是可直接改写进论文初稿的结构化素材。文中实验数字来自 M1-M5 已生成和验证的材料，不新增实验结果，不夸大系统能力。

## 0. 材料来源与使用方式

建议论文写作时把本文件作为主线材料，并结合以下文件核对事实：

- 项目基线：`PROJECT_BASELINE.md`
- 可控实验数据：`data/experiments/m1_stroke/`
- 检测评估：`data/experiments/m2_stroke_detection/`
- 修复评估：`data/experiments/m3_stroke_repair/`
- 核心回归测试：`tests/python_engine/test_m4_core_regression.py`
- 答辩演示流程：`DEFENSE_DEMO_RUNBOOK.md`
- 路线图总纲：`GRADUATION_PROJECT_ROADMAP.md`

人工写论文时仍需补充学校格式、摘要、关键词、参考文献、图表编号、最终截图和导师要求章节。

## 1. 研究背景与问题定义

随着医疗、金融、业务运营等领域中表格数据规模不断增长，数据质量问题会直接影响后续统计分析、机器学习建模和业务决策。实际表格数据往往同时包含数值字段、类别字段、布尔字段、时间字段和派生字段，这类数据可称为混合类型表格数据。相比单一数值矩阵，混合类型数据的异常检测与修复更复杂，因为不同字段的异常形态、判断标准和修复方式并不一致。

本项目关注的问题是：用户输入一份混合类型 CSV 表格后，系统如何自动识别数据中的典型异常，生成可解释的问题清单，并对可自动处理的问题执行受控修复，同时保留修复前后的对比和回滚依据。

本课题的研究对象不是单纯训练一个分类模型，而是一个围绕数据质量治理的工程闭环。该闭环包括数据读取、字段识别、异常扫描、异常记录、批量修复、修复评估、回滚和答辩演示。通过这种方式，系统既能展示算法能力，也能说明工程实现中如何保障结果可复现、过程可解释和操作可回退。

## 2. 需求分析材料

系统的核心用户是需要处理表格数据质量问题的学生、研究者或业务分析人员。用户希望在不手工逐行检查 CSV 的情况下，快速知道数据中是否存在缺失值、异常数值、罕见类别、重复记录和字段间不一致问题，并获得可控的修复结果。

功能需求可以概括为：

- 数据输入：支持读取混合类型 CSV 数据，保留列名、行号和关键字段信息。
- 异常扫描：识别缺失值、数值离群、稀有类别、重复记录和跨列一致性异常。
- 问题清单：输出异常类型、影响列、行索引、置信信息和简要解释。
- 批量修复：对缺失值、数值离群和稀有类别等当前支持的异常执行自动修复。
- 人工复核：对重复记录和跨列一致性这类高风险异常保留 manual review 口径，不强行自动修复。
- 回滚能力：批量修复时保存回滚清单，使用户可以恢复修复前数据。
- 评估能力：使用可控 ground truth 评估检测和修复效果。
- 演示能力：提供稳定的命令行主演示路径，并在前端或 Node/npm 不可用时保留兜底流程。

非功能需求包括：

- 可复现性：实验数据生成、异常注入和评估使用固定输入与固定随机种子。
- 可解释性：检测结果和修复结果需能说明“为什么被标记”和“修复后产生了什么变化”。
- 可测试性：核心行为由 Python engine 回归测试覆盖，Go 后端关键包也有验证命令。
- 稳定性：M5 主演示不依赖当前不可靠的 Node/npm 环境，而使用已验证的 Python engine 路径。

## 3. 系统架构说明

项目当前采用双轨入口和分层架构：

| 层次 | 主要文件或目录 | 作用 |
|---|---|---|
| Legacy 演示入口 | `app.py` | 保留 Streamlit 演示路径，便于算法演示和历史兼容 |
| Python 核心算法 | `src/` | 承担训练、修复候选生成、Gower/KNN 等核心逻辑 |
| Python engine | `appshell/core/python_engine/` | 提供 JSON 协议入口，统一处理 `health`、`train`、`scan_file`、`repair`、`repair_batch`、`rollback_repair_batch` |
| Go 后端 | `appshell/backend/` | 负责任务编排、Python 子进程调用、超时、取消、状态记录和 Wails 绑定 |
| Wails/frontend | `appshell/frontend/` | 提供桌面应用界面和任务结果展示能力，目前 Node/npm 环境不是可靠基线 |
| 实验与报告 | `data/experiments/`、`demo/`、`thesis-defense/` | 保存实验数据、评估报告、演示材料和答辩资产 |

主数据流可以描述为：

1. 用户提供 CSV 数据。
2. 系统读取数据并识别字段。
3. Python engine 对整表执行异常扫描。
4. 系统返回结构化问题清单和扫描摘要。
5. 用户选择可修复异常或使用预设修复请求。
6. `repair_batch` 执行批量修复并输出修复后 CSV。
7. 系统保存修复记录、对比结果和 rollback manifest。
8. 用户可以查看修复效果，也可以根据回滚清单恢复输出文件。

这种设计的优点是把算法逻辑、协议入口、任务编排和界面展示分离。论文中可以将 Python engine 作为系统核心边界，将 Go 后端作为工程化任务管理层，将 Wails/frontend 作为展示层。

## 4. 算法方法说明

当前系统覆盖的主要异常类型如下：

| 异常类型 | 检测或处理思路 | 当前修复口径 |
|---|---|---|
| 缺失值 | 扫描空值或缺失单元格 | 数值字段可用中位数等策略，类别字段可用众数等策略 |
| 数值离群 | 使用当前检测规则标记异常高低值 | 当前批量修复会对选中 issue 执行裁剪或替换，但 M2 已记录较多误报 |
| 稀有类别 | 检测类别字段中的极低频取值 | 可替换为更常见类别 |
| 重复记录 | 基于指定字段组合识别重复组 | 当前作为人工复核项，不计入自动修复主口径 |
| 跨列一致性 | 通过规则检查字段关系，如 `record_start_day <= record_end_day` | 当前作为人工复核项，不强行自动修改 |
| Gower/KNN 修复 | 面向混合类型数据，用近邻信息生成修复建议 | M4 已验证类别建议返回原始标签，数值字段可给出建议 |

论文写作时应强调：系统没有把所有异常都视为同等可自动修复。缺失值、数值离群和稀有类别适合规则型批量处理；重复记录和跨列一致性通常需要结合业务语义，因此当前系统把它们标记为 manual review。这种边界有助于避免为了展示效果而做不安全的自动修改。

## 5. 实验设计说明

M1 使用 `data/raw/healthcare-dataset-stroke-data.csv` 作为主数据源，因为该数据集包含数值字段和类别字段，体量适中，适合展示混合类型数据异常检测与修复。

M1 实验数据生成过程包括：

- 生成保守 clean subset，减少原始噪声对 ground truth 的影响。
- 使用固定随机种子 `20260503`，保证重复运行结果稳定。
- 为实验辅助新增 `row_id`、`source_row_id`、`record_start_day`、`record_end_day`。
- 对 clean 数据注入五类异常，并输出 corrupted 数据和 ground truth。

M1 生成结果：

| 文件 | 说明 |
|---|---|
| `clean.csv` | 保守 clean subset，4228 行、16 列 |
| `corrupted.csv` | 注入异常后的实验数据，4240 行、16 列 |
| `ground_truth.csv` | 100 条异常注入记录 |
| `injection_summary.json` | 注入统计和生成配置 |

异常注入数量：

| 类型 | 数量 |
|---|---:|
| `missing_values` | 30 |
| `numeric_outlier` | 24 |
| `rare_category` | 18 |
| `duplicate_record` | 12 |
| `cross_column_consistency` | 16 |
| **合计** | **100** |

M2 检测评估以 `ground_truth.csv` 作为唯一真值来源。缺失值、数值离群和稀有类别按异常类型、行索引和列名精确匹配；跨列一致性按异常类型和行索引匹配；重复记录按 `source_row_id` 组匹配。由于 M1 未注入 `time_series_shift`，M2 主评分中关闭该类型。

M3 修复评估只把 M1 中 `repairable=True` 的 72 条作为主修复成功率分母。`duplicate_record` 和 `cross_column_consistency` 共 28 条被记录为 manual review，不作为自动修复失败计算。

## 6. 实验结果摘要

### 6.1 检测效果

M2 检测评估结果如下：

| Type | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `missing_values` | 30 | 30 | 30 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| `numeric_outlier` | 24 | 146 | 24 | 122 | 0 | 0.164384 | 1.000000 | 0.282353 |
| `rare_category` | 18 | 18 | 18 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| `duplicate_record` | 12 | 12 | 12 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| `cross_column_consistency` | 16 | 16 | 16 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| **Overall** | 100 | 222 | 100 | 122 | 0 | 0.450450 | 1.000000 | 0.621118 |

可以在论文中这样概括：系统对 M1 注入的 100 条异常全部召回，整体 recall 为 `1.000000`，说明当前扫描规则能覆盖实验设计中的五类异常。但整体 precision 为 `0.450450`，主要原因是数值离群检测产生 122 条 false positive。该结果体现出当前检测器对异常敏感，但数值离群阈值仍有优化空间。

### 6.2 修复效果

M3 修复评估结果如下：

| Type | GT | Changed | Exact | Improved/Exact | Exact Rate | Improved/Exact Rate |
|---|---:|---:|---:|---:|---:|---:|
| `missing_values` | 30 | 30 | 7 | 7 | 0.233333 | 0.233333 |
| `numeric_outlier` | 24 | 24 | 0 | 24 | 0.000000 | 1.000000 |
| `rare_category` | 18 | 18 | 10 | 10 | 0.555556 | 0.555556 |
| **Overall** | 72 | 72 | 17 | 41 | 0.236111 | 0.569444 |

可以在论文中这样概括：在 72 条可自动修复真值中，系统全部执行了修改，其中 17 条完全恢复到原始值，exact restoration rate 为 `0.236111`；41 条达到完全恢复或误差改善，improved/exact rate 为 `0.569444`。数值离群项没有完全恢复到原值，但 24 条均降低了绝对误差，说明规则型修复可以缓解极端数值影响，但不等同于恢复真实原值。

修复前后扫描摘要：

- Before issue count：12
- After issue count：4
- Resolved issue count：8
- `repair_batch` 修改单元格总数：194
- 非 ground truth 单元格修改：122
- manual review：`cross_column_consistency=16`，`duplicate_record=12`

论文中应如实说明：非 ground truth 单元格修改主要来自 M2 已记录的数值离群误报。该现象不是系统成功率，而是当前检测阈值和修复联动带来的副作用。

## 7. 系统测试说明

M4 补充了 Python 核心回归测试，重点覆盖：

- `scan_file` 对缺失值、数值离群、稀有类别、跨列一致性和重复记录的扫描行为。
- `repair_batch` 对选中 issue 的修复、`write_output=false`、`plan_only=true` 和 skipped issue 的处理。
- `rollback_repair_batch` 对 rollback manifest 的恢复能力，以及 missing/invalid manifest 的错误响应。
- `src.repair_module.AnomalyRepairer` 的 Gower/KNN 修复建议，确认混合类型数据返回原始类别标签。
- 缺失 CSV、非法 `scan_config`、非法 `repair_strategy` 等错误输入响应。

最近一次 M5 验证中，Python engine 全量测试结果为：

```text
33 passed, 12 warnings in 52.81s
```

Go 关键包验证结果为：

```text
appshell/backend/internal/engine ok
appshell/backend/internal/task ok
appshell/backend/cmd/wails ok
```

测试边界也应写入论文或答辩说明：

- 当前 Node/npm 在本机不可作为可靠基线，因此前端构建不是已验证结果。
- Windows clean-machine 安装包验证尚未完成。
- pandas deprecation warning 仍存在，但不影响当前测试通过。
- 默认 shell 的 Python 入口不稳定，Go 测试需要临时把 `.\.venv-win\Scripts` 加入 `PATH`。

## 8. 总结与展望素材

本项目完成了一个面向混合类型表格数据的异常检测与修复系统原型。系统不仅能识别常见数据质量问题，还形成了从可控实验数据、ground truth、检测评估、修复评估、核心回归测试到答辩演示的完整证明链条。相比只展示模型训练结果，该项目更强调数据质量处理流程的可复现性、可解释性和可回退性。

当前系统的主要价值包括：

- 能处理同时包含数值和类别字段的 CSV 数据。
- 能输出结构化异常清单，覆盖缺失值、数值离群、稀有类别、重复记录和跨列一致性。
- 能对部分异常执行批量修复，并保留修复后文件和回滚依据。
- 有可控异常注入数据和 ground truth，便于客观评估检测和修复效果。
- 有核心回归测试和答辩演示材料，便于交付和复现。

当前局限包括：

- 数值离群检测召回高但误报较多，precision 需要进一步优化。
- 批量修复对数值离群可以降低误差，但不能保证恢复真实原值。
- 重复记录和跨列一致性目前更适合人工复核，尚未形成安全的自动修复策略。
- 前端构建、Node/npm 环境和 Windows 安装包验证仍未形成可靠闭环。
- 论文最终格式、参考文献和图表排版仍需人工整理。

后续可改进方向：

- 优化数值离群检测阈值，降低 false positive。
- 为重复记录和跨列一致性设计更谨慎的交互式修复流程。
- 补齐前端执行回滚入口和历史任务回放能力。
- 在干净 Windows 环境中验证安装包和桌面应用启动流程。
- 将 M1-M3 的实验图表进一步整理为论文图和答辩图。

## 9. 可改写为论文目录的建议结构

可将论文组织为以下章节：

1. 绪论
   - 研究背景
   - 研究意义
   - 国内外相关工作
   - 本文主要工作
2. 需求分析
   - 功能需求
   - 非功能需求
   - 数据质量问题定义
3. 系统设计
   - 总体架构
   - Python engine 协议设计
   - Go 后端任务编排设计
   - 前端展示与演示流程
4. 异常检测与修复方法
   - 混合类型字段处理
   - 异常检测规则
   - 批量修复策略
   - 回滚机制
5. 实验设计与结果分析
   - 实验数据构造
   - 检测效果评估
   - 修复效果评估
   - 结果局限分析
6. 系统测试与演示
   - 核心行为测试
   - 命令行演示流程
   - 已知环境边界
7. 总结与展望
   - 工作总结
   - 系统不足
   - 后续改进方向

## 10. 仍需人工补充

以下内容不应由 M6 自动编造，需要人工按学校要求补齐：

- 中英文摘要和关键词。
- 参考文献及引用格式。
- 学校模板中的封面、诚信声明、致谢和附录。
- 国内外研究现状中的正式文献综述。
- 最终论文图号、表号和截图。
- 导师要求的章节标题、字数比例和格式细节。
- 若需要提交 Word 版本，还需将本 Markdown 内容人工或单独任务转换为 `.docx` 并检查排版。

## 2026-05-13 Cross-Dataset Thesis Update

The final thesis source file was not found in this repository. The following
material is therefore maintained here as the paper-facing update source for
Chapter 5, Chapter 6, and Chapter 7. All numbers below come from the generated
CSV artifacts under `artifacts/experiments/cross_dataset/`.

### Chapter 5: Supplementary Cross-dataset Validation

The original M1/M2/M3 stroke experiment remains the main controlled experiment.
The supplementary validation adds `orders_transactions` and `user_device_logs`
to check whether the same mixed-type CSV scanning, controlled repair, review
boundary, and side-effect accounting can be reproduced beyond one data source.

Use these table sources directly:

| Thesis table | Artifact source |
|---|---|
| Cross-dataset detection metrics summary | `artifacts/experiments/cross_dataset/summary_detection_metrics.csv` |
| Cross-dataset repair metrics summary | `artifacts/experiments/cross_dataset/summary_repair_metrics.csv` |
| Numeric outlier threshold sensitivity summary | `artifacts/experiments/cross_dataset/threshold_sensitivity_numeric_outlier.csv` |

Detection overall metrics from the 2026-05-13 run:

| Dataset | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stroke` | 100 | 222 | 100 | 122 | 0 | 0.450450 | 1.000000 | 0.621118 |
| `orders_transactions` | 100 | 140 | 100 | 40 | 0 | 0.714286 | 1.000000 | 0.833333 |
| `user_device_logs` | 100 | 100 | 100 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |

Issue-level discussion:

- `missing_values` reached recall `1.000000` on all three datasets. Precision
  was `1.000000` on `stroke` and `user_device_logs`, and `0.967742` on
  `orders_transactions`.
- `rare_category` reached precision, recall, and F1 of `1.000000` on all three
  datasets in this controlled setup.
- `numeric_outlier` remained the main source of false positives. With the
  default thresholds, `stroke` produced 122 numeric false positives and
  `orders_transactions` produced 39 numeric false positives. `user_device_logs`
  did not produce numeric false positives in this run.
- `duplicate_record` and `cross_column_consistency` were detected correctly in
  these controlled datasets, but they remain manual-review issue types.

Repair overall metrics from the 2026-05-13 run:

| Dataset | Repairable GT | Changed | Exact | Improved/Exact | Exact Rate | Improved/Exact Rate | Non-GT Modified | Skipped Review-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stroke` | 72 | 72 | 17 | 41 | 0.236111 | 0.569444 | 122 | 28 |
| `orders_transactions` | 72 | 72 | 5 | 29 | 0.069444 | 0.402778 | 40 | 28 |
| `user_device_logs` | 72 | 72 | 20 | 44 | 0.277778 | 0.611111 | 0 | 28 |

The repair discussion should stay conservative. Exact restoration is limited:
only 17, 5, and 20 of the 72 repairable ground-truth anomalies were restored
exactly for `stroke`, `orders_transactions`, and `user_device_logs`. The
improved-or-exact metric is more suitable for numeric outlier repair because
clipping or median-style repair can reduce error without recovering the true
original value. Non-GT modified cells are not repair successes; they reveal side
effects caused mainly by numeric false positives.

Numeric threshold sensitivity:

- On `stroke`, the default `iqr_factor=1.5` produced numeric precision
  `0.164384` and recall `1.000000`. Raising the IQR factor to `2.0` reduced
  false positives to 25 and improved precision to `0.489796`. The strict
  `iqr_factor=3.0, robust_z_threshold=4.5` setting removed numeric false
  positives but reduced recall to `0.625000`.
- On `orders_transactions`, `iqr_factor=2.0, robust_z_threshold=4.5` and stricter
  comparable settings reached precision, recall, and F1 of `1.000000`.
- On `user_device_logs`, all tested threshold combinations reached precision,
  recall, and F1 of `1.000000`.

This supports the paper claim that numeric outlier detection is threshold
sensitive and should be tuned with domain context rather than treated as a
universal automatic-repair trigger.

### Chapter 6: Extended Sample and Scale Testing

Rename or extend the previous section as:

```text
6.6 Extended Sample and Scale Testing
```

Use this table source:

```text
artifacts/experiments/cross_dataset/summary_scale_metrics.csv
```

Scale results from the 2026-05-13 run:

| Dataset Name | Rows | Columns | Scan Time (s) | Repair Time (s) | Detected Issues | Changed Cells | Output Size MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `orders_transactions_scale` | 5000 | 13 | 0.252031 | 0.188720 | 152 | 111 | 0.566670 |
| `orders_transactions_scale` | 10000 | 13 | 0.206418 | 0.217076 | 301 | 221 | 1.133484 |
| `orders_transactions_scale` | 50000 | 13 | 0.452200 | 0.964246 | 611 | 531 | 5.654707 |
| `orders_transactions_scale` | 100000 | 13 | 1.053711 | 1.899857 | 974 | 894 | 11.303867 |

These results show that the command-line experimental pipeline can process
larger mixed CSV files in this local environment. The scale test is a system
throughput check, not an accuracy proof. Peak memory is not reported because the
pipeline avoids adding platform-specific memory-measurement dependencies.

### Chapter 7: Limitations and Future Work Update

The conclusion should remain cautious. A suitable statement is:

> The supplementary cross-dataset experiments strengthen the evidence that the
> system is not tied to a single stroke dataset. Across stroke,
> orders_transactions, and user_device_logs, the same pipeline produced
> reproducible scanning, controlled repair, rollback-compatible outputs,
> review-only handling for duplicate and cross-column issues, and side-effect
> statistics.

Avoid claiming that the system is production-ready, applicable to all real
business data, able to repair every anomaly automatically, or able to recover
all original values.

Keep these limitations:

- Numeric outlier detection still needs threshold tuning and domain rules.
- Repair quality depends on anomaly type and repair strategy.
- `duplicate_record` and `cross_column_consistency` still require human
  confirmation.
- More real-domain datasets are future work.
- Front-end, deployment, and full user workflow validation can still be
  improved.
