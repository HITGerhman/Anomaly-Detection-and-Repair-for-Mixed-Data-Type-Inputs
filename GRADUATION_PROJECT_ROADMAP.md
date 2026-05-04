# Graduation Project Roadmap

## 1. 文档定位

本文档是本项目面向本科毕业设计收尾阶段的总纲路线图。

它的目的不是描述具体实现细节，而是明确：

- 项目为什么还需要升级；
- 每个阶段要解决什么问题；
- 每个阶段完成后应当产生什么结果；
- 哪些事情现在不应该做；
- 如何避免项目在后续迭代中变成黑盒。

后续使用 Codex、人工开发或答辩材料整理时，应优先参考本文档，确保每一次修改都服务于毕业设计的核心目标：

> 让“混合数据类型异常检测与修复系统”变得可证明、可复现、可测试、可演示、可写论文。

---

## 2. 项目当前定位

本项目的核心主题是：

> 面向混合数据类型表格数据的异常检测与自动修复系统设计与实现。

项目目前已经具备一定工程基础，包括：

- Python 算法与引擎层；
- Streamlit 演示入口；
- Wails 桌面应用路径；
- Go 后端任务编排；
- 异常扫描、模型训练、批量修复、回滚等核心动作；
- SHAP、LightGBM、Gower/KNN 等算法资产；
- 任务历史、结果表达、可观测性等工程资产。

因此，后续升级不应以“继续堆新功能”为主，而应以“补齐毕业设计证明链条”为主。

---

## 3. 总体升级目标

本轮升级的总目标是将项目从“功能已经较多的工程原型”推进为“能够支撑毕业论文与答辩演示的完整系统”。

具体来说，升级完成后，项目应当能够清楚回答以下问题：

1. 系统要解决什么问题？
2. 为什么混合类型数据异常检测与修复有实际意义？
3. 系统采用了怎样的总体架构？
4. 异常检测是如何完成的？
5. 异常修复是如何完成的？
6. 修复是否真的有效？
7. 系统是否可以复现运行？
8. 系统是否有基本测试保证？
9. 答辩现场如何稳定演示？
10. 如果系统出现运行问题，是否有备用展示方案？

---

## 4. 全局原则

### 4.1 不以重构为默认方向

当前项目已经形成较完整的 Python Engine、Go Backend、Wails Frontend 与 Streamlit 演示路径。后续升级应尽量在现有结构上补充实验、测试、文档和演示支撑，避免大规模重构。

### 4.2 不为炫技引入新技术

本阶段重点不是增加新的模型、框架或复杂 agent 机制，而是让现有系统更可证明、更可解释、更适合答辩。

### 4.3 每个阶段必须可验收

每个 Milestone 都应当有明确结果，而不是“优化一下”“完善一下”这类模糊目标。

每个阶段都应该回答：

- 做这个阶段的目的是什么？
- 完成后项目会多出什么能力？
- 会产生哪些可检查的文件或结果？
- 怎样判断这个阶段完成了？

### 4.4 优先增加旁路资产，谨慎修改核心逻辑

优先新增：

- 实验脚本；
- 评估脚本；
- 测试文件；
- 演示文档；
- 论文支撑材料。

谨慎修改：

- 核心算法流程；
- Python Engine action 协议；
- Go 后端任务语义；
- Wails 主流程；
- 已经可运行的 Streamlit 演示路径。

### 4.5 不编造实验结果

所有实验指标必须来自实际运行结果。若某项结果尚未生成，应明确标注为“待实验生成”，不能为了论文完整性虚构数据。

### 4.6 避免项目黑盒化

每次升级都应增强项目可理解性，而不是只增加代码复杂度。

系统最终应能解释清楚：

- 输入是什么；
- 经过了哪些步骤；
- 每一步的作用是什么；
- 输出是什么；
- 为什么这个输出可信；
- 如果输出有风险，如何回滚或验证。

---

## 5. 最终期望结果

升级完成后，项目应具备以下毕业设计交付能力。

### 5.1 可证明

系统不仅能运行，还能用实验数据证明异常检测和修复的有效性。

应当具备：

- 可控的实验数据集；
- 可计算的检测指标；
- 可计算的修复指标；
- 修复前后对比结果；
- 必要的消融或对比说明。

### 5.2 可复现

其他人在相同环境下按照文档操作，应能复现主要流程。

应当具备：

- 环境说明；
- 运行命令；
- 测试命令；
- 示例数据；
- 输出目录说明。

### 5.3 可测试

项目核心路径应有自动化测试支撑，避免只依赖手动演示。

应当覆盖：

- 异常扫描；
- 批量修复；
- Gower 修复路径；
- 回滚；
- 错误输入处理。

### 5.4 可演示

答辩现场应有稳定、简洁、可控的演示流程。

应当具备：

- 主演示路径；
- 备用演示路径；
- 演示数据；
- 演示步骤；
- 常见失败兜底方案。

### 5.5 可写论文

项目应能支撑完整毕业论文结构。

论文中应能自然展开：

- 研究背景；
- 需求分析；
- 系统设计；
- 算法设计；
- 系统实现；
- 实验评价；
- 系统测试；
- 总结与展望。

---

## 6. Milestone 总览

### M0：项目基线确认

**状态：DONE**

**完成说明：**

已于 2026-05-03 完成当前项目基线确认，基线说明文档见 `PROJECT_BASELINE.md`。本次仅确认并记录当前可运行入口、环境状态、测试命令、已知问题和高风险模块；未执行 M1 及之后任务，未重构主架构，未引入新依赖。

**验证命令：**

```powershell
git status --short --branch
.\.venv-win\Scripts\python.exe --version
.\.venv-win\Scripts\python.exe -m pytest --collect-only tests/python_engine -q
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q
'{"task_id":"health-m0","action":"health","payload":{}}' | .\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py
Push-Location appshell\backend; go test ./internal/engine ./internal/task ./cmd/wails; Pop-Location
$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH; Push-Location appshell\backend; go test ./internal/engine ./internal/task ./cmd/wails; Pop-Location
node --version
npm --version
```

**验证结果摘要：**

Python engine 测试通过：`21 passed in 33.86s`。Engine health 返回 `status=ok`，并确认当前支持 `health/train/repair/scan_file/repair_batch/rollback_repair_batch`。Go 关键包在默认环境下因 Python 子进程入口返回 `exit status 9009` 失败，但临时将 `.\.venv-win\Scripts` 加入 `PATH` 后通过。Node 当前 `Access is denied`，npm 当前不可用，前端构建不纳入本次已通过基线。

**目的：**

确认当前项目的实际状态，明确哪些入口可运行、哪些测试可执行、哪些问题已知存在。

**为什么需要：**

在继续升级前，必须先知道项目当前基线。否则后续 Codex 或人工修改可能会破坏已有可运行能力，却难以及时发现。

**完成后应当得到：**

- 当前主入口说明；
- 当前环境准备说明；
- 当前测试命令说明；
- 当前已知问题说明；
- 当前不建议随意修改的核心模块说明。

**结果形态：**

- 一份基线说明文档；
- 一个清楚的“当前项目能做什么”的快照。

**验收标准：**

阅读基线文档后，开发者应能知道项目如何启动、如何测试、哪些地方是高风险区域。

---

### M1：实验数据与异常注入体系

**状态：DONE**

**完成说明：**

已于 2026-05-03 完成 M1 实验数据与异常注入体系建设，新增可复现生成脚本 `scripts/generate_m1_experiment_data.py`，基于 `data/raw/healthcare-dataset-stroke-data.csv` 生成 `data/experiments/m1_stroke/` 下的 clean、corrupted、ground truth、注入统计和说明文档。本次仅构造可控实验数据与真实注入记录，未执行 M2 及之后任务，未计算检测或修复指标，未修改核心算法、Python engine 协议、Go 后端或 Wails 前端，未引入新依赖。

**验证命令：**

```powershell
.\.venv-win\Scripts\python.exe scripts\generate_m1_experiment_data.py --output-dir data\experiments\m1_stroke --seed 20260503
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine/test_m1_experiment_data.py -q
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q
'{"task_id":"m1-scan","action":"scan_file","payload":{"csv_path":"data/experiments/m1_stroke/corrupted.csv","scan_config":{"enable_cross_column_consistency":true,"consistency_rules":[{"name":"record_start_before_end","type":"lte","left_col":"record_start_day","right_col":"record_end_day"}],"enable_duplicate_record":true,"duplicate_subset":["source_row_id"]}}}' | .\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py
```

**验证结果摘要：**

- 生成器完成，`clean.csv` 为 4228 行、16 列，`corrupted.csv` 为 4240 行、16 列，`ground_truth.csv` 为 100 条注入记录。
- 注入统计为：`missing_values=30`、`numeric_outlier=24`、`rare_category=18`、`duplicate_record=12`、`cross_column_consistency=16`。
- M1 专项测试通过：`2 passed in 2.70s`。
- Python engine 既有回归通过：`23 passed in 25.58s`。
- scan smoke test 返回 `status=ok`，读取 4240 行、16 列，汇总 `issue_count=19`，包含缺失值、数值离群、稀有类别、重复记录和跨列一致性问题。

**目的：**

构造可用于评价的实验数据，使系统检测和修复结果有真实参照。

**为什么需要：**

毕业设计不能只展示系统能运行，还要证明系统有效。若没有可控实验数据，就很难计算检测准确率和修复效果。

**完成后应当得到：**

- 干净数据；
- 注入异常后的数据；
- 异常注入记录；
- ground truth；
- 每类异常的数量统计。

**结果形态：**

- 可重复生成的实验数据；
- 清楚记录异常来源和异常类型的报告。

**验收标准：**

系统后续能基于这些数据计算检测效果和修复效果，而不是只能凭主观观察判断结果好坏。

---

### M2：异常检测效果评估

**状态：DONE**

**完成说明：**

已于 2026-05-03 完成 M2 异常检测效果评估，新增 `scripts/evaluate_m2_detection.py`，基于 M1 的 `data/experiments/m1_stroke/` 产物生成 `data/experiments/m2_stroke_detection/` 下的检测指标、匹配明细和评估报告。本次仅评估当前扫描器对 M1 五类注入异常的检测效果，未执行 M3 及之后任务，未计算修复效果，未修改核心算法、Python engine 协议、Go 后端或 Wails 前端，未引入新依赖。

**验证命令：**

```powershell
.\.venv-win\Scripts\python.exe scripts\evaluate_m2_detection.py --m1-dir data\experiments\m1_stroke --output-dir data\experiments\m2_stroke_detection
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine/test_m2_detection_evaluation.py -q
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q
git status --short --branch
Select-String -Path .\GRADUATION_PROJECT_ROADMAP.md -Pattern '^\| M[0-6] \|'
```

**验证结果摘要：**

- M2 评估脚本执行成功，生成 `detection_metrics.json`、`detection_matches.json` 和 `README.md`。
- 总体检测指标：ground truth `100`，predicted `222`，TP `100`，FP `122`，FN `0`，precision `0.450450`，recall `1.000000`，F1 `0.621118`。
- 分类型指标：`missing_values`、`rare_category`、`duplicate_record`、`cross_column_consistency` 均为 precision/recall/F1 `1.000000`；`numeric_outlier` 为 TP `24`、FP `122`、FN `0`、precision `0.164384`、recall `1.000000`、F1 `0.282353`。
- M2 专项测试通过：`2 passed in 2.45s`。
- Python engine 回归通过：`25 passed in 31.37s`。

**目的：**

量化评估系统发现异常的能力。

**为什么需要：**

异常检测模块是本项目核心之一。论文和答辩中必须能够说明系统检测效果如何，而不仅是展示若干检测结果样例。

**完成后应当得到：**

- 检测准确性指标；
- 不同异常类型的检测效果；
- 检测结果与 ground truth 的对比；
- 可写入论文的实验结果摘要。

**结果形态：**

- 机器可读的评估结果；
- 人类可读的评估报告；
- 可用于论文或答辩的指标表。

**验收标准：**

能够回答：系统检测出了多少真实异常？漏掉了多少？误报了多少？哪些异常类型检测较好，哪些较弱？

---

### M3：异常修复效果评估

**状态：DONE**

**完成说明：**

已于 2026-05-04 完成 M3 异常修复效果评估，新增 `scripts/evaluate_m3_repair.py`，基于 M1 的 `data/experiments/m1_stroke/` 与 M2 的 `data/experiments/m2_stroke_detection/` 产物生成 `data/experiments/m3_stroke_repair/` 下的修复后数据、修复指标、逐条明细和评估报告。本次仅评估当前 `repair_batch` 对缺失值、数值离群和稀有类别的修复效果，未执行 M4 及之后任务，未修改核心算法、Python engine 协议、Go 后端或 Wails 前端，未引入新依赖。

**验证命令：**

```powershell
.\.venv-win\Scripts\python.exe scripts\evaluate_m3_repair.py --m1-dir data\experiments\m1_stroke --m2-dir data\experiments\m2_stroke_detection --output-dir data\experiments\m3_stroke_repair
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine/test_m3_repair_evaluation.py -q
.\.venv-win\Scripts\python.exe -m pytest tests/python_engine -q
git status --short --branch
Select-String -Path .\GRADUATION_PROJECT_ROADMAP.md -Pattern '^\| M[0-6] \|'
```

**验证结果摘要：**

- M3 评估脚本执行成功，生成 `repaired.csv`、`repair_metrics.json`、`repair_details.json` 和 `README.md`。
- 主评分口径为 M1 中 `repairable=True` 的 72 条真值；28 条不可自动修复真值单独记录为 manual review，其中 `cross_column_consistency=16`、`duplicate_record=12`。
- 总体修复结果：72 条可修复真值均被修改，exact restored `17`，exact restoration rate `0.236111`，improved/exact `41`，improved/exact rate `0.569444`。
- 分类型结果：`missing_values` exact `7/30`；`numeric_outlier` exact `0/24`，但 `24/24` 均降低绝对误差，平均误差从 `180.506667` 降至 `55.003698`；`rare_category` exact `10/18`。
- 修复前扫描 issue count 为 `12`，修复后为 `4`，resolved issue count 为 `8`。
- `repair_batch` 共修改 `194` 个单元格，其中 `72` 个对应可修复 ground truth，`122` 个为非 ground truth 单元格修改，主要来自 M2 已记录的 numeric outlier 误报副作用；本次只记录现象，不调参、不修改核心检测逻辑。
- M3 专项测试通过：`2 passed in 4.25s`。
- Python engine 回归通过：`27 passed in 45.52s`。

**目的：**

量化评估系统修复异常数据的能力。

**为什么需要：**

本项目不只是检测异常，还强调自动修复。如果不能证明修复后数据质量提升，系统贡献就会打折扣。

**完成后应当得到：**

- 修复前后问题数量对比；
- 数值字段修复误差；
- 类别字段修复准确率；
- 修复成功率；
- 未修复或跳过问题的统计说明。

**结果形态：**

- 修复评估报告；
- 修复前后对比数据；
- 可用于论文实验章节的结果表。

**验收标准：**

能够回答：系统修改了哪些异常？修复后是否更接近真实值？哪些异常不适合自动修复？

---

### M4：核心行为回归测试

**状态：DONE**

**完成说明：**

已于 2026-05-04 完成 M4 核心行为回归测试补充，新增 `tests/python_engine/test_m4_core_regression.py`，围绕 Python engine 与算法核心路径建立固定输入、固定行为的回归测试。本次仅补充 Python 核心测试并运行 Go 既有关键包验证，未执行 M5 及之后任务，未新增 Go 测试，未重构主架构，未引入新依赖，未修改 Wails 前端。

**验证命令：**

```powershell
.\.venv-win\Scripts\python.exe -m pytest tests\python_engine\test_m4_core_regression.py -q
.\.venv-win\Scripts\python.exe -m pytest tests\python_engine -q
$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH
Push-Location appshell\backend
go test ./internal/engine ./internal/task ./cmd/wails
Pop-Location
git status --short --branch
Select-String -Path .\GRADUATION_PROJECT_ROADMAP.md -Pattern '^\| M[0-6] \|'
```

**验证结果摘要：**

- M4 专项测试通过：`6 passed, 12 warnings in 9.50s`。
- Python engine 全量回归通过：`33 passed, 12 warnings in 38.29s`。
- Go 既有关键包验证通过：`appshell/backend/internal/engine`、`appshell/backend/internal/task`、`appshell/backend/cmd/wails` 均为 `ok`。
- 新增测试覆盖 `scan_file` 混合异常扫描合同、`repair_batch` 选中 issue 修复与 manual-review 跳过、`plan_only` 不落盘行为、`rollback_repair_batch` 恢复与错误 manifest、Gower/KNN 修复建议原始标签保留，以及缺失 CSV、非法 scan_config、非法 repair_strategy 等结构化错误响应。
- 当前警告来自 `src/repair_module.py` 中 `pd.api.types.is_categorical_dtype` 的 pandas deprecation warning；本次 M4 只记录该现象，不在回归测试任务中修改核心逻辑。

**目的：**

为系统核心功能建立自动化测试，降低后续修改破坏主流程的风险。

**为什么需要：**

毕业设计项目不仅要能演示，还要体现基本工程质量。测试可以证明系统不是完全依赖手动操作和偶然成功。

**完成后应当得到：**

- 异常扫描测试；
- 批量修复测试；
- Gower 修复测试；
- 回滚测试；
- 错误输入处理测试。

**结果形态：**

- 自动化测试文件；
- 可重复执行的测试命令；
- 测试通过记录。

**验收标准：**

核心功能修改后，可以通过测试快速判断是否破坏已有行为。

---

### M5：答辩演示流程收口

**状态：DONE**

**完成说明：**

已于 2026-05-04 完成 M5 答辩演示流程收口，新增 `DEFENSE_DEMO_RUNBOOK.md` 和 `demo/m5/` 演示请求材料。主演示路径采用 AppShell/Engine 能力链路，围绕 `data/experiments/m1_stroke/corrupted.csv` 展示扫描、批量修复、结果查看和回滚说明；命令行 Python engine 作为可靠兜底，Wails/前端作为可选展示。本次未执行 M6，未新增实验指标，未修改核心算法，未引入新依赖。

**验证命令：**

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input demo\m5\scan_request.json
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input demo\m5\repair_request.json
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input <临时rollback请求.json>
.\.venv-win\Scripts\python.exe -m pytest tests\python_engine -q
$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH
Push-Location appshell\backend
go test ./internal/engine ./internal/task ./cmd/wails
Pop-Location
git status --short --branch
Select-String -Path .\GRADUATION_PROJECT_ROADMAP.md -Pattern '^\| M[0-6] \|'
```

**验证结果摘要：**

- scan 演示请求执行成功，`issue_count=12`，问题类型统计为 `numeric_outlier=3`、`duplicate_record=1`、`cross_column_consistency=1`、`missing_values=4`、`rare_category=3`。
- repair 演示请求执行成功，`selected_issue_count=10`、`applied_issue_count=10`、`total_cells_modified=194`，输出到 `outputs/demo/m5/repair/corrupted.repaired.csv`，并生成 rollback manifest。
- rollback 临时请求执行成功，使用 `restore_target=output_csv` 验证了回滚清单可恢复演示输出文件；之后已重跑 repair，使演示输出保持修复后状态。
- Python engine 全量回归通过：`33 passed, 12 warnings in 52.81s`。
- Go 既有关键包验证通过：`appshell/backend/internal/engine`、`appshell/backend/internal/task`、`appshell/backend/cmd/wails` 均为 `ok`。

**目的：**

形成清晰、稳定、可控的答辩演示路径。

**为什么需要：**

答辩现场不是开发调试现场。演示流程必须短、稳、清楚，并且有失败兜底方案。

**完成后应当得到：**

- 主演示流程；
- 备用演示流程；
- 演示数据说明；
- 演示结果说明；
- 常见失败处理方案。

**结果形态：**

- 答辩演示脚本；
- 备用演示方案；
- 可复用的演示输入与输出。

**验收标准：**

按照演示文档操作，可以在有限时间内完整展示“选择数据、发现异常、修复异常、查看结果、说明回滚”的闭环。

---

### M6：论文支撑材料整理

**状态：DONE**

**完成说明：**

已于 2026-05-04 完成 M6 论文支撑材料整理，新增 `THESIS_SUPPORT_MATERIALS.md`。该文档基于 M0-M5 的真实产物整理研究背景、需求分析、系统架构、算法方法、实验设计、实验结果、系统测试、总结展望和仍需人工补充内容，可作为毕业论文初稿的结构化素材。本次未修改核心算法、Python engine 协议、Go 后端或 Wails 前端，未新增依赖，未新增实验指标，未重新生成 M1-M3 数据。

**验证命令：**

```powershell
Get-Content -Raw -Encoding UTF8 .\THESIS_SUPPORT_MATERIALS.md
Select-String -Path .\GRADUATION_PROJECT_ROADMAP.md -Pattern '^\| M[0-6] \|'
git status --short --branch
.\.venv-win\Scripts\python.exe -m pytest tests\python_engine -q
$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH
Push-Location appshell\backend
go test ./internal/engine ./internal/task ./cmd/wails
Pop-Location
```

**验证结果：**

- `THESIS_SUPPORT_MATERIALS.md` 已覆盖 M6 要求的论文素材类别，并明确标记学校格式、摘要、关键词、参考文献、图号和最终排版等仍需人工补充内容。
- 文档中的实验数字来自 M1-M5 已验证材料，包括 M2 overall precision `0.450450`、recall `1.000000`、numeric outlier FP `122`，以及 M3 exact restoration rate `0.236111`、improved/exact rate `0.569444`。
- 与远端最新 `origin/main` rebase 后，Python engine 回归通过：`41 passed, 12 warnings in 51.55s`。
- 与远端最新 `origin/main` rebase 后，Go 关键包验证通过：`internal/engine`、`internal/task`、`cmd/wails` 均为 `ok`。
- M0-M6 当前均已完成。

**目的：**

将项目已有事实、系统设计、算法思路、实验结果和测试情况整理为论文写作素材。

**为什么需要：**

代码完成不等于论文完成。毕业设计最终需要通过文字说明系统价值、设计逻辑和实验结论。

**完成后应当得到：**

- 研究背景材料；
- 需求分析材料；
- 系统架构说明；
- 算法方法说明；
- 实验设计说明；
- 实验结果摘要；
- 系统测试说明；
- 总结与展望素材。

**结果形态：**

- 论文支撑文档；
- 可直接改写进毕业论文的章节草稿；
- 仍需人工补充内容的明确标记。

**验收标准：**

阅读该材料后，可以较顺利地展开毕业论文初稿，而不是重新从代码中反推项目逻辑。

---

## 7. 建议执行顺序

推荐执行顺序如下：

1. M0：项目基线确认；
2. M1：实验数据与异常注入体系；
3. M2：异常检测效果评估；
4. M3：异常修复效果评估；
5. M4：核心行为回归测试；
6. M5：答辩演示流程收口；
7. M6：论文支撑材料整理。

不建议跳过 M1 直接做 M2 或 M3，因为没有 ground truth，评估结果会缺少可信基础。

不建议优先做界面美化或 agent 扩展，因为它们对毕业设计核心证明链条的贡献不如实验与评估直接。

---

## 8. 当前阶段的非目标

以下内容不是本轮升级优先目标：

- 大规模重构系统架构；
- 更换核心技术栈；
- 引入新的复杂模型；
- 全面重写前端界面；
- 强行落地完整 multi-agent 系统；
- 追求商业级安装包；
- 为了展示效果编造实验数据；
- 为了论文完整性夸大系统能力。

这些内容可以作为后续扩展或展望，但不应影响当前毕业设计收尾主线。

---

## 9. Codex 使用规则

后续使用 Codex 时，应遵守以下规则。

### 9.1 每次只执行一个 Milestone

Codex 每次只能被授权执行一个明确阶段，不应一次性执行整个路线图。

推荐提示方式：

> 请阅读 `GRADUATION_PROJECT_ROADMAP.md`。本次只执行 M1，不要执行 M2 及之后任务。

### 9.2 明确禁止事项

每次任务都应明确说明：

- 不重构主架构；
- 不修改无关文件；
- 不引入不必要依赖；
- 不执行未来阶段；
- 不编造实验结果。

### 9.3 明确验收结果

每次任务都应明确要求 Codex 输出：

- 新增或修改了哪些文件；
- 如何运行；
- 如何验证；
- 哪些问题尚未解决；
- 是否更新了路线图状态。

### 9.4 人工必须审查 Diff

Codex 完成任务后，必须人工查看变更。重点检查：

- 是否修改了无关核心逻辑；
- 是否引入了多余复杂度；
- 是否生成了无法复现的结果；
- 是否将未来阶段提前混入当前任务；
- 是否破坏已有运行路径。

---

## 10. 项目理解主线

为了避免项目变成黑盒，后续所有文档、实验和答辩说明都应围绕以下主线展开：

1. 用户提供混合类型表格数据；
2. 系统读取数据并识别字段特征；
3. 系统扫描缺失、离群、稀有类别、重复记录、跨列不一致等异常；
4. 系统生成问题清单和风险摘要；
5. 系统根据规则、模型或近邻信息生成修复建议；
6. 系统执行可控修复并保留回滚信息；
7. 系统对修复前后结果进行对比；
8. 系统输出可解释的结果、图表和报告；
9. 用户可以查看修复效果，也可以根据回滚信息恢复原数据。

这条主线应贯穿：

- README；
- 实验文档；
- 演示脚本；
- 论文；
- 答辩 PPT；
- 后续 Codex 任务。

---

## 11. 最终交付状态定义

当以下条件基本满足时，可以认为项目已经达到毕业设计收尾要求：

- 有清楚的系统目标和问题定义；
- 有稳定的主运行入口；
- 有可控实验数据；
- 有异常检测评估结果；
- 有异常修复评估结果；
- 有核心行为测试；
- 有答辩演示脚本；
- 有备用演示方案；
- 有论文支撑材料；
- 能讲清楚系统每一步在做什么；
- 能用实验和演示证明系统有效。

如果这些条件满足，即使系统没有继续增加复杂 agent 或更炫界面，也已经足以支撑一个完成度较高的本科毕业设计。

---

## 12. 状态记录

| Milestone | 名称 | 状态 | 说明 |
|---|---|---|---|
| M0 | 项目基线确认 | DONE | 已完成当前可运行入口、环境状态、测试命令、已知问题和风险区域确认；详见 `PROJECT_BASELINE.md` |
| M1 | 实验数据与异常注入体系 | DONE | 已生成可复现 clean/corrupted/ground truth 数据、注入统计和说明文档；详见 `data/experiments/m1_stroke/` |
| M2 | 异常检测效果评估 | DONE | 已生成检测指标、ground truth 匹配明细和评估报告；详见 `data/experiments/m2_stroke_detection/` |
| M3 | 异常修复效果评估 | DONE | 已生成修复后数据、修复指标、逐条明细和评估报告；详见 `data/experiments/m3_stroke_repair/` |
| M4 | 核心行为回归测试 | DONE | 已补充 Python 核心回归测试，覆盖扫描、批量修复、Gower 修复、回滚和错误输入处理 |
| M5 | 答辩演示流程收口 | DONE | 已形成 AppShell/Engine 主演示 runbook、备用演示方案和可复用 JSON 请求 |
| M6 | 论文支撑材料整理 | DONE | 已形成可改写进毕业论文初稿的结构化支撑材料；详见 `THESIS_SUPPORT_MATERIALS.md` |

---

## 13. 维护说明

本文档应随着项目推进持续更新，但更新时应保持“总纲”定位。

如果某个阶段需要详细实现方案，应另建独立文档或在对应任务中说明，不应把大量代码级细节塞进本文档。

本文档最重要的作用是让项目始终保持清楚方向：

> 不是为了让项目更复杂，而是为了让项目更可信、更可讲、更能通过毕业设计答辩。
