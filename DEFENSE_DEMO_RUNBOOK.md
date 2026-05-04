# Defense Demo Runbook

本文件用于 M5 答辩演示流程收口。目标是在 3-5 分钟内稳定展示：

> 混合类型表格数据输入后，系统可以扫描异常、生成问题清单、执行可控修复、输出结果，并保留回滚依据。

本次主演示路径采用 AppShell/Engine 能力链路。Wails/前端可以作为界面截图或可选展示，但不作为必需前提；命令行 Python engine 是现场兜底路径。

## 演示前准备

推荐在答辩前提前打开 PowerShell，位于项目根目录：

```powershell
cd "D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs"
.\.venv-win\Scripts\python.exe --version
```

确认关键文件存在：

```powershell
Test-Path .\data\experiments\m1_stroke\corrupted.csv
Test-Path .\data\experiments\m2_stroke_detection\detection_metrics.json
Test-Path .\data\experiments\m3_stroke_repair\repair_metrics.json
Test-Path .\demo\m5\scan_request.json
Test-Path .\demo\m5\repair_request.json
```

## 3-5 分钟主演示脚本

### 0. 开场说明

可以这样开场：

> 这个系统面向混合类型表格数据，重点不是只训练一个模型，而是形成异常扫描、修复、评估和回滚的完整闭环。下面用中风数据集的可控异常实验数据演示。

说明演示数据：

- 输入数据：`data/experiments/m1_stroke/corrupted.csv`
- 真值记录：`data/experiments/m1_stroke/ground_truth.csv`
- 检测评估：`data/experiments/m2_stroke_detection/`
- 修复评估：`data/experiments/m3_stroke_repair/`

### 1. 扫描异常

运行：

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input demo\m5\scan_request.json
```

讲解重点：

- 输入是 corrupted CSV，不是干净数据。
- 扫描配置启用了缺失值、数值离群、稀有类别、跨列一致性和重复记录检查。
- 输出中重点看 `status=ok`、`result.issue_count`、`result.scan_summary.issue_type_counts`。

答辩口径：

> M2 评估中，系统对 100 条注入异常全部召回，recall 为 1.0；数值离群存在额外误报，这是后续修复阶段需要解释的风险。

### 2. 执行批量修复

运行：

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input demo\m5\repair_request.json
```

讲解重点：

- 修复只选择当前系统支持自动处理的问题：缺失值、数值离群、稀有类别。
- 重复记录和跨列一致性不强行自动修复，作为人工复核项解释。
- 输出中重点看：
  - `result.selected_issue_count`
  - `result.applied_issue_count`
  - `result.total_cells_modified`
  - `result.output_csv`
  - `result.rollback.manifest_path`

答辩口径：

> M3 评估中，72 条可自动修复真值全部被修改；其中 41 条达到 exact 或误差改善。数值离群全部降低绝对误差，但由于 M2 存在数值离群误报，修复也带来了额外单元格修改，这在报告里明确记录为副作用。

### 3. 展示修复结果

查看输出文件：

```powershell
Get-ChildItem .\outputs\demo\m5\repair
Get-Content .\data\experiments\m3_stroke_repair\README.md
```

可展示的关键指标：

- `repairable_ground_truth_count=72`
- `exact_restored_count=17`
- `improved_or_exact_count=41`
- `non_ground_truth_cells_modified=122`
- `cross_column_consistency=16` 和 `duplicate_record=12` 是 manual review 项

### 4. 说明回滚

从 repair 输出复制 `result.rollback.manifest_path`，写入临时 rollback 请求。也可以参考：

```powershell
Get-Content .\demo\m5\rollback_request.template.json
```

建议现场只说明：

> 每次批量修复会保存原 CSV 的备份和回滚清单。若用户不满意修复结果，可以调用 `rollback_repair_batch` 恢复到修复前状态。

若需要现场执行，可用 `restore_target=output_csv` 演示，这只会把演示输出文件恢复为原始输入内容，不改 M1 数据资产。

### 5. 收尾总结

可以这样收尾：

> 这个演示展示的是项目的工程闭环：有可控实验数据、有检测指标、有修复指标、有核心回归测试，也有回滚和失败兜底。它不是黑盒模型展示，而是可复现、可解释、可验证的异常检测与修复系统。

## 备用演示路径

### 备用 A：只展示已有实验报告

如果现场命令无法运行，直接打开：

- `data/experiments/m2_stroke_detection/README.md`
- `data/experiments/m3_stroke_repair/README.md`
- `GRADUATION_PROJECT_ROADMAP.md`

讲清楚 M1-M4 已经生成的真实结果和测试记录。

### 备用 B：只运行 health

如果扫描或修复耗时异常，先证明 engine 可用：

```powershell
'{"task_id":"defense-health","action":"health","payload":{}}' | .\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py
```

再切换到已有 M2/M3 报告展示。

### 备用 C：Go/AppShell 后端验证

如果需要说明 AppShell 后端不是摆设：

```powershell
$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH
Push-Location appshell\backend
go test ./internal/engine ./internal/task ./cmd/wails
Pop-Location
```

说明 Go 后端负责任务编排、超时、取消、历史记录和 Wails 绑定。

## 常见失败处理

| 问题 | 处理 |
|---|---|
| `python` 不可用 | 使用 `.\.venv-win\Scripts\python.exe`，不要依赖系统 `python` |
| Go 测试 Python 子进程失败 | 先执行 `$env:PATH = (Resolve-Path '.\.venv-win\Scripts').Path + ';' + $env:PATH` |
| Node/npm 不可用 | 不作为主演示前提，改用 Python engine 命令行闭环 |
| Wails 窗口打不开 | 改用命令行主演示，并展示 `appshell/frontend/index.html` 或截图式说明 |
| 修复输出目录已有文件 | `repair_batch` 会覆盖演示输出；必要时删除 `outputs/demo/m5/repair/` 后重跑 |
| 现场时间不足 | 只运行 scan request，修复结果直接展示 M3 README |
| 老师追问误报 | 说明 M2 已记录 numeric outlier FP=122，M3 已记录对应副作用，没有编造或隐藏 |

## 不在 M5 现场承诺的内容

- 不承诺 Node/npm 构建已恢复。
- 不承诺 Windows 安装包已经在干净机器验证。
- 不把重复记录和跨列一致性说成已经自动修复。
- 不把 M2/M3 指标夸大为生产级效果，只作为毕业设计实验结果说明。
