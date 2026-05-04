# Auto Agent Enhancement Roadmap

## 1. 文档定位

本文档用于指导 `feat/auto-agent-planner` 分支上的 agent 智能增强工作。

它不是毕业设计 M0-M6 主线的一部分，也不是当前论文成果的替代品。`main` 分支上的 M0-M6 已经作为毕业设计阶段性成果冻结，后续 agent 工作必须在独立分支中进行，避免破坏已有实验数据、评估结果、测试记录、演示材料和论文支撑文档。

本文档的目标是明确：

- agent 智能增强最终希望达到什么效果；
- 哪些能力应该复用现有确定性工具；
- 哪些事情 agent 可以做，哪些事情 agent 不能做；
- 如何分阶段推进，避免一次性大改造成黑盒；
- Codex 后续每次应该只执行哪个阶段、产出什么结果、如何验收。

---

## 2. 总体目标

本分支的目标是探索一个可选的 Auto Agent 编排层，使系统从“用户手动选择 issue 的半自动工具”升级为“可以自动规划、谨慎执行、复扫验证、生成解释报告的数据质量助手”。

最终期望体验是：

> 用户选择一个 CSV 文件后，系统自动完成数据扫描、异常理解、修复计划生成、低风险修复执行、修复后复扫、结果验证、失败兜底和自然语言解释。

需要注意：

> agent 不是为了替代现有算法，也不是为了让项目显得更像 AI。agent 的价值在于降低用户决策负担、提升流程自动化、增强解释能力和安全审计能力。

---

## 3. 与毕业设计主线的关系

### 3.1 主线已经冻结

`main` 分支上的 M0-M6 已经完成：

- M0 项目基线确认；
- M1 实验数据与异常注入；
- M2 异常检测评估；
- M3 异常修复评估；
- M4 核心回归测试；
- M5 答辩演示流程；
- M6 论文支撑材料。

这些成果用于毕业论文和答辩，不应被 agent 分支的实验性开发破坏。

### 3.2 agent 是后续增强，不是当前论文主贡献

当前毕业设计主贡献应继续围绕：

- 混合类型数据异常检测；
- 批量修复；
- ground truth 评估；
- 修复前后对比；
- 回滚机制；
- 工程化测试与演示闭环。

agent 自动化可以作为后续扩展方向、增强原型或未来工作，不应反向修改论文主线的实验结论。

---

## 4. 核心原则

### 4.1 工具在下，Agent 在上

系统必须保持清晰分层：

- 下层：确定性工具层，负责扫描、修复、复扫、回滚和结果落盘；
- 上层：agent 编排层，负责理解扫描结果、生成修复计划、判断风险、组织解释。

agent 不直接替代现有算法，不重新实现扫描逻辑，不绕开工具层写文件。

### 4.2 执行真相唯一

同一类扫描、修复、回滚逻辑只能有一个真实执行来源。

当前应继续复用现有 Python Engine actions：

- `health`
- `scan_file`
- `repair_batch`
- `rollback_repair_batch`

后续如需调用训练、单样本修复或 Gower/KNN 能力，也应通过现有工具或旁路包装接入，而不是复制一套不一致逻辑。

### 4.3 自动化必须伴随安全增强

自动化越强，安全边界越要明确。

必须保证：

- 自动执行前有修复计划；
- 自动执行后必须复扫；
- 修复效果必须被验证；
- 高风险 issue 不能默认自动修复；
- 每次写出修复文件都必须保留 rollback manifest；
- LLM API 失败、超时或输出非法时必须 fallback；
- agent 不能直接修改 CSV。

### 4.4 外部 API 是增强，不是唯一依赖

外部 LLM API 可用于生成更自然的计划和解释，但系统不能完全依赖它。

必须支持：

- 无 API 时 deterministic planner 正常工作；
- API 超时时 fallback；
- API 返回非法 JSON 时 fallback；
- API key 不写入代码、测试或提交记录；
- LLM 只输出 plan，不直接执行 repair。

### 4.5 不破坏 M0-M6 产物

本分支默认禁止修改：

- `data/experiments/m1_stroke/`
- `data/experiments/m2_stroke_detection/`
- `data/experiments/m3_stroke_repair/`
- `PROJECT_BASELINE.md`
- `GRADUATION_PROJECT_ROADMAP.md`
- `THESIS_SUPPORT_MATERIALS.md`
- `DEFENSE_DEMO_RUNBOOK.md`

除非后续明确要求，否则 agent 分支只应新增旁路模块、旁路测试和旁路文档。

---

## 5. 目标用户体验

当前系统的典型流程是：

1. 用户选择 CSV；
2. 用户手动扫描；
3. 用户查看 issue；
4. 用户选择 issue；
5. 用户执行修复；
6. 用户自己判断修复效果。

agent 增强后的目标流程是：

1. 用户选择 CSV；
2. 系统自动扫描；
3. agent 自动理解数据结构和异常分布；
4. agent 自动区分低风险、中风险和高风险问题；
5. agent 生成结构化修复计划；
6. 系统预览或执行低风险修复；
7. 系统复扫修复后的 CSV；
8. validation gate 判断修复是否可接受；
9. 系统输出修复后 CSV、回滚清单和解释报告；
10. 若验证失败，则提示回滚或自动回滚。

最终输出应包括：

- 修复后的 CSV；
- 原始文件备份或回滚依据；
- rollback manifest；
- 扫描摘要；
- 修复计划；
- 修复前后对比；
- 自动修复项列表；
- 人工复核项列表；
- 风险说明；
- 用户可读的 Markdown 报告。

---

## 6. 异常类型处理策略

结合当前 M2/M3 实验结果，自动化策略必须保守。

### 6.1 默认可自动修复

| 类型 | 默认策略 | 原因 |
|---|---|---|
| `missing_values` | 自动修复 | 缺失值修复风险相对低，可使用 median/mode 等确定性策略 |
| `rare_category` | 谨慎自动修复 | 可替换为常见类别，但仍应记录解释和回滚 |

### 6.2 默认谨慎处理

| 类型 | 默认策略 | 原因 |
|---|---|---|
| `numeric_outlier` | 默认 cautious / plan_only / 需要验证 | M2 中数值离群误报较多，M3 中已记录非 ground truth 单元格修改副作用 |

### 6.3 默认人工复核

| 类型 | 默认策略 | 原因 |
|---|---|---|
| `duplicate_record` | manual_review | 重复记录是否删除需要业务语义 |
| `cross_column_consistency` | manual_review | 跨列不一致需要判断应该修改哪一列，不能无脑自动修 |

### 6.4 默认阻断

未知 `issue_type` 默认进入 `blocked_issue_ids`，不自动执行。

---

## 7. 推荐 Auto Agent 流程

### Step 1：Scan

调用 `scan_file`，获得：

- data profile；
- issue list；
- scan summary；
- issue type counts；
- issue ids；
- column-level risk information。

### Step 2：Profile

agent 或 deterministic planner 对扫描结果进行总结：

- 数据规模；
- 列类型；
- 问题类型分布；
- 高风险列；
- 可自动修复 issue；
- 人工复核 issue。

### Step 3：Plan

生成结构化修复计划。

计划必须区分：

- `auto_repair_issue_ids`
- `manual_review_issue_ids`
- `cautious_issue_ids`
- `blocked_issue_ids`
- `repair_strategy`
- `risk_notes`
- `explanation`

### Step 4：Preview / Plan Only

在真正写出前，优先支持 dry-run 或 plan-only。

目标是让系统先知道：

- 预计修复哪些 issue；
- 预计跳过哪些 issue；
- 预计使用什么策略；
- 是否存在明显高风险。

### Step 5：Execute

只对低风险 issue 调用 `repair_batch`。

默认策略：

- missing values：可执行；
- rare category：可执行但记录风险；
- numeric outlier：默认谨慎，不无脑执行；
- duplicate / consistency：不自动执行。

### Step 6：Rescan

对修复后的 CSV 再次调用 `scan_file`。

### Step 7：Validate

validation gate 判断结果是否可接受：

- 修复后 issue count 是否下降；
- 是否出现新问题；
- 修改单元格数量是否异常；
- 是否包含高风险 issue 的自动修改；
- 是否需要回滚；
- 是否需要人工确认。

### Step 8：Explain

生成用户可读报告，说明：

- 发现了什么；
- 自动修复了什么；
- 跳过了什么；
- 为什么跳过；
- 修复前后有什么变化；
- 有哪些风险；
- 如何回滚。

### Step 9：Rollback if Needed

若验证失败，应调用或提示调用 `rollback_repair_batch`。

---

## 8. Repair Plan 建议结构

建议 planner 输出如下 JSON 结构：

```json
{
  "mode": "auto_agent_plan",
  "planner": "deterministic|llm",
  "auto_repair_issue_ids": [],
  "manual_review_issue_ids": [],
  "cautious_issue_ids": [],
  "blocked_issue_ids": [],
  "repair_strategy": {
    "missing_numeric": "median",
    "missing_categorical": "mode",
    "rare_category": "mode",
    "outlier": "clip",
    "conflict_policy": "first_wins"
  },
  "risk_notes": [],
  "explanation": "",
  "trace": []
}
```

字段含义：

- `planner`：说明计划来自 deterministic planner 还是 LLM planner；
- `auto_repair_issue_ids`：允许自动修复的低风险 issue；
- `manual_review_issue_ids`：必须人工复核的 issue；
- `cautious_issue_ids`：需要谨慎处理或 plan-only 的 issue；
- `blocked_issue_ids`：未知或不安全 issue；
- `repair_strategy`：传给 `repair_batch` 的策略；
- `risk_notes`：风险说明；
- `explanation`：面向用户的计划解释；
- `trace`：记录 planner 做出决策的依据。

---

## 9. 外部 API 接入目标

外部 LLM API 的作用是增强规划和解释，而不是执行修复。

### 9.1 建议环境变量

继续沿用项目已有方向：

```powershell
$env:APPSHELL_LANGGRAPH_LLM_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:APPSHELL_LANGGRAPH_LLM_API_KEY = "your-api-key"
$env:APPSHELL_LANGGRAPH_LLM_MODEL = "your-model-name"
$env:APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS = "4000"
```

### 9.2 LLM Planner 输入

LLM planner 可接收：

- data profile；
- scan summary；
- issue list；
- issue type counts；
- current known risks from M2/M3；
- user policy，例如是否允许自动修复 numeric outlier。

### 9.3 LLM Planner 输出

LLM 必须输出结构化 JSON plan。

禁止 LLM 输出任意自然语言后直接执行。自然语言只能作为解释，不可作为执行依据。

### 9.4 Fallback 要求

以下情况必须 fallback 到 deterministic planner：

- 环境变量缺失；
- API key 缺失；
- endpoint 不可达；
- API 超时；
- 返回非 200；
- 返回非法 JSON；
- JSON schema 校验失败；
- plan 试图自动修复高风险 issue。

---

## 10. Validation Gate 目标

Validation Gate 是全自动化能否可信的关键。

它必须基于修复前后结果判断：

- `before_issue_count`
- `after_issue_count`
- `resolved_issue_count`
- `total_cells_modified`
- skipped issues；
- manual review issues；
- rollback manifest 是否存在；
- 是否修复了不应该自动修复的 issue。

建议输出结构：

```json
{
  "verdict": "accept|warn|reject|rollback_recommended",
  "before_issue_count": 0,
  "after_issue_count": 0,
  "resolved_issue_count": 0,
  "risk_notes": [],
  "rollback_recommended": false,
  "explanation": ""
}
```

默认规则：

- 修复后 issue count 下降：倾向 accept 或 warn；
- 修复后 issue count 上升：reject；
- 修改单元格异常多：warn；
- 自动修复 high-risk issue：reject；
- rollback manifest 缺失：warn 或 reject；
- repair_batch 报错：reject 并建议 rollback。

---

## 11. 分阶段实施计划

### A0：分支与冻结点确认

**目的：**

确保 agent 增强在独立分支上进行，不影响毕业设计主线。

**预期结果：**

- 从 `origin/main` 创建 `feat/auto-agent-planner`；
- 不修改 `main`；
- 不修改 M0-M6 产物；
- 本文档作为 agent 分支路线图存在。

**验收标准：**

- 当前分支为 `feat/auto-agent-planner`；
- `AUTO_AGENT_ROADMAP.md` 存在；
- 与 `main` 的差异仅为 agent 路线图或后续 agent 相关旁路文件。

---

### A1：Auto Agent 设计文档

**目的：**

在写代码前，先冻结 agent 的目标、边界、输入输出、安全策略和执行流程。

**预期结果：**

新增：

- `AUTO_AGENT_DESIGN.md`

文档应说明：

- Auto Agent 是什么；
- 与 M0-M6 主线的关系；
- agent 能做什么、不能做什么；
- 如何调用现有工具；
- 如何生成 repair plan；
- 如何接入外部 API；
- 如何 fallback；
- 如何验证和回滚。

**验收标准：**

- 只新增设计文档；
- 不写代码；
- 不修改核心逻辑；
- 不修改实验结果；
- 文档明确 agent 不直接修改 CSV。

---

### A2：Deterministic Planner 原型

**目的：**

先实现一个不依赖外部 API 的规则版 planner，保证系统在没有 LLM 时仍然可用。

**预期结果：**

新增旁路 planner 模块和测试。

planner 输入：

- `scan_file` 返回的 scan result。

planner 输出：

- 结构化 repair plan。

默认规则：

- `missing_values` → `auto_repair_issue_ids`
- `rare_category` → `auto_repair_issue_ids`
- `numeric_outlier` → `cautious_issue_ids`
- `duplicate_record` → `manual_review_issue_ids`
- `cross_column_consistency` → `manual_review_issue_ids`
- unknown → `blocked_issue_ids`

**验收标准：**

- 不调用外部 API；
- 不修改现有 engine action 协议；
- 不修改 `repair_batch` 核心逻辑；
- 有测试覆盖五类 issue；
- Python 测试通过；
- M1-M3 实验结果不变。

---

### A3：LLM Planner Adapter

**目的：**

在 deterministic planner 之外增加可选 LLM planner，使外部 API 能生成更灵活的计划和解释。

**预期结果：**

新增 LLM adapter，支持 OpenAI-compatible endpoint。

必须支持环境变量：

- `APPSHELL_LANGGRAPH_LLM_BASE_URL`
- `APPSHELL_LANGGRAPH_LLM_API_KEY`
- `APPSHELL_LANGGRAPH_LLM_MODEL`
- `APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS`

**安全要求：**

- 不硬编码 API key；
- 不真实调用 API 进行单元测试；
- 使用 mock 测试；
- 输出必须 JSON schema 校验；
- LLM 失败必须 fallback；
- LLM 不能直接执行修复；
- 高风险 issue 不能因 LLM 建议而绕过安全策略。

**验收标准：**

测试覆盖：

- 环境变量缺失 fallback；
- API 超时 fallback；
- 非 200 fallback；
- 非法 JSON fallback；
- 合法 JSON plan 通过校验；
- LLM 试图自动修复高风险 issue 时被拦截或降级。

---

### A4：Validation Gate

**目的：**

实现自动修复后的安全判定逻辑，避免“修了但变差”或“误修过多”。

**预期结果：**

新增 validation gate 模块。

输入：

- 修复前 scan summary；
- repair_batch result；
- 修复后 scan summary；
- repair plan；
- rollback manifest 信息。

输出：

- `accept`
- `warn`
- `reject`
- `rollback_recommended`

**验收标准：**

测试覆盖：

- issue count 下降 → accept 或 warn；
- issue count 上升 → reject；
- 修改单元格异常多 → warn；
- 自动修复 manual review issue → reject；
- rollback manifest 缺失 → warn/reject；
- repair error → reject。

---

### A5：Auto Agent CLI Demo

**目的：**

提供一个最小可演示入口，让一条命令完成自动化闭环。

**预期命令示例：**

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\auto_agent_cli.py --csv data\experiments\m1_stroke\corrupted.csv --output-dir outputs\auto_agent\m1_stroke
```

**预期流程：**

1. 调用 `scan_file`；
2. 生成 repair plan；
3. 执行低风险修复；
4. 对修复后 CSV 复扫；
5. validation gate 给出 verdict；
6. 输出 report；
7. 保留 rollback manifest。

**预期输出：**

- repaired CSV；
- repair plan JSON；
- validation result JSON；
- auto agent trace JSON；
- Markdown report；
- rollback manifest。

**验收标准：**

- 无外部 API 时可使用 deterministic planner 跑通；
- 有 mock 或测试覆盖 CLI 主流程；
- 不修改 M1-M3 既有实验结果；
- 不修改 M5 原答辩 runbook；
- 输出目录独立于 M0-M6 产物。

---

### A6：可选 Wails / Go 编排接入

**目的：**

在 CLI 原型稳定后，再考虑接入 Go backend 或 Wails 前端。

**预期结果：**

- Go 侧可以发起 auto agent session；
- 任务历史中能看到 agent trace；
- 前端可展示 agent plan、风险说明和验证结果。

**注意：**

A6 不是当前最小目标。只有在 A1-A5 稳定后才考虑。

**验收标准：**

- 不破坏现有 Wails 基础流程；
- Go 关键包测试通过；
- Python 测试通过；
- 失败时能 fallback 到原有手动扫描/修复流程。

---

## 12. Codex 工作规则

后续每次交给 Codex 的任务必须遵守：

1. 每次只执行一个阶段；
2. 先阅读本文档；
3. 不修改 `main`；
4. 不修改 M0-M6 产物；
5. 不引入无必要依赖；
6. 不写入真实 API key；
7. 不让 LLM 直接执行修复；
8. 每次都说明修改了哪些文件、如何测试、哪些风险尚未解决。

推荐提示格式：

```text
请阅读 `AUTO_AGENT_ROADMAP.md`。
当前分支是 `feat/auto-agent-planner`。
本次只执行 A2，不执行 A3 及之后任务。
不要修改 M0-M6 产物。
不要修改现有核心 engine action 协议。
完成后运行相关测试，并说明修改文件、测试命令和已知风险。
```

---

## 13. 当前非目标

以下内容不是当前 agent 增强的首要目标：

- 一次性实现 12-agent 协作；
- 大规模重构 Python Engine；
- 重写 `repair_batch`；
- 替换规则扫描；
- 替换 Gower/KNN；
- 直接把外部 API 绑定为强依赖；
- 自动修复所有 issue；
- 自动删除重复记录；
- 自动修改跨列一致性问题；
- 修改毕业论文主线实验结果；
- 为了展示 AI 感牺牲可复现性和可审计性。

---

## 14. 最终完成定义

当 A1-A5 完成后，本分支应能证明：

- agent 层可以读取现有扫描结果；
- agent 层可以生成结构化修复计划；
- 无外部 API 时 deterministic planner 可用；
- 有外部 API 时 LLM planner 可选增强；
- LLM 失败时可 fallback；
- 低风险 issue 可以自动执行修复；
- 高风险 issue 默认人工复核；
- 修复后能复扫和验证；
- 结果能生成用户可读报告；
- 全流程保留 rollback 和 trace；
- 不破坏 M0-M6 毕设成果。

一句话：

> 本分支成功的标准不是“agent 数量多”，而是系统真正降低了用户决策负担，同时保持扫描、修复、验证和回滚的确定性、安全性与可审计性。
