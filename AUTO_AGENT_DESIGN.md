# Auto Agent 设计文档

## 1. 文档定位

本文档是 `AUTO_AGENT_ROADMAP.md` 中 A1 阶段的设计冻结文档，用于说明 Auto Agent 的目标、边界、输入输出、安全策略、执行流程和后续阶段约束。

本文档不替代：

- `AUTO_AGENT_ROADMAP.md`：阶段路线图与验收入口；
- `MULTI_AGENT_BLUEPRINT.md`：长期智能化升级蓝图；
- `GRADUATION_PROJECT_ROADMAP.md`、`THESIS_SUPPORT_MATERIALS.md` 和 M0-M6 产物：毕业设计冻结主线。

当前 `main` 分支上的 M0-M6 已冻结。Auto Agent 后续工作只在 `feat/auto-agent-planner` 分支上作为旁路增强推进，不反向修改论文主线实验结论、评估数据、演示材料和答辩支撑文档。

## 2. 目标体验与非目标

### 2.1 目标体验

Auto Agent 的目标是把当前“用户手动扫描、手动选择 issue、手动判断修复效果”的半自动流程，升级为“系统自动规划、谨慎执行、复扫验证、生成解释”的数据质量助手。

目标流程是：

1. 用户选择 CSV 文件；
2. 系统调用现有扫描工具；
3. agent 理解扫描摘要、异常类型和风险分布；
4. agent 生成结构化 repair plan；
5. 系统预览候选修复方案；
6. 低风险 issue 可自动执行，中高风险 issue 进入审批或人工复核；
7. 系统对修复后 CSV 复扫；
8. validation gate 判断是否接受、警告、拒绝或建议回滚；
9. 系统输出修复文件、rollback manifest、计划、轨迹和用户可读解释。

### 2.2 非目标

Auto Agent 不是为了替代现有算法，也不是为了让项目表面上更像 AI。它是确定性工具层之上的编排层。

默认非目标如下：

- 不修改 `main` 分支冻结成果；
- 不修改 M0-M6 实验数据、评估报告、测试文件、演示材料和论文支撑文档；
- 不重写 `scan_file`、`repair_batch`、`repair_with_gower` 或 `rollback_repair_batch`；
- 不改变现有 Python engine action 协议；
- 不让 LLM 直接执行修复；
- 不让 agent 绕过工具层写文件；
- 不把真实 API key 写入代码、文档、测试或提交记录。

最重要的边界是：agent 不直接修改 CSV。CSV 写入只能由确定性工具层在受控参数、回滚清单和验证流程下完成。

## 3. 当前系统基线

### 3.1 分支与冻结点

- 当前实验分支：`feat/auto-agent-planner`
- 当前分支用途：探索 Auto Agent 自动化增强方向；
- 冻结主线：`origin/main`
- 冻结原则：`main` 上 M0-M6 作为毕业设计阶段性成果，不被 agent 实验污染。

### 3.2 已有确定性工具

Auto Agent 必须复用现有 Python engine actions 作为真实执行来源：

- `scan_file`：扫描 CSV 并返回 issue、summary、profile 等结构化结果；
- `repair_batch`：对 selected issue 执行规则型批量修复；
- `repair_with_gower`：使用 Gower/KNN 候选执行混合类型修复；
- `rollback_repair_batch`：根据 rollback manifest 恢复修复产物；
- `health`、`train`、`repair`：可作为后续扩展工具，但 A1 不改变其协议。

### 3.3 已有 Go Agent Runtime

当前 Go 侧已经存在 `appshell/backend/internal/agent` 运行时基础，设计文档应承认这一现实，而不是把 Auto Agent 描述成完全从零开始。

当前可引用的边界包括：

- Go agent actions：`agent.session.plan`、`agent.session.execute`、`agent.session.auto`、`agent.session.approve`；
- planning boundary：`PlanningInput`；
- plan/session 类型：`AgentPlan`、`RepairCandidate`、`AgentSession`；
- trace 类型：`AgentTraceEvent`；
- tool registry：`engine.scan_table`、`engine.repair_batch`、`engine.repair_with_gower`、`engine.rollback_batch` 等工具 ID。

A1 不要求修改这些类型。本文档只把它们作为后续实现必须尊重的边界。

### 3.4 LangGraph 与外部 LLM 现实状态

当前仓库已有 LangGraph sidecar 方向的基础结构，Go 侧通过 `APPSHELL_LANGGRAPH_*` 配置本地 sidecar 的启动、host、port、script、Python 解释器和超时。

外部 LLM provider 配置应使用 `APPSHELL_LANGGRAPH_LLM_*` 变量描述，例如：

- `APPSHELL_LANGGRAPH_LLM_BASE_URL`
- `APPSHELL_LANGGRAPH_LLM_API_KEY`
- `APPSHELL_LANGGRAPH_LLM_MODEL`
- `APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS`

两类配置必须区分：

- `APPSHELL_LANGGRAPH_*`：控制本地 sidecar 是否启用、如何启动、如何被 Go 调用；
- `APPSHELL_LANGGRAPH_LLM_*`：控制 sidecar 内部是否连接 OpenAI-compatible 外部模型。

真实 API key 只能保存在本地未跟踪配置中，不进入设计文档、样例输出和测试断言。

## 4. 分层架构

Auto Agent 采用“工具在下，agent 在上”的分层结构。

### 4.1 User Experience Layer

负责用户入口和结果呈现：

- 文件选择；
- 自动化任务启动；
- 计划、风险、修复结果和回滚入口展示；
- agent trace 和解释报告展示。

### 4.2 Go Agent Runtime Layer

负责所有任务生命周期和安全控制：

- 创建 session；
- 调用扫描、预览、修复、复扫和回滚工具；
- 组装 `PlanningInput`；
- 调用 deterministic planner 或 LangGraph/LLM planner；
- 保存 `AgentSession` 和 `AgentTraceEvent`；
- 执行 approval gate、validation gate 和 rollback gate；
- 保证超时、取消、错误响应和持久化一致。

Go runtime 是执行控制平面。Planner 只能提出计划，不能直接执行工具。

### 4.3 Planner / Cognition Layer

负责理解扫描结果并生成计划：

- deterministic planner：无外部 API 时的默认可用路径；
- LangGraph/LLM planner：可选增强，用于更自然的计划选择和解释；
- fallback：任何外部认知失败都回到 deterministic planner。

Planner 的职责是返回结构化 `AgentPlan` 或兼容计划，不负责写文件、不负责调用工具、不负责回滚。

### 4.4 Deterministic Tool Layer

负责真实执行：

- 扫描；
- 修复预览；
- 规则修复；
- Gower/KNN 修复；
- 复扫；
- rollback manifest 生成；
- rollback 执行。

同一类扫描、修复和回滚逻辑只能有一个真实来源，避免 Go、Python、LLM 各自实现一套不一致逻辑。

### 4.5 Algorithm & Artifact Layer

负责算法和产物：

- LightGBM 模型；
- 规则扫描结果；
- Gower/KNN 邻居证据；
- 修复后 CSV；
- rollback manifest；
- presentation bundle；
- SQLite task/agent 历史和 trace。

Auto Agent 只组织和解释这些资产，不把它们替换成黑盒输出。

## 5. Agent 能做什么、不能做什么

### 5.1 Agent 可以做

- 读取扫描结果和 profile；
- 统计 issue 类型和风险分布；
- 将 issue 分为自动修复、谨慎处理、人工复核和阻断；
- 在已有候选方案中选择更合适的 repair candidate；
- 生成结构化 repair plan；
- 生成风险说明和用户可读解释；
- 请求人工审批；
- 建议 rollback；
- 记录 trace，方便回放和审计。

### 5.2 Agent 不能做

- 不能直接修改 CSV；
- 不能绕过 `repair_batch`、`repair_with_gower` 等工具写出文件；
- 不能修改 M0-M6 冻结产物；
- 不能修改 Python engine action 协议；
- 不能在没有 rollback manifest 的情况下默认接受写出修复结果；
- 不能因为 LLM 建议而自动修复高风险 issue；
- 不能把自然语言解释当成执行依据；
- 不能将 API key、用户本地路径中的敏感信息或私有配置写入报告。

## 6. Repair Plan 设计边界

本节是概念映射/设计说明，不要求 A1 或 A2 修改现有 wire shape。

后续 repair plan 应至少表达：

```json
{
  "mode": "auto_agent_plan",
  "planner": "deterministic|langgraph|llm",
  "auto_repair_issue_ids": [],
  "cautious_issue_ids": [],
  "manual_review_issue_ids": [],
  "blocked_issue_ids": [],
  "selected_candidate_id": "",
  "repair_strategy": {
    "missing_numeric": "median",
    "missing_categorical": "mode",
    "rare_category": "mode",
    "outlier": "clip"
  },
  "risk_notes": [],
  "explanation": "",
  "trace": []
}
```

与现有 Go 类型的概念映射：

- `auto_repair_issue_ids` 可映射到 `AgentPlan.SelectedIssueIDs` 或候选 payload 中的 selected issues；
- `manual_review_issue_ids`、`cautious_issue_ids`、`blocked_issue_ids` 可映射到 `AgentPlan.SkippedIssues`、reason codes 或 approval context；
- `selected_candidate_id` 可映射到 `AgentPlan.SelectedCandidateID`；
- 候选执行路径可映射到 `RepairCandidate.ToolSequence`、`PlanPayloads`、`ExecutePayloads`；
- 用户解释可映射到 `AgentPlan.UserExplanation`、`ReasoningSummary` 和 presentation 层。

后续实现应优先复用现有结构，不为了文档示例强行改协议。

## 7. 推荐运行流程

### Step 1: Scan

Go runtime 调用 `scan_file` / `engine.scan_table`，得到原始扫描结果、issue 列表、summary 和 profile。

### Step 2: Profile

系统总结数据规模、字段类型、issue 类型分布、高风险列和可修复范围。

### Step 3: Candidate Preview

系统使用 plan-only 或 preview 方式生成候选修复方案。候选可以包括：

- rule candidate：来自 `repair_batch`；
- gower candidate：来自 `repair_with_gower`；
- hybrid candidate：按 issue 来源组合规则和 Gower/KNN 能力。

预览阶段默认不写出最终 CSV。

### Step 4: Plan

Planner 基于 `PlanningInput` 和候选预览生成 `AgentPlan`。

默认风险分类：

- `missing_values`：低风险，允许自动修复；
- `rare_category`：中低风险，可谨慎自动修复并保留解释；
- `numeric_outlier`：谨慎处理，默认需要 plan-only、验证或审批；
- `duplicate_record`：人工复核；
- `cross_column_consistency`：人工复核；
- unknown issue：阻断，不自动执行。

### Step 5: Approval / Execute

低风险计划可进入自动执行。包含谨慎或高风险项时，默认进入审批或人工复核。

执行只能通过确定性工具层完成。Planner 不直接调用修复工具。

### Step 6: Rescan

系统对修复后的 CSV 再次调用扫描工具，获得 after scan。

### Step 7: Validate

Validation gate 比较 before/after：

- issue count 是否下降；
- 是否出现新 issue；
- 修改单元格数量是否异常；
- 是否自动修改了不应自动修改的高风险 issue；
- rollback manifest 是否存在；
- repair tool 是否返回结构化错误。

### Step 8: Rollback / Explain

若验证通过，输出修复结果、风险说明和解释报告。

若验证失败，应执行或推荐 `rollback_repair_batch`。自动回滚后的 rejected output snapshot 应尽量保留，便于复盘。

## 8. 外部 API 与 fallback 策略

外部 LLM API 只能增强规划和解释，不能成为唯一依赖。

### 8.1 sidecar 配置

`APPSHELL_LANGGRAPH_*` 负责 sidecar 本身：

- `APPSHELL_LANGGRAPH_ENABLED`
- `APPSHELL_LANGGRAPH_HOST`
- `APPSHELL_LANGGRAPH_PORT`
- `APPSHELL_LANGGRAPH_SCRIPT`
- `APPSHELL_LANGGRAPH_PYTHON_BIN`
- `APPSHELL_LANGGRAPH_STARTUP_TIMEOUT_MS`
- `APPSHELL_LANGGRAPH_REQUEST_TIMEOUT_MS`

### 8.2 LLM provider 配置

`APPSHELL_LANGGRAPH_LLM_*` 负责 OpenAI-compatible provider：

- `APPSHELL_LANGGRAPH_LLM_BASE_URL`
- `APPSHELL_LANGGRAPH_LLM_API_KEY`
- `APPSHELL_LANGGRAPH_LLM_MODEL`
- `APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS`

文档和测试只能使用占位说明，不能包含真实 key。

### 8.3 fallback 触发条件

以下情况必须 fallback 到 deterministic planner：

- sidecar 禁用；
- sidecar 脚本缺失；
- sidecar 启动失败；
- healthcheck 失败；
- provider 环境变量缺失；
- API key 缺失；
- endpoint 不可达；
- 请求超时；
- 返回非 200；
- 返回非法 JSON；
- schema 校验失败；
- 返回的 candidate 不存在；
- 计划试图自动修复高风险 issue；
- 计划试图绕过 rollback、validation 或 approval gate。

Fallback 不是错误兜底的装饰语，而是默认安全路径。只要外部认知不可信，就继续使用 deterministic planner。

## 9. 验证与回滚策略

### 9.1 默认风险策略

| issue 类型 | 默认策略 | 说明 |
|---|---|---|
| `missing_values` | 自动修复 | 风险相对低，适合 median/mode 等确定性策略 |
| `rare_category` | 谨慎自动修复 | 可自动处理，但必须记录解释和回滚 |
| `numeric_outlier` | 谨慎 / 审批 / plan-only | 既有实验记录了误报和副作用，不能无脑自动修 |
| `duplicate_record` | 人工复核 | 是否删除或合并需要业务语义 |
| `cross_column_consistency` | 人工复核 | 需要判断应修改哪一列 |
| unknown | 阻断 | 未知类型默认不自动执行 |

### 9.2 Validation Gate 输出

Validation gate 至少应能表达：

```json
{
  "verdict": "accept|warn|reject|rollback_recommended",
  "before_issue_count": 0,
  "after_issue_count": 0,
  "resolved_issue_count": 0,
  "total_cells_modified": 0,
  "risk_notes": [],
  "rollback_recommended": false,
  "explanation": ""
}
```

该结构同样是概念映射/设计说明，不要求 A1 改现有 response schema。

### 9.3 回滚原则

- 每次写出修复 CSV 都必须尽量生成 rollback manifest；
- rollback manifest 缺失时，自动化结果不得默认为完全可信；
- 验证失败时应优先保护原始数据和用户可追溯性；
- 回滚动作只能通过 `rollback_repair_batch` 或已有受控工具执行；
- 回滚结果也应进入 agent trace 和最终解释。

## 10. 后续阶段约束

A1 只新增设计文档，不写业务代码。

后续阶段必须按顺序推进：

- A2：deterministic planner 原型；
- A3：LLM planner adapter；
- A4：自动执行、验证与回滚闭环；
- A5：报告与 UI 集成。

每次只能执行一个阶段。除非用户明确要求，否则不得跳过阶段，不得混入 unrelated refactor，不得修改 M0-M6 冻结产物。

## 11. A1 验收标准

A1 完成后应满足：

- 根目录存在 `AUTO_AGENT_DESIGN.md`；
- 文档说明 Auto Agent 是什么；
- 文档说明它与 M0-M6 主线的关系；
- 文档说明 agent 能做什么、不能做什么；
- 文档说明如何调用现有工具；
- 文档说明 repair plan 的概念结构；
- 文档说明外部 API 和 fallback 策略；
- 文档说明验证和回滚策略；
- 文档明确 agent 不直接修改 CSV；
- 不写业务代码；
- 不修改核心逻辑；
- 不修改实验结果。
