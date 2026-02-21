# MEMO

Last updated: 2026-02-21 21:03:40

## 最终目标

将当前“混合类型异常检测与修复”项目产品化为可交付桌面应用：
- Python 负责稳定可复用的算法内核。
- Go 负责任务生命周期与进程编排。
- Wails 前端负责可用、可观测、可导出的用户流程。
- 全链路具备可追溯日志与历史任务持久化能力。

## 当前总体方法

1. 分层解耦：算法层（Python）与服务/编排层（Go）和展示层（Wails）彻底分离。
2. 协议优先：统一 `input.json -> output.json` 协议、错误码与结构化日志。
3. 任务化运行：前端不直接连算法脚本，通过 Go 的任务服务调度 Python 引擎。
4. 可观测性先行：`task_id` 贯穿前端事件、Go 日志、Python 日志，并持久化历史。

## 已完成阶段

### 阶段1：Python 内核产品化（已完成）

- 核心训练逻辑从脚本抽离到 `src/training_core.py`。
- 保留兼容导出 `src/utils.py`，避免影响 `app.py` 现有调用。
- 完成 Python 引擎三层拆分：
  - `appshell/core/python_engine/engine_core.py`
  - `appshell/core/python_engine/engine_service.py`
  - `appshell/core/python_engine/engine_main.py`
- 完成统一协议与错误码：`appshell/core/python_engine/engine_protocol.py`。
- 完成结构化日志：`appshell/core/python_engine/engine_logging.py`。
- 测试覆盖核心验收：同输入 10 次一致、异常输入返回明确错误码、核心算法测试通过。

### 阶段2：Go 后端编排层（已完成）

- 在 `appshell/backend/internal/task/service.go` 实现：
  - `RunTask` / `CancelTask` / `GetTaskStatus`
  - 并发队列与 worker 池（默认并发 3）
  - 超时、取消、状态流转与兼容旧接口
- `appshell/backend/internal/engine/runner.go` 基于 `exec.CommandContext` 调 Python。
- 验收达成：至少 3 并发、取消 2 秒内生效、超时后无僵尸进程。

### 阶段3：Wails 前端 MVP（已完成）

- 完成 UI 流程：参数配置 -> 启动任务 -> 进度展示 -> 结果查看 -> 导出。
- 新增 Wails 绑定入口与调用方法（`RunTask`/`GetTaskStatus`/`CancelTask` 等）。
- 完成 macOS 风格视觉重构与 3D Wizard 卡片翻转交互。
- 新增 CSV 列自动识别：前端从后端读取可用列，避免手输目标列。

### 阶段4：数据与可观测性（已完成）

- Go 结构化日志落盘：`outputs/appshell/go_backend.log`（可环境变量覆盖）。
- Python `stderr` 结构化日志转发到 Go 日志流。
- 任务历史 sqlite 持久化：`outputs/appshell/task_history.sqlite`。
- 重启后可查询最近历史；支持仅保留最近 N 条任务记录。

## 本次改动记录（算法优化）

- 改动日期：2026-02-21 21:03:40
- 改动内容简述：
  - 目标：提升异常检测在中等难度数据上的有效性与可解释性，减少“只看准确率”导致的漏检风险。
  - 动机：此前策略在 `simple_medium_anomaly.csv` 上出现高精度但召回偏低（漏检偏多）。
  - 方法：引入“类不平衡处理 + 阈值调优 + 指标体系升级”的组合优化。
- 相关模块/文件：
  - `src/training_core.py`
  - `src/utils.py`
  - `app.py`
  - `appshell/core/python_engine/engine_core.py`
  - `tests/python_engine/test_training_core.py`
  - `MEMO.md`
- 已解决的问题/新增功能：
  - 新增类不平衡训练策略：`class_weight`（默认 `balanced`）。
  - 新增阈值调优流程：
    - 在训练集内再划分验证集。
    - 以 `F-beta`（默认 `beta=2`）为主目标，在最小精度约束下选阈值。
    - 选出的阈值写入模型属性 `decision_threshold`。
  - 新增阈值推理接口：`predict_with_threshold()` 与 `get_decision_threshold()`。
  - `app.py` 与引擎汇总改为阈值感知预测，不再固定 `0.5`。
  - 指标体系升级：
    - 二分类输出 anomaly-focused 指标（`precision/recall/f1` 指向异常类）
    - 保留 `*_weighted` 指标
    - 输出 `decision_threshold` 与 `threshold_optimization` 元信息
  - 新增/更新测试：验证阈值字段、异常指标字段、阈值预测一致性与确定性。
- 待处理事项：
  - 当前仍以“离散标签（分类）”为主，连续目标（如 `bmi`）暂不支持直接训练。
  - 后续可增加自动任务类型判断（分类/回归）并扩展回归型异常评分路径。
  - 可加入阈值校准模式开关（偏召回 / 偏精度 / 平衡）供前端选择。

## 算法优化思路重点说明

1. 不再只追求 Accuracy
- 异常检测通常类不平衡，准确率高不代表漏检少。
- 本轮将“异常类召回”提升到与精度同等优先级。

2. 训练阶段先处理类不平衡
- `class_weight=balanced` 让模型对少数类（异常）更敏感。
- 目的不是盲目提高召回，而是降低模型对多数类的偏置。

3. 推理阶段做阈值优化而不是固定 0.5
- 概率阈值直接影响 FP/FN 平衡。
- 使用验证集搜索阈值：在满足最小精度前提下，最大化 `F-beta`（默认偏向召回）。
- 这样可以把“业务偏好”显式参数化，而不是隐藏在模型默认行为里。

4. 指标分层输出，便于决策
- anomaly-focused 指标用于回答“异常抓得怎样”。
- weighted 指标用于回答“整体预测稳定性怎样”。
- 两组同时看，避免单指标误导。

## 本轮实测结果（中等难度样本）

数据：`data/raw/simple_medium_anomaly.csv`（目标列 `stroke`）

- 优化前（历史记录）：
  - `accuracy=0.875`
  - `precision=1.0000`
  - `recall=0.3750`
  - `confusion_matrix=[[64,0],[10,6]]`
- 优化后（当前实现）：
  - `decision_threshold=0.4898679324`
  - `accuracy=0.8375`
  - `precision=0.6364`
  - `recall=0.4375`
  - `f1=0.5185`
  - `auc=0.7852`
  - `confusion_matrix=[[60,4],[9,7]]`
  - `threshold_optimization.selected_fbeta=0.6098`

结论：
- 召回从 `0.3750` 提升到 `0.4375`（漏检减少）。
- 精度与准确率有所下降，属于“更偏召回”的可控权衡。
- 该权衡符合异常检测“宁可多报少漏”的常见目标，可继续按业务需求调节阈值策略。

## 已验证结果（本轮）

- `python -m pytest -q tests/python_engine` -> `10 passed`
- 中等样本重复 10 次训练/评估签名一致（阈值与核心指标稳定）。

## 当前正在解决的问题

1. 连续目标列（如 `bmi`）不适用当前分类训练路径。
2. 需要把“阈值策略”进一步产品化为可配置选项，并在 UI 中可解释展示。
3. 需要在后续阶段补充打包与部署（阶段5）链路，完成交付闭环。

## 下一步建议（按优先级）

1. 阶段5：Windows 打包落地（Python 引擎可执行化 + Wails 打包 + 安装器）。
2. 算法扩展：补充回归目标的异常评分方案（如残差/分位数/孤立森林混合路径）。
3. 体验增强：前端增加阈值策略选择与“精度-召回”说明，支持重跑对比。
