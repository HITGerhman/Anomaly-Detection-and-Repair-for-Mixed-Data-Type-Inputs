"""Minimal Auto Agent CLI demo wrapper.

This script intentionally reuses the Go agent runtime instead of rebuilding
planner or validation logic in Python. It only launches the existing demo
runner, extracts the structured response, and writes demo artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_response_from_stdout(stdout: str) -> dict[str, Any] | None:
    marker = "response:"
    start_at = stdout.rfind(marker)
    if start_at >= 0:
        segment = stdout[start_at + len(marker) :]
        parsed = decode_first_json_object(segment)
        if parsed is not None:
            return parsed

    last: dict[str, Any] | None = None
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "task_id" in value and "status" in value:
            last = value
    return last


def decode_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def build_command(args: argparse.Namespace, history_db: Path) -> list[str]:
    timeout = f"{int(args.timeout_seconds)}s"
    cmd = [
        args.go_bin,
        "run",
        "./cmd/demo",
        "-action",
        "agent.session.auto",
        "-csv",
        str(args.csv.resolve()),
        "-output",
        str(args.output_dir.resolve()),
        "-history-db",
        str(history_db.resolve()),
        "-engine",
        str((repo_root_from_here() / "appshell" / "core" / "python_engine" / "engine_main.py").resolve()),
        "-timeout",
        timeout,
    ]
    if args.goal:
        cmd.extend(["-goal", args.goal])
    if args.model_dir:
        cmd.extend(["-model-dir", str(args.model_dir.resolve())])
    return cmd


def run_go_demo(cmd: list[str], backend_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    venv_scripts = repo_root_from_here() / ".venv-win" / "Scripts"
    if venv_scripts.exists():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        cmd,
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def read_trace(history_db: Path, session_id: str) -> list[dict[str, Any]]:
    if not history_db.exists() or not session_id:
        return []
    query = """
        SELECT id, session_id, task_id, seq, agent_name, trace_type, summary, payload_json, created_at_unix_ms
        FROM agent_trace
        WHERE session_id = ?
        ORDER BY seq ASC, id ASC;
    """
    try:
        with sqlite3.connect(history_db) as conn:
            rows = conn.execute(query, (session_id,)).fetchall()
    except sqlite3.Error:
        return []

    events: list[dict[str, Any]] = []
    for row in rows:
        payload: Any = {}
        try:
            payload = json.loads(row[7] or "{}")
        except json.JSONDecodeError:
            payload = {}
        events.append(
            {
                "id": row[0],
                "session_id": row[1],
                "task_id": row[2],
                "seq": row[3],
                "agent_name": row[4],
                "trace_type": row[5],
                "summary": row[6],
                "payload": payload,
                "created_at_unix_ms": row[8],
            }
        )
    return events


def nested_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_count(plan: dict[str, Any], key: str) -> int:
    value = plan.get(key)
    return len(value) if isinstance(value, list) else 0


def rollback_manifest(execution: dict[str, Any]) -> str:
    rollback = nested_map(execution.get("rollback"))
    return str(rollback.get("manifest_path") or execution.get("rollback_manifest_path") or "")


def default_metric_definitions() -> dict[str, str]:
    return {
        "before_issue_items": "修复前 scan 发现的问题条目数。",
        "after_issue_items": "修复后 rescan 仍存在的问题条目数。",
        "resolved_issue_items": "问题条目减少数，计算方式为 max(before_issue_items - after_issue_items, 0)。",
        "modified_cell_count": "实际被修复流程改写的 CSV 单元格数量。",
        "rollback_manifest_created": "是否为本次写出结果生成了可回滚 manifest。",
        "rollback_recommended": "validation gate 是否认为当前输出不安全，需要回滚或建议回滚。",
        "resolved_issue_count": "兼容旧字段；新报告不使用该字段作为简历或答辩口径。",
        "total_cells_modified": "兼容旧字段；等价于 modified_cell_count。",
    }


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalized_post_validation(post_validation: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(post_validation)
    before = _to_int(normalized.get("before_issue_items", normalized.get("before_issue_count", 0)))
    after = _to_int(normalized.get("after_issue_items", normalized.get("after_issue_count", 0)))
    modified = _to_int(normalized.get("modified_cell_count", normalized.get("total_cells_modified", normalized.get("changed_cell_count", 0))))
    normalized["before_issue_items"] = before
    normalized["after_issue_items"] = after
    normalized["resolved_issue_items"] = _to_int(normalized.get("resolved_issue_items", max(before - after, 0)))
    normalized["modified_cell_count"] = modified
    definitions = nested_map(normalized.get("metric_definitions"))
    normalized["metric_definitions"] = definitions if definitions else default_metric_definitions()
    return normalized


def issue_explanations(plan: dict[str, Any]) -> dict[str, Any]:
    blocked = plan.get("blocked_issue_details")
    cautious = plan.get("cautious_issue_details")
    reason_counts = plan.get("blocked_reason_counts")
    if isinstance(blocked, list) and isinstance(cautious, list) and isinstance(reason_counts, dict):
        return {
            "blocked_issue_details": blocked,
            "cautious_issue_details": cautious,
            "blocked_reason_counts": reason_counts,
        }

    blocked_items: list[dict[str, Any]] = []
    cautious_items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for item in plan.get("skipped_issues") or []:
        if not isinstance(item, dict):
            continue
        details = nested_map(item.get("details"))
        bucket = str(details.get("bucket") or "").strip()
        reason = str(item.get("reason") or details.get("reason") or "").strip()
        if bucket == "blocked" or reason in {"unsupported_issue_type", "missing_issue_id", "blocked_issue_type"}:
            blocked_reason = "missing_issue_id" if reason == "missing_issue_id" else "unsupported_issue_type"
            blocked_items.append(
                {
                    "issue_id": item.get("issue_id", ""),
                    "issue_type": item.get("issue_type", ""),
                    "column": item.get("column", ""),
                    "blocked_reason": blocked_reason,
                    "blocked_by_rule": "a2_deterministic_issue_bucket_policy",
                    "suggested_next_action": details.get("recommended_action", "block_until_supported"),
                }
            )
            counts[blocked_reason] = counts.get(blocked_reason, 0) + 1
        elif bucket == "cautious" or reason in {"requires_human_review_before_auto_repair", "cautious_issue_type"}:
            cautious_items.append(
                {
                    "issue_id": item.get("issue_id", ""),
                    "issue_type": item.get("issue_type", ""),
                    "column": item.get("column", ""),
                    "risk_reason": "requires_human_review_before_auto_repair",
                    "approval_required": True,
                    "suggested_action": details.get("recommended_action", "review_before_auto_repair"),
                }
            )
    return {
        "blocked_issue_details": blocked_items,
        "cautious_issue_details": cautious_items,
        "blocked_reason_counts": counts,
    }


def merged_timings(response: dict[str, Any], plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    for source in (nested_map(plan.get("timings_ms")), nested_map(execution.get("timings_ms"))):
        for key, value in source.items():
            timings[key] = value
    if "total_duration_ms" not in timings:
        timings["total_duration_ms"] = response.get("duration_ms")
    for key in [
        "scan_duration_ms",
        "retrieve_duration_ms",
        "llm_plan_duration_ms",
        "llm_explain_duration_ms",
        "repair_duration_ms",
        "validation_duration_ms",
        "rollback_manifest_duration_ms",
        "total_duration_ms",
    ]:
        timings.setdefault(key, None)
    return timings


def build_report(csv_path: Path, response: dict[str, Any] | None, trace: list[dict[str, Any]], error: dict[str, Any] | None = None) -> str:
    if response is None:
        reason = nested_map(error).get("message", "Go demo did not return a structured agent response.")
        return "\n".join(
            [
                "# Auto Agent CLI Demo Report",
                "",
                f"- 输入 CSV：`{csv_path}`",
                "- 状态：failed",
                f"- 失败原因：{reason}",
                "",
            ]
        )

    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    plan = nested_map(agent.get("plan"))
    validation = nested_map(agent.get("validation"))
    post_validation = normalized_post_validation(nested_map(validation.get("post_execute")))
    execution = nested_map(agent.get("execution"))
    safety = nested_map(result.get("safety"))
    explanations = issue_explanations(plan)
    timings = merged_timings(response, plan, execution)

    output_csv = execution.get("output_csv") or ""
    manifest = rollback_manifest(execution)
    manifest_created = bool(manifest)
    risk_notes = post_validation.get("risk_notes") or post_validation.get("risk_flags") or []
    if not isinstance(risk_notes, list):
        risk_notes = []
    blocked_reason_counts = explanations["blocked_reason_counts"]

    timing_rows = [f"| `{key}` | {'' if value is None else value} |" for key, value in timings.items()]
    lines = [
        "# Auto Agent CLI Demo Report",
        "",
        "## 概览",
        "",
        f"- 输入 CSV：`{csv_path}`",
        f"- 修复输出 CSV：`{output_csv}`",
        f"- task status：`{response.get('status', '')}`",
        f"- session id：`{agent.get('session_id', '')}`",
        f"- final verdict：`{safety.get('final_verdict', '')}`",
        f"- validation verdict：`{post_validation.get('verdict', '')}`",
        "",
        "## Repair Plan",
        "",
        f"- selected source：`{plan.get('selected_source', '')}`",
        f"- auto repair issues：{list_count(plan, 'auto_repair_issue_ids')}",
        f"- cautious issues：{list_count(plan, 'cautious_issue_ids')}",
        f"- manual review issues：{list_count(plan, 'manual_review_issue_ids')}",
        f"- blocked issues：{list_count(plan, 'blocked_issue_ids')}",
        f"- blocked reason 分布：{json.dumps(blocked_reason_counts, ensure_ascii=False) if blocked_reason_counts else '{}'}",
        "",
        "## Validation",
        "",
        f"- before_issue_items：{post_validation.get('before_issue_items', '')}",
        f"- after_issue_items：{post_validation.get('after_issue_items', '')}",
        f"- resolved_issue_items：{post_validation.get('resolved_issue_items', '')}",
        f"- modified_cell_count：{post_validation.get('modified_cell_count', '')}",
        f"- rollback_recommended：{post_validation.get('rollback_recommended', '')}",
        f"- risk notes：{', '.join(str(item) for item in risk_notes) if risk_notes else 'none'}",
        "",
        "## Metric Definitions",
        "",
        "- `before_issue_items` / `after_issue_items` 是 scan 问题条目数，不是单元格数。",
        "- `resolved_issue_items` 是问题条目减少数。",
        "- `modified_cell_count` 是实际被修改的单元格数。",
        "- `resolved_issue_count` 是兼容旧字段，新报告不使用它作为简历或答辩口径。",
        "",
        "## Rollback 说明",
        "",
        f"- rollback_manifest_created：{manifest_created}",
        f"- rollback_recommended：{post_validation.get('rollback_recommended', '')}",
        f"- rollback_manifest_path：`{manifest}`",
        "- rollback manifest 是写出修复结果后的恢复凭证；rollback_recommended 只有在 validation gate 判定输出不安全时才为 true。",
        "",
        "## Timings",
        "",
        "| stage | duration_ms |",
        "|---|---:|",
        *timing_rows,
        "",
        "## Trace",
        "",
        f"- trace events：{len(trace)}",
        "",
        "## 回滚提示",
        "",
        "如果 validation verdict 为 `reject` 或 `rollback_recommended`，应优先使用保留的 rollback manifest 恢复或人工复核。",
        "",
    ]
    return "\n".join(lines)


def artifact_blocks(response: dict[str, Any], trace: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    plan = nested_map(agent.get("plan"))
    validation = nested_map(agent.get("validation"))
    execution = nested_map(agent.get("execution"))
    post_validation = normalized_post_validation(nested_map(validation.get("post_execute")))
    explanations = issue_explanations(plan)
    timings = merged_timings(response, plan, execution)
    timings["trace_event_count"] = len(trace)
    return post_validation, explanations, timings


def write_artifacts(output_dir: Path, csv_path: Path, history_db: Path, response: dict[str, Any] | None, error: dict[str, Any] | None = None) -> bool:
    if response is None:
        if error is not None:
            write_json(output_dir / "error.json", error)
        (output_dir / "report.md").write_text(build_report(csv_path, None, [], error), encoding="utf-8")
        return False

    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    plan = nested_map(agent.get("plan"))
    validation = nested_map(agent.get("validation"))
    execution = nested_map(agent.get("execution"))
    safety = nested_map(result.get("safety"))
    session_id = str(agent.get("session_id") or "")
    trace = read_trace(history_db, session_id)
    post_validation, explanations, timings = artifact_blocks(response, trace)
    validation["post_execute"] = post_validation
    agent["validation"] = validation
    result["agent"] = agent
    response["result"] = result

    write_json(output_dir / "response.json", response)
    write_json(output_dir / "repair_plan.json", plan)
    write_json(output_dir / "validation_result.json", post_validation)
    write_json(output_dir / "metric_definitions.json", post_validation.get("metric_definitions", default_metric_definitions()))
    write_json(output_dir / "issue_explanations.json", explanations)
    write_json(output_dir / "timings.json", timings)
    write_json(output_dir / "execution.json", execution)
    write_json(output_dir / "safety.json", safety)
    write_json(output_dir / "auto_agent_trace.json", trace)
    (output_dir / "report.md").write_text(build_report(csv_path, response, trace), encoding="utf-8")

    complete = bool(agent and session_id and plan and validation)
    if not complete:
        write_json(
            output_dir / "error.json",
            {
                "message": "Agent response is missing session, plan, or validation fields.",
                "has_agent": bool(agent),
                "session_id": session_id,
                "has_plan": bool(plan),
                "has_validation": bool(validation),
            },
        )
    return complete


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = repo_root_from_here()
    parser = argparse.ArgumentParser(description="Auto Agent CLI demo")
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Independent output directory for demo artifacts")
    parser.add_argument("--goal", default="", help="Optional user goal for agent.session.auto")
    parser.add_argument("--model-dir", type=Path, default=None, help="Optional model artifacts directory")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Go demo task timeout in seconds")
    parser.add_argument("--go-bin", default="go", help="Go executable")
    parser.add_argument("--backend-dir", type=Path, default=repo_root / "appshell" / "backend", help="AppShell Go backend directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.csv = args.csv.resolve()
    args.output_dir = args.output_dir.resolve()
    args.backend_dir = args.backend_dir.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_db = args.output_dir / "auto_agent.sqlite"
    cmd = build_command(args, history_db)

    completed = run_go_demo(cmd, args.backend_dir)
    (args.output_dir / "run_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (args.output_dir / "run_stderr.log").write_text(completed.stderr or "", encoding="utf-8")

    response = extract_response_from_stdout(completed.stdout or "")
    error: dict[str, Any] | None = None
    if response is None:
        error = {
            "message": "Go demo did not return a structured response.",
            "returncode": completed.returncode,
        }
    artifacts_complete = write_artifacts(args.output_dir, args.csv, history_db, response, error)
    if completed.returncode != 0:
        return completed.returncode
    return 0 if artifacts_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
