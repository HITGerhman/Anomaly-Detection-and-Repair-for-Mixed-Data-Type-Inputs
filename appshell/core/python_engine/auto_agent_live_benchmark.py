"""Run repeated Auto Agent CLI demos and summarize live stability metrics.

The benchmark intentionally delegates every run to ``auto_agent_cli.py`` so it
does not duplicate planner, validation, rollback, or reporting logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any

def _load_auto_agent_cli() -> ModuleType:
    cli_path = Path(__file__).resolve().with_name("auto_agent_cli.py")
    spec = importlib.util.spec_from_file_location("_auto_agent_cli_for_live_benchmark", cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load auto_agent_cli.py from {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auto_agent_cli = _load_auto_agent_cli()


FALLBACK_STATUSES = {"fallback", "disabled", "unavailable", "degraded"}
TIMING_KEYS = [
    "llm_plan_duration_ms",
    "llm_explain_duration_ms",
    "scan_duration_ms",
    "retrieve_duration_ms",
    "repair_duration_ms",
    "validation_duration_ms",
    "rollback_manifest_duration_ms",
    "total_duration_ms",
]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def nested_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile95(values: list[float]) -> float | None:
    return percentile(values, 0.95)


def percentile50(values: list[float]) -> float | None:
    return percentile(values, 0.50)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def env_summary() -> dict[str, Any]:
    return {
        "env_var_names": [
            "APPSHELL_LANGGRAPH_ENABLED",
            "APPSHELL_LANGGRAPH_LLM_BASE_URL",
            "APPSHELL_LANGGRAPH_LLM_MODEL",
            "APPSHELL_LANGGRAPH_LLM_API_KEY",
            "APPSHELL_LANGGRAPH_REQUEST_TIMEOUT_MS",
            "APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS",
        ],
        "langgraph_enabled": os.getenv("APPSHELL_LANGGRAPH_ENABLED", ""),
        "llm_base_url": os.getenv("APPSHELL_LANGGRAPH_LLM_BASE_URL", ""),
        "llm_model": os.getenv("APPSHELL_LANGGRAPH_LLM_MODEL", ""),
        "llm_api_key_configured": bool(os.getenv("APPSHELL_LANGGRAPH_LLM_API_KEY")),
        "langgraph_request_timeout_ms": os.getenv("APPSHELL_LANGGRAPH_REQUEST_TIMEOUT_MS", ""),
        "llm_timeout_ms": os.getenv("APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS", ""),
    }


def cli_argv(args: argparse.Namespace, run_dir: Path) -> list[str]:
    argv = [
        "--csv",
        str(args.csv),
        "--output-dir",
        str(run_dir),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.model_dir is not None:
        argv.extend(["--model-dir", str(args.model_dir)])
    if args.goal:
        argv.extend(["--goal", args.goal])
    if args.go_bin:
        argv.extend(["--go-bin", args.go_bin])
    if args.backend_dir is not None:
        argv.extend(["--backend-dir", str(args.backend_dir)])
    return argv


def rollback_manifest_path(response: dict[str, Any]) -> str:
    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    execution = nested_map(agent.get("execution"))
    rollback = nested_map(execution.get("rollback"))
    return str(rollback.get("manifest_path") or execution.get("rollback_manifest_path") or "")


def summarize_run(run_index: int, run_dir: Path, returncode: int, wall_ms: int) -> dict[str, Any]:
    response = read_json(run_dir / "response.json")
    timings = read_json(run_dir / "timings.json")
    trace = read_json(run_dir / "auto_agent_trace.json")
    error = read_json(run_dir / "error.json")

    response_map = nested_map(response)
    result = nested_map(response_map.get("result"))
    agent = nested_map(result.get("agent"))
    plan = nested_map(agent.get("plan"))
    validation = nested_map(agent.get("validation"))
    post_validation = nested_map(validation.get("post_execute"))
    safety = nested_map(result.get("safety"))
    cognition = nested_map(plan.get("cognition"))
    timings_map = nested_map(timings)

    fallback_reason = str(cognition.get("fallback_reason_code") or "")
    cognition_status = str(cognition.get("status") or "")
    fallback = cognition_status in FALLBACK_STATUSES or bool(fallback_reason)
    total_ms = as_number(timings_map.get("total_duration_ms"))
    if total_ms is None:
        total_ms = as_number(response_map.get("duration_ms"))
    if total_ms is None:
        total_ms = float(wall_ms)

    validation_verdict = str(post_validation.get("verdict") or "")
    final_verdict = str(safety.get("final_verdict") or "")
    trace_count = len(trace) if isinstance(trace, list) else 0
    manifest = rollback_manifest_path(response_map)

    run_result = {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "status": "success" if returncode == 0 and response_map else "failed",
        "duration_ms": total_ms,
        "wall_duration_ms": wall_ms,
        "final_verdict": final_verdict,
        "validation_verdict": validation_verdict,
        "cognition_provider": str(cognition.get("provider") or ""),
        "cognition_status": cognition_status,
        "planner_mode": str(cognition.get("planner_mode") or ""),
        "llm_mode": str(cognition.get("llm_mode") or ""),
        "fallback_reason_code": fallback_reason,
        "fallback": fallback,
        "accepted": final_verdict == "accepted" and validation_verdict in {"accept", "warn"},
        "validation_rejected": validation_verdict in {"reject", "rollback_recommended"},
        "trace_event_count": trace_count,
        "rollback_manifest_created": bool(manifest),
        "rollback_manifest_path": manifest,
        "error": error,
    }
    for key in TIMING_KEYS:
        run_result[key] = timings_map.get(key)
    return run_result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success_runs = sum(1 for item in results if item.get("status") == "success")
    accepted_runs = sum(1 for item in results if item.get("accepted") is True)
    fallback_runs = sum(1 for item in results if item.get("fallback") is True)
    validation_reject_runs = sum(1 for item in results if item.get("validation_rejected") is True)
    manifest_runs = sum(1 for item in results if item.get("rollback_manifest_created") is True)
    total_ms = [float(item["duration_ms"]) for item in results if as_number(item.get("duration_ms")) is not None]
    trace_counts = [float(item.get("trace_event_count", 0)) for item in results]

    rate = lambda count: (count / total) if total else 0.0
    return {
        "total_runs": total,
        "success_runs": success_runs,
        "success_rate": rate(success_runs),
        "accepted_runs": accepted_runs,
        "accepted_rate": rate(accepted_runs),
        "fallback_runs": fallback_runs,
        "fallback_rate": rate(fallback_runs),
        "validation_reject_runs": validation_reject_runs,
        "avg_total_ms": mean(total_ms),
        "p95_total_ms": percentile95(total_ms),
        "rollback_manifest_created_rate": rate(manifest_runs),
        "avg_trace_event_count": mean(trace_counts),
        "fallback_reason_counts": fallback_reason_counts(results),
    }


def fallback_reason_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        if not item.get("fallback"):
            continue
        reason = str(item.get("fallback_reason_code") or item.get("cognition_status") or "fallback")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def aggregate_timings(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = len(results)
    timings_by_stage: dict[str, list[float]] = {key: [] for key in TIMING_KEYS}
    timings_file_runs = 0

    for item in results:
        run_dir = Path(str(item.get("run_dir") or ""))
        timings = read_json(run_dir / "timings.json")
        if isinstance(timings, dict):
            timings_file_runs += 1
        else:
            timings = {}
        for key in TIMING_KEYS:
            value = as_number(timings.get(key))
            if value is not None:
                timings_by_stage[key].append(value)

    stage_stats: dict[str, dict[str, Any]] = {}
    total_avg = mean(timings_by_stage["total_duration_ms"])
    for key in TIMING_KEYS:
        values = timings_by_stage[key]
        avg_value = mean(values)
        missing_runs = total_runs - len(values)
        share: float | None = None
        if key == "total_duration_ms" and avg_value is not None:
            share = 1.0
        elif avg_value is not None and total_avg not in (None, 0):
            share = avg_value / float(total_avg)
        stage_stats[key] = {
            "stage": key,
            "avg_ms": avg_value,
            "p50_ms": percentile50(values),
            "p95_ms": percentile95(values),
            "max_ms": max(values) if values else None,
            "present_runs": len(values),
            "missing_runs": missing_runs,
            "missing": missing_runs > 0,
            "share_of_total_avg": share,
        }

    top_slowest = [
        {
            "stage": item["stage"],
            "avg_ms": item["avg_ms"],
            "share_of_total_avg": item["share_of_total_avg"],
        }
        for item in sorted(
            (stage_stats[key] for key in TIMING_KEYS if key != "total_duration_ms" and stage_stats[key]["avg_ms"] is not None),
            key=lambda value: float(value["avg_ms"]),
            reverse=True,
        )[:3]
    ]
    missing_fields = {key: value["missing_runs"] for key, value in stage_stats.items() if value["missing_runs"] > 0}
    dominant_area, dominant_avg = dominant_timing_area(stage_stats)
    return {
        "total_runs": total_runs,
        "timings_file_runs": timings_file_runs,
        "stages": stage_stats,
        "top_slowest_stages": top_slowest,
        "missing_fields": missing_fields,
        "dominant_area": dominant_area,
        "dominant_area_avg_ms": dominant_avg,
    }


def dominant_timing_area(stage_stats: dict[str, dict[str, Any]]) -> tuple[str, float | None]:
    groups = {
        "LLM": ["llm_plan_duration_ms", "llm_explain_duration_ms"],
        "scan": ["scan_duration_ms"],
        "retrieve": ["retrieve_duration_ms"],
        "repair": ["repair_duration_ms"],
        "validation": ["validation_duration_ms"],
        "rollback": ["rollback_manifest_duration_ms"],
    }
    totals: dict[str, float] = {}
    for group, keys in groups.items():
        values = [as_number(stage_stats.get(key, {}).get("avg_ms")) for key in keys]
        present = [float(value) for value in values if value is not None]
        if present:
            totals[group] = sum(present)
    if not totals:
        return "", None
    winner = max(totals.items(), key=lambda item: item[1])
    return winner[0], winner[1]


def build_timings_summary_md(timings_summary: dict[str, Any]) -> str:
    stages = nested_map(timings_summary.get("stages"))
    top_stages = list_from_any(timings_summary.get("top_slowest_stages"))
    top_rank = {str(item.get("stage")): index + 1 for index, item in enumerate(top_stages) if isinstance(item, dict)}
    missing_fields = nested_map(timings_summary.get("missing_fields"))
    dominant = str(timings_summary.get("dominant_area") or "")
    dominant_avg = timings_summary.get("dominant_area_avg_ms")
    conclusion = "No timing data was available."
    if dominant:
        conclusion = f"Primary latency contributor: {dominant} (avg {dominant_avg} ms)."

    lines = [
        "# Auto Agent Live Benchmark Timings Summary",
        "",
        f"- total_runs: `{timings_summary.get('total_runs')}`",
        f"- timings_file_runs: `{timings_summary.get('timings_file_runs')}`",
        f"- missing_fields: `{json.dumps(missing_fields, ensure_ascii=False)}`",
        f"- conclusion: {conclusion}",
        "",
        "## Top 3 Slowest Stages",
        "",
    ]
    if top_stages:
        for index, item in enumerate(top_stages, start=1):
            lines.append(f"{index}. `{item.get('stage')}` avg `{item.get('avg_ms')}` ms, share `{item.get('share_of_total_avg')}`")
    else:
        lines.append("- missing")
    lines.extend(
        [
            "",
            "## Stage Breakdown",
            "",
            "| stage | top | avg_ms | p50_ms | p95_ms | max_ms | present_runs | missing_runs | missing | share_of_total_avg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for key in TIMING_KEYS:
        item = nested_map(stages.get(key))
        top_label = f"top {top_rank[key]}" if key in top_rank else ""
        lines.append(
            f"| `{key}` | {top_label} | {item.get('avg_ms')} | {item.get('p50_ms')} | "
            f"{item.get('p95_ms')} | {item.get('max_ms')} | {item.get('present_runs')} | "
            f"{item.get('missing_runs')} | {item.get('missing')} | {item.get('share_of_total_avg')} |"
        )
    lines.append("")
    return "\n".join(lines)


def list_from_any(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_summary_md(args: argparse.Namespace, summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    env = summary["environment"]
    lines = [
        "# Auto Agent Live API Stability Benchmark",
        "",
        "## 配置",
        "",
        f"- 输入 CSV：`{args.csv}`",
        f"- runs：`{args.runs}`",
        f"- output dir：`{args.output_dir}`",
        f"- LangGraph enabled：`{env.get('langgraph_enabled', '')}`",
        f"- LLM base URL：`{env.get('llm_base_url', '')}`",
        f"- LLM model：`{env.get('llm_model', '')}`",
        f"- API key configured：`{env.get('llm_api_key_configured', False)}`",
        "",
        "## 汇总指标",
        "",
        f"- total_runs：`{summary['total_runs']}`",
        f"- success_runs / success_rate：`{summary['success_runs']}` / `{summary['success_rate']:.2%}`",
        f"- accepted_runs / accepted_rate：`{summary['accepted_runs']}` / `{summary['accepted_rate']:.2%}`",
        f"- fallback_runs / fallback_rate：`{summary['fallback_runs']}` / `{summary['fallback_rate']:.2%}`",
        f"- validation_reject_runs：`{summary['validation_reject_runs']}`",
        f"- avg_total_ms：`{summary['avg_total_ms']}`",
        f"- p95_total_ms：`{summary['p95_total_ms']}`",
        f"- rollback_manifest_created_rate：`{summary['rollback_manifest_created_rate']:.2%}`",
        f"- avg_trace_event_count：`{summary['avg_trace_event_count']}`",
        f"- fallback_reason_counts：`{json.dumps(summary['fallback_reason_counts'], ensure_ascii=False)}`",
        "",
        "## 单次结果",
        "",
        "| run | status | final | validation | fallback | total_ms | trace | rollback_manifest |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['run_index']} | {item['status']} | {item['final_verdict']} | {item['validation_verdict']} | "
            f"{item['fallback']} | {item['duration_ms']} | {item['trace_event_count']} | {item['rollback_manifest_created']} |"
        )
    lines.append("")
    lines.append("环境变量报告只记录变量名、模型名和是否配置 API key，不记录 API key 值。")
    lines.append("")
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> int:
    args.csv = args.csv.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
    if args.backend_dir is not None:
        args.backend_dir = args.backend_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    exit_code = 0
    for index in range(1, args.runs + 1):
        run_dir = args.output_dir / f"run_{index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        returncode = auto_agent_cli.main(cli_argv(args, run_dir))
        wall_ms = int((time.perf_counter() - started) * 1000)
        run_result = summarize_run(index, run_dir, returncode, wall_ms)
        write_json(run_dir / "run_result.json", run_result)
        results.append(run_result)
        if returncode != 0:
            exit_code = returncode
            if not args.continue_on_error:
                break

    summary = aggregate(results)
    summary["environment"] = env_summary()
    summary["input_csv"] = str(args.csv)
    summary["output_dir"] = str(args.output_dir)
    timings_summary = aggregate_timings(results)
    timings_summary_path = args.output_dir / "timings_summary.json"
    write_json(timings_summary_path, timings_summary)
    (args.output_dir / "timings_summary.md").write_text(build_timings_summary_md(timings_summary), encoding="utf-8")
    summary["timings_summary_path"] = str(timings_summary_path)
    summary["timings_top_slowest_stages"] = timings_summary["top_slowest_stages"]
    summary["timings_dominant_area"] = timings_summary["dominant_area"]
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(build_summary_md(args, summary, results), encoding="utf-8")
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = auto_agent_cli.repo_root_from_here()
    parser = argparse.ArgumentParser(description="Repeated Auto Agent live benchmark")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--goal", default="")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--go-bin", default="go")
    parser.add_argument("--backend-dir", type=Path, default=repo_root / "appshell" / "backend")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
