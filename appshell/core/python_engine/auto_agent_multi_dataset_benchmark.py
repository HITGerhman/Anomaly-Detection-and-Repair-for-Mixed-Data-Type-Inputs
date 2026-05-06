"""Run Auto Agent against multiple mixed-type datasets and summarize stability.

The benchmark delegates each run to ``auto_agent_cli.py``. This module only
selects datasets, creates isolated run directories, and aggregates the artifacts
that the CLI already writes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path
from types import ModuleType
from typing import Any


FALLBACK_STATUSES = {"fallback", "disabled", "unavailable", "degraded"}
REQUIRED_SUMMARY_FIELDS = [
    "dataset_name",
    "row_count",
    "column_count",
    "total_runs",
    "success_rate",
    "accepted_rate",
    "fallback_rate",
    "before_issue_items_avg",
    "after_issue_items_avg",
    "resolved_issue_items_avg",
    "modified_cell_count_avg",
    "blocked_issue_count_avg",
    "cautious_issue_count_avg",
    "rollback_manifest_created_rate",
    "avg_total_ms",
    "p95_total_ms",
    "avg_trace_event_count",
]


def _load_auto_agent_cli() -> ModuleType:
    cli_path = Path(__file__).resolve().with_name("auto_agent_cli.py")
    spec = importlib.util.spec_from_file_location("_auto_agent_cli_for_multi_dataset_benchmark", cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load auto_agent_cli.py from {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auto_agent_cli = _load_auto_agent_cli()


def repo_root() -> Path:
    return auto_agent_cli.repo_root_from_here()


def default_datasets() -> list[tuple[str, Path]]:
    root = repo_root()
    return [
        ("m1_stroke", root / "data" / "experiments" / "m1_stroke" / "corrupted.csv"),
        (
            "orders_transactions",
            root / "data" / "experiments" / "auto_agent_multi_dataset" / "orders_transactions" / "corrupted.csv",
        ),
        (
            "user_device_logs",
            root / "data" / "experiments" / "auto_agent_multi_dataset" / "user_device_logs" / "corrupted.csv",
        ),
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


def list_from_any(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    number = as_number(value)
    return int(number) if number is not None else 0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def resolve_path(path: Path) -> Path:
    raw = path.expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (repo_root() / raw).resolve()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, len(header)


def safe_dir_name(dataset_name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name.strip())
    return text.strip("._-") or "dataset"


def parse_dataset_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit("--dataset must use name=path format")
    name, path_text = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit("--dataset name must be non-empty")
    path = resolve_path(Path(path_text.strip()))
    return name, path


def selected_datasets(args: argparse.Namespace) -> list[tuple[str, Path]]:
    datasets = default_datasets()
    order = [name for name, _ in datasets]
    by_name = {name: path for name, path in datasets}
    for raw in args.dataset or []:
        name, path = parse_dataset_spec(raw)
        if name not in by_name:
            order.append(name)
        by_name[name] = path
    result = [(name, by_name[name]) for name in order]
    missing = [(name, path) for name, path in result if not path.exists()]
    if missing:
        details = ", ".join(f"{name}={path}" for name, path in missing)
        raise SystemExit(f"dataset CSV does not exist: {details}")
    return result


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


def cli_argv(args: argparse.Namespace, csv_path: Path, run_dir: Path) -> list[str]:
    argv = [
        "--csv",
        str(csv_path),
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


def plan_from_response(response: dict[str, Any]) -> dict[str, Any]:
    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    return nested_map(agent.get("plan"))


def safety_from_response(response: dict[str, Any]) -> dict[str, Any]:
    result = nested_map(response.get("result"))
    return nested_map(result.get("safety"))


def post_validation_from_response(response: dict[str, Any]) -> dict[str, Any]:
    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    validation = nested_map(agent.get("validation"))
    return nested_map(validation.get("post_execute"))


def fallback_reason_code(plan: dict[str, Any]) -> str:
    cognition = nested_map(plan.get("cognition"))
    for key in ("fallback_reason_code", "reason_code"):
        code = str(cognition.get(key) or "").strip()
        if code:
            return code
    for source in (cognition.get("reason_codes"), plan.get("reason_codes")):
        for item in list_from_any(source):
            code = str(item).strip()
            if code:
                return code
    return ""


def count_list(plan: dict[str, Any], key: str) -> int:
    return len(list_from_any(plan.get(key)))


def dict_counts_from_any(value: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(value, dict):
        return result
    for key, raw_count in value.items():
        text = str(key).strip()
        if not text:
            continue
        result[text] = result.get(text, 0) + as_int(raw_count)
    return result


def increment_count(counts: dict[str, int], key: str, amount: int = 1) -> None:
    text = str(key or "").strip()
    if not text:
        return
    counts[text] = counts.get(text, 0) + amount


def skipped_issue_rollups(plan: dict[str, Any]) -> dict[str, dict[str, int]]:
    rollups = {
        "manual_review_reason_counts": {},
        "manual_review_issue_type_counts": {},
        "blocked_reason_counts": {},
        "blocked_issue_type_counts": {},
        "cautious_reason_counts": {},
        "cautious_issue_type_counts": {},
    }
    for item in list_from_any(plan.get("skipped_issues")):
        if not isinstance(item, dict):
            continue
        details = nested_map(item.get("details"))
        bucket = str(details.get("bucket") or "").strip()
        reason = str(item.get("reason") or details.get("reason") or "").strip()
        issue_type = str(item.get("issue_type") or details.get("issue_type") or "").strip()
        if bucket == "manual_review" or reason == "manual_review_required":
            increment_count(rollups["manual_review_reason_counts"], reason or "manual_review_required")
            increment_count(rollups["manual_review_issue_type_counts"], issue_type or "unknown")
        elif bucket == "blocked" or reason in {"unsupported_issue_type", "missing_issue_id", "blocked_issue_type"}:
            increment_count(rollups["blocked_reason_counts"], reason or "unsupported_issue_type")
            increment_count(rollups["blocked_issue_type_counts"], issue_type or "unknown")
        elif bucket == "cautious" or reason in {"requires_human_review_before_auto_repair", "cautious_issue_type"}:
            increment_count(rollups["cautious_reason_counts"], reason or "requires_human_review_before_auto_repair")
            increment_count(rollups["cautious_issue_type_counts"], issue_type or "unknown")
    return rollups


def merge_counts(*sources: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for source in sources:
        for key, count in source.items():
            increment_count(merged, key, count)
    return merged


def summarize_run(dataset_name: str, run_index: int, run_dir: Path, returncode: int, wall_ms: int) -> dict[str, Any]:
    response = nested_map(read_json(run_dir / "response.json"))
    plan = nested_map(read_json(run_dir / "repair_plan.json")) or plan_from_response(response)
    validation = nested_map(read_json(run_dir / "validation_result.json")) or post_validation_from_response(response)
    explanations = nested_map(read_json(run_dir / "issue_explanations.json"))
    timings = nested_map(read_json(run_dir / "timings.json"))
    trace = read_json(run_dir / "auto_agent_trace.json")
    error = read_json(run_dir / "error.json")

    cognition = nested_map(plan.get("cognition"))
    cognition_status = str(cognition.get("status") or "")
    fallback_reason = fallback_reason_code(plan)
    fallback = cognition_status in FALLBACK_STATUSES or bool(fallback_reason)
    final_verdict = str(safety_from_response(response).get("final_verdict") or "")
    validation_verdict = str(validation.get("verdict") or "")
    trace_count = len(trace) if isinstance(trace, list) else 0
    total_ms = as_number(timings.get("total_duration_ms"))
    if total_ms is None:
        total_ms = as_number(response.get("duration_ms"))
    if total_ms is None:
        total_ms = float(wall_ms)

    skipped_rollups = skipped_issue_rollups(plan)
    explanation_blocked_counts = dict_counts_from_any(explanations.get("blocked_reason_counts"))
    blocked_counts = explanation_blocked_counts or skipped_rollups["blocked_reason_counts"]
    cautious_counts: dict[str, int] = {}
    for item in list_from_any(explanations.get("cautious_issue_details")):
        if isinstance(item, dict):
            increment_count(cautious_counts, str(item.get("risk_reason") or "requires_human_review_before_auto_repair"))
    if not cautious_counts:
        cautious_counts = dict(skipped_rollups["cautious_reason_counts"])

    result = {
        "dataset_name": dataset_name,
        "run_index": run_index,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "status": "success" if returncode == 0 and response else "failed",
        "duration_ms": total_ms,
        "wall_duration_ms": wall_ms,
        "final_verdict": final_verdict,
        "validation_verdict": validation_verdict,
        "accepted": final_verdict == "accepted" and validation_verdict in {"accept", "warn"},
        "fallback": fallback,
        "fallback_reason_code": fallback_reason,
        "cognition_provider": str(cognition.get("provider") or ""),
        "cognition_status": cognition_status,
        "planner_mode": str(cognition.get("planner_mode") or ""),
        "before_issue_items": as_int(validation.get("before_issue_items", validation.get("before_issue_count"))),
        "after_issue_items": as_int(validation.get("after_issue_items", validation.get("after_issue_count"))),
        "resolved_issue_items": as_int(validation.get("resolved_issue_items", validation.get("resolved_issue_count"))),
        "modified_cell_count": as_int(validation.get("modified_cell_count", validation.get("total_cells_modified"))),
        "blocked_issue_count": count_list(plan, "blocked_issue_ids"),
        "cautious_issue_count": count_list(plan, "cautious_issue_ids"),
        "manual_review_issue_count": count_list(plan, "manual_review_issue_ids"),
        "blocked_reason_counts": blocked_counts,
        "blocked_issue_type_counts": skipped_rollups["blocked_issue_type_counts"],
        "cautious_reason_counts": cautious_counts,
        "cautious_issue_type_counts": skipped_rollups["cautious_issue_type_counts"],
        "manual_review_reason_counts": skipped_rollups["manual_review_reason_counts"],
        "manual_review_issue_type_counts": skipped_rollups["manual_review_issue_type_counts"],
        "trace_event_count": trace_count,
        "rollback_manifest_created": bool(rollback_manifest_path(response)),
        "rollback_manifest_path": rollback_manifest_path(response),
        "error": error,
    }
    return result


def aggregate_counts(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        for reason, count in nested_map(item.get(key)).items():
            increment_count(counts, str(reason), as_int(count))
    return counts


def number_values(results: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in results:
        value = as_number(item.get(key))
        if value is not None:
            values.append(float(value))
    return values


def dataset_summary(dataset_name: str, csv_path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    row_count, column_count = csv_shape(csv_path)
    total = len(results)
    success_runs = sum(1 for item in results if item.get("status") == "success")
    accepted_runs = sum(1 for item in results if item.get("accepted") is True)
    fallback_runs = sum(1 for item in results if item.get("fallback") is True)
    manifest_runs = sum(1 for item in results if item.get("rollback_manifest_created") is True)
    summary = {
        "dataset_name": dataset_name,
        "csv_path": str(csv_path),
        "row_count": row_count,
        "column_count": column_count,
        "total_runs": total,
        "success_runs": success_runs,
        "success_rate": rate(success_runs, total),
        "accepted_runs": accepted_runs,
        "accepted_rate": rate(accepted_runs, total),
        "fallback_runs": fallback_runs,
        "fallback_rate": rate(fallback_runs, total),
        "before_issue_items_avg": mean(number_values(results, "before_issue_items")),
        "after_issue_items_avg": mean(number_values(results, "after_issue_items")),
        "resolved_issue_items_avg": mean(number_values(results, "resolved_issue_items")),
        "modified_cell_count_avg": mean(number_values(results, "modified_cell_count")),
        "blocked_issue_count_avg": mean(number_values(results, "blocked_issue_count")),
        "cautious_issue_count_avg": mean(number_values(results, "cautious_issue_count")),
        "manual_review_issue_count_avg": mean(number_values(results, "manual_review_issue_count")),
        "rollback_manifest_created_rate": rate(manifest_runs, total),
        "avg_total_ms": mean(number_values(results, "duration_ms")),
        "p95_total_ms": percentile95(number_values(results, "duration_ms")),
        "avg_trace_event_count": mean(number_values(results, "trace_event_count")),
        "fallback_reason_counts": aggregate_counts(results, "fallback_reason_counts"),
        "blocked_reason_counts": aggregate_counts(results, "blocked_reason_counts"),
        "blocked_issue_type_counts": aggregate_counts(results, "blocked_issue_type_counts"),
        "cautious_reason_counts": aggregate_counts(results, "cautious_reason_counts"),
        "cautious_issue_type_counts": aggregate_counts(results, "cautious_issue_type_counts"),
        "manual_review_reason_counts": aggregate_counts(results, "manual_review_reason_counts"),
        "manual_review_issue_type_counts": aggregate_counts(results, "manual_review_issue_type_counts"),
    }
    missing_fields = [field for field in REQUIRED_SUMMARY_FIELDS if field not in summary]
    if missing_fields:
        raise RuntimeError(f"dataset summary missing required fields: {missing_fields}")
    return summary


def add_fallback_reason_counts(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for item in results:
        if not item.get("fallback"):
            continue
        reason = str(item.get("fallback_reason_code") or item.get("cognition_status") or "fallback")
        increment_count(counts, reason)
    summary["fallback_reason_counts"] = counts


def build_remaining_reason_text(summary: dict[str, Any]) -> str:
    cautious_counts = nested_map(summary.get("cautious_reason_counts"))
    blocked_counts = nested_map(summary.get("blocked_reason_counts"))
    blocked_types = nested_map(summary.get("blocked_issue_type_counts"))
    manual_types = nested_map(summary.get("manual_review_issue_type_counts"))
    parts = [
        "`numeric_outlier` is placed in the cautious bucket because the current policy requires human confirmation before automatic outlier repair.",
        "`duplicate_record` and `cross_column_consistency` are manual-review issues, so their persistence is expected and is not counted as an Auto Agent run failure.",
    ]
    if blocked_counts or blocked_types:
        parts.append(
            "Blocked leftovers mainly reflect unsupported automatic repair types, for example `time_series_shift`; "
            f"blocked reasons={json.dumps(blocked_counts, ensure_ascii=False)}, issue_types={json.dumps(blocked_types, ensure_ascii=False)}."
        )
    else:
        parts.append("No blocked leftovers were reported for this dataset.")
    if cautious_counts:
        parts.append(f"Cautious reasons={json.dumps(cautious_counts, ensure_ascii=False)}.")
    if manual_types:
        parts.append(f"Manual-review issue types={json.dumps(manual_types, ensure_ascii=False)}.")
    return " ".join(parts)


def build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Auto Agent Multi-Dataset Benchmark",
        "",
        "This benchmark runs the existing Auto Agent CLI against multiple mixed-type datasets.",
        "Outputs are written under the requested output directory; large run artifacts should remain ignored under `outputs/`.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | rows | cols | runs | success | accepted | fallback | before_avg | after_avg | resolved_avg | modified_avg | blocked_avg | cautious_avg | rollback_manifest | avg_ms | p95_ms | avg_trace |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in list_from_any(summary.get("datasets")):
        lines.append(
            f"| `{item['dataset_name']}` | {item['row_count']} | {item['column_count']} | {item['total_runs']} | "
            f"{item['success_rate']:.2%} | {item['accepted_rate']:.2%} | {item['fallback_rate']:.2%} | "
            f"{item['before_issue_items_avg']} | {item['after_issue_items_avg']} | {item['resolved_issue_items_avg']} | "
            f"{item['modified_cell_count_avg']} | {item['blocked_issue_count_avg']} | {item['cautious_issue_count_avg']} | "
            f"{item['rollback_manifest_created_rate']:.2%} | {item['avg_total_ms']} | {item['p95_total_ms']} | {item['avg_trace_event_count']} |"
        )

    lines.extend(["", "## Remaining Issue Reasons", ""])
    for item in list_from_any(summary.get("datasets")):
        lines.extend(
            [
                f"### {item['dataset_name']}",
                "",
                build_remaining_reason_text(item),
                "",
                f"- blocked_reason_counts: `{json.dumps(item.get('blocked_reason_counts', {}), ensure_ascii=False)}`",
                f"- cautious_reason_counts: `{json.dumps(item.get('cautious_reason_counts', {}), ensure_ascii=False)}`",
                f"- manual_review_issue_type_counts: `{json.dumps(item.get('manual_review_issue_type_counts', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- `before_issue_items`, `after_issue_items`, and `resolved_issue_items` are issue-item counts, not modified-cell counts.",
            "- `modified_cell_count` is reported separately to avoid mixing scan findings with repair writes.",
            "- Environment reporting records variable names, model names, and whether an API key was configured; it does not record API key values.",
            "",
        ]
    )
    return "\n".join(lines)


def run_dataset(args: argparse.Namespace, dataset_name: str, csv_path: Path, dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    exit_code = 0
    for index in range(1, args.runs_per_dataset + 1):
        run_dir = dataset_dir / f"run_{index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        returncode = auto_agent_cli.main(cli_argv(args, csv_path, run_dir))
        wall_ms = int((time.perf_counter() - started) * 1000)
        run_result = summarize_run(dataset_name, index, run_dir, returncode, wall_ms)
        write_json(run_dir / "run_result.json", run_result)
        results.append(run_result)
        if returncode != 0:
            exit_code = returncode or 1
            if not args.continue_on_error:
                break

    summary = dataset_summary(dataset_name, csv_path, results)
    add_fallback_reason_counts(summary, results)
    write_json(dataset_dir / "runs.json", results)
    write_json(dataset_dir / "dataset_summary.json", summary)
    return summary, results, exit_code


def run_benchmark(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
    if args.backend_dir is not None:
        args.backend_dir = args.backend_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = selected_datasets(args)
    dataset_summaries: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    exit_code = 0
    for dataset_name, csv_path in datasets:
        summary, results, dataset_exit = run_dataset(args, dataset_name, csv_path, args.output_dir / safe_dir_name(dataset_name))
        dataset_summaries.append(summary)
        all_results.extend(results)
        if dataset_exit != 0:
            exit_code = dataset_exit
            if not args.continue_on_error:
                break

    summary = {
        "benchmark": "auto_agent_multi_dataset",
        "output_dir": str(args.output_dir),
        "runs_per_dataset": args.runs_per_dataset,
        "total_datasets": len(dataset_summaries),
        "total_runs": sum(int(item["total_runs"]) for item in dataset_summaries),
        "datasets": dataset_summaries,
        "environment": env_summary(),
        "required_dataset_summary_fields": REQUIRED_SUMMARY_FIELDS,
    }
    write_json(args.output_dir / "all_run_results.json", all_results)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(build_summary_md(summary), encoding="utf-8")
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto Agent multi-dataset benchmark")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs-per-dataset", type=int, default=5)
    parser.add_argument("--dataset", action="append", default=[], help="Dataset override or append in name=path format")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--goal", default="")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--go-bin", default="go")
    parser.add_argument("--backend-dir", type=Path, default=repo_root() / "appshell" / "backend")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs_per_dataset <= 0:
        raise SystemExit("--runs-per-dataset must be positive")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
