"""Benchmark Auto Agent fallback behavior when LLM/API access fails.

The script reuses auto_agent_cli.py for every run. It only patches environment
variables, hosts local mock OpenAI-compatible endpoints, and aggregates the
resulting artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


DUMMY_API_KEY = "-".join(["fallback", "benchmark", "dummy", "key"])
FALLBACK_STATUSES = {"fallback", "disabled", "unavailable", "degraded"}
FALLBACK_PREFIXES = ("llm_", "plan_")
FALLBACK_CODES = {
    "disabled",
    "planner_mode_fallback",
    "healthcheck_failed",
    "script_missing",
    "startup_failed",
    "port_occupied",
    "invalid_candidate",
}


@dataclass(frozen=True)
class FallbackScenario:
    name: str
    expected_reason: str
    mock_mode: str = ""
    enabled: bool = True
    wrong_base_url: bool = False
    model: str = "fallback-benchmark-model"
    llm_timeout_ms: int = 1000
    request_timeout_ms: int = 15000


SCENARIOS: list[FallbackScenario] = [
    FallbackScenario(name="langgraph_disabled", expected_reason="disabled", enabled=False),
    FallbackScenario(name="api_base_url_wrong", expected_reason="llm_unavailable", wrong_base_url=True),
    FallbackScenario(name="api_timeout", expected_reason="llm_timeout", mock_mode="timeout", llm_timeout_ms=500, request_timeout_ms=8000),
    FallbackScenario(name="invalid_json_response", expected_reason="llm_invalid_json", mock_mode="invalid_json"),
    FallbackScenario(name="empty_response", expected_reason="llm_empty_response", mock_mode="empty_response"),
    FallbackScenario(name="wrong_model_or_mock_404", expected_reason="llm_non_200", mock_mode="http_404", model="missing-model"),
]


def _load_auto_agent_cli() -> ModuleType:
    cli_path = Path(__file__).resolve().with_name("auto_agent_cli.py")
    spec = importlib.util.spec_from_file_location("_auto_agent_cli_for_fallback_benchmark", cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load auto_agent_cli.py from {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auto_agent_cli = _load_auto_agent_cli()


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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MockLLMServer:
    def __init__(self, mode: str):
        self.mode = mode
        self.port = free_port()
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "MockLLMServer":
        mode = self.mode

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                _ = self.rfile.read(content_length)
                if mode == "timeout":
                    time.sleep(2.0)
                    body = _json_bytes({"choices": [{"message": {"content": "{}"}}]})
                    self._write(200, body)
                    return
                if mode == "invalid_json":
                    self._write(200, b"not-json")
                    return
                if mode == "empty_response":
                    self._write(200, _json_bytes({"choices": [{"message": {"content": ""}}]}))
                    return
                if mode == "http_404":
                    self._write(404, _json_bytes({"error": "model_not_found"}))
                    return
                self._write(500, _json_bytes({"error": "unexpected_mock_mode"}))

            def _write(self, status_code: int, body: bytes) -> None:
                try:
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True).encode("utf-8")


@contextlib.contextmanager
def patched_environ(updates: dict[str, str]) -> Iterator[None]:
    original: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def scenario_env(scenario: FallbackScenario, run_index: int, llm_base_url: str) -> dict[str, str]:
    sidecar_port = free_port()
    env = {
        "APPSHELL_LANGGRAPH_ENABLED": "1" if scenario.enabled else "0",
        "APPSHELL_LANGGRAPH_HOST": "127.0.0.1",
        "APPSHELL_LANGGRAPH_PORT": str(sidecar_port),
        "APPSHELL_LANGGRAPH_LLM_BASE_URL": llm_base_url,
        "APPSHELL_LANGGRAPH_LLM_API_KEY": DUMMY_API_KEY,
        "APPSHELL_LANGGRAPH_LLM_MODEL": scenario.model,
        "APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS": str(scenario.llm_timeout_ms),
        "APPSHELL_LANGGRAPH_REQUEST_TIMEOUT_MS": str(scenario.request_timeout_ms),
        "APPSHELL_LANGGRAPH_STARTUP_TIMEOUT_MS": "10000",
        "AUTO_AGENT_FALLBACK_BENCHMARK_RUN": str(run_index),
    }
    if not scenario.enabled:
        env["APPSHELL_LANGGRAPH_LLM_BASE_URL"] = ""
        env["APPSHELL_LANGGRAPH_LLM_API_KEY"] = ""
        env["APPSHELL_LANGGRAPH_LLM_MODEL"] = ""
    return env


def fallback_reason_codes(plan: dict[str, Any], run_result: dict[str, Any]) -> list[str]:
    cognition = nested_map(plan.get("cognition"))
    candidates: list[str] = []
    candidates.append(str(run_result.get("fallback_reason_code") or ""))
    candidates.append(str(cognition.get("fallback_reason_code") or ""))
    for source in (cognition.get("reason_codes"), plan.get("reason_codes")):
        for item in list_from_any(source):
            candidates.append(str(item))

    reasons: list[str] = []
    for candidate in candidates:
        code = candidate.strip()
        if not code:
            continue
        if code in FALLBACK_CODES or code.startswith(FALLBACK_PREFIXES):
            if code not in reasons:
                reasons.append(code)
    return reasons


def rollback_manifest_path(response: dict[str, Any]) -> str:
    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    execution = nested_map(agent.get("execution"))
    rollback = nested_map(execution.get("rollback"))
    return str(rollback.get("manifest_path") or execution.get("rollback_manifest_path") or "")


def output_csv_path(response: dict[str, Any]) -> str:
    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    execution = nested_map(agent.get("execution"))
    return str(execution.get("output_csv") or "")


def summarize_run(scenario: FallbackScenario, run_index: int, run_dir: Path, returncode: int, wall_ms: int) -> dict[str, Any]:
    response = nested_map(read_json(run_dir / "response.json"))
    timings = nested_map(read_json(run_dir / "timings.json"))
    trace = read_json(run_dir / "auto_agent_trace.json")
    error = read_json(run_dir / "error.json")

    result = nested_map(response.get("result"))
    agent = nested_map(result.get("agent"))
    plan = nested_map(agent.get("plan"))
    validation = nested_map(agent.get("validation"))
    post_validation = nested_map(validation.get("post_execute"))
    safety = nested_map(result.get("safety"))
    cognition = nested_map(plan.get("cognition"))
    trace_count = len(trace) if isinstance(trace, list) else 0
    manifest = rollback_manifest_path(response)
    output_csv = output_csv_path(response)
    total_ms = as_number(timings.get("total_duration_ms"))
    if total_ms is None:
        total_ms = as_number(response.get("duration_ms"))
    if total_ms is None:
        total_ms = float(wall_ms)

    base_run = {
        "fallback_reason_code": str(cognition.get("fallback_reason_code") or ""),
    }
    reason_codes = fallback_reason_codes(plan, base_run)
    cognition_status = str(cognition.get("status") or "")
    fallback = cognition_status in FALLBACK_STATUSES or bool(reason_codes)
    validation_verdict = str(post_validation.get("verdict") or "")
    structured_success = returncode == 0 and bool(response)
    manifest_required = bool(output_csv)
    manifest_ok = bool(manifest) if manifest_required else True
    fallback_success = bool(fallback and structured_success and validation_verdict and trace_count > 0 and manifest_ok)

    return {
        "scenario": scenario.name,
        "expected_reason": scenario.expected_reason,
        "run_index": run_index,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "status": "success" if structured_success else "failed",
        "duration_ms": total_ms,
        "wall_duration_ms": wall_ms,
        "final_verdict": str(safety.get("final_verdict") or ""),
        "validation_verdict": validation_verdict,
        "cognition_provider": str(cognition.get("provider") or ""),
        "cognition_status": cognition_status,
        "planner_mode": str(cognition.get("planner_mode") or ""),
        "llm_mode": str(cognition.get("llm_mode") or ""),
        "fallback": fallback,
        "fallback_reason_code": reason_codes[0] if reason_codes else "",
        "fallback_reason_codes": reason_codes,
        "fallback_success": fallback_success,
        "accepted": str(safety.get("final_verdict") or "") == "accepted" and validation_verdict in {"accept", "warn"},
        "validation_rejected": validation_verdict in {"reject", "rollback_recommended"},
        "trace_event_count": trace_count,
        "rollback_manifest_created": bool(manifest),
        "rollback_manifest_path": manifest,
        "output_csv": output_csv,
        "error": error,
    }


def aggregate_scenario(scenario: FallbackScenario, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success_runs = sum(1 for item in results if item.get("status") == "success")
    fallback_runs = sum(1 for item in results if item.get("fallback") is True)
    fallback_success_runs = sum(1 for item in results if item.get("fallback_success") is True)
    accepted_runs = sum(1 for item in results if item.get("accepted") is True)
    validation_reject_runs = sum(1 for item in results if item.get("validation_rejected") is True)
    manifest_runs = sum(1 for item in results if item.get("rollback_manifest_created") is True)
    total_ms = [float(item["duration_ms"]) for item in results if as_number(item.get("duration_ms")) is not None]
    trace_counts = [float(item.get("trace_event_count", 0)) for item in results]

    rate = lambda count: (count / total) if total else 0.0
    return {
        "scenario": scenario.name,
        "expected_reason": scenario.expected_reason,
        "total_runs": total,
        "success_runs": success_runs,
        "success_rate": rate(success_runs),
        "fallback_runs": fallback_runs,
        "fallback_rate": rate(fallback_runs),
        "fallback_success_rate": rate(fallback_success_runs),
        "accepted_runs": accepted_runs,
        "accepted_rate": rate(accepted_runs),
        "validation_reject_runs": validation_reject_runs,
        "rollback_manifest_created_rate": rate(manifest_runs),
        "avg_trace_event_count": mean(trace_counts),
        "fallback_reason_counts": fallback_reason_counts(results),
        "avg_total_ms": mean(total_ms),
        "p95_total_ms": percentile95(total_ms),
        "degraded_successfully": fallback_success_runs == total and total > 0,
    }


def fallback_reason_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        for reason in list_from_any(item.get("fallback_reason_codes")):
            code = str(reason).strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    return counts


def build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Auto Agent LLM Fallback Benchmark",
        "",
        "This benchmark validates that Auto Agent falls back to deterministic planning when LLM/API access fails.",
        "Each run still uses the full CLI flow: plan, execute, rescan, validation gate, trace export, and rollback manifest checks.",
        "",
        "## Scenario Results",
        "",
        "| scenario | degraded | success_rate | fallback_rate | fallback_success_rate | accepted_rate | rollback_manifest_rate | avg_trace | p95_total_ms | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["scenarios"]:
        lines.append(
            f"| `{item['scenario']}` | {item['degraded_successfully']} | {item['success_rate']:.2%} | "
            f"{item['fallback_rate']:.2%} | {item['fallback_success_rate']:.2%} | {item['accepted_rate']:.2%} | "
            f"{item['rollback_manifest_created_rate']:.2%} | {item['avg_trace_event_count']} | "
            f"{item['p95_total_ms']} | `{json.dumps(item['fallback_reason_counts'], ensure_ascii=False)}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `fallback_success_rate` requires fallback detection, a structured response, a validation verdict, trace events, and rollback manifest presence whenever an output CSV is written.",
            "- Environment reporting records variable names, model names, and whether an API key was configured; it does not record API key values.",
            "- All LLM/API failures are produced by local mock endpoints or disabled configuration; no external LLM API is called.",
            "",
        ]
    )
    return "\n".join(lines)


def run_scenario(args: argparse.Namespace, scenario: FallbackScenario, scenario_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    mock_context: contextlib.AbstractContextManager[Any]
    llm_base_url = ""
    if scenario.mock_mode:
        mock_context = MockLLMServer(scenario.mock_mode)
    else:
        mock_context = contextlib.nullcontext()

    with mock_context as mock_server:
        if isinstance(mock_server, MockLLMServer):
            llm_base_url = mock_server.base_url
        elif scenario.wrong_base_url:
            llm_base_url = f"http://127.0.0.1:{free_port()}"

        results: list[dict[str, Any]] = []
        for run_index in range(1, args.runs_per_scenario + 1):
            run_dir = scenario_dir / f"run_{run_index:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            env_updates = scenario_env(scenario, run_index, llm_base_url)
            started = time.perf_counter()
            with patched_environ(env_updates):
                returncode = auto_agent_cli.main(cli_argv(args, run_dir))
            wall_ms = int((time.perf_counter() - started) * 1000)
            run_result = summarize_run(scenario, run_index, run_dir, returncode, wall_ms)
            write_json(run_dir / "run_result.json", run_result)
            results.append(run_result)
            if returncode != 0 and not args.continue_on_error:
                break

    scenario_summary = aggregate_scenario(scenario, results)
    write_json(scenario_dir / "scenario_summary.json", scenario_summary)
    return scenario_summary, results


def run_benchmark(args: argparse.Namespace) -> int:
    args.csv = args.csv.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
    if args.backend_dir is not None:
        args.backend_dir = args.backend_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario_summaries: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    exit_code = 0
    for scenario in SCENARIOS:
        summary, results = run_scenario(args, scenario, args.output_dir / scenario.name)
        scenario_summaries.append(summary)
        all_results.extend(results)
        if summary["success_runs"] < summary["total_runs"]:
            exit_code = 1
            if not args.continue_on_error:
                break

    summary = {
        "benchmark": "auto_agent_llm_fallback",
        "input_csv": str(args.csv),
        "output_dir": str(args.output_dir),
        "runs_per_scenario": args.runs_per_scenario,
        "scenarios": scenario_summaries,
        "total_runs": sum(item["total_runs"] for item in scenario_summaries),
        "all_scenarios_degraded_successfully": all(item["degraded_successfully"] for item in scenario_summaries),
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(build_summary_md(summary), encoding="utf-8")
    write_json(args.output_dir / "all_run_results.json", all_results)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = auto_agent_cli.repo_root_from_here()
    parser = argparse.ArgumentParser(description="Auto Agent LLM fallback benchmark")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs-per-scenario", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--goal", default="")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--go-bin", default="go")
    parser.add_argument("--backend-dir", type=Path, default=repo_root / "appshell" / "backend")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs_per_scenario <= 0:
        raise SystemExit("--runs-per-scenario must be positive")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
