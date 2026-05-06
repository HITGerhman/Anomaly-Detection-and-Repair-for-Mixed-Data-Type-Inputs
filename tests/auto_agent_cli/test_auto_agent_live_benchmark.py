from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "appshell" / "core" / "python_engine" / "auto_agent_live_benchmark.py"


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_agent_live_benchmark", BENCH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fake_timings(index: int) -> dict[str, Any]:
    return {
        "llm_plan_duration_ms": 600 + index,
        "llm_explain_duration_ms": 200 + index,
        "scan_duration_ms": 10 + index,
        "retrieve_duration_ms": 90 + index,
        "repair_duration_ms": 30 + index,
        "validation_duration_ms": 40 + index,
        "rollback_manifest_duration_ms": 1,
        "total_duration_ms": 1000 + index,
    }


def fake_response(index: int, *, fallback: bool = False, validation_verdict: str = "accept", final_verdict: str = "accepted") -> dict[str, Any]:
    cognition = {
        "provider": "langgraph" if not fallback else "deterministic",
        "status": "engaged" if not fallback else "fallback",
        "fallback_reason_code": "" if not fallback else "plan_timeout",
    }
    return {
        "task_id": f"task-{index}",
        "status": "ok",
        "duration_ms": 1000 + index,
        "result": {
            "agent": {
                "session_id": f"session-{index}",
                "plan": {"cognition": cognition},
                "validation": {"post_execute": {"verdict": validation_verdict}},
                "execution": {
                    "rollback": {"manifest_path": f"run_{index}/.rollback/repair.json"},
                },
            },
            "safety": {"final_verdict": final_verdict},
        },
    }


def test_live_benchmark_aggregates_success_fallback_and_reject(tmp_path: Path, monkeypatch) -> None:
    bench = load_benchmark()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age\n1\n", encoding="utf-8")
    output_dir = tmp_path / "bench"
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        run_dir = Path(argv[argv.index("--output-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        index = len(calls)
        if index == 1:
            response = fake_response(index)
        elif index == 2:
            response = fake_response(index, fallback=True)
        else:
            response = fake_response(index, validation_verdict="reject", final_verdict="rolled_back")
        write_json(run_dir / "response.json", response)
        write_json(run_dir / "timings.json", fake_timings(index))
        write_json(run_dir / "auto_agent_trace.json", [{"seq": 1}, {"seq": 2}])
        return 0

    monkeypatch.setattr(bench.auto_agent_cli, "main", fake_main)
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_API_KEY", "secret-for-test")

    rc = bench.main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--runs",
            "3",
            "--continue-on-error",
        ]
    )

    assert rc == 0
    assert len(calls) == 3
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_runs"] == 3
    assert summary["success_runs"] == 3
    assert summary["success_rate"] == 1
    assert summary["accepted_runs"] == 2
    assert summary["fallback_runs"] == 1
    assert summary["validation_reject_runs"] == 1
    assert summary["rollback_manifest_created_rate"] == 1
    assert summary["avg_trace_event_count"] == 2
    assert summary["environment"]["llm_model"] == "deepseek-chat"
    assert summary["environment"]["llm_api_key_configured"] is True
    assert "secret-for-test" not in (output_dir / "summary.json").read_text(encoding="utf-8")
    assert (output_dir / "run_001" / "run_result.json").exists()
    assert (output_dir / "summary.md").exists()
    timings_summary = json.loads((output_dir / "timings_summary.json").read_text(encoding="utf-8"))
    assert timings_summary["stages"]["llm_plan_duration_ms"]["avg_ms"] == 602
    assert timings_summary["stages"]["llm_plan_duration_ms"]["p50_ms"] == 602
    assert timings_summary["stages"]["llm_plan_duration_ms"]["p95_ms"] == 603
    assert timings_summary["stages"]["llm_plan_duration_ms"]["max_ms"] == 603
    assert timings_summary["stages"]["llm_plan_duration_ms"]["share_of_total_avg"] == 602 / 1002
    assert timings_summary["top_slowest_stages"][0]["stage"] == "llm_plan_duration_ms"
    assert "timings_summary_path" in summary
    assert "llm_plan_duration_ms" in (output_dir / "timings_summary.md").read_text(encoding="utf-8")


def test_live_benchmark_stops_on_error_without_continue(tmp_path: Path, monkeypatch) -> None:
    bench = load_benchmark()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age\n1\n", encoding="utf-8")
    output_dir = tmp_path / "bench"
    calls = 0

    def fake_main(argv: list[str]) -> int:
        nonlocal calls
        calls += 1
        run_dir = Path(argv[argv.index("--output-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "error.json", {"message": "failed"})
        return 5

    monkeypatch.setattr(bench.auto_agent_cli, "main", fake_main)

    rc = bench.main(["--csv", str(csv_path), "--output-dir", str(output_dir), "--runs", "3"])

    assert rc == 5
    assert calls == 1
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_runs"] == 1
    assert summary["success_runs"] == 0
    assert (output_dir / "run_001" / "run_result.json").exists()


def test_live_benchmark_timings_summary_marks_missing_fields(tmp_path: Path, monkeypatch) -> None:
    bench = load_benchmark()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age\n1\n", encoding="utf-8")
    output_dir = tmp_path / "bench"
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        run_dir = Path(argv[argv.index("--output-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        index = len(calls)
        timings = {
            "llm_plan_duration_ms": [600, 700, 650][index - 1],
            "scan_duration_ms": [10, 20, 30][index - 1],
            "retrieve_duration_ms": [100, 110, 90][index - 1],
            "repair_duration_ms": [30, 40, 50][index - 1],
            "validation_duration_ms": [40, 50, 60][index - 1],
            "rollback_manifest_duration_ms": [1, 2, 3][index - 1],
            "total_duration_ms": [1000, 1200, 1100][index - 1],
        }
        if index != 2:
            timings["llm_explain_duration_ms"] = [200, 0, 250][index - 1]
        write_json(run_dir / "response.json", fake_response(index))
        write_json(run_dir / "timings.json", timings)
        write_json(run_dir / "auto_agent_trace.json", [{"seq": 1}])
        return 0

    monkeypatch.setattr(bench.auto_agent_cli, "main", fake_main)

    rc = bench.main(["--csv", str(csv_path), "--output-dir", str(output_dir), "--runs", "3"])

    assert rc == 0
    timings_summary = json.loads((output_dir / "timings_summary.json").read_text(encoding="utf-8"))
    plan_stage = timings_summary["stages"]["llm_plan_duration_ms"]
    assert plan_stage["avg_ms"] == 650
    assert plan_stage["p50_ms"] == 650
    assert plan_stage["p95_ms"] == 700
    assert plan_stage["max_ms"] == 700
    assert plan_stage["share_of_total_avg"] == 650 / 1100
    explain_stage = timings_summary["stages"]["llm_explain_duration_ms"]
    assert explain_stage["present_runs"] == 2
    assert explain_stage["missing_runs"] == 1
    assert explain_stage["missing"] is True
    top = [item["stage"] for item in timings_summary["top_slowest_stages"]]
    assert top == ["llm_plan_duration_ms", "llm_explain_duration_ms", "retrieve_duration_ms"]
    assert "total_duration_ms" not in top
    assert timings_summary["dominant_area"] == "LLM"
    report = (output_dir / "timings_summary.md").read_text(encoding="utf-8")
    assert "missing" in report
    assert "top 1" in report
