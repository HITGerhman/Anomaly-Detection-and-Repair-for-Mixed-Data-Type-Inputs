from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "appshell" / "core" / "python_engine" / "auto_agent_fallback_benchmark.py"


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_agent_fallback_benchmark", BENCH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


SCENARIO_REASON_PLACEMENTS = {
    "langgraph_disabled": ("fallback_reason_code", "disabled"),
    "api_base_url_wrong": ("cognition_reason_codes", "llm_unavailable"),
    "api_timeout": ("plan_reason_codes", "llm_timeout"),
    "invalid_json_response": ("cognition_reason_codes", "llm_invalid_json"),
    "empty_response": ("plan_reason_codes", "llm_empty_response"),
    "wrong_model_or_mock_404": ("cognition_reason_codes", "llm_non_200"),
}


def fake_response(scenario: str, run_index: int) -> dict[str, Any]:
    placement, reason = SCENARIO_REASON_PLACEMENTS[scenario]
    cognition: dict[str, Any] = {
        "provider": "deterministic",
        "status": "fallback",
        "fallback_reason_code": reason if placement == "fallback_reason_code" else "",
        "reason_codes": [reason] if placement == "cognition_reason_codes" else [],
    }
    plan: dict[str, Any] = {
        "cognition": cognition,
        "reason_codes": [reason] if placement == "plan_reason_codes" else [],
    }
    return {
        "task_id": f"task-{scenario}-{run_index}",
        "status": "ok",
        "duration_ms": 1000 + run_index,
        "result": {
            "agent": {
                "session_id": f"session-{scenario}-{run_index}",
                "plan": plan,
                "validation": {"post_execute": {"verdict": "accept"}},
                "execution": {
                    "output_csv": f"outputs/{scenario}/repaired.csv",
                    "rollback": {"manifest_path": f"outputs/{scenario}/.rollback/repair.json"},
                },
            },
            "safety": {"final_verdict": "accepted"},
        },
    }


def test_fallback_benchmark_runs_all_scenarios_and_redacts_dummy_key(tmp_path: Path, monkeypatch) -> None:
    bench = load_benchmark()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age\n1\n", encoding="utf-8")
    output_dir = tmp_path / "fallback"
    calls: list[list[str]] = []
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_API_KEY", "original-secret")

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        run_dir = Path(argv[argv.index("--output-dir") + 1])
        scenario = run_dir.parent.name
        run_index = int(run_dir.name.split("_")[1])
        assert os.environ["APPSHELL_LANGGRAPH_PORT"]
        if scenario == "langgraph_disabled":
            assert os.environ["APPSHELL_LANGGRAPH_ENABLED"] == "0"
            assert os.environ["APPSHELL_LANGGRAPH_LLM_API_KEY"] == ""
        else:
            assert os.environ["APPSHELL_LANGGRAPH_ENABLED"] == "1"
            assert os.environ["APPSHELL_LANGGRAPH_LLM_API_KEY"] == bench.DUMMY_API_KEY
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "response.json", fake_response(scenario, run_index))
        write_json(run_dir / "timings.json", {"total_duration_ms": 1000 + run_index})
        write_json(run_dir / "auto_agent_trace.json", [{"seq": 1}, {"seq": 2}])
        return 0

    monkeypatch.setattr(bench.auto_agent_cli, "main", fake_main)

    rc = bench.main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--runs-per-scenario",
            "3",
            "--continue-on-error",
        ]
    )

    assert rc == 0
    assert len(calls) == 18
    assert os.environ["APPSHELL_LANGGRAPH_LLM_API_KEY"] == "original-secret"
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_runs"] == 18
    assert summary["all_scenarios_degraded_successfully"] is True
    assert len(summary["scenarios"]) == 6
    by_name = {item["scenario"]: item for item in summary["scenarios"]}
    for scenario, (_, reason) in SCENARIO_REASON_PLACEMENTS.items():
        item = by_name[scenario]
        assert item["total_runs"] == 3
        assert item["success_runs"] == 3
        assert item["success_rate"] == 1
        assert item["fallback_runs"] == 3
        assert item["fallback_rate"] == 1
        assert item["fallback_success_rate"] == 1
        assert item["accepted_runs"] == 3
        assert item["accepted_rate"] == 1
        assert item["validation_reject_runs"] == 0
        assert item["rollback_manifest_created_rate"] == 1
        assert item["avg_trace_event_count"] == 2
        assert item["fallback_reason_counts"] == {reason: 3}
        assert item["degraded_successfully"] is True
    assert (output_dir / "langgraph_disabled" / "run_001" / "run_result.json").exists()
    summary_text = (output_dir / "summary.json").read_text(encoding="utf-8") + (output_dir / "summary.md").read_text(encoding="utf-8")
    assert bench.DUMMY_API_KEY not in summary_text
    assert "original-secret" not in summary_text


def test_fallback_success_requires_validation_trace_and_rollback_manifest() -> None:
    bench = load_benchmark()
    scenario = bench.FallbackScenario(name="api_timeout", expected_reason="llm_timeout")
    results = [
        {
            "status": "success",
            "fallback": True,
            "fallback_success": True,
            "accepted": True,
            "validation_rejected": False,
            "rollback_manifest_created": True,
            "trace_event_count": 2,
            "duration_ms": 100,
            "fallback_reason_codes": ["llm_timeout"],
        },
        {
            "status": "success",
            "fallback": True,
            "fallback_success": False,
            "accepted": True,
            "validation_rejected": False,
            "rollback_manifest_created": False,
            "trace_event_count": 2,
            "duration_ms": 120,
            "fallback_reason_codes": ["llm_timeout"],
        },
        {
            "status": "success",
            "fallback": True,
            "fallback_success": False,
            "accepted": True,
            "validation_rejected": False,
            "rollback_manifest_created": True,
            "trace_event_count": 0,
            "duration_ms": 140,
            "fallback_reason_codes": ["llm_timeout"],
        },
    ]

    summary = bench.aggregate_scenario(scenario, results)

    assert summary["fallback_rate"] == 1
    assert summary["fallback_success_rate"] == 1 / 3
    assert summary["rollback_manifest_created_rate"] == 2 / 3
    assert summary["avg_trace_event_count"] == 4 / 3


def test_fallback_reason_codes_scan_cognition_and_plan_reason_codes() -> None:
    bench = load_benchmark()
    plan = {
        "cognition": {
            "fallback_reason_code": "",
            "reason_codes": ["selected_rule", "llm_unavailable"],
        },
        "reason_codes": ["plan_timeout", "selected_hybrid"],
    }

    reasons = bench.fallback_reason_codes(plan, {"fallback_reason_code": ""})

    assert reasons == ["llm_unavailable", "plan_timeout"]
