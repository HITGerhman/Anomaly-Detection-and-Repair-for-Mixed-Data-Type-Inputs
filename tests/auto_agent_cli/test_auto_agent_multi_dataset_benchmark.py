from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "appshell" / "core" / "python_engine" / "auto_agent_multi_dataset_benchmark.py"


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_agent_multi_dataset_benchmark", BENCH_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "id,category,amount,window_start,window_end",
                "1,a,10,1,2",
                "2,b,20,2,3",
                "2,b,20,2,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def fake_artifacts(run_dir: Path, *, fallback: bool = False) -> None:
    cognition = {
        "provider": "langgraph" if not fallback else "deterministic",
        "status": "engaged" if not fallback else "fallback",
        "fallback_reason_code": "" if not fallback else "disabled",
    }
    plan = {
        "selected_source": "hybrid",
        "cognition": cognition,
        "auto_repair_issue_ids": ["missing-1", "rare-1"],
        "cautious_issue_ids": ["outlier-1"],
        "manual_review_issue_ids": ["dup-1", "cross-1"],
        "blocked_issue_ids": ["shift-1"],
        "skipped_issues": [
            {
                "issue_id": "outlier-1",
                "issue_type": "numeric_outlier",
                "column": "amount",
                "reason": "requires_human_review_before_auto_repair",
                "details": {
                    "bucket": "cautious",
                    "reason": "requires_human_review_before_auto_repair",
                    "recommended_action": "review_before_auto_repair",
                },
            },
            {
                "issue_id": "dup-1",
                "issue_type": "duplicate_record",
                "column": "id",
                "reason": "manual_review_required",
                "details": {
                    "bucket": "manual_review",
                    "reason": "manual_review_required",
                    "recommended_action": "manual_review",
                },
            },
            {
                "issue_id": "cross-1",
                "issue_type": "cross_column_consistency",
                "column": "window_start",
                "reason": "manual_review_required",
                "details": {
                    "bucket": "manual_review",
                    "reason": "manual_review_required",
                    "recommended_action": "manual_review",
                },
            },
            {
                "issue_id": "shift-1",
                "issue_type": "time_series_shift",
                "column": "amount",
                "reason": "unsupported_issue_type",
                "details": {
                    "bucket": "blocked",
                    "reason": "unsupported_issue_type",
                    "recommended_action": "block_until_supported",
                },
            },
        ],
    }
    validation = {
        "verdict": "accept",
        "before_issue_items": 12,
        "after_issue_items": 7,
        "resolved_issue_items": 5,
        "modified_cell_count": 6,
        "rollback_recommended": False,
    }
    response = {
        "task_id": "task-1",
        "status": "ok",
        "duration_ms": 1000,
        "result": {
            "agent": {
                "session_id": "session-1",
                "plan": plan,
                "validation": {"post_execute": validation},
                "execution": {
                    "output_csv": str(run_dir / "corrupted.repaired.hybrid.csv"),
                    "rollback": {"manifest_path": str(run_dir / ".rollback" / "repair.json")},
                },
            },
            "safety": {"final_verdict": "accepted"},
        },
    }
    write_json(run_dir / "response.json", response)
    write_json(run_dir / "repair_plan.json", plan)
    write_json(run_dir / "validation_result.json", validation)
    write_json(
        run_dir / "issue_explanations.json",
        {
            "blocked_issue_details": [
                {
                    "issue_id": "shift-1",
                    "issue_type": "time_series_shift",
                    "blocked_reason": "unsupported_issue_type",
                    "suggested_next_action": "block_until_supported",
                }
            ],
            "cautious_issue_details": [
                {
                    "issue_id": "outlier-1",
                    "issue_type": "numeric_outlier",
                    "risk_reason": "requires_human_review_before_auto_repair",
                    "approval_required": True,
                }
            ],
            "blocked_reason_counts": {"unsupported_issue_type": 1},
        },
    )
    write_json(run_dir / "timings.json", {"total_duration_ms": 1000, "trace_event_count": 3})
    write_json(run_dir / "auto_agent_trace.json", [{"seq": 1}, {"seq": 2}, {"seq": 3}])


def test_multi_dataset_benchmark_aggregates_required_fields_and_reasons(tmp_path: Path, monkeypatch) -> None:
    bench = load_benchmark()
    datasets = {
        "m1_stroke": tmp_path / "m1.csv",
        "orders_transactions": tmp_path / "orders.csv",
        "user_device_logs": tmp_path / "logs.csv",
    }
    for path in datasets.values():
        write_csv(path)
    output_dir = tmp_path / "bench"
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        run_dir = Path(argv[argv.index("--output-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(argv[argv.index("--csv") + 1])
        fallback = csv_path == datasets["user_device_logs"] and len([call for call in calls if str(csv_path) in call]) == 1
        fake_artifacts(run_dir, fallback=fallback)
        return 0

    monkeypatch.setattr(bench.auto_agent_cli, "main", fake_main)
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_API_KEY", "benchmark-test-token")
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_MODEL", "deepseek-chat")

    rc = bench.main(
        [
            "--output-dir",
            str(output_dir),
            "--runs-per-dataset",
            "2",
            "--dataset",
            f"m1_stroke={datasets['m1_stroke']}",
            "--dataset",
            f"orders_transactions={datasets['orders_transactions']}",
            "--dataset",
            f"user_device_logs={datasets['user_device_logs']}",
        ]
    )

    assert rc == 0
    assert len(calls) == 6
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_datasets"] == 3
    assert summary["total_runs"] == 6
    for item in summary["datasets"]:
        for field in bench.REQUIRED_SUMMARY_FIELDS:
            assert field in item
        assert item["row_count"] == 3
        assert item["column_count"] == 5
        assert item["total_runs"] == 2
        assert item["success_rate"] == 1
        assert item["accepted_rate"] == 1
        assert item["before_issue_items_avg"] == 12
        assert item["after_issue_items_avg"] == 7
        assert item["resolved_issue_items_avg"] == 5
        assert item["modified_cell_count_avg"] == 6
        assert item["blocked_issue_count_avg"] == 1
        assert item["cautious_issue_count_avg"] == 1
        assert item["rollback_manifest_created_rate"] == 1
        assert item["avg_total_ms"] == 1000
        assert item["p95_total_ms"] == 1000
        assert item["avg_trace_event_count"] == 3
        assert item["blocked_reason_counts"] == {"unsupported_issue_type": 2}
        assert item["cautious_reason_counts"] == {"requires_human_review_before_auto_repair": 2}
        assert item["manual_review_issue_type_counts"] == {"duplicate_record": 2, "cross_column_consistency": 2}

    logs_summary = next(item for item in summary["datasets"] if item["dataset_name"] == "user_device_logs")
    assert logs_summary["fallback_rate"] == 0.5
    assert logs_summary["fallback_reason_counts"] == {"disabled": 1}
    assert (output_dir / "m1_stroke" / "run_001" / "run_result.json").exists()
    assert (output_dir / "orders_transactions" / "runs.json").exists()
    assert (output_dir / "user_device_logs" / "dataset_summary.json").exists()
    report = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "numeric_outlier" in report
    assert "duplicate_record" in report
    assert "cross_column_consistency" in report
    assert "time_series_shift" in report
    assert "unsupported_issue_type" in report
    assert "benchmark-test-token" not in (output_dir / "summary.json").read_text(encoding="utf-8")
    assert "benchmark-test-token" not in report


def test_committed_multi_dataset_samples_are_small_and_mixed_type() -> None:
    bench = load_benchmark()
    sample_paths = [
        ROOT / "data" / "experiments" / "auto_agent_multi_dataset" / "orders_transactions" / "corrupted.csv",
        ROOT / "data" / "experiments" / "auto_agent_multi_dataset" / "user_device_logs" / "corrupted.csv",
    ]
    for path in sample_paths:
        rows, cols = bench.csv_shape(path)
        text = path.read_text(encoding="utf-8")
        assert rows == 30
        assert cols >= 10
        assert ",," in text or ",\n" in text
        assert path.stat().st_size < 10_000
