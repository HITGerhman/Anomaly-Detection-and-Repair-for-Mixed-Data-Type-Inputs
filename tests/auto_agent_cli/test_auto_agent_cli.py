from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "appshell" / "core" / "python_engine" / "auto_agent_cli.py"


def load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_agent_cli", CLI_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_response() -> dict[str, Any]:
    return {
        "task_id": "task-1",
        "status": "ok",
        "duration_ms": 200,
        "result": {
            "agent": {
                "session_id": "session-1",
                "plan": {
                    "plan_id": "plan-1",
                    "selected_source": "rule",
                    "auto_repair_issue_ids": ["auto-1", "auto-2"],
                    "cautious_issue_ids": ["cautious-1"],
                    "manual_review_issue_ids": ["manual-1"],
                    "blocked_issue_ids": ["blocked-1"],
                    "cautious_issue_details": [
                        {
                            "issue_id": "cautious-1",
                            "issue_type": "numeric_outlier",
                            "column": "bmi",
                            "risk_reason": "requires_human_review_before_auto_repair",
                            "approval_required": True,
                            "suggested_action": "review_before_auto_repair",
                        }
                    ],
                    "blocked_issue_details": [
                        {
                            "issue_id": "blocked-1",
                            "issue_type": "time_series_shift",
                            "column": "date",
                            "blocked_reason": "unsupported_issue_type",
                            "blocked_by_rule": "a2_deterministic_issue_bucket_policy",
                            "suggested_next_action": "block_until_supported",
                        }
                    ],
                    "blocked_reason_counts": {"unsupported_issue_type": 1},
                    "timings_ms": {
                        "scan_duration_ms": 10,
                        "retrieve_duration_ms": 20,
                        "llm_plan_duration_ms": 30,
                        "llm_explain_duration_ms": 40,
                    },
                    "cognition": {"provider": "langgraph", "status": "engaged"},
                },
                "validation": {
                    "post_execute": {
                        "verdict": "accept",
                        "before_issue_count": 2,
                        "after_issue_count": 1,
                        "resolved_issue_count": 1,
                        "before_issue_items": 2,
                        "after_issue_items": 1,
                        "resolved_issue_items": 1,
                        "modified_cell_count": 2,
                        "total_cells_modified": 2,
                        "rollback_recommended": False,
                        "risk_notes": [],
                    }
                },
                "execution": {
                    "output_csv": "outputs/auto_agent/demo/repaired.csv",
                    "rollback": {"manifest_path": "outputs/auto_agent/demo/.rollback/repair.json"},
                    "timings_ms": {
                        "repair_duration_ms": 50,
                        "validation_duration_ms": 60,
                        "rollback_manifest_duration_ms": 7,
                        "total_duration_ms": 200,
                    },
                },
            },
            "safety": {"final_verdict": "accepted"},
        },
    }


def go_stdout(response: dict[str, Any]) -> str:
    return "task submitted: task-1\ntask=task-1 status=succeeded\ntask=task-1 response:\n" + json.dumps(response, indent=2) + "\n"


def create_trace_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE agent_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                trace_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_unix_ms INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO agent_trace (
                session_id, task_id, seq, agent_name, trace_type, summary, payload_json, created_at_unix_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "session-1",
                "task-1",
                1,
                "validator",
                "validation",
                "Validation gate accepted",
                json.dumps({"phase": "post_execute", "verdict": "accept"}),
                123,
            ),
        )


def test_cli_builds_go_demo_command_and_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    cli = load_cli()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age,city\n1,a\n", encoding="utf-8")
    output_dir = tmp_path / "auto"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        history_db = Path(cmd[cmd.index("-history-db") + 1])
        create_trace_db(history_db)
        return subprocess.CompletedProcess(cmd, 0, stdout=go_stdout(sample_response()), stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["--csv", str(csv_path), "--output-dir", str(output_dir), "--go-bin", "go-test"])

    assert rc == 0
    cmd = calls[0]
    assert cmd[:3] == ["go-test", "run", "./cmd/demo"]
    assert cmd[cmd.index("-action") + 1] == "agent.session.auto"
    assert Path(cmd[cmd.index("-history-db") + 1]) == output_dir / "auto_agent.sqlite"
    assert Path(cmd[cmd.index("-output") + 1]) == output_dir
    assert (output_dir / "response.json").exists()
    assert (output_dir / "repair_plan.json").exists()
    assert (output_dir / "validation_result.json").exists()
    assert (output_dir / "metric_definitions.json").exists()
    assert (output_dir / "issue_explanations.json").exists()
    assert (output_dir / "timings.json").exists()
    assert (output_dir / "auto_agent_trace.json").exists()


def test_extract_response_from_go_stdout() -> None:
    cli = load_cli()
    response = sample_response()

    parsed = cli.extract_response_from_stdout(go_stdout(response))

    assert parsed == response


def test_trace_export_and_report_content(tmp_path: Path, monkeypatch) -> None:
    cli = load_cli()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age,city\n1,a\n", encoding="utf-8")
    output_dir = tmp_path / "auto"

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        create_trace_db(Path(cmd[cmd.index("-history-db") + 1]))
        return subprocess.CompletedProcess(cmd, 0, stdout=go_stdout(sample_response()), stderr="sidecar log\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["--csv", str(csv_path), "--output-dir", str(output_dir)])

    assert rc == 0
    trace = json.loads((output_dir / "auto_agent_trace.json").read_text(encoding="utf-8"))
    assert trace[0]["trace_type"] == "validation"
    validation = json.loads((output_dir / "validation_result.json").read_text(encoding="utf-8"))
    assert validation["before_issue_items"] == 2
    assert validation["resolved_issue_items"] == 1
    assert validation["modified_cell_count"] == 2
    assert "metric_definitions" in validation
    explanations = json.loads((output_dir / "issue_explanations.json").read_text(encoding="utf-8"))
    assert explanations["blocked_issue_details"][0]["blocked_reason"] == "unsupported_issue_type"
    timings = json.loads((output_dir / "timings.json").read_text(encoding="utf-8"))
    assert timings["repair_duration_ms"] == 50
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "validation verdict" in report
    assert "`accept`" in report
    assert "outputs/auto_agent/demo/.rollback/repair.json" in report
    assert "outputs/auto_agent/demo/repaired.csv" in report
    assert "before_issue_items" in report
    assert "resolved_issue_items" in report
    assert "modified_cell_count" in report
    assert "resolved_issue_count" in report
    assert "rollback_manifest_created" in report
    assert "unsupported_issue_type" in report
    assert "| `repair_duration_ms` | 50 |" in report
    assert (output_dir / "run_stderr.log").read_text(encoding="utf-8") == "sidecar log\n"


def test_nonzero_go_exit_writes_logs_and_error(tmp_path: Path, monkeypatch) -> None:
    cli = load_cli()
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("age,city\n1,a\n", encoding="utf-8")
    output_dir = tmp_path / "auto"

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 7, stdout="no structured response\n", stderr="boom\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["--csv", str(csv_path), "--output-dir", str(output_dir)])

    assert rc == 7
    error = json.loads((output_dir / "error.json").read_text(encoding="utf-8"))
    assert error["returncode"] == 7
    assert (output_dir / "run_stdout.log").read_text(encoding="utf-8") == "no structured response\n"
    assert "failed" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_report_avoids_old_resolved_issue_count_as_primary_metric() -> None:
    cli = load_cli()

    report = cli.build_report(Path("input.csv"), sample_response(), [])

    assert "resolved_issue_items" in report
    assert "modified_cell_count" in report
    assert "resolved_issue_count" in report
    assert "resolved issue count" not in report.lower()
