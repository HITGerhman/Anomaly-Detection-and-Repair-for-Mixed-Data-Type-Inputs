from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.repair_module import AnomalyRepairer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_MAIN = PROJECT_ROOT / "appshell" / "core" / "python_engine" / "engine_main.py"


def _run_engine(payload: dict[str, object] | str) -> dict[str, object]:
    payload_text = payload if isinstance(payload, str) else json.dumps(payload)
    proc = subprocess.run(
        [sys.executable, str(ENGINE_MAIN)],
        input=payload_text,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"engine stdout is empty; stderr={proc.stderr}"
    return json.loads(lines[-1])


def _write_mixed_regression_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "record_id": ["r01", "r02", "dup", "dup", "r05", "r06", "r07", "r08", "r09", "r10", "r11", "r12"],
            "age": [40, 41, 42, 42, 43, 44, 45, 46, 47, 48, 49, 250],
            "bmi": [22.1, np.nan, 23.0, 23.0, 23.4, 23.6, 23.8, 24.0, 24.2, 24.4, 24.6, 24.8],
            "work_type": [
                "Private",
                "Private",
                "Private",
                "Private",
                "Private",
                "Private",
                "Private",
                "Private",
                "Govt_job",
                "Self-employed",
                "Private",
                "Private",
            ],
            "start_day": [1, 2, 3, 3, 5, 6, 9, 8, 9, 10, 11, 12],
            "end_day": [2, 3, 4, 4, 6, 7, 7, 9, 10, 11, 12, 13],
        }
    ).to_csv(path, index=False)


def _scan_payload(csv_path: Path) -> dict[str, object]:
    return {
        "task_id": "m4-scan",
        "action": "scan_file",
        "payload": {
            "csv_path": str(csv_path),
            "scan_config": {
                "max_bins": 40,
                "max_issues": 500,
                "enable_time_series_shift": False,
                "enable_cross_column_consistency": True,
                "consistency_rules": [
                    {
                        "name": "start_before_end",
                        "type": "lte",
                        "left_col": "start_day",
                        "right_col": "end_day",
                    }
                ],
                "enable_duplicate_record": True,
                "duplicate_subset": ["record_id"],
            },
        },
    }


def _scan(csv_path: Path) -> dict[str, object]:
    response = _run_engine(_scan_payload(csv_path))
    assert response["status"] == "ok"
    return response["result"]  # type: ignore[return-value]


def _issue_ids_by_type(scan_result: dict[str, object]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for issue in scan_result["issues"]:  # type: ignore[index]
        issue_type = str(issue["issue_type"])
        grouped.setdefault(issue_type, []).append(str(issue["issue_id"]))
    return grouped


def test_m4_scan_file_fixed_mixed_issue_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_scan.csv"
    _write_mixed_regression_csv(csv_path)

    result = _scan(csv_path)
    issue_types = {issue["issue_type"] for issue in result["issues"]}  # type: ignore[index]

    assert {
        "missing_values",
        "numeric_outlier",
        "rare_category",
        "cross_column_consistency",
        "duplicate_record",
    }.issubset(issue_types)
    assert result["data_profile"]["rows"] == 12  # type: ignore[index]
    assert result["scan_config"]["max_bins"] == 40  # type: ignore[index]
    assert result["scan_config"]["duplicate_subset"] == ["record_id"]  # type: ignore[index]

    summary = result["scan_summary"]  # type: ignore[index]
    for issue_type in issue_types:
        assert summary["issue_type_counts"][issue_type] >= 1

    for issue in result["issues"]:  # type: ignore[index]
        assert issue["issue_id"]
        assert issue["column"]
        assert "issue_score" in issue
        assert "confidence" in issue
        assert "explain_features" in issue
        assert "detail" in issue
        assert "segments" in issue


def test_m4_repair_batch_applies_selected_issues_and_skips_manual_review(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_repair.csv"
    _write_mixed_regression_csv(csv_path)
    scan_result = _scan(csv_path)
    ids_by_type = _issue_ids_by_type(scan_result)
    selected_ids = [
        ids_by_type["missing_values"][0],
        ids_by_type["numeric_outlier"][0],
        ids_by_type["rare_category"][0],
        ids_by_type["cross_column_consistency"][0],
        ids_by_type["duplicate_record"][0],
    ]

    response = _run_engine(
        {
            "task_id": "m4-repair-batch",
            "action": "repair_batch",
            "payload": {
                "csv_path": str(csv_path),
                "issue_ids": selected_ids,
                "write_output": False,
                "scan_config": _scan_payload(csv_path)["payload"]["scan_config"],  # type: ignore[index]
            },
        }
    )

    assert response["status"] == "ok"
    result = response["result"]
    assert result["selected_issue_count"] == len(selected_ids)
    assert result["write_output"] is False
    assert result["output_csv"] is None
    assert result["applied_issue_count"] >= 3
    assert result["total_cells_modified"] >= 3
    assert result["comparison"]["before_issue_count"] >= result["comparison"]["after_issue_count"]

    skipped = {item["issue_id"]: item for item in result["skipped_issues"]}
    assert ids_by_type["cross_column_consistency"][0] in skipped
    assert ids_by_type["duplicate_record"][0] in skipped
    assert skipped[ids_by_type["cross_column_consistency"][0]]["reason"] == "unsupported_issue_type"
    assert skipped[ids_by_type["duplicate_record"][0]]["reason"] == "unsupported_issue_type"


def test_m4_repair_batch_plan_only_keeps_files_unwritten(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_plan_only.csv"
    output_csv = tmp_path / "should_not_exist.csv"
    _write_mixed_regression_csv(csv_path)
    scan_result = _scan(csv_path)
    ids_by_type = _issue_ids_by_type(scan_result)
    selected_ids = [ids_by_type["missing_values"][0], ids_by_type["numeric_outlier"][0]]

    response = _run_engine(
        {
            "task_id": "m4-plan-only",
            "action": "repair_batch",
            "payload": {
                "csv_path": str(csv_path),
                "issue_ids": selected_ids,
                "plan_only": True,
                "write_output": True,
                "output_csv": str(output_csv),
                "scan_config": _scan_payload(csv_path)["payload"]["scan_config"],  # type: ignore[index]
            },
        }
    )

    assert response["status"] == "ok"
    result = response["result"]
    assert result["plan_only"] is True
    assert result["execution_mode"] == "plan_only"
    assert result["write_output"] is False
    assert result["output_csv"] is None
    assert result["comparison"]["changed_cell_count"] >= 2
    assert not output_csv.exists()


def test_m4_rollback_repair_batch_restores_source_and_reports_manifest_errors(tmp_path: Path) -> None:
    csv_path = tmp_path / "rollback_source.csv"
    output_csv = tmp_path / "rollback_output.csv"
    rollback_dir = tmp_path / "rollback_meta"
    _write_mixed_regression_csv(csv_path)
    original_content = csv_path.read_text(encoding="utf-8")
    scan_result = _scan(csv_path)
    ids_by_type = _issue_ids_by_type(scan_result)

    repair_response = _run_engine(
        {
            "task_id": "m4-rollback-repair",
            "action": "repair_batch",
            "payload": {
                "csv_path": str(csv_path),
                "issue_ids": [ids_by_type["missing_values"][0], ids_by_type["numeric_outlier"][0]],
                "write_output": True,
                "output_csv": str(output_csv),
                "enable_rollback": True,
                "rollback_dir": str(rollback_dir),
                "scan_config": _scan_payload(csv_path)["payload"]["scan_config"],  # type: ignore[index]
            },
        }
    )
    assert repair_response["status"] == "ok"
    manifest_path = Path(repair_response["result"]["rollback"]["manifest_path"])
    assert manifest_path.exists()

    csv_path.write_text("changed,data\n1,2\n", encoding="utf-8")
    rollback_response = _run_engine(
        {
            "task_id": "m4-rollback",
            "action": "rollback_repair_batch",
            "payload": {
                "manifest_path": str(manifest_path),
                "restore_target": "source_csv",
            },
        }
    )
    assert rollback_response["status"] == "ok"
    assert csv_path.read_text(encoding="utf-8") == original_content

    missing_response = _run_engine(
        {
            "task_id": "m4-missing-manifest",
            "action": "rollback_repair_batch",
            "payload": {"manifest_path": str(tmp_path / "missing.json")},
        }
    )
    assert missing_response["status"] == "error"
    assert missing_response["error"]["code"] == "FILE_NOT_FOUND"

    invalid_manifest = tmp_path / "invalid_manifest.json"
    invalid_manifest.write_text("[]\n", encoding="utf-8")
    invalid_response = _run_engine(
        {
            "task_id": "m4-invalid-manifest",
            "action": "rollback_repair_batch",
            "payload": {"manifest_path": str(invalid_manifest)},
        }
    )
    assert invalid_response["status"] == "error"
    assert invalid_response["error"]["code"] == "ROLLBACK_FAILED"


def test_m4_gower_repairer_returns_original_labels_and_numeric_suggestion() -> None:
    normal_data = pd.DataFrame(
        {
            "work_type": pd.Categorical(
                ["Private", "Private", "Private", "Govt_job", "Self-employed"],
                categories=["Private", "Govt_job", "Self-employed"],
            ),
            "age": [40, 41, 42, 70, 72],
            "smoking_status": ["never", "never", "never", "smokes", "smokes"],
        }
    )
    repairer = AnomalyRepairer(normal_data)
    anomaly_sample = pd.DataFrame(
        {
            "work_type": pd.Categorical(["Govt_job"], categories=normal_data["work_type"].cat.categories),
            "age": [41],
            "smoking_status": ["never"],
        }
    )

    categorical_suggestion, categorical_neighbors = repairer.generate_repair_suggestion(
        anomaly_sample,
        "work_type",
        k=3,
    )
    assert categorical_suggestion["Suggested Value"] == "Private"
    assert not isinstance(categorical_suggestion["Suggested Value"], (int, np.integer))
    assert list(categorical_neighbors["work_type"].astype(str)) == ["Private", "Private", "Private"]

    numeric_suggestion, _ = repairer.generate_repair_suggestion(anomaly_sample, "age", k=3)
    assert numeric_suggestion["Suggested Value"] == 41


def test_m4_engine_structured_errors_for_invalid_inputs(tmp_path: Path) -> None:
    missing_csv_response = _run_engine(
        {
            "task_id": "m4-missing-csv",
            "action": "scan_file",
            "payload": {"csv_path": str(tmp_path / "missing.csv")},
        }
    )
    assert missing_csv_response["status"] == "error"
    assert missing_csv_response["error"]["code"] == "FILE_NOT_FOUND"

    csv_path = tmp_path / "invalid_inputs.csv"
    _write_mixed_regression_csv(csv_path)

    invalid_scan_response = _run_engine(
        {
            "task_id": "m4-invalid-scan-config",
            "action": "scan_file",
            "payload": {"csv_path": str(csv_path), "scan_config": {"max_bins": 1}},
        }
    )
    assert invalid_scan_response["status"] == "error"
    assert invalid_scan_response["error"]["code"] == "INVALID_INPUT"
    assert invalid_scan_response["error"]["details"]["field"] == "max_bins"

    invalid_strategy_response = _run_engine(
        {
            "task_id": "m4-invalid-repair-strategy",
            "action": "repair_batch",
            "payload": {
                "csv_path": str(csv_path),
                "repair_strategy": {"conflict_policy": "merge_everything"},
            },
        }
    )
    assert invalid_strategy_response["status"] == "error"
    assert invalid_strategy_response["error"]["code"] == "INVALID_INPUT"
    assert invalid_strategy_response["error"]["details"]["field"] == "repair_strategy.conflict_policy"

