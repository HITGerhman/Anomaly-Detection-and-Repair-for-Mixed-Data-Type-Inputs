from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from src.training_core import load_system_state, predict_with_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_MAIN = PROJECT_ROOT / "appshell" / "core" / "python_engine" / "engine_main.py"


def _run_engine(payload_text: str) -> dict[str, object]:
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


def test_invalid_json_returns_invalid_json_code() -> None:
    resp = _run_engine("{bad-json")
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "INVALID_JSON"


def test_missing_action_returns_invalid_input_code() -> None:
    resp = _run_engine(json.dumps({"task_id": "t-1", "payload": {}}))
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "INVALID_INPUT"


def test_unknown_action_returns_unknown_action_code() -> None:
    payload = {"task_id": "t-2", "action": "not-supported", "payload": {}}
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "UNKNOWN_ACTION"


def test_scan_file_returns_issue_catalog(tmp_path: Path) -> None:
    csv_path = tmp_path / "scan_input.csv"
    pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 400, np.nan],
            "bmi": [21.1, 22.3, np.nan, 24.2, 23.9, 25.1, 24.8],
            "gender": ["M", "F", "F", "M", "Z", "M", None],
        }
    ).to_csv(csv_path, index=False)

    payload = {
        "task_id": "scan-1",
        "action": "scan_file",
        "payload": {
            "csv_path": str(csv_path),
            "max_bins": 24,
            "max_issues": 200,
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "ok"
    result = resp["result"]
    assert result["data_profile"]["rows"] == 7
    assert result["data_profile"]["columns"] == 3
    assert result["issue_count"] >= 2
    assert result["scan_config"]["max_bins"] == 24
    assert "scan_summary" in result
    issue_types = {item["issue_type"] for item in result["issues"]}
    assert "missing_values" in issue_types
    assert "numeric_outlier" in issue_types
    assert any(item["column"] == "age" for item in result["column_thumbnails"])
    assert all("issue_score" in item for item in result["issues"])
    assert all("risk_score" in item for item in result["column_thumbnails"])
    assert all("hot_segments" in item for item in result["column_thumbnails"])
    scores = [float(item["issue_score"]) for item in result["issues"]]
    assert scores == sorted(scores, reverse=True)


def test_scan_file_honors_custom_scan_thresholds(tmp_path: Path) -> None:
    csv_path = tmp_path / "scan_thresholds.csv"
    pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 200, 201, 202],
            "cat": ["a", "a", "a", "a", "a", "a", "b", "c", "d"],
        }
    ).to_csv(csv_path, index=False)

    payload = {
        "task_id": "scan-threshold-1",
        "action": "scan_file",
        "payload": {
            "csv_path": str(csv_path),
            "scan_config": {
                "max_bins": 30,
                "rare_ratio_threshold": 0.2,
                "rare_count_floor": 1,
                "numeric_iqr_factor": 1.2,
            },
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "ok"
    config = resp["result"]["scan_config"]
    assert config["max_bins"] == 30
    assert abs(float(config["rare_ratio_threshold"]) - 0.2) < 1e-9
    assert abs(float(config["numeric_iqr_factor"]) - 1.2) < 1e-9


def test_repair_batch_applies_selected_issues_without_writing_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "repair_batch_input.csv"
    pd.DataFrame(
        {
            "age": [19, 20, 21, 22, 23, 24, 430, np.nan],
            "bmi": [19.5, np.nan, 21.0, 22.3, 22.9, 23.1, 23.8, 24.2],
            "gender": ["M", "F", "F", "M", "M", "F", "X", None],
        }
    ).to_csv(csv_path, index=False)

    scan_payload = {
        "task_id": "scan-for-repair-batch",
        "action": "scan_file",
        "payload": {"csv_path": str(csv_path)},
    }
    scan_resp = _run_engine(json.dumps(scan_payload))
    assert scan_resp["status"] == "ok"
    scan_issues = scan_resp["result"]["issues"]
    selected_ids = [
        issue["issue_id"]
        for issue in scan_issues
        if issue["issue_type"] in {"missing_values", "numeric_outlier"}
    ][:3]
    assert selected_ids

    repair_payload = {
        "task_id": "repair-batch-1",
        "action": "repair_batch",
        "payload": {
            "csv_path": str(csv_path),
            "issue_ids": selected_ids,
            "write_output": False,
            "scan_config": {
                "max_bins": 48,
                "max_issues": 1200,
            },
        },
    }
    repair_resp = _run_engine(json.dumps(repair_payload))
    assert repair_resp["status"] == "ok"
    result = repair_resp["result"]
    assert result["selected_issue_count"] == len(selected_ids)
    assert result["applied_issue_count"] >= 1
    assert result["total_cells_modified"] >= 1
    assert result["write_output"] is False
    assert result["output_csv"] is None
    assert result["scan_config"]["max_bins"] == 48
    assert "comparison" in result
    assert result["comparison"]["before_issue_count"] >= result["comparison"]["after_issue_count"]


def test_repair_batch_plan_only_returns_real_comparison(tmp_path: Path) -> None:
    csv_path = tmp_path / "repair_batch_plan_only.csv"
    pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 420, np.nan],
            "bmi": [20.1, np.nan, 21.5, 22.2, 23.1, 24.2, 25.3, 24.7],
            "work_type": ["Private", "Private", "Govt", "Private", "X", "Y", "Private", None],
        }
    ).to_csv(csv_path, index=False)

    scan_payload = {
        "task_id": "scan-for-plan-only",
        "action": "scan_file",
        "payload": {"csv_path": str(csv_path)},
    }
    scan_resp = _run_engine(json.dumps(scan_payload))
    assert scan_resp["status"] == "ok"
    selected_ids = [item["issue_id"] for item in scan_resp["result"]["issues"]][:4]
    assert selected_ids

    repair_payload = {
        "task_id": "repair-plan-only",
        "action": "repair_batch",
        "payload": {
            "csv_path": str(csv_path),
            "issue_ids": selected_ids,
            "plan_only": True,
            "write_output": True,
            "repair_strategy": {
                "conflict_policy": "last_wins",
                "missing_numeric": "median",
                "missing_categorical": "mode",
            },
        },
    }
    repair_resp = _run_engine(json.dumps(repair_payload))
    assert repair_resp["status"] == "ok"
    result = repair_resp["result"]
    assert result["plan_only"] is True
    assert result["write_output"] is False
    assert result["output_csv"] is None
    assert result["execution_mode"] == "plan_only"
    assert result["comparison"]["before_issue_count"] >= result["comparison"]["after_issue_count"]
    assert result["comparison"]["changed_cell_count"] >= 1
    assert "repair_strategy" in result
    assert "conflict_summary" in result


def test_rollback_repair_batch_restores_source_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "rollback_input.csv"
    pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 420, np.nan],
            "bmi": [20.1, np.nan, 21.5, 22.2, 23.1, 24.2, 25.3, 24.7],
            "gender": ["M", "F", "F", "M", "M", "F", "X", None],
        }
    ).to_csv(csv_path, index=False)
    original_content = csv_path.read_text(encoding="utf-8")

    scan_payload = {
        "task_id": "scan-for-rollback",
        "action": "scan_file",
        "payload": {"csv_path": str(csv_path)},
    }
    scan_resp = _run_engine(json.dumps(scan_payload))
    assert scan_resp["status"] == "ok"
    selected_ids = [item["issue_id"] for item in scan_resp["result"]["issues"]][:3]
    assert selected_ids

    repaired_csv = tmp_path / "rollback_output.csv"
    repair_payload = {
        "task_id": "repair-for-rollback",
        "action": "repair_batch",
        "payload": {
            "csv_path": str(csv_path),
            "issue_ids": selected_ids,
            "write_output": True,
            "output_csv": str(repaired_csv),
            "enable_rollback": True,
            "rollback_dir": str(tmp_path / "rollback_meta"),
        },
    }
    repair_resp = _run_engine(json.dumps(repair_payload))
    assert repair_resp["status"] == "ok"
    result = repair_resp["result"]
    assert result["output_csv"] == str(repaired_csv)
    assert repaired_csv.exists()
    assert result["rollback"] is not None

    manifest_path = Path(result["rollback"]["manifest_path"])
    assert manifest_path.exists()

    csv_path.write_text("corrupted,data\n1,2\n", encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8") != original_content

    rollback_payload = {
        "task_id": "rollback-run",
        "action": "rollback_repair_batch",
        "payload": {
            "manifest_path": str(manifest_path),
            "restore_target": "source_csv",
        },
    }
    rollback_resp = _run_engine(json.dumps(rollback_payload))
    assert rollback_resp["status"] == "ok"
    assert rollback_resp["result"]["restored_to"] == str(csv_path.resolve())
    assert csv_path.read_text(encoding="utf-8") == original_content


def test_invalid_target_returns_invalid_target_column_code(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"a": [1, 2], "b": [0, 1]}).to_csv(csv_path, index=False)

    payload = {
        "task_id": "t-3",
        "action": "train",
        "payload": {
            "csv_path": str(csv_path),
            "target_col": "stroke",
            "output_dir": str(tmp_path / "out"),
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "INVALID_TARGET_COLUMN"


def test_continuous_numeric_target_trains_as_regression(tmp_path: Path) -> None:
    rows = 80
    rng = np.random.default_rng(20260225)
    csv_path = tmp_path / "continuous.csv"
    pd.DataFrame(
        {
            "feature_a": list(range(rows)),
            "feature_b": rng.normal(50, 4, size=rows),
            "bmi": [18.5 + i * 0.08 for i in range(rows)],
        }
    ).to_csv(csv_path, index=False)

    payload = {
        "task_id": "t-4",
        "action": "train",
        "payload": {
            "csv_path": str(csv_path),
            "target_col": "bmi",
            "output_dir": str(tmp_path / "out"),
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "ok"
    metrics = resp["result"]["metrics"]
    assert metrics["task_type"] == "regression"
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "prediction_confidence_mean" in metrics
    assert resp["result"]["data_profile"]["task_type"] == "regression"


def test_continuous_numeric_target_with_classification_mode_returns_unsupported(tmp_path: Path) -> None:
    rows = 40
    csv_path = tmp_path / "continuous_classification.csv"
    pd.DataFrame(
        {
            "feature_a": list(range(rows)),
            "bmi": [18.0 + i * 0.2 for i in range(rows)],
        }
    ).to_csv(csv_path, index=False)

    payload = {
        "task_id": "t-4c",
        "action": "train",
        "payload": {
            "csv_path": str(csv_path),
            "target_col": "bmi",
            "task_type": "classification",
            "output_dir": str(tmp_path / "out"),
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "UNSUPPORTED_TARGET_TYPE"


def test_target_with_missing_values_returns_invalid_input(tmp_path: Path) -> None:
    csv_path = tmp_path / "target_nan.csv"
    pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "stroke": [0, 1, None, 0, 1],
        }
    ).to_csv(csv_path, index=False)

    payload = {
        "task_id": "t-5",
        "action": "train",
        "payload": {
            "csv_path": str(csv_path),
            "target_col": "stroke",
            "output_dir": str(tmp_path / "out"),
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "INVALID_INPUT"
    assert resp["error"]["details"]["missing_count"] == 1


def test_scan_file_reports_extended_issue_types_and_explanations(tmp_path: Path) -> None:
    csv_path = tmp_path / "scan_extended.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12],
            "start_day": [1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12],
            "end_day": [2, 3, 4, 4, 6, 7, 6, 9, 10, 11, 12, 13],
            "sensor": [10.0, 10.4, 10.9, 10.9, 11.2, 11.6, 30.0, 31.0, 31.4, 12.1, 12.5, 12.9],
            "group": ["a", "a", "b", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
        }
    ).to_csv(csv_path, index=False)

    payload = {
        "task_id": "scan-extended-1",
        "action": "scan_file",
        "payload": {
            "csv_path": str(csv_path),
            "scan_config": {
                "time_series_min_points": 6,
                "time_series_z_threshold": 2.2,
                "consistency_rules": [
                    {
                        "name": "start_before_end",
                        "type": "lte",
                        "left_col": "start_day",
                        "right_col": "end_day",
                    }
                ],
                "enable_duplicate_record": True,
                "duplicate_subset": ["id", "start_day", "end_day", "sensor", "group"],
            },
        },
    }
    resp = _run_engine(json.dumps(payload))
    assert resp["status"] == "ok"
    result = resp["result"]
    issue_types = {item["issue_type"] for item in result["issues"]}
    assert "time_series_shift" in issue_types
    assert "cross_column_consistency" in issue_types
    assert "duplicate_record" in issue_types
    assert all("confidence" in item for item in result["issues"])
    assert all("explain_features" in item for item in result["issues"])
    assert "issue_type_counts" in result["scan_summary"]


def test_repair_action_returns_repair_summary(tmp_path: Path) -> None:
    rows = 160
    csv_path = tmp_path / "repair_input.csv"
    rng = np.random.default_rng(20260221)
    df = pd.DataFrame(
        {
            "age": np.concatenate([rng.integers(18, 50, size=rows - 32), rng.integers(62, 85, size=32)]),
            "avg_glucose_level": np.concatenate(
                [rng.normal(95, 10, size=rows - 32), rng.normal(185, 12, size=32)]
            ),
            "hypertension": np.concatenate([rng.integers(0, 2, size=rows - 32), np.ones(32)]),
            "stroke": np.concatenate([np.zeros(rows - 32, dtype=int), np.ones(32, dtype=int)]),
        }
    )
    df.to_csv(csv_path, index=False)

    model_dir = tmp_path / "model_out"
    train_payload = {
        "task_id": "repair-train",
        "action": "train",
        "payload": {
            "csv_path": str(csv_path),
            "target_col": "stroke",
            "output_dir": str(model_dir),
        },
    }
    train_resp = _run_engine(json.dumps(train_payload))
    assert train_resp["status"] == "ok"

    model, x_test, _ = load_system_state(model_dir)
    pred, prob = predict_with_threshold(model, x_test)
    anomaly_idx = np.where(pred == 1)[0]
    sample_index = int(anomaly_idx[0]) if anomaly_idx.size > 0 else int(np.argmax(prob))

    repair_payload = {
        "task_id": "repair-run",
        "action": "repair",
        "payload": {
            "model_dir": str(model_dir),
            "sample_index": sample_index,
            "max_changes": 3,
            "k_neighbors": 9,
        },
    }
    repair_resp = _run_engine(json.dumps(repair_payload))
    assert repair_resp["status"] == "ok"
    result = repair_resp["result"]
    assert result["sample_index"] == sample_index
    assert "repair_summary" in result
    assert "repair_changes" in result
    assert "repaired_sample" in result
