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


def test_continuous_numeric_target_returns_unsupported_target_type(tmp_path: Path) -> None:
    rows = 40
    csv_path = tmp_path / "continuous.csv"
    pd.DataFrame(
        {
            "feature_a": list(range(rows)),
            "bmi": [18.5 + i * 0.1 for i in range(rows)],
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
    assert resp["status"] == "error"
    assert resp["error"]["code"] == "UNSUPPORTED_TARGET_TYPE"
    assert "continuous" in resp["error"]["message"].lower()


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
