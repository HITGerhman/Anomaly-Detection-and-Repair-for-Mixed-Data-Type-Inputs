from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


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

