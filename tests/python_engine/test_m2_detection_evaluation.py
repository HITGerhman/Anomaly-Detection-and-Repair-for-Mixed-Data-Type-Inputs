from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_m2_detection.py"
M1_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"

EXPECTED_FILES = {
    "detection_metrics.json",
    "detection_matches.json",
    "README.md",
}

EXPECTED_TYPES = {
    "cross_column_consistency",
    "duplicate_record",
    "missing_values",
    "numeric_outlier",
    "rare_category",
}


def _run_evaluator(output_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--m1-dir",
            str(M1_DIR),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"overall"' in proc.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m2_evaluator_creates_metrics_and_matches(tmp_path: Path) -> None:
    output_dir = tmp_path / "m2"
    _run_evaluator(output_dir)

    assert EXPECTED_FILES.issubset({item.name for item in output_dir.iterdir()})

    metrics = json.loads((output_dir / "detection_metrics.json").read_text(encoding="utf-8"))
    matches = json.loads((output_dir / "detection_matches.json").read_text(encoding="utf-8"))

    assert metrics["milestone"] == "M2"
    assert metrics["dataset"] == "m2_stroke_detection"
    assert metrics["scan_config"]["enable_time_series_shift"] is False
    assert set(metrics["metrics"]["by_type"]) == EXPECTED_TYPES

    overall = metrics["metrics"]["overall"]
    by_type = metrics["metrics"]["by_type"]
    assert overall["ground_truth_count"] == 100
    assert matches["summary"]["truth_match_rows"] == 100
    assert matches["summary"]["false_positive_rows"] == overall["fp"]
    assert matches["summary"]["false_negative_rows"] == overall["fn"]

    summed = {
        "ground_truth_count": sum(item["ground_truth_count"] for item in by_type.values()),
        "predicted_count": sum(item["predicted_count"] for item in by_type.values()),
        "tp": sum(item["tp"] for item in by_type.values()),
        "fp": sum(item["fp"] for item in by_type.values()),
        "fn": sum(item["fn"] for item in by_type.values()),
    }
    for field, value in summed.items():
        assert overall[field] == value

    for metric in [overall, *by_type.values()]:
        assert metric["tp"] + metric["fn"] == metric["ground_truth_count"]
        assert metric["tp"] + metric["fp"] == metric["predicted_count"]
        assert 0.0 <= metric["precision"] <= 1.0
        assert 0.0 <= metric["recall"] <= 1.0
        assert 0.0 <= metric["f1"] <= 1.0

    assert by_type["duplicate_record"]["ground_truth_count"] == 12
    assert by_type["duplicate_record"]["recall"] == 1.0
    assert by_type["cross_column_consistency"]["ground_truth_count"] == 16


def test_m2_evaluator_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_evaluator(first)
    _run_evaluator(second)

    for filename in EXPECTED_FILES:
        assert _sha256(first / filename) == _sha256(second / filename)
