from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_m3_repair.py"
M1_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
M2_DIR = PROJECT_ROOT / "data" / "experiments" / "m2_stroke_detection"

EXPECTED_FILES = {
    "repaired.csv",
    "repair_metrics.json",
    "repair_details.json",
    "README.md",
}

REPAIRABLE_TYPES = {"missing_values", "numeric_outlier", "rare_category"}


def _run_evaluator(output_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--m1-dir",
            str(M1_DIR),
            "--m2-dir",
            str(M2_DIR),
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


def test_m3_evaluator_creates_repair_metrics_and_details(tmp_path: Path) -> None:
    output_dir = tmp_path / "m3"
    _run_evaluator(output_dir)

    assert EXPECTED_FILES.issubset({item.name for item in output_dir.iterdir()})

    metrics = json.loads((output_dir / "repair_metrics.json").read_text(encoding="utf-8"))
    details = json.loads((output_dir / "repair_details.json").read_text(encoding="utf-8"))

    assert metrics["milestone"] == "M3"
    assert metrics["dataset"] == "m3_stroke_repair"
    assert set(metrics["metrics"]["by_type"]) == REPAIRABLE_TYPES
    assert metrics["scan_config"]["enable_time_series_shift"] is False
    assert metrics["repair_batch"]["output_csv"] == "repaired.csv"

    overall = metrics["metrics"]["overall"]
    assert overall["total_ground_truth_count"] == 100
    assert overall["repairable_ground_truth_count"] == 72
    assert overall["skipped_non_repairable_ground_truth_count"] == 28
    assert details["summary"]["repairable_truth_rows"] == 72
    assert details["summary"]["skipped_non_repairable_truth_rows"] == 28

    by_type = metrics["metrics"]["by_type"]
    assert by_type["missing_values"]["ground_truth_count"] == 30
    assert by_type["numeric_outlier"]["ground_truth_count"] == 24
    assert by_type["rare_category"]["ground_truth_count"] == 18

    skipped = metrics["metrics"]["skipped_non_repairable_by_type"]
    assert skipped["cross_column_consistency"] == 16
    assert skipped["duplicate_record"] == 12

    comparison = metrics["repair_batch"]["comparison"]
    assert comparison["before_issue_count"] >= comparison["after_issue_count"]
    assert comparison["changed_cell_count"] == overall["total_cells_modified"]
    assert details["summary"]["changed_cell_rows"] == overall["changed_cells_observed"]
    assert details["summary"]["non_ground_truth_changed_cell_rows"] == overall["non_ground_truth_cells_modified"]

    assert 0.0 <= overall["exact_restoration_rate"] <= 1.0
    assert 0.0 <= overall["improved_or_exact_rate"] <= 1.0
    assert overall["non_ground_truth_cells_modified"] >= 0


def test_m3_evaluator_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_evaluator(first)
    _run_evaluator(second)

    for filename in EXPECTED_FILES:
        assert _sha256(first / filename) == _sha256(second / filename)
