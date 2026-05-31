from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_large_labeled_validation.py"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_large_labeled_validation as llv  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_streaming_generation_is_deterministic_and_labeled(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary = llv.generate_streaming_orders_dataset(
        dataset_name="orders_transactions_1m_labeled",
        row_count=260,
        seed=20260531,
        dataset_work_dir=first,
    )
    second_summary = llv.generate_streaming_orders_dataset(
        dataset_name="orders_transactions_1m_labeled",
        row_count=260,
        seed=20260531,
        dataset_work_dir=second,
    )

    assert first_summary["ground_truth_rows"] == 100
    assert first_summary["repairable_ground_truth_rows"] == 72
    assert first_summary["non_repairable_ground_truth_rows"] == 28
    assert first_summary["corrupted_rows"] == 272
    assert first_summary["injection_counts_by_type"] == {
        "missing_values": 30,
        "numeric_outlier": 24,
        "rare_category": 18,
        "duplicate_record": 12,
        "cross_column_consistency": 16,
    }
    assert second_summary["corrupted_rows"] == first_summary["corrupted_rows"]
    assert _sha256(first / "corrupted.csv") == _sha256(second / "corrupted.csv")
    assert _sha256(first / "ground_truth.csv") == _sha256(second / "ground_truth.csv")

    ground_truth = pd.read_csv(first / "ground_truth.csv")
    assert len(ground_truth) == 100
    assert set(ground_truth["expected_issue_type"]) == {
        "missing_values",
        "numeric_outlier",
        "rare_category",
        "duplicate_record",
        "cross_column_consistency",
    }
    assert int((ground_truth["expected_issue_type"] == "duplicate_record").sum()) == 12


def test_large_labeled_runner_small_outputs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    work_dir = tmp_path / "work"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--run",
            "both",
            "--rows-1m",
            "320",
            "--rows-10m",
            "340",
            "--output-dir",
            str(artifact_dir),
            "--work-dir",
            str(work_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "output_dir" in proc.stdout

    detection = pd.read_csv(artifact_dir / "summary_detection_metrics.csv")
    assert set(detection["dataset"]) == {
        "orders_transactions_1m_labeled",
        "orders_transactions_10m_labeled",
    }
    overall = detection[detection["issue_type"] == "Overall"]
    assert set(overall["gt"]) == {100}
    assert set(overall["fn"]) == {0}

    repair = pd.read_csv(artifact_dir / "summary_repair_metrics.csv")
    assert set(repair["dataset"]) == {"orders_transactions_1m_labeled"}
    repair_overall = repair[repair["issue_type"] == "Overall"].iloc[0]
    assert int(repair_overall["repairable_gt"]) == 72
    assert int(repair_overall["skipped_non_repairable_count"]) == 28

    runtime = pd.read_csv(artifact_dir / "summary_runtime_memory.csv")
    assert set(runtime["stage"]) == {"generate_labeled_csv", "scan_detect", "repair_evaluate"}
    assert {"peak_working_set_mb", "peak_private_memory_mb", "wall_seconds"}.issubset(runtime.columns)

    one_m_summary = json.loads((artifact_dir / "orders_transactions_1m_labeled" / "injection_summary.json").read_text(encoding="utf-8"))
    ten_m_repair = json.loads((artifact_dir / "orders_transactions_10m_labeled" / "repair_run_summary.json").read_text(encoding="utf-8"))
    assert one_m_summary["corrupted_rows"] == 332
    assert ten_m_repair["repair_evaluated"] is False
