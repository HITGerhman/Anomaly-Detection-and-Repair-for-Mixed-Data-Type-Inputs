from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_cross_dataset_validation.py"
DATASETS = ["stroke", "orders_transactions", "user_device_logs"]


def _run_runner(args: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "output_dir" in proc.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_args(output_dir: Path, work_dir: Path) -> list[str]:
    return [
        "--output-dir",
        str(output_dir),
        "--scale-work-dir",
        str(work_dir),
        "--synthetic-rows",
        "240",
        "--injections",
        "25",
        "--scale-rows",
        "120",
        "--threshold-config",
        "1.5:3.5",
    ]


def test_cross_dataset_validation_small_all_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "cross_dataset"
    work_dir = tmp_path / "work"
    _run_runner(["--all", *_small_args(output_dir, work_dir)])

    expected_dataset_files = {
        "clean.csv",
        "corrupted.csv",
        "ground_truth.csv",
        "injection_summary.json",
        "detection_metrics.csv",
        "repair_metrics.csv",
        "side_effect_summary.csv",
    }
    for dataset in DATASETS:
        dataset_dir = output_dir / dataset
        assert expected_dataset_files.issubset({item.name for item in dataset_dir.iterdir()})
        ground_truth = pd.read_csv(dataset_dir / "ground_truth.csv")
        assert list(ground_truth.columns) == [
            "anomaly_id",
            "dataset",
            "expected_issue_type",
            "row_index",
            "column_name",
            "original_value",
            "corrupted_value",
            "repairable",
            "source_row_id",
            "duplicate_group",
            "consistency_rule_name",
            "created_by_seed",
            "notes",
        ]
        expected_rows = 100 if dataset == "stroke" else 25
        assert len(ground_truth) == expected_rows
        assert set(ground_truth["expected_issue_type"]) == {
            "missing_values",
            "numeric_outlier",
            "rare_category",
            "duplicate_record",
            "cross_column_consistency",
        }

        side_effects = pd.read_csv(dataset_dir / "side_effect_summary.csv")
        review_only = side_effects[side_effects["metric"] == "review_only_skipped"]
        assert set(review_only["issue_type"]) == {"duplicate_record", "cross_column_consistency"}
        assert int(review_only["count"].sum()) > 0

    detection = pd.read_csv(output_dir / "summary_detection_metrics.csv")
    assert {"dataset", "issue_type", "precision", "recall", "f1"}.issubset(detection.columns)
    assert set(detection["dataset"]) == set(DATASETS)
    for dataset in DATASETS:
        assert set(detection[detection["dataset"] == dataset]["issue_type"]) == {
            "missing_values",
            "numeric_outlier",
            "rare_category",
            "duplicate_record",
            "cross_column_consistency",
            "Overall",
        }

    repair = pd.read_csv(output_dir / "summary_repair_metrics.csv")
    assert {"exact_rate", "improved_or_exact_rate", "non_gt_modified"}.issubset(repair.columns)
    for dataset in DATASETS:
        assert set(repair[repair["dataset"] == dataset]["issue_type"]) == {
            "missing_values",
            "numeric_outlier",
            "rare_category",
            "Overall",
        }
        overall = repair[(repair["dataset"] == dataset) & (repair["issue_type"] == "Overall")].iloc[0]
        assert int(overall["skipped_non_repairable_count"]) > 0

    threshold = pd.read_csv(output_dir / "threshold_sensitivity_numeric_outlier.csv")
    assert {"dataset", "iqr_factor", "robust_z_threshold", "precision", "recall", "f1"}.issubset(threshold.columns)
    assert len(threshold) == len(DATASETS)

    scale = pd.read_csv(output_dir / "summary_scale_metrics.csv")
    assert {"rows", "scan_time_seconds", "repair_time_seconds", "detected_issue_count"}.issubset(scale.columns)
    assert list(scale["rows"]) == [120]


def test_cross_dataset_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_runner(["--generate", *_small_args(first, tmp_path / "work1")])
    _run_runner(["--generate", *_small_args(second, tmp_path / "work2")])

    for dataset in DATASETS:
        for filename in ["clean.csv", "corrupted.csv", "ground_truth.csv", "injection_summary.json"]:
            assert _sha256(first / dataset / filename) == _sha256(second / dataset / filename)
