from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_gower_k_sensitivity.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_gower_k_sensitivity", EVALUATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r6_evaluator_is_importable() -> None:
    module = _load_evaluator()

    assert callable(module.evaluate)
    assert callable(module.build_k_metrics)
    assert callable(module.build_report)


def test_r6_k_metrics_include_mean_neighbor_confidence(tmp_path: Path) -> None:
    module = _load_evaluator()
    corrupted = pd.DataFrame({"metric": [1.0, 10.0], "category": ["x", "y"]})
    repaired = pd.DataFrame({"metric": [1.0, 2.0], "category": ["z", "y"]})
    ground_truth = pd.DataFrame(
        [
            {
                "injection_id": "m1-0001",
                "anomaly_type": "numeric_outlier",
                "expected_issue_type": "numeric_outlier",
                "row_id": "r2",
                "source_row_id": "2",
                "row_index": 1,
                "column": "metric",
                "original_value": "2.0",
                "corrupted_value": "10.0",
                "repairable": True,
                "notes": "numeric outlier",
            }
        ]
    )
    repair_result = {
        "neighbor_evidence": [{"candidate_confidence": 0.8}, {"candidate_confidence": 0.6}],
        "skipped_issues": [],
        "total_cells_modified": 2,
    }

    metrics = module.build_k_metrics(
        5,
        corrupted,
        repaired,
        ground_truth,
        {"issue_count": 2},
        {"issue_count": 1},
        repair_result,
        tmp_path / "repaired.csv",
    )

    assert metrics["k_neighbors"] == 5
    assert metrics["mean_neighbor_confidence"] == 0.7
    assert metrics["resolved_issue_count"] == 1
    assert metrics["total_cells_modified"] == 2
    assert metrics["exact_restoration_rate"] == 1.0
    assert metrics["non_ground_truth_cells_modified"] == 1


def test_r6_report_explains_k_tradeoffs_and_failures() -> None:
    module = _load_evaluator()
    report = module.build_report(
        {
            "source": {
                "clean_csv": "clean.csv",
                "corrupted_csv": "corrupted.csv",
                "ground_truth_csv": "ground_truth.csv",
            },
            "selected_issue_ids": ["issue-1"],
            "default_k_assessment": {
                "supports_default_k": True,
                "reason": "K=5 is near the best result.",
            },
            "k_results": {
                "3": {
                    "k_neighbors": 3,
                    "status": "ok",
                    "before_issue_count": 2,
                    "after_issue_count": 1,
                    "resolved_issue_count": 1,
                    "total_cells_modified": 3,
                    "exact_restored_count": 1,
                    "exact_restoration_rate": 0.5,
                    "improved_or_exact_count": 1,
                    "improved_or_exact_rate": 0.5,
                    "non_ground_truth_cells_modified": 1,
                    "mean_neighbor_confidence": 0.7,
                    "notes": ["small k note"],
                },
                "15": {
                    "k_neighbors": 15,
                    "status": "failed",
                    "before_issue_count": 2,
                    "after_issue_count": None,
                    "resolved_issue_count": None,
                    "total_cells_modified": None,
                    "exact_restored_count": None,
                    "exact_restoration_rate": None,
                    "improved_or_exact_count": None,
                    "improved_or_exact_rate": None,
                    "non_ground_truth_cells_modified": None,
                    "mean_neighbor_confidence": None,
                    "notes": ["k=15 failed: boom"],
                    "error": "boom",
                },
            },
        }
    )

    assert "# R6 Gower K Sensitivity" in report
    assert "If K is too small" in report
    assert "If K is too large" in report
    assert "sqrt(n)" in report
    assert "Continue default `K=5`: **Yes**" in report
    assert "`K=15` failure reason: boom" in report
