from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_repair_strategy_comparison.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_repair_strategy_comparison", EVALUATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r5_evaluator_is_importable() -> None:
    module = _load_evaluator()

    assert callable(module.evaluate)
    assert callable(module.compute_strategy_metrics)
    assert callable(module.build_report)


def test_r5_strategy_metrics_count_success_and_side_effects(tmp_path: Path) -> None:
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

    metrics = module.compute_strategy_metrics(
        "toy",
        corrupted,
        repaired,
        ground_truth,
        {"issue_count": 2},
        {"issue_count": 1},
        {"skipped_issues": [{"issue_id": "skip-1"}], "total_cells_modified": 99},
        tmp_path / "repaired.csv",
        ["toy note"],
    )

    assert metrics["before_issue_count"] == 2
    assert metrics["after_issue_count"] == 1
    assert metrics["resolved_issue_count"] == 1
    assert metrics["total_cells_modified"] == 2
    assert metrics["engine_total_cells_modified"] == 99
    assert metrics["exact_restored_count"] == 1
    assert metrics["exact_restoration_rate"] == 1.0
    assert metrics["improved_or_exact_count"] == 1
    assert metrics["improved_or_exact_rate"] == 1.0
    assert metrics["non_ground_truth_cells_modified"] == 1
    assert metrics["skipped_issue_count"] == 1
    assert metrics["notes"] == ["toy note"]


def test_r5_report_includes_success_and_failure_notes() -> None:
    module = _load_evaluator()
    report = module.build_report(
        {
            "source": {
                "clean_csv": "clean.csv",
                "corrupted_csv": "corrupted.csv",
                "ground_truth_csv": "ground_truth.csv",
            },
            "selected_issue_ids": ["issue-1"],
            "strategies": {
                "rule-only": {
                    "strategy": "rule-only",
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
                    "skipped_issue_count": 0,
                    "notes": ["rule note"],
                },
                "gower-only": {
                    "strategy": "gower-only",
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
                    "skipped_issue_count": None,
                    "notes": ["gower failed: boom"],
                    "error": "boom",
                },
            },
        }
    )

    assert "# R5 Repair Strategy Comparison" in report
    assert "| rule-only | ok | 2 | 1 | 1 | 3 | 1 | 0.500000 | 1 | 0.500000 | 1 | 0 |" in report
    assert "| gower-only | failed | 2 | - | - | - | - | - | - | - | - | - |" in report
    assert "`rule-only`: rule note" in report
    assert "`gower-only` failure reason: boom" in report
