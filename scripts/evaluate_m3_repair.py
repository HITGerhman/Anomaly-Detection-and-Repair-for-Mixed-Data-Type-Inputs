"""Evaluate M3 repair quality against the M1 ground truth.

The evaluator reuses the existing rule-based repair_batch action. It does not
change the Python engine protocol or tune detector thresholds; it records the
current repair behavior against the controlled M1/M2 experiment data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENGINE_DIR = PROJECT_ROOT / "appshell" / "core" / "python_engine"
DEFAULT_M1_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
DEFAULT_M2_DIR = PROJECT_ROOT / "data" / "experiments" / "m2_stroke_detection"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "m3_stroke_repair"

if str(PYTHON_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_ENGINE_DIR))

from engine_core import action_repair_batch, _to_builtin  # noqa: E402


REPAIRABLE_TYPES = ["missing_values", "numeric_outlier", "rare_category"]
NON_REPAIRABLE_TYPES = ["cross_column_consistency", "duplicate_record"]
M3_SCAN_CONFIG = {
    "max_issues": 1000,
    "preview_limit": 20,
    "enable_time_series_shift": False,
    "enable_cross_column_consistency": True,
    "consistency_rules": [
        {
            "name": "record_start_before_end",
            "type": "lte",
            "left_col": "record_start_day",
            "right_col": "record_end_day",
        }
    ],
    "enable_duplicate_record": True,
    "duplicate_subset": ["source_row_id"],
}
M3_REPAIR_STRATEGY = {
    "conflict_policy": "first_wins",
    "issue_priority": REPAIRABLE_TYPES,
    "missing_numeric": "median",
    "missing_categorical": "mode",
    "outlier": "clip",
    "rare_category": "mode",
    "preview_limit": 20,
}


def _resolve_path(path_text: str | Path) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return resolved.name


def _value_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


def _number_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def _values_equal(left: Any, right: Any) -> bool:
    left_number = _number_or_none(left)
    right_number = _number_or_none(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-9)
    return _value_text(left) == _value_text(right)


def _cell_changed(before: Any, after: Any) -> bool:
    return not _values_equal(before, after)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _load_inputs(m1_dir: Path, m2_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    clean_csv = m1_dir / "clean.csv"
    corrupted_csv = m1_dir / "corrupted.csv"
    ground_truth_csv = m1_dir / "ground_truth.csv"
    m1_summary_json = m1_dir / "injection_summary.json"
    m2_metrics_json = m2_dir / "detection_metrics.json"
    required = [clean_csv, corrupted_csv, ground_truth_csv, m1_summary_json, m2_metrics_json]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"M3 inputs are missing required files: {missing}")
    clean = pd.read_csv(clean_csv)
    corrupted = pd.read_csv(corrupted_csv)
    ground_truth = pd.read_csv(ground_truth_csv)
    m1_summary = json.loads(m1_summary_json.read_text(encoding="utf-8"))
    m2_metrics = json.loads(m2_metrics_json.read_text(encoding="utf-8"))
    return clean, corrupted, ground_truth, m1_summary, m2_metrics


def _repairable_issue_ids(m2_metrics: dict[str, Any]) -> list[str]:
    issue_ids: list[str] = []
    for issue in m2_metrics.get("scan_issues", []):
        if str(issue.get("issue_type")) not in REPAIRABLE_TYPES:
            continue
        issue_id = str(issue.get("issue_id") or "").strip()
        if issue_id and issue_id not in issue_ids:
            issue_ids.append(issue_id)
    if not issue_ids:
        raise ValueError("No repairable issue_ids found in M2 detection_metrics.json")
    return issue_ids


def _run_repair_batch(corrupted_csv: Path, output_csv: Path, issue_ids: list[str]) -> dict[str, Any]:
    payload = {
        "csv_path": str(corrupted_csv),
        "issue_ids": issue_ids,
        "scan_config": M3_SCAN_CONFIG,
        "repair_strategy": M3_REPAIR_STRATEGY,
        "plan_only": False,
        "write_output": True,
        "enable_rollback": False,
        "output_csv": str(output_csv),
    }
    return action_repair_batch(payload)


def _ground_truth_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "injection_id": str(record["injection_id"]),
        "anomaly_type": str(record["anomaly_type"]),
        "expected_issue_type": str(record["expected_issue_type"]),
        "row_id": str(record["row_id"]),
        "source_row_id": _value_text(record["source_row_id"]),
        "row_index": int(record["row_index"]),
        "column": str(record["column"]),
        "original_value": _value_text(record["original_value"]),
        "corrupted_value": _value_text(record["corrupted_value"]),
        "repairable": _to_bool(record["repairable"]),
        "notes": str(record["notes"]),
    }


def _evaluate_repairable_truth(
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[list[dict[str, Any]], set[tuple[int, str]]]:
    rows: list[dict[str, Any]] = []
    ground_truth_cells: set[tuple[int, str]] = set()

    for raw_record in ground_truth.sort_values("injection_id").to_dict(orient="records"):
        record = _ground_truth_record(raw_record)
        if not record["repairable"]:
            continue
        anomaly_type = record["anomaly_type"]
        column = record["column"]
        row_index = int(record["row_index"])
        if anomaly_type not in REPAIRABLE_TYPES:
            continue
        if column not in corrupted.columns or column not in repaired.columns:
            continue

        before = corrupted.at[row_index, column]
        after = repaired.at[row_index, column]
        original = record["original_value"]
        before_number = _number_or_none(before)
        after_number = _number_or_none(after)
        original_number = _number_or_none(original)
        before_error = None
        after_error = None
        if original_number is not None and before_number is not None:
            before_error = abs(before_number - original_number)
        if original_number is not None and after_number is not None:
            after_error = abs(after_number - original_number)

        changed = _cell_changed(before, after)
        exact_restored = _values_equal(original, after)
        improved = before_error is not None and after_error is not None and after_error < before_error
        improved_or_exact = exact_restored or improved
        if exact_restored:
            status = "exact_restored"
        elif improved:
            status = "improved_not_exact"
        elif changed:
            status = "changed_not_improved"
        else:
            status = "unchanged"

        ground_truth_cells.add((row_index, column))
        rows.append(
            {
                **record,
                "before_value": _value_text(before),
                "after_value": _value_text(after),
                "changed": changed,
                "exact_restored": exact_restored,
                "improved": improved,
                "improved_or_exact": improved_or_exact,
                "before_abs_error": None if before_error is None else round(before_error, 6),
                "after_abs_error": None if after_error is None else round(after_error, 6),
                "status": status,
            }
        )
    return rows, ground_truth_cells


def _skipped_non_repairable_truth(ground_truth: pd.DataFrame) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for raw_record in ground_truth.sort_values("injection_id").to_dict(orient="records"):
        record = _ground_truth_record(raw_record)
        if record["repairable"]:
            continue
        anomaly_type = record["anomaly_type"]
        if anomaly_type == "cross_column_consistency":
            reason = "manual_review_required"
        elif anomaly_type == "duplicate_record":
            reason = "manual_review_required"
        else:
            reason = "not_marked_repairable"
        skipped.append({**record, "skip_reason": reason})
    return skipped


def _changed_cells(corrupted: pd.DataFrame, repaired: pd.DataFrame, ground_truth_cells: set[tuple[int, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changed: list[dict[str, Any]] = []
    non_ground_truth: list[dict[str, Any]] = []
    for row_index in range(len(corrupted)):
        for column in corrupted.columns:
            if column not in repaired.columns:
                continue
            before = corrupted.at[row_index, column]
            after = repaired.at[row_index, column]
            if not _cell_changed(before, after):
                continue
            row = {
                "row_index": int(row_index),
                "column": str(column),
                "before": _value_text(before),
                "after": _value_text(after),
                "is_ground_truth_repairable_cell": (row_index, str(column)) in ground_truth_cells,
            }
            changed.append(row)
            if not row["is_ground_truth_repairable_cell"]:
                non_ground_truth.append(row)
    return changed, non_ground_truth


def _build_by_type_metrics(repairable_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for anomaly_type in REPAIRABLE_TYPES:
        rows = [row for row in repairable_rows if row["anomaly_type"] == anomaly_type]
        before_errors = [float(row["before_abs_error"]) for row in rows if row["before_abs_error"] is not None]
        after_errors = [float(row["after_abs_error"]) for row in rows if row["after_abs_error"] is not None]
        comparable_errors = [
            float(row["before_abs_error"]) - float(row["after_abs_error"])
            for row in rows
            if row["before_abs_error"] is not None and row["after_abs_error"] is not None
        ]
        by_type[anomaly_type] = {
            "ground_truth_count": len(rows),
            "changed_count": sum(1 for row in rows if row["changed"]),
            "exact_restored_count": sum(1 for row in rows if row["exact_restored"]),
            "improved_count": sum(1 for row in rows if row["improved"]),
            "improved_or_exact_count": sum(1 for row in rows if row["improved_or_exact"]),
            "unchanged_count": sum(1 for row in rows if not row["changed"]),
            "exact_restoration_rate": _rate(sum(1 for row in rows if row["exact_restored"]), len(rows)),
            "improved_or_exact_rate": _rate(sum(1 for row in rows if row["improved_or_exact"]), len(rows)),
            "mean_before_abs_error": _mean_or_none(before_errors),
            "mean_after_abs_error": _mean_or_none(after_errors),
            "mean_abs_error_delta": _mean_or_none(comparable_errors),
        }
    return by_type


def _count_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        anomaly_type = str(row["anomaly_type"])
        counts[anomaly_type] = counts.get(anomaly_type, 0) + 1
    return {key: counts.get(key, 0) for key in [*REPAIRABLE_TYPES, *NON_REPAIRABLE_TYPES]}


def _sanitize_repair_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_issue_count": int(result.get("selected_issue_count", 0)),
        "applied_issue_count": int(result.get("applied_issue_count", 0)),
        "total_cells_modified": int(result.get("total_cells_modified", 0)),
        "selected_issue_ids": list(result.get("selected_issue_ids", [])),
        "applied_repairs": result.get("applied_repairs", []),
        "skipped_issues": result.get("skipped_issues", []),
        "comparison": result.get("comparison", {}),
        "conflict_summary": result.get("conflict_summary", {}),
        "write_output": bool(result.get("write_output", False)),
        "output_csv": "repaired.csv",
    }


def _write_readme(output_dir: Path, metrics_doc: dict[str, Any]) -> None:
    overall = metrics_doc["metrics"]["overall"]
    by_type = metrics_doc["metrics"]["by_type"]
    skipped = metrics_doc["metrics"]["skipped_non_repairable_by_type"]
    comparison = metrics_doc["repair_batch"]["comparison"]
    rows = [
        "| Type | GT | Changed | Exact | Improved/Exact | Exact Rate | Improved/Exact Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for anomaly_type in REPAIRABLE_TYPES:
        row = by_type[anomaly_type]
        rows.append(
            f"| `{anomaly_type}` | {row['ground_truth_count']} | {row['changed_count']} | "
            f"{row['exact_restored_count']} | {row['improved_or_exact_count']} | "
            f"{row['exact_restoration_rate']:.6f} | {row['improved_or_exact_rate']:.6f} |"
        )
    rows.append(
        f"| **Overall** | {overall['repairable_ground_truth_count']} | {overall['repairable_changed_count']} | "
        f"{overall['exact_restored_count']} | {overall['improved_or_exact_count']} | "
        f"{overall['exact_restoration_rate']:.6f} | {overall['improved_or_exact_rate']:.6f} |"
    )

    text = f"""# M3 Stroke Repair Evaluation

This directory contains the M3 repair evaluation based on the M1 stroke experiment data and M2 detection output.

## Inputs

- M1 directory: `{metrics_doc["source"]["m1_dir"]}`
- M2 directory: `{metrics_doc["source"]["m2_dir"]}`
- Repaired data: `repaired.csv`
- Primary denominator: repairable M1 ground truth rows only

## Scoring Policy

- The primary repair success rate uses the 72 M1 rows marked `repairable=True`.
- `duplicate_record` and `cross_column_consistency` are reported as skipped/manual-review items because the current rule-based batch repair does not auto-repair them.
- Missing values, numeric outliers, and rare categories are evaluated by row index and column.
- Numeric repairs report before/after absolute error when both values are numeric.
- Extra changed cells outside repairable ground truth are counted as side effects, not successes.

## Repair Metrics

{chr(10).join(rows)}

## Before/After Scan Summary

- Before issue count: {comparison.get("before_issue_count")}
- After issue count: {comparison.get("after_issue_count")}
- Resolved issue count: {comparison.get("resolved_issue_count")}
- Total cells modified by `repair_batch`: {overall["total_cells_modified"]}
- Non-ground-truth cells modified: {overall["non_ground_truth_cells_modified"]}

## Skipped Manual-Review Items

- `cross_column_consistency`: {skipped.get("cross_column_consistency", 0)}
- `duplicate_record`: {skipped.get("duplicate_record", 0)}

## Notes

- M3 evaluates repair quality only. It does not tune detection thresholds or repair algorithms.
- The current numeric outlier detector produced false positives in M2; when those issue IDs are repaired, their resulting cell changes are recorded as side effects.
- Detailed per-row repair outcomes and side effects are listed in `repair_details.json`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def evaluate(m1_dir: Path, m2_dir: Path, output_dir: Path) -> dict[str, Any]:
    m1_dir = _resolve_path(m1_dir)
    m2_dir = _resolve_path(m2_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, corrupted, ground_truth, m1_summary, m2_metrics = _load_inputs(m1_dir, m2_dir)
    output_csv = output_dir / "repaired.csv"
    issue_ids = _repairable_issue_ids(m2_metrics)
    repair_result = _run_repair_batch(m1_dir / "corrupted.csv", output_csv, issue_ids)
    repaired = pd.read_csv(output_csv)

    repairable_rows, ground_truth_cells = _evaluate_repairable_truth(corrupted, repaired, ground_truth)
    skipped_rows = _skipped_non_repairable_truth(ground_truth)
    changed_cells, non_ground_truth_changes = _changed_cells(corrupted, repaired, ground_truth_cells)
    by_type = _build_by_type_metrics(repairable_rows)

    repairable_count = len(repairable_rows)
    exact_count = sum(1 for row in repairable_rows if row["exact_restored"])
    improved_or_exact_count = sum(1 for row in repairable_rows if row["improved_or_exact"])
    changed_count = sum(1 for row in repairable_rows if row["changed"])
    overall = {
        "total_ground_truth_count": int(len(ground_truth)),
        "repairable_ground_truth_count": repairable_count,
        "skipped_non_repairable_ground_truth_count": len(skipped_rows),
        "repairable_changed_count": changed_count,
        "exact_restored_count": exact_count,
        "improved_or_exact_count": improved_or_exact_count,
        "exact_restoration_rate": _rate(exact_count, repairable_count),
        "improved_or_exact_rate": _rate(improved_or_exact_count, repairable_count),
        "total_cells_modified": int(repair_result.get("total_cells_modified", len(changed_cells))),
        "changed_cells_observed": len(changed_cells),
        "non_ground_truth_cells_modified": len(non_ground_truth_changes),
    }

    metrics_doc = {
        "milestone": "M3",
        "dataset": "m3_stroke_repair",
        "source": {
            "m1_dir": _display_path(m1_dir),
            "m2_dir": _display_path(m2_dir),
            "clean_csv": "clean.csv",
            "corrupted_csv": "corrupted.csv",
            "ground_truth_csv": "ground_truth.csv",
            "m2_metrics_json": "detection_metrics.json",
        },
        "data_profile": {
            "clean_rows": int(len(clean)),
            "clean_columns": int(len(clean.columns)),
            "corrupted_rows": int(len(corrupted)),
            "corrupted_columns": int(len(corrupted.columns)),
            "repaired_rows": int(len(repaired)),
            "repaired_columns": int(len(repaired.columns)),
            "ground_truth_rows": int(len(ground_truth)),
            "m1_injection_counts_by_type": m1_summary.get("injection_counts_by_type", {}),
        },
        "scoring_policy": {
            "primary_denominator": "M1 ground_truth rows where repairable=True",
            "repairable_types": REPAIRABLE_TYPES,
            "manual_review_types": NON_REPAIRABLE_TYPES,
            "numeric_metric": "before/after absolute error against original_value when numeric values are available",
            "side_effect_policy": "changed cells outside repairable ground truth are counted separately",
        },
        "scan_config": M3_SCAN_CONFIG,
        "repair_strategy": M3_REPAIR_STRATEGY,
        "selected_issue_ids": issue_ids,
        "repair_batch": _sanitize_repair_result(repair_result),
        "metrics": {
            "overall": overall,
            "by_type": by_type,
            "ground_truth_by_type": _count_by_type([_ground_truth_record(row) for row in ground_truth.to_dict(orient="records")]),
            "skipped_non_repairable_by_type": _count_by_type(skipped_rows),
        },
        "notes": [
            "M3 evaluates current repair behavior only; it does not change detector thresholds or repair algorithms.",
            "The primary success rate excludes duplicate and cross-column consistency rows because they are not supported by current automatic repair.",
            "Numeric outlier false-positive side effects from M2 are recorded as non-ground-truth cell changes.",
        ],
    }

    details_doc = {
        "milestone": "M3",
        "dataset": "m3_stroke_repair",
        "summary": {
            "repairable_truth_rows": len(repairable_rows),
            "skipped_non_repairable_truth_rows": len(skipped_rows),
            "changed_cell_rows": len(changed_cells),
            "non_ground_truth_changed_cell_rows": len(non_ground_truth_changes),
        },
        "repairable_truth_results": repairable_rows,
        "skipped_non_repairable_truth": skipped_rows,
        "changed_cells": changed_cells,
        "non_ground_truth_changed_cells": non_ground_truth_changes,
    }

    (output_dir / "repair_metrics.json").write_text(
        json.dumps(_to_builtin(metrics_doc), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "repair_details.json").write_text(
        json.dumps(_to_builtin(details_doc), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_readme(output_dir, metrics_doc)
    return _to_builtin(metrics_doc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate M3 repair quality against M1/M2 experiment data.")
    parser.add_argument("--m1-dir", default=str(DEFAULT_M1_DIR), help="M1 experiment data directory.")
    parser.add_argument("--m2-dir", default=str(DEFAULT_M2_DIR), help="M2 detection evaluation directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="M3 output directory.")
    args = parser.parse_args()

    metrics = evaluate(Path(args.m1_dir), Path(args.m2_dir), Path(args.output_dir))
    print(json.dumps(metrics["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
