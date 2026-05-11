"""Evaluate rule, Gower, and hybrid repair strategies on the M1 stroke data.

This is an R5 side experiment. It does not modify M1-M3 experiment outputs and
does not change the Python engine action protocol. The hybrid strategy is a
deterministic approximation of the Auto Agent mock planner: it previews rule and
Gower repairs, chooses the better source per issue, then executes the selected
sources in sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENGINE_DIR = PROJECT_ROOT / "appshell" / "core" / "python_engine"
DEFAULT_M1_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "r5_repair_strategy_comparison"

if str(PYTHON_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_ENGINE_DIR))

from engine_core import action_repair_batch, action_repair_with_gower, action_scan_file, _to_builtin  # noqa: E402


REPAIRABLE_TYPES = ["missing_values", "numeric_outlier", "rare_category"]
NON_REPAIRABLE_TYPES = ["cross_column_consistency", "duplicate_record"]
R5_SCAN_CONFIG = {
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
R5_REPAIR_STRATEGY = {
    "conflict_policy": "first_wins",
    "issue_priority": REPAIRABLE_TYPES,
    "missing_numeric": "median",
    "missing_categorical": "mode",
    "outlier": "clip",
    "rare_category": "mode",
    "preview_limit": 20,
}
R5_GOWER_STRATEGY = {
    "k_neighbors": 5,
    "weight_mode": "uniform",
    "max_candidates": 512,
    "preview_limit": 20,
}


def _resolve_path(path_text: str | Path) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
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


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _int_from_any(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _float_from_any(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _load_inputs(m1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_csv = m1_dir / "clean.csv"
    corrupted_csv = m1_dir / "corrupted.csv"
    ground_truth_csv = m1_dir / "ground_truth.csv"
    required = [clean_csv, corrupted_csv, ground_truth_csv]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R5 inputs are missing required files: {missing}")
    return pd.read_csv(clean_csv), pd.read_csv(corrupted_csv), pd.read_csv(ground_truth_csv)


def _scan(csv_path: Path) -> dict[str, Any]:
    return action_scan_file({"csv_path": str(csv_path), "scan_config": R5_SCAN_CONFIG})


def _repairable_issue_ids(scan_result: dict[str, Any]) -> list[str]:
    issue_ids: list[str] = []
    for issue in scan_result.get("issues", []):
        if str(issue.get("issue_type")) not in REPAIRABLE_TYPES:
            continue
        issue_id = str(issue.get("issue_id") or "").strip()
        if issue_id and issue_id not in issue_ids:
            issue_ids.append(issue_id)
    if not issue_ids:
        raise ValueError("No repairable issue_ids found in baseline scan")
    return issue_ids


def _ground_truth_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "injection_id": str(record.get("injection_id", "")),
        "anomaly_type": str(record.get("anomaly_type", "")),
        "expected_issue_type": str(record.get("expected_issue_type", "")),
        "row_id": str(record.get("row_id", "")),
        "source_row_id": _value_text(record.get("source_row_id", "")),
        "row_index": int(record.get("row_index", 0)),
        "column": str(record.get("column", "")),
        "original_value": _value_text(record.get("original_value", "")),
        "corrupted_value": _value_text(record.get("corrupted_value", "")),
        "repairable": _to_bool(record.get("repairable", False)),
        "notes": str(record.get("notes", "")),
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
            }
        )
    return rows, ground_truth_cells


def _changed_cells(
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    ground_truth_cells: set[tuple[int, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def compute_strategy_metrics(
    strategy_name: str,
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    ground_truth: pd.DataFrame,
    before_scan: dict[str, Any],
    after_scan: dict[str, Any],
    repair_result: dict[str, Any],
    output_csv: Path,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute the R5 comparison metrics for one repaired dataframe."""

    repairable_rows, ground_truth_cells = _evaluate_repairable_truth(corrupted, repaired, ground_truth)
    changed_cells, non_ground_truth_changes = _changed_cells(corrupted, repaired, ground_truth_cells)
    exact_count = sum(1 for row in repairable_rows if row["exact_restored"])
    improved_or_exact_count = sum(1 for row in repairable_rows if row["improved_or_exact"])
    before_issue_count = int(before_scan.get("issue_count", len(before_scan.get("issues", []))))
    after_issue_count = int(after_scan.get("issue_count", len(after_scan.get("issues", []))))
    resolved_issue_count = max(0, before_issue_count - after_issue_count)
    skipped_issue_count = len(repair_result.get("skipped_issues", []))

    by_type: dict[str, dict[str, Any]] = {}
    for issue_type in REPAIRABLE_TYPES:
        rows = [row for row in repairable_rows if row["anomaly_type"] == issue_type]
        by_type[issue_type] = {
            "ground_truth_count": len(rows),
            "changed_count": sum(1 for row in rows if row["changed"]),
            "exact_restored_count": sum(1 for row in rows if row["exact_restored"]),
            "improved_or_exact_count": sum(1 for row in rows if row["improved_or_exact"]),
            "exact_restoration_rate": _rate(sum(1 for row in rows if row["exact_restored"]), len(rows)),
            "improved_or_exact_rate": _rate(sum(1 for row in rows if row["improved_or_exact"]), len(rows)),
        }

    return {
        "strategy": strategy_name,
        "status": "ok",
        "output_csv": _display_path(output_csv),
        "before_issue_count": before_issue_count,
        "after_issue_count": after_issue_count,
        "resolved_issue_count": resolved_issue_count,
        "total_cells_modified": len(changed_cells),
        "engine_total_cells_modified": _int_from_any(repair_result.get("total_cells_modified")),
        "exact_restored_count": exact_count,
        "exact_restoration_rate": _rate(exact_count, len(repairable_rows)),
        "improved_or_exact_count": improved_or_exact_count,
        "improved_or_exact_rate": _rate(improved_or_exact_count, len(repairable_rows)),
        "non_ground_truth_cells_modified": len(non_ground_truth_changes),
        "skipped_issue_count": skipped_issue_count,
        "repairable_ground_truth_count": len(repairable_rows),
        "by_type": by_type,
        "notes": notes or [],
        "repair_summary": {
            "selected_issue_count": _int_from_any(repair_result.get("selected_issue_count")),
            "applied_issue_count": _int_from_any(repair_result.get("applied_issue_count")),
            "selected_issue_ids": list(repair_result.get("selected_issue_ids", [])),
            "applied_repairs": repair_result.get("applied_repairs", []),
            "skipped_issues": repair_result.get("skipped_issues", []),
            "comparison": repair_result.get("comparison", {}),
            "issue_source_map": repair_result.get("issue_source_map", {}),
            "execution_steps": repair_result.get("execution_steps", []),
        },
    }


def failed_strategy_metrics(strategy_name: str, before_scan: dict[str, Any], error: Exception) -> dict[str, Any]:
    before_issue_count = int(before_scan.get("issue_count", len(before_scan.get("issues", []))))
    return {
        "strategy": strategy_name,
        "status": "failed",
        "output_csv": None,
        "before_issue_count": before_issue_count,
        "after_issue_count": None,
        "resolved_issue_count": None,
        "total_cells_modified": None,
        "engine_total_cells_modified": None,
        "exact_restored_count": None,
        "exact_restoration_rate": None,
        "improved_or_exact_count": None,
        "improved_or_exact_rate": None,
        "non_ground_truth_cells_modified": None,
        "skipped_issue_count": None,
        "repairable_ground_truth_count": None,
        "by_type": {},
        "notes": [f"{strategy_name} failed: {error}"],
        "error": str(error),
    }


def _repair_batch_payload(csv_path: Path, issue_ids: list[str], output_csv: Path, *, plan_only: bool) -> dict[str, Any]:
    return {
        "csv_path": str(csv_path),
        "issue_ids": issue_ids,
        "scan_config": R5_SCAN_CONFIG,
        "repair_strategy": R5_REPAIR_STRATEGY,
        "plan_only": plan_only,
        "write_output": not plan_only,
        "enable_rollback": False,
        "output_csv": str(output_csv),
    }


def _gower_payload(csv_path: Path, issue_ids: list[str], output_csv: Path, *, plan_only: bool) -> dict[str, Any]:
    return {
        "csv_path": str(csv_path),
        "issue_ids": issue_ids,
        "scan_config": R5_SCAN_CONFIG,
        "gower_strategy": R5_GOWER_STRATEGY,
        "plan_only": plan_only,
        "write_output": not plan_only,
        "enable_rollback": False,
        "output_csv": str(output_csv),
    }


def _run_rule_only(corrupted_csv: Path, output_csv: Path, issue_ids: list[str]) -> dict[str, Any]:
    return action_repair_batch(_repair_batch_payload(corrupted_csv, issue_ids, output_csv, plan_only=False))


def _run_gower_only(corrupted_csv: Path, output_csv: Path, issue_ids: list[str]) -> dict[str, Any]:
    return action_repair_with_gower(_gower_payload(corrupted_csv, issue_ids, output_csv, plan_only=False))


def _issue_metrics_from_preview(preview: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    metrics: dict[str, dict[str, Any]] = {}
    skipped: set[str] = set()
    for item in preview.get("applied_repairs", []):
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id") or "").strip()
        if issue_id:
            metrics[issue_id] = dict(item)
    for item in preview.get("neighbor_evidence", []):
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id") or "").strip()
        if not issue_id:
            continue
        existing = metrics.setdefault(issue_id, {})
        existing.update(item)
    for item in preview.get("skipped_issues", []):
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id") or "").strip()
        if issue_id:
            skipped.add(issue_id)
    return metrics, skipped


def _issue_rows_touched(issue: dict[str, Any] | None) -> int:
    if not issue:
        return 0
    return _int_from_any(issue.get("rows_touched"))


def _issue_confidence(issue: dict[str, Any] | None) -> float:
    if not issue:
        return 0.0
    return _float_from_any(issue.get("candidate_confidence"))


def choose_issue_source(
    rule_issue: dict[str, Any] | None,
    has_rule: bool,
    gower_issue: dict[str, Any] | None,
    has_gower: bool,
) -> str:
    """Mirror appshell/backend/internal/agent/mock_planner.go chooseIssueSource."""

    if has_rule and not has_gower:
        return "rule"
    if has_gower and not has_rule:
        return "gower"
    if not has_rule and not has_gower:
        return ""

    rule_resolved = _int_from_any((rule_issue or {}).get("resolved_count"))
    gower_resolved = _int_from_any((gower_issue or {}).get("resolved_count"))
    if rule_resolved > gower_resolved:
        return "rule"
    if gower_resolved > rule_resolved:
        return "gower"

    rule_confidence = _issue_confidence(rule_issue)
    gower_confidence = _issue_confidence(gower_issue)
    if rule_confidence > gower_confidence:
        return "rule"
    if gower_confidence > rule_confidence:
        return "gower"

    rule_rows = _issue_rows_touched(rule_issue)
    gower_rows = _issue_rows_touched(gower_issue)
    if rule_rows < gower_rows:
        return "rule"
    if gower_rows < rule_rows:
        return "gower"
    return "rule"


def _run_hybrid(corrupted_csv: Path, output_csv: Path, issue_ids: list[str]) -> dict[str, Any]:
    preview_csv = output_csv.with_suffix(".preview.csv")
    rule_preview = action_repair_batch(_repair_batch_payload(corrupted_csv, issue_ids, preview_csv, plan_only=True))
    gower_preview = action_repair_with_gower(_gower_payload(corrupted_csv, issue_ids, preview_csv, plan_only=True))

    rule_issues, rule_skipped = _issue_metrics_from_preview(rule_preview)
    gower_issues, gower_skipped = _issue_metrics_from_preview(gower_preview)
    issue_source_map: dict[str, str] = {}
    for issue_id in issue_ids:
        has_rule = issue_id in rule_issues and issue_id not in rule_skipped
        has_gower = issue_id in gower_issues and issue_id not in gower_skipped
        source = choose_issue_source(rule_issues.get(issue_id), has_rule, gower_issues.get(issue_id), has_gower)
        if source:
            issue_source_map[issue_id] = source

    hybrid_rule_ids = [issue_id for issue_id in issue_ids if issue_source_map.get(issue_id) == "rule"]
    hybrid_gower_ids = [issue_id for issue_id in issue_ids if issue_source_map.get(issue_id) == "gower"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    execution_steps: list[dict[str, Any]] = []
    applied_repairs: list[dict[str, Any]] = []
    skipped_issues: list[dict[str, Any]] = []
    current_csv = corrupted_csv

    if not issue_source_map:
        shutil.copy2(corrupted_csv, output_csv)
    else:
        with tempfile.TemporaryDirectory(prefix="r5-hybrid-") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            if hybrid_rule_ids:
                rule_output = temp_dir / "hybrid_step_1_rule.csv" if hybrid_gower_ids else output_csv
                rule_result = action_repair_batch(
                    _repair_batch_payload(current_csv, hybrid_rule_ids, rule_output, plan_only=False)
                )
                current_csv = Path(str(rule_result.get("output_csv") or rule_output)).resolve()
                applied_repairs.extend(rule_result.get("applied_repairs", []))
                skipped_issues.extend(rule_result.get("skipped_issues", []))
                execution_steps.append(
                    {
                        "step": len(execution_steps) + 1,
                        "tool_id": "engine.repair_batch",
                        "selected_issue_ids": hybrid_rule_ids,
                        "output_csv": _display_path(current_csv),
                        "comparison": rule_result.get("comparison", {}),
                    }
                )
            if hybrid_gower_ids:
                gower_result = action_repair_with_gower(
                    _gower_payload(current_csv, hybrid_gower_ids, output_csv, plan_only=False)
                )
                current_csv = Path(str(gower_result.get("output_csv") or output_csv)).resolve()
                applied_repairs.extend(gower_result.get("applied_repairs", []))
                skipped_issues.extend(gower_result.get("skipped_issues", []))
                execution_steps.append(
                    {
                        "step": len(execution_steps) + 1,
                        "tool_id": "engine.repair_with_gower",
                        "selected_issue_ids": hybrid_gower_ids,
                        "output_csv": _display_path(current_csv),
                        "comparison": gower_result.get("comparison", {}),
                    }
                )
            if current_csv.resolve() != output_csv.resolve() and current_csv.exists():
                shutil.copy2(current_csv, output_csv)

    return {
        "selected_issue_ids": list(issue_source_map),
        "selected_issue_count": len(issue_source_map),
        "applied_issue_count": len(applied_repairs),
        "applied_repairs": applied_repairs,
        "skipped_issues": skipped_issues,
        "issue_source_map": issue_source_map,
        "execution_steps": execution_steps,
        "comparison": {
            "source": "deterministic_auto_agent_hybrid_approximation",
            "rule_issue_count": len(hybrid_rule_ids),
            "gower_issue_count": len(hybrid_gower_ids),
        },
        "total_cells_modified": sum(_issue_rows_touched(item) for item in applied_repairs if isinstance(item, dict)),
        "output_csv": str(output_csv),
    }


def _run_and_score_strategy(
    strategy_name: str,
    corrupted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    corrupted_csv: Path,
    strategy_dir: Path,
    issue_ids: list[str],
    before_scan: dict[str, Any],
) -> dict[str, Any]:
    output_csv = strategy_dir / "repaired.csv"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    try:
        if strategy_name == "rule-only":
            repair_result = _run_rule_only(corrupted_csv, output_csv, issue_ids)
            notes = ["rule-only uses engine.repair_batch with the shared R5 scan config."]
        elif strategy_name == "gower-only":
            repair_result = _run_gower_only(corrupted_csv, output_csv, issue_ids)
            notes = ["gower-only uses engine.repair_with_gower with k=5 and max_candidates=512."]
        elif strategy_name == "hybrid":
            repair_result = _run_hybrid(corrupted_csv, output_csv, issue_ids)
            notes = [
                "hybrid is a deterministic Auto Agent approximation, not a full CLI session.",
                "issue source selection mirrors mock_planner.go: resolved count, confidence, rows touched, then rule.",
            ]
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        repaired = pd.read_csv(output_csv)
        after_scan = _scan(output_csv)
        return compute_strategy_metrics(
            strategy_name,
            corrupted,
            repaired,
            ground_truth,
            before_scan,
            after_scan,
            repair_result,
            output_csv,
            notes,
        )
    except Exception as exc:
        return failed_strategy_metrics(strategy_name, before_scan, exc)


def build_report(metrics_doc: dict[str, Any]) -> str:
    rows = [
        "| Strategy | Status | Before | After | Resolved | Modified Cells | Exact | Exact Rate | Improved/Exact | Improved/Exact Rate | Non-GT Modified | Skipped |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in metrics_doc.get("strategies", {}).values():
        rows.append(
            "| {strategy} | {status} | {before} | {after} | {resolved} | {modified} | {exact} | {exact_rate} | "
            "{improved} | {improved_rate} | {non_gt} | {skipped} |".format(
                strategy=strategy.get("strategy", ""),
                status=strategy.get("status", ""),
                before=_format_cell(strategy.get("before_issue_count")),
                after=_format_cell(strategy.get("after_issue_count")),
                resolved=_format_cell(strategy.get("resolved_issue_count")),
                modified=_format_cell(strategy.get("total_cells_modified")),
                exact=_format_cell(strategy.get("exact_restored_count")),
                exact_rate=_format_rate(strategy.get("exact_restoration_rate")),
                improved=_format_cell(strategy.get("improved_or_exact_count")),
                improved_rate=_format_rate(strategy.get("improved_or_exact_rate")),
                non_gt=_format_cell(strategy.get("non_ground_truth_cells_modified")),
                skipped=_format_cell(strategy.get("skipped_issue_count")),
            )
        )

    notes: list[str] = []
    for strategy in metrics_doc.get("strategies", {}).values():
        for note in strategy.get("notes", []):
            notes.append(f"- `{strategy.get('strategy')}`: {note}")
        if strategy.get("status") == "failed" and strategy.get("error"):
            notes.append(f"- `{strategy.get('strategy')}` failure reason: {strategy['error']}")

    source = metrics_doc.get("source", {})
    selected_ids = metrics_doc.get("selected_issue_ids", [])
    return f"""# R5 Repair Strategy Comparison

This report compares rule-only, gower-only, and hybrid repair strategies on the same M1 corrupted CSV.

## Inputs

- Clean CSV: `{source.get("clean_csv", "")}`
- Corrupted CSV: `{source.get("corrupted_csv", "")}`
- Ground truth CSV: `{source.get("ground_truth_csv", "")}`
- Repairable issue IDs selected: {len(selected_ids)}

## Scoring Policy

- The primary denominator is M1 `ground_truth.csv` rows where `repairable=True` and the type is one of `{", ".join(REPAIRABLE_TYPES)}`.
- `exact_restored_count` requires the repaired value to match `original_value`.
- `improved_or_exact_count` also counts numeric repairs that reduce absolute error versus the corrupted value.
- Changed cells outside repairable ground-truth cells are counted as side effects.
- Hybrid is a deterministic Auto Agent approximation; it does not run a full Go CLI session.

## Metrics

{chr(10).join(rows)}

## Notes

{chr(10).join(notes) if notes else "- No additional notes."}
"""


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _format_rate(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def evaluate(m1_dir: Path, output_dir: Path) -> dict[str, Any]:
    m1_dir = _resolve_path(m1_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, corrupted, ground_truth = _load_inputs(m1_dir)
    corrupted_csv = m1_dir / "corrupted.csv"
    before_scan = _scan(corrupted_csv)
    issue_ids = _repairable_issue_ids(before_scan)

    strategies: dict[str, dict[str, Any]] = {}
    for strategy_name in ["rule-only", "gower-only", "hybrid"]:
        strategies[strategy_name] = _run_and_score_strategy(
            strategy_name=strategy_name,
            corrupted=corrupted,
            ground_truth=ground_truth,
            corrupted_csv=corrupted_csv,
            strategy_dir=output_dir / strategy_name,
            issue_ids=issue_ids,
            before_scan=before_scan,
        )

    metrics_doc = {
        "milestone": "R5",
        "dataset": "r5_repair_strategy_comparison",
        "source": {
            "m1_dir": _display_path(m1_dir),
            "clean_csv": _display_path(m1_dir / "clean.csv"),
            "corrupted_csv": _display_path(corrupted_csv),
            "ground_truth_csv": _display_path(m1_dir / "ground_truth.csv"),
        },
        "data_profile": {
            "clean_rows": int(len(clean)),
            "clean_columns": int(len(clean.columns)),
            "corrupted_rows": int(len(corrupted)),
            "corrupted_columns": int(len(corrupted.columns)),
            "ground_truth_rows": int(len(ground_truth)),
        },
        "scan_config": R5_SCAN_CONFIG,
        "repair_strategy": R5_REPAIR_STRATEGY,
        "gower_strategy": R5_GOWER_STRATEGY,
        "selected_issue_ids": issue_ids,
        "strategies": strategies,
        "notes": [
            "R5 is a side experiment and does not overwrite M1-M3 outputs.",
            "All metrics are computed from actual repaired CSV outputs.",
            "Failed strategies are reported with failure reasons instead of fabricated metrics.",
        ],
    }

    metrics_path = output_dir / "strategy_comparison_metrics.json"
    report_path = output_dir / "strategy_comparison_report.md"
    metrics_path.write_text(
        json.dumps(_to_builtin(metrics_doc), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_path.write_text(build_report(metrics_doc), encoding="utf-8", newline="\n")
    return _to_builtin(metrics_doc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rule, Gower, and hybrid repair strategies on M1 data.")
    parser.add_argument("--m1-dir", default=str(DEFAULT_M1_DIR), help="M1 experiment data directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="R5 output directory.")
    args = parser.parse_args()

    metrics = evaluate(Path(args.m1_dir), Path(args.output_dir))
    summary = {
        name: {
            "status": result.get("status"),
            "resolved_issue_count": result.get("resolved_issue_count"),
            "total_cells_modified": result.get("total_cells_modified"),
            "exact_restoration_rate": result.get("exact_restoration_rate"),
            "improved_or_exact_rate": result.get("improved_or_exact_rate"),
            "non_ground_truth_cells_modified": result.get("non_ground_truth_cells_modified"),
        }
        for name, result in metrics["strategies"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
