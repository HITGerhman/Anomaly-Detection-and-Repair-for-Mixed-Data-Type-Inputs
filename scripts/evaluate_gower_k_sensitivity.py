"""Evaluate repair_with_gower sensitivity to k_neighbors on the M1 stroke data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENGINE_DIR = PROJECT_ROOT / "appshell" / "core" / "python_engine"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_M1_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "r6_gower_k_sensitivity"
DEFAULT_K_VALUES = [3, 5, 7, 9, 15]
DEFAULT_K = 5

for path in [PYTHON_ENGINE_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engine_core import action_repair_with_gower, _to_builtin  # noqa: E402
from evaluate_repair_strategy_comparison import (  # noqa: E402
    R5_GOWER_STRATEGY,
    R5_SCAN_CONFIG,
    _display_path,
    _format_cell,
    _format_rate,
    _float_from_any,
    _load_inputs,
    _repairable_issue_ids,
    _scan,
    compute_strategy_metrics,
)


def _rate_value(value: Any) -> float:
    if value is None:
        return 0.0
    return _float_from_any(value)


def mean_neighbor_confidence(repair_result: dict[str, Any]) -> float | None:
    confidences: list[float] = []
    for source_name in ["neighbor_evidence", "applied_repairs"]:
        for item in repair_result.get(source_name, []):
            if not isinstance(item, dict):
                continue
            if "candidate_confidence" not in item:
                continue
            confidences.append(_float_from_any(item.get("candidate_confidence")))
        if confidences:
            break
    if not confidences:
        return None
    return round(sum(confidences) / float(len(confidences)), 6)


def _gower_payload(csv_path: Path, issue_ids: list[str], output_csv: Path, k_neighbors: int) -> dict[str, Any]:
    strategy = dict(R5_GOWER_STRATEGY)
    strategy["k_neighbors"] = int(k_neighbors)
    return {
        "csv_path": str(csv_path),
        "issue_ids": issue_ids,
        "scan_config": R5_SCAN_CONFIG,
        "gower_strategy": strategy,
        "plan_only": False,
        "write_output": True,
        "enable_rollback": False,
        "output_csv": str(output_csv),
    }


def build_k_metrics(
    k_neighbors: int,
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    ground_truth: pd.DataFrame,
    before_scan: dict[str, Any],
    after_scan: dict[str, Any],
    repair_result: dict[str, Any],
    output_csv: Path,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    metrics = compute_strategy_metrics(
        f"gower-k-{k_neighbors}",
        corrupted,
        repaired,
        ground_truth,
        before_scan,
        after_scan,
        repair_result,
        output_csv,
        notes or [f"repair_with_gower executed with k_neighbors={k_neighbors}."],
    )
    metrics["k_neighbors"] = int(k_neighbors)
    metrics["mean_neighbor_confidence"] = mean_neighbor_confidence(repair_result)
    return metrics


def failed_k_metrics(k_neighbors: int, before_scan: dict[str, Any], error: Exception) -> dict[str, Any]:
    before_issue_count = int(before_scan.get("issue_count", len(before_scan.get("issues", []))))
    return {
        "strategy": f"gower-k-{k_neighbors}",
        "k_neighbors": int(k_neighbors),
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
        "mean_neighbor_confidence": None,
        "skipped_issue_count": None,
        "repairable_ground_truth_count": None,
        "by_type": {},
        "notes": [f"k={k_neighbors} failed: {error}"],
        "error": str(error),
    }


def assess_default_k(k_results: dict[str, dict[str, Any]], default_k: int = DEFAULT_K) -> dict[str, Any]:
    successful = [item for item in k_results.values() if item.get("status") == "ok"]
    default_result = k_results.get(str(default_k))
    if not successful or not default_result or default_result.get("status") != "ok":
        return {
            "default_k": default_k,
            "supports_default_k": False,
            "reason": f"K={default_k} did not complete successfully, so the experiment cannot support it.",
        }

    best_quality = max(_rate_value(item.get("improved_or_exact_rate")) for item in successful)
    min_side_effects = min(int(item.get("non_ground_truth_cells_modified") or 0) for item in successful)
    max_resolved = max(int(item.get("resolved_issue_count") or 0) for item in successful)
    default_quality = _rate_value(default_result.get("improved_or_exact_rate"))
    default_side_effects = int(default_result.get("non_ground_truth_cells_modified") or 0)
    default_resolved = int(default_result.get("resolved_issue_count") or 0)

    quality_gap = round(best_quality - default_quality, 6)
    side_effect_gap = default_side_effects - min_side_effects
    resolved_gap = max_resolved - default_resolved
    supports = quality_gap <= 0.02 and side_effect_gap <= 10 and resolved_gap <= 1
    if supports:
        reason = (
            f"K={default_k} is within 2 percentage points of the best improved/exact rate, "
            "does not add meaningful side effects, and resolves nearly as many issues as the best K."
        )
    else:
        reason = (
            f"K={default_k} is not the strongest compromise in this run: "
            f"quality_gap={quality_gap}, side_effect_gap={side_effect_gap}, resolved_gap={resolved_gap}."
        )

    return {
        "default_k": default_k,
        "supports_default_k": supports,
        "best_improved_or_exact_rate": round(best_quality, 6),
        "default_improved_or_exact_rate": round(default_quality, 6),
        "quality_gap": quality_gap,
        "min_non_ground_truth_cells_modified": min_side_effects,
        "default_non_ground_truth_cells_modified": default_side_effects,
        "side_effect_gap": side_effect_gap,
        "max_resolved_issue_count": max_resolved,
        "default_resolved_issue_count": default_resolved,
        "resolved_gap": resolved_gap,
        "reason": reason,
    }


def _run_and_score_k(
    k_neighbors: int,
    corrupted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    corrupted_csv: Path,
    output_dir: Path,
    issue_ids: list[str],
    before_scan: dict[str, Any],
) -> dict[str, Any]:
    k_dir = output_dir / f"k_{k_neighbors}"
    output_csv = k_dir / "repaired.csv"
    k_dir.mkdir(parents=True, exist_ok=True)
    try:
        repair_result = action_repair_with_gower(_gower_payload(corrupted_csv, issue_ids, output_csv, k_neighbors))
        repaired = pd.read_csv(output_csv)
        after_scan = _scan(output_csv)
        return build_k_metrics(
            k_neighbors,
            corrupted,
            repaired,
            ground_truth,
            before_scan,
            after_scan,
            repair_result,
            output_csv,
        )
    except Exception as exc:
        return failed_k_metrics(k_neighbors, before_scan, exc)


def _sorted_k_results(metrics_doc: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        metrics_doc.get("k_results", {}).values(),
        key=lambda item: int(item.get("k_neighbors", 0)),
    )


def build_report(metrics_doc: dict[str, Any]) -> str:
    rows = [
        "| K | Status | Before | After | Resolved | Modified Cells | Exact | Exact Rate | Improved/Exact | Improved/Exact Rate | Non-GT Modified | Mean Confidence |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in _sorted_k_results(metrics_doc):
        rows.append(
            "| {k} | {status} | {before} | {after} | {resolved} | {modified} | {exact} | {exact_rate} | "
            "{improved} | {improved_rate} | {non_gt} | {confidence} |".format(
                k=item.get("k_neighbors"),
                status=item.get("status"),
                before=_format_cell(item.get("before_issue_count")),
                after=_format_cell(item.get("after_issue_count")),
                resolved=_format_cell(item.get("resolved_issue_count")),
                modified=_format_cell(item.get("total_cells_modified")),
                exact=_format_cell(item.get("exact_restored_count")),
                exact_rate=_format_rate(item.get("exact_restoration_rate")),
                improved=_format_cell(item.get("improved_or_exact_count")),
                improved_rate=_format_rate(item.get("improved_or_exact_rate")),
                non_gt=_format_cell(item.get("non_ground_truth_cells_modified")),
                confidence=_format_rate(item.get("mean_neighbor_confidence")),
            )
        )

    notes: list[str] = []
    for item in _sorted_k_results(metrics_doc):
        for note in item.get("notes", []):
            notes.append(f"- `K={item.get('k_neighbors')}`: {note}")
        if item.get("status") == "failed" and item.get("error"):
            notes.append(f"- `K={item.get('k_neighbors')}` failure reason: {item['error']}")

    source = metrics_doc.get("source", {})
    assessment = metrics_doc.get("default_k_assessment", {})
    support_text = "Yes" if assessment.get("supports_default_k") else "No"
    return f"""# R6 Gower K Sensitivity

This report evaluates `repair_with_gower` with several `k_neighbors` values on the same M1 corrupted CSV.

## Inputs

- Clean CSV: `{source.get("clean_csv", "")}`
- Corrupted CSV: `{source.get("corrupted_csv", "")}`
- Ground truth CSV: `{source.get("ground_truth_csv", "")}`
- Repairable issue IDs selected: {len(metrics_doc.get("selected_issue_ids", []))}

## Metrics

{chr(10).join(rows)}

## Default K Assessment

- Continue default `K=5`: **{support_text}**
- Reason: {assessment.get("reason", "")}

## Interpretation

- If K is too small, a single neighbor can dominate the candidate value, so the repair is more sensitive to local noise and may be less stable.
- If K is too large, less similar rows enter the neighbor set, pushing numeric medians and categorical modes toward global population behavior and weakening local similarity.
- This project should not use `sqrt(n)` directly because Gower KNN here is not a standard KNN classifier. It is used to generate repair candidates for mixed-type data, where local similarity matters more than broad voting coverage. On a dataset with several thousand rows, `sqrt(n)` would make K much larger than the local neighborhood needed for repair.

## Notes

{chr(10).join(notes) if notes else "- No additional notes."}
"""


def evaluate(m1_dir: Path, output_dir: Path, k_values: list[int] | None = None) -> dict[str, Any]:
    m1_dir = Path(m1_dir).expanduser()
    if not m1_dir.is_absolute():
        m1_dir = (PROJECT_ROOT / m1_dir).resolve()
    else:
        m1_dir = m1_dir.resolve()
    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_k_values = [int(value) for value in (k_values or DEFAULT_K_VALUES)]
    clean, corrupted, ground_truth = _load_inputs(m1_dir)
    corrupted_csv = m1_dir / "corrupted.csv"
    before_scan = _scan(corrupted_csv)
    issue_ids = _repairable_issue_ids(before_scan)

    k_results: dict[str, dict[str, Any]] = {}
    for k_neighbors in resolved_k_values:
        k_results[str(k_neighbors)] = _run_and_score_k(
            k_neighbors,
            corrupted,
            ground_truth,
            corrupted_csv,
            output_dir,
            issue_ids,
            before_scan,
        )

    metrics_doc = {
        "milestone": "R6",
        "dataset": "r6_gower_k_sensitivity",
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
        "gower_strategy_base": {key: value for key, value in R5_GOWER_STRATEGY.items() if key != "k_neighbors"},
        "k_values": resolved_k_values,
        "selected_issue_ids": issue_ids,
        "k_results": k_results,
        "default_k_assessment": assess_default_k(k_results, default_k=DEFAULT_K),
        "notes": [
            "R6 is a side experiment and does not overwrite M1-M3 outputs.",
            "All K metrics are computed from actual repaired CSV outputs.",
            "Failed K values are reported with failure reasons instead of fabricated metrics.",
        ],
    }

    metrics_path = output_dir / "k_sensitivity_metrics.json"
    report_path = output_dir / "k_sensitivity_report.md"
    metrics_path.write_text(
        json.dumps(_to_builtin(metrics_doc), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_path.write_text(build_report(metrics_doc), encoding="utf-8", newline="\n")
    return _to_builtin(metrics_doc)


def _parse_k_values(raw_values: list[str] | None) -> list[int] | None:
    if not raw_values:
        return None
    values: list[int] = []
    for raw in raw_values:
        for item in str(raw).split(","):
            text = item.strip()
            if not text:
                continue
            values.append(int(text))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gower repair sensitivity to k_neighbors on M1 data.")
    parser.add_argument("--m1-dir", default=str(DEFAULT_M1_DIR), help="M1 experiment data directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="R6 output directory.")
    parser.add_argument("--k", action="append", help="K value or comma-separated K values. Defaults to 3,5,7,9,15.")
    args = parser.parse_args()

    metrics = evaluate(Path(args.m1_dir), Path(args.output_dir), _parse_k_values(args.k))
    summary = {
        key: {
            "status": result.get("status"),
            "resolved_issue_count": result.get("resolved_issue_count"),
            "total_cells_modified": result.get("total_cells_modified"),
            "exact_restoration_rate": result.get("exact_restoration_rate"),
            "improved_or_exact_rate": result.get("improved_or_exact_rate"),
            "non_ground_truth_cells_modified": result.get("non_ground_truth_cells_modified"),
            "mean_neighbor_confidence": result.get("mean_neighbor_confidence"),
        }
        for key, result in metrics["k_results"].items()
    }
    summary["default_k_assessment"] = metrics["default_k_assessment"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
