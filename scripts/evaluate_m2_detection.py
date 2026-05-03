"""Evaluate M2 detection quality against the M1 ground truth.

The evaluator intentionally reuses the current Python engine detector without
changing the public engine protocol. It imports the detector internals only to
read complete boolean masks that are not exposed by the scan_file JSON output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENGINE_DIR = PROJECT_ROOT / "appshell" / "core" / "python_engine"
DEFAULT_M1_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "m2_stroke_detection"

if str(PYTHON_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_ENGINE_DIR))

from engine_core import (  # noqa: E402
    _detect_issues_for_frame,
    _load_dataframe_module,
    _scan_config_from_payload,
    _to_builtin,
)


M1_ANOMALY_TYPES = [
    "missing_values",
    "numeric_outlier",
    "rare_category",
    "duplicate_record",
    "cross_column_consistency",
]

M2_SCAN_CONFIG = {
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


def _resolve_path(path_text: str | Path) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


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


def _metric_counts(ground_truth_keys: set[str], prediction_keys: set[str]) -> dict[str, Any]:
    tp = len(ground_truth_keys & prediction_keys)
    fp = len(prediction_keys - ground_truth_keys)
    fn = len(ground_truth_keys - prediction_keys)
    predicted_count = len(prediction_keys)
    ground_truth_count = len(ground_truth_keys)
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / ground_truth_count if ground_truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "ground_truth_count": ground_truth_count,
        "predicted_count": predicted_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _truth_key(record: dict[str, Any]) -> str:
    anomaly_type = str(record["anomaly_type"])
    if anomaly_type in {"missing_values", "numeric_outlier", "rare_category"}:
        return f"{anomaly_type}|row={int(record['row_index'])}|col={record['column']}"
    if anomaly_type == "cross_column_consistency":
        return f"{anomaly_type}|row={int(record['row_index'])}"
    if anomaly_type == "duplicate_record":
        return f"{anomaly_type}|source_row_id={_value_text(record['source_row_id'])}"
    raise ValueError(f"unsupported anomaly_type in ground truth: {anomaly_type}")


def _prediction_key(anomaly_type: str, *, row_index: int | None = None, column: str | None = None, source_row_id: str | None = None) -> str:
    if anomaly_type in {"missing_values", "numeric_outlier", "rare_category"}:
        return f"{anomaly_type}|row={int(row_index or 0)}|col={column}"
    if anomaly_type == "cross_column_consistency":
        return f"{anomaly_type}|row={int(row_index or 0)}"
    if anomaly_type == "duplicate_record":
        return f"{anomaly_type}|source_row_id={source_row_id}"
    raise ValueError(f"unsupported predicted anomaly_type: {anomaly_type}")


def _mask_positions(mask: Any) -> list[int]:
    return [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]


def _issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    mask = issue["mask"]
    return _to_builtin(
        {
            "issue_id": issue["issue_id"],
            "issue_type": issue["issue_type"],
            "column": issue["column"],
            "count": int(issue["count"]),
            "detected_rows": len(_mask_positions(mask)),
            "ratio": round(float(issue["ratio"]), 6),
            "issue_score": round(float(issue["issue_score"]), 6),
            "severity": issue["severity"],
            "confidence": round(float(issue.get("confidence", 0.0)), 6),
            "detail": issue.get("detail", {}),
        }
    )


def _load_inputs(m1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clean_csv = m1_dir / "clean.csv"
    corrupted_csv = m1_dir / "corrupted.csv"
    ground_truth_csv = m1_dir / "ground_truth.csv"
    summary_json = m1_dir / "injection_summary.json"
    missing = [path.name for path in [clean_csv, corrupted_csv, ground_truth_csv, summary_json] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"M1 directory is missing required files: {missing}")
    clean = pd.read_csv(clean_csv)
    corrupted = pd.read_csv(corrupted_csv)
    ground_truth = pd.read_csv(ground_truth_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    return clean, corrupted, ground_truth, summary


def _build_truth_events(ground_truth: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    truth_events = {anomaly_type: {} for anomaly_type in M1_ANOMALY_TYPES}
    for record in ground_truth.sort_values("injection_id").to_dict(orient="records"):
        anomaly_type = str(record["anomaly_type"])
        if anomaly_type not in truth_events:
            continue
        key = _truth_key(record)
        truth_events[anomaly_type][key] = {
            "key": key,
            "injection_id": str(record["injection_id"]),
            "anomaly_type": anomaly_type,
            "expected_issue_type": str(record["expected_issue_type"]),
            "row_id": str(record["row_id"]),
            "source_row_id": _value_text(record["source_row_id"]),
            "row_index": int(record["row_index"]),
            "column": str(record["column"]),
            "original_value": _value_text(record["original_value"]),
            "corrupted_value": _value_text(record["corrupted_value"]),
            "repairable": bool(record["repairable"]),
            "notes": str(record["notes"]),
        }
    return truth_events


def _add_prediction(predictions: dict[str, dict[str, dict[str, Any]]], anomaly_type: str, key: str, event: dict[str, Any]) -> None:
    existing = predictions[anomaly_type].get(key)
    if existing is None:
        predictions[anomaly_type][key] = event
        return
    issue_id = event.get("issue_id")
    if issue_id and issue_id not in existing["issue_ids"]:
        existing["issue_ids"].append(issue_id)


def _build_prediction_events(corrupted: pd.DataFrame, issues: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    predictions = {anomaly_type: {} for anomaly_type in M1_ANOMALY_TYPES}
    issue_summaries: list[dict[str, Any]] = []

    for issue in issues:
        anomaly_type = str(issue["issue_type"])
        if anomaly_type not in predictions:
            continue

        issue_id = str(issue["issue_id"])
        column = str(issue["column"])
        positions = _mask_positions(issue["mask"])
        issue_summaries.append(_issue_summary(issue))

        if anomaly_type in {"missing_values", "numeric_outlier", "rare_category"}:
            for row_index in positions:
                key = _prediction_key(anomaly_type, row_index=row_index, column=column)
                value = corrupted.at[row_index, column] if column in corrupted.columns else ""
                _add_prediction(
                    predictions,
                    anomaly_type,
                    key,
                    {
                        "key": key,
                        "anomaly_type": anomaly_type,
                        "row_index": int(row_index),
                        "column": column,
                        "value": _value_text(value),
                        "issue_id": issue_id,
                        "issue_ids": [issue_id],
                    },
                )
        elif anomaly_type == "cross_column_consistency":
            for row_index in positions:
                key = _prediction_key(anomaly_type, row_index=row_index)
                _add_prediction(
                    predictions,
                    anomaly_type,
                    key,
                    {
                        "key": key,
                        "anomaly_type": anomaly_type,
                        "row_index": int(row_index),
                        "column": "record_start_day,record_end_day",
                        "values": {
                            "record_start_day": _value_text(corrupted.at[row_index, "record_start_day"]),
                            "record_end_day": _value_text(corrupted.at[row_index, "record_end_day"]),
                        },
                        "issue_id": issue_id,
                        "issue_ids": [issue_id],
                    },
                )
        elif anomaly_type == "duplicate_record":
            grouped_rows: dict[str, list[int]] = {}
            for row_index in positions:
                source_row_id = _value_text(corrupted.at[row_index, "source_row_id"])
                grouped_rows.setdefault(source_row_id, []).append(int(row_index))
            for source_row_id, row_indices in grouped_rows.items():
                key = _prediction_key(anomaly_type, source_row_id=source_row_id)
                _add_prediction(
                    predictions,
                    anomaly_type,
                    key,
                    {
                        "key": key,
                        "anomaly_type": anomaly_type,
                        "source_row_id": source_row_id,
                        "marked_row_indices": sorted(row_indices),
                        "marked_row_count": len(row_indices),
                        "issue_id": issue_id,
                        "issue_ids": [issue_id],
                    },
                )

    return predictions, sorted(issue_summaries, key=lambda item: (item["issue_type"], item["column"], item["issue_id"]))


def _build_matches(
    truth_events: dict[str, dict[str, dict[str, Any]]],
    prediction_events: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    truth_matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []

    for anomaly_type in M1_ANOMALY_TYPES:
        truth_by_key = truth_events[anomaly_type]
        predictions_by_key = prediction_events[anomaly_type]
        for key in sorted(truth_by_key):
            truth = truth_by_key[key]
            prediction = predictions_by_key.get(key)
            row = {
                "key": key,
                "detected": prediction is not None,
                "truth": truth,
                "prediction": prediction,
            }
            truth_matches.append(row)
            if prediction is None:
                false_negatives.append(row)
        for key in sorted(set(predictions_by_key).difference(truth_by_key)):
            false_positives.append(
                {
                    "key": key,
                    "anomaly_type": anomaly_type,
                    "prediction": predictions_by_key[key],
                }
            )

    return {
        "truth_matches": truth_matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _build_metrics(
    truth_events: dict[str, dict[str, dict[str, Any]]],
    prediction_events: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_type: dict[str, dict[str, Any]] = {}
    all_truth_keys: set[str] = set()
    all_prediction_keys: set[str] = set()
    for anomaly_type in M1_ANOMALY_TYPES:
        truth_keys = set(truth_events[anomaly_type])
        prediction_keys = set(prediction_events[anomaly_type])
        by_type[anomaly_type] = _metric_counts(truth_keys, prediction_keys)
        all_truth_keys.update(truth_keys)
        all_prediction_keys.update(prediction_keys)
    return _metric_counts(all_truth_keys, all_prediction_keys), by_type


def _write_readme(output_dir: Path, metrics_doc: dict[str, Any]) -> None:
    overall = metrics_doc["metrics"]["overall"]
    by_type = metrics_doc["metrics"]["by_type"]
    table_rows = [
        "| Type | GT | Pred | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for anomaly_type in M1_ANOMALY_TYPES:
        row = by_type[anomaly_type]
        table_rows.append(
            f"| `{anomaly_type}` | {row['ground_truth_count']} | {row['predicted_count']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['f1']:.6f} |"
        )
    table_rows.append(
        f"| **Overall** | {overall['ground_truth_count']} | {overall['predicted_count']} | "
        f"{overall['tp']} | {overall['fp']} | {overall['fn']} | {overall['precision']:.6f} | "
        f"{overall['recall']:.6f} | {overall['f1']:.6f} |"
    )

    text = f"""# M2 Stroke Detection Evaluation

This directory contains the M2 detection evaluation based on the M1 stroke experiment data.

## Inputs

- M1 directory: `{metrics_doc["source"]["m1_dir"]}`
- Ground truth rows: {overall["ground_truth_count"]}
- Corrupted rows: {metrics_doc["data_profile"]["corrupted_rows"]}
- Scored anomaly types: {", ".join(f"`{item}`" for item in M1_ANOMALY_TYPES)}

## Scoring Policy

- Missing values, numeric outliers, and rare categories are matched by anomaly type, row index, and column.
- Cross-column consistency is matched by anomaly type and row index.
- Duplicate records are matched by anomaly type and `source_row_id` group; marked row counts are reported separately.
- `time_series_shift` is disabled for M2 scoring because M1 did not inject that anomaly type.

## Metrics

{chr(10).join(table_rows)}

## Notes

- M2 evaluates detection only. Repair quality is intentionally left for M3.
- False positives and false negatives are listed in `detection_matches.json`.
- Numeric outlier precision may be lower when the current detector also flags natural high-end values in the corrupted dataset. M2 records this behavior without tuning detector thresholds.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def evaluate(m1_dir: Path, output_dir: Path) -> dict[str, Any]:
    m1_dir = _resolve_path(m1_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, corrupted, ground_truth, injection_summary = _load_inputs(m1_dir)
    frame_pd = _load_dataframe_module("M2DetectionEvaluation")
    scan_config = _scan_config_from_payload({"scan_config": M2_SCAN_CONFIG})
    issues = _detect_issues_for_frame(corrupted, frame_pd, scan_config=scan_config)

    truth_events = _build_truth_events(ground_truth)
    prediction_events, issue_summaries = _build_prediction_events(corrupted, issues)
    matches = _build_matches(truth_events, prediction_events)
    overall, by_type = _build_metrics(truth_events, prediction_events)

    metrics_doc = {
        "milestone": "M2",
        "dataset": "m2_stroke_detection",
        "source": {
            "m1_dir": str(m1_dir.relative_to(PROJECT_ROOT) if m1_dir.is_relative_to(PROJECT_ROOT) else m1_dir),
            "clean_csv": "clean.csv",
            "corrupted_csv": "corrupted.csv",
            "ground_truth_csv": "ground_truth.csv",
        },
        "data_profile": {
            "clean_rows": int(len(clean)),
            "clean_columns": int(len(clean.columns)),
            "corrupted_rows": int(len(corrupted)),
            "corrupted_columns": int(len(corrupted.columns)),
            "ground_truth_rows": int(len(ground_truth)),
            "m1_injection_counts_by_type": injection_summary.get("injection_counts_by_type", {}),
        },
        "scoring_policy": {
            "event_matching": {
                "missing_values": "anomaly_type + row_index + column",
                "numeric_outlier": "anomaly_type + row_index + column",
                "rare_category": "anomaly_type + row_index + column",
                "cross_column_consistency": "anomaly_type + row_index",
                "duplicate_record": "anomaly_type + source_row_id group",
            },
            "time_series_shift": "excluded from M2 scoring because M1 did not inject this anomaly type",
        },
        "scan_config": scan_config,
        "scan_issue_count": len(issue_summaries),
        "scan_issues": issue_summaries,
        "metrics": {
            "overall": overall,
            "by_type": by_type,
        },
        "notes": [
            "M2 evaluates detection only; repair quality belongs to M3.",
            "The evaluator imports engine_core internals only to access complete masks.",
            "No detector threshold tuning or core algorithm changes are performed by M2.",
        ],
    }

    matches_doc = {
        "milestone": "M2",
        "dataset": "m2_stroke_detection",
        "scoring_policy": metrics_doc["scoring_policy"],
        "summary": {
            "truth_match_rows": len(matches["truth_matches"]),
            "false_positive_rows": len(matches["false_positives"]),
            "false_negative_rows": len(matches["false_negatives"]),
        },
        **matches,
    }

    (output_dir / "detection_metrics.json").write_text(
        json.dumps(_to_builtin(metrics_doc), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "detection_matches.json").write_text(
        json.dumps(_to_builtin(matches_doc), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_readme(output_dir, metrics_doc)
    return _to_builtin(metrics_doc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate M2 detection quality against M1 ground truth.")
    parser.add_argument("--m1-dir", default=str(DEFAULT_M1_DIR), help="M1 experiment data directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="M2 output directory.")
    args = parser.parse_args()

    metrics = evaluate(Path(args.m1_dir), Path(args.output_dir))
    print(json.dumps(metrics["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
