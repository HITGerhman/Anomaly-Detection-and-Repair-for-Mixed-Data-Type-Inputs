"""Generate M1 clean/corrupted/ground-truth experiment data.

This script is intentionally standalone and only depends on the project's
existing pandas runtime. It does not call or modify the engine.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
DEFAULT_SEED = 20260503

GROUND_TRUTH_COLUMNS = [
    "injection_id",
    "anomaly_type",
    "expected_issue_type",
    "row_id",
    "source_row_id",
    "row_index",
    "column",
    "original_value",
    "corrupted_value",
    "repairable",
    "notes",
]

INJECTION_PLAN = {
    "missing_values": 30,
    "numeric_outlier": 24,
    "rare_category": 18,
    "duplicate_record": 12,
    "cross_column_consistency": 16,
}


def _resolve_path(path_text: str | Path) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _cell_text(value: Any) -> str:
    value = _json_safe(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _filter_iqr(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result = result[result[column].between(lower, upper)].copy()
    return result


def build_clean_subset(source_csv: Path) -> pd.DataFrame:
    source = pd.read_csv(source_csv)
    required_columns = {
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    missing = sorted(required_columns.difference(source.columns))
    if missing:
        raise ValueError(f"source CSV missing required columns: {missing}")

    clean = source.dropna().copy()
    clean = clean[(clean["gender"] != "Other") & (clean["work_type"] != "Never_worked")].copy()
    clean = _filter_iqr(clean, ["age", "avg_glucose_level", "bmi"])
    clean = clean.sort_values("id").reset_index(drop=True)

    clean.insert(0, "row_id", [f"m1-row-{idx:05d}" for idx in range(len(clean))])
    clean.insert(1, "source_row_id", clean["id"].astype(str))

    start_day = [(idx % 120) + 10 for idx in range(len(clean))]
    clean["record_start_day"] = start_day
    clean["record_end_day"] = [value + 5 for value in start_day]

    return clean


def _new_truth_record(
    records: list[dict[str, Any]],
    *,
    injection_id: int,
    anomaly_type: str,
    expected_issue_type: str,
    frame: pd.DataFrame,
    row_index: int,
    column: str,
    original_value: Any,
    corrupted_value: Any,
    repairable: bool,
    notes: str,
) -> None:
    row = frame.iloc[row_index]
    records.append(
        {
            "injection_id": f"m1-{injection_id:04d}",
            "anomaly_type": anomaly_type,
            "expected_issue_type": expected_issue_type,
            "row_id": str(row["row_id"]),
            "source_row_id": str(row["source_row_id"]),
            "row_index": int(row_index),
            "column": column,
            "original_value": _cell_text(original_value),
            "corrupted_value": _cell_text(corrupted_value),
            "repairable": bool(repairable),
            "notes": notes,
        }
    )


def inject_anomalies(clean: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    corrupted = clean.copy(deep=True)
    truth_records: list[dict[str, Any]] = []
    used_cells: set[tuple[int, str]] = set()
    injection_id = 1

    def choose_rows(count: int, *, excluded: set[int] | None = None) -> list[int]:
        excluded = excluded or set()
        candidates = [idx for idx in range(len(corrupted)) if idx not in excluded]
        if count > len(candidates):
            raise ValueError(f"cannot choose {count} rows from {len(candidates)} candidates")
        return rng.sample(candidates, count)

    def choose_cell(row_index: int, columns: list[str]) -> str:
        candidates = [col for col in columns if (row_index, col) not in used_cells]
        if not candidates:
            raise ValueError(f"no available columns for row {row_index}")
        column = rng.choice(candidates)
        used_cells.add((row_index, column))
        return column

    missing_columns = ["bmi", "avg_glucose_level", "smoking_status", "work_type"]
    for row_index in choose_rows(INJECTION_PLAN["missing_values"]):
        column = choose_cell(row_index, missing_columns)
        original = corrupted.at[row_index, column]
        corrupted.at[row_index, column] = pd.NA
        _new_truth_record(
            truth_records,
            injection_id=injection_id,
            anomaly_type="missing_values",
            expected_issue_type="missing_values",
            frame=corrupted,
            row_index=row_index,
            column=column,
            original_value=original,
            corrupted_value="",
            repairable=True,
            notes="Cell was replaced with a missing value.",
        )
        injection_id += 1

    outlier_values = {
        "age": 140,
        "avg_glucose_level": 420.0,
        "bmi": 95.0,
    }
    outlier_columns = list(outlier_values)
    for row_index in choose_rows(INJECTION_PLAN["numeric_outlier"]):
        column = choose_cell(row_index, outlier_columns)
        original = corrupted.at[row_index, column]
        new_value = outlier_values[column]
        corrupted.at[row_index, column] = new_value
        _new_truth_record(
            truth_records,
            injection_id=injection_id,
            anomaly_type="numeric_outlier",
            expected_issue_type="numeric_outlier",
            frame=corrupted,
            row_index=row_index,
            column=column,
            original_value=original,
            corrupted_value=new_value,
            repairable=True,
            notes="Numeric value was moved outside the clean subset range.",
        )
        injection_id += 1

    rare_columns = ["smoking_status", "work_type", "gender"]
    for offset, row_index in enumerate(choose_rows(INJECTION_PLAN["rare_category"])):
        column = choose_cell(row_index, rare_columns)
        original = corrupted.at[row_index, column]
        new_value = f"__M1_RARE_{column.upper()}_{offset:03d}__"
        corrupted.at[row_index, column] = new_value
        _new_truth_record(
            truth_records,
            injection_id=injection_id,
            anomaly_type="rare_category",
            expected_issue_type="rare_category",
            frame=corrupted,
            row_index=row_index,
            column=column,
            original_value=original,
            corrupted_value=new_value,
            repairable=True,
            notes="Categorical value was replaced with a deterministic singleton category.",
        )
        injection_id += 1

    for row_index in choose_rows(INJECTION_PLAN["cross_column_consistency"]):
        original_start = corrupted.at[row_index, "record_start_day"]
        original_end = corrupted.at[row_index, "record_end_day"]
        new_start = int(original_end) + 10
        corrupted.at[row_index, "record_start_day"] = new_start
        _new_truth_record(
            truth_records,
            injection_id=injection_id,
            anomaly_type="cross_column_consistency",
            expected_issue_type="cross_column_consistency",
            frame=corrupted,
            row_index=row_index,
            column="record_start_day,record_end_day",
            original_value=f"{original_start}<={original_end}",
            corrupted_value=f"{new_start}>{original_end}",
            repairable=False,
            notes="record_start_day was made larger than record_end_day.",
        )
        injection_id += 1

    duplicate_source_rows = choose_rows(INJECTION_PLAN["duplicate_record"])
    duplicate_rows = corrupted.iloc[duplicate_source_rows].copy(deep=True)
    duplicate_rows["row_id"] = [f"m1-dup-{idx:05d}" for idx in range(len(duplicate_rows))]

    corrupted = pd.concat([corrupted, duplicate_rows], ignore_index=True)
    for duplicate_offset, source_row_index in enumerate(duplicate_source_rows):
        duplicate_row_index = len(clean) + duplicate_offset
        original_row = clean.iloc[source_row_index]
        duplicate_row = corrupted.iloc[duplicate_row_index]
        _new_truth_record(
            truth_records,
            injection_id=injection_id,
            anomaly_type="duplicate_record",
            expected_issue_type="duplicate_record",
            frame=corrupted,
            row_index=duplicate_row_index,
            column="source_row_id",
            original_value=original_row["row_id"],
            corrupted_value=duplicate_row["row_id"],
            repairable=False,
            notes="A duplicate row was appended with the same source_row_id.",
        )
        injection_id += 1

    ground_truth = pd.DataFrame(truth_records, columns=GROUND_TRUTH_COLUMNS)
    return corrupted.reset_index(drop=True), ground_truth


def build_summary(
    source_csv: Path,
    clean: pd.DataFrame,
    corrupted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    by_type = ground_truth["anomaly_type"].value_counts().sort_index().to_dict()
    by_expected = ground_truth["expected_issue_type"].value_counts().sort_index().to_dict()
    return {
        "milestone": "M1",
        "dataset": "m1_stroke",
        "seed": int(seed),
        "source_csv": str(source_csv.relative_to(PROJECT_ROOT) if source_csv.is_relative_to(PROJECT_ROOT) else source_csv),
        "clean_rows": int(len(clean)),
        "clean_columns": int(len(clean.columns)),
        "corrupted_rows": int(len(corrupted)),
        "corrupted_columns": int(len(corrupted.columns)),
        "ground_truth_rows": int(len(ground_truth)),
        "injection_counts_by_type": {str(k): int(v) for k, v in by_type.items()},
        "expected_issue_counts": {str(k): int(v) for k, v in by_expected.items()},
        "repairable_ground_truth_rows": int(ground_truth["repairable"].astype(bool).sum()),
        "non_repairable_ground_truth_rows": int((~ground_truth["repairable"].astype(bool)).sum()),
        "notes": [
            "M1 only constructs experiment data and ground truth.",
            "Detection and repair metrics are intentionally left for M2/M3.",
            "No synthetic metric values are fabricated by this generator.",
        ],
    }


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# M1 Stroke Experiment Data

This directory contains the M1 experiment data generated from the stroke dataset.

## Files

- `clean.csv`: conservative clean subset used as the reference table.
- `corrupted.csv`: clean subset plus deterministic injected anomalies.
- `ground_truth.csv`: row/cell-level injection records.
- `injection_summary.json`: machine-readable generation summary.

## Generation

```powershell
.\\.venv-win\\Scripts\\python.exe scripts\\generate_m1_experiment_data.py --output-dir data\\experiments\\m1_stroke --seed {summary["seed"]}
```

## Injection Types

{json.dumps(summary["injection_counts_by_type"], ensure_ascii=False, indent=2)}

M1 only creates controlled data and ground truth. Detection metrics belong to M2, and repair metrics belong to M3.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def generate(source_csv: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    source_csv = _resolve_path(source_csv)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = build_clean_subset(source_csv)
    corrupted, ground_truth = inject_anomalies(clean, seed)
    summary = build_summary(source_csv, clean, corrupted, ground_truth, seed)

    clean.to_csv(output_dir / "clean.csv", index=False, lineterminator="\n", float_format="%.6f")
    corrupted.to_csv(output_dir / "corrupted.csv", index=False, lineterminator="\n", float_format="%.6f")
    ground_truth.to_csv(output_dir / "ground_truth.csv", index=False, lineterminator="\n")
    (output_dir / "injection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate M1 controlled experiment data.")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE), help="Source CSV path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed.")
    args = parser.parse_args()

    summary = generate(Path(args.source_csv), Path(args.output_dir), int(args.seed))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
