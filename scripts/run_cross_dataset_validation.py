"""Run reproducible cross-dataset validation experiments.

This script is intentionally an experiment harness around the existing Python
engine. It generates or copies controlled CSV artifacts, calls the current scan
and repair logic, scores results against ground truth, and writes paper-ready
CSV summaries. It does not modify engine defaults or core algorithms.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENGINE_DIR = PROJECT_ROOT / "appshell" / "core" / "python_engine"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "cross_dataset"
DEFAULT_SCALE_WORK_DIR = PROJECT_ROOT / "outputs" / "cross_dataset_validation"
M1_STROKE_DIR = PROJECT_ROOT / "data" / "experiments" / "m1_stroke"
DEFAULT_SEED = 20260513
DEFAULT_SYNTHETIC_ROWS = 5000
DEFAULT_TOTAL_INJECTIONS = 100
DEFAULT_SCALE_ROWS = [5000, 10000, 50000, 100000]

if str(PYTHON_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_ENGINE_DIR))

from engine_core import (  # noqa: E402
    _detect_issues_for_frame,
    _load_dataframe_module,
    _scan_config_from_payload,
    _to_builtin,
    action_repair_batch,
    action_scan_file,
)


ISSUE_TYPES = [
    "missing_values",
    "numeric_outlier",
    "rare_category",
    "duplicate_record",
    "cross_column_consistency",
]
REPAIRABLE_TYPES = ["missing_values", "numeric_outlier", "rare_category"]
NON_REPAIRABLE_TYPES = ["duplicate_record", "cross_column_consistency"]
GROUND_TRUTH_COLUMNS = [
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
DETECTION_COLUMNS = ["dataset", "issue_type", "gt", "pred", "tp", "fp", "fn", "precision", "recall", "f1"]
REPAIR_COLUMNS = [
    "dataset",
    "issue_type",
    "repairable_gt",
    "changed",
    "exact",
    "improved_or_exact",
    "exact_rate",
    "improved_or_exact_rate",
    "non_gt_modified",
    "skipped_non_repairable_count",
]
THRESHOLD_COLUMNS = [
    "dataset",
    "iqr_factor",
    "robust_z_threshold",
    "gt",
    "pred",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "status",
    "error",
]
SCALE_COLUMNS = [
    "dataset_name",
    "rows",
    "columns",
    "scan_time_seconds",
    "repair_time_seconds",
    "detected_issue_count",
    "changed_cell_count",
    "output_file_size_mb",
]
DEFAULT_INJECTION_PLAN = {
    "missing_values": 30,
    "numeric_outlier": 24,
    "rare_category": 18,
    "duplicate_record": 12,
    "cross_column_consistency": 16,
}
DATASET_SEED_OFFSETS = {
    "stroke": 0,
    "orders_transactions": 101,
    "user_device_logs": 202,
}
DEFAULT_DATASETS = ["stroke", "orders_transactions", "user_device_logs"]
DEFAULT_THRESHOLD_CONFIGS = [
    (1.5, 3.5),
    (1.5, 4.5),
    (1.5, 5.0),
    (2.0, 3.5),
    (2.0, 4.5),
    (2.0, 5.0),
    (3.0, 3.5),
    (3.0, 4.5),
    (3.0, 5.0),
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    consistency_rule_name: str
    consistency_left: str
    consistency_right: str
    duplicate_subset: tuple[str, ...] = ("source_row_id",)


DATASET_CONFIGS = {
    "stroke": DatasetConfig(
        name="stroke",
        consistency_rule_name="record_start_before_end",
        consistency_left="record_start_day",
        consistency_right="record_end_day",
    ),
    "orders_transactions": DatasetConfig(
        name="orders_transactions",
        consistency_rule_name="paid_pay_time_not_before_order_time",
        consistency_left="order_time",
        consistency_right="pay_time",
    ),
    "user_device_logs": DatasetConfig(
        name="user_device_logs",
        consistency_rule_name="logout_not_before_login",
        consistency_left="login_time",
        consistency_right="logout_time",
    ),
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
        return str(resolved)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _value_text(value: Any) -> str:
    value = _json_safe(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
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


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    _write_csv(path, pd.DataFrame(rows, columns=columns))


def scaled_injection_plan(total: int = DEFAULT_TOTAL_INJECTIONS) -> dict[str, int]:
    if total == DEFAULT_TOTAL_INJECTIONS:
        return dict(DEFAULT_INJECTION_PLAN)
    if total <= 0:
        raise ValueError("total injections must be positive")

    weights = {key: value / DEFAULT_TOTAL_INJECTIONS for key, value in DEFAULT_INJECTION_PLAN.items()}
    raw = {key: total * weight for key, weight in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(DEFAULT_INJECTION_PLAN, key=lambda key: (raw[key] - counts[key], DEFAULT_INJECTION_PLAN[key]), reverse=True)
    for index in range(remainder):
        counts[order[index % len(order)]] += 1
    if total >= len(DEFAULT_INJECTION_PLAN):
        for key in DEFAULT_INJECTION_PLAN:
            if counts[key] == 0:
                donor = max(counts, key=lambda item: counts[item])
                counts[donor] -= 1
                counts[key] = 1
    return counts


def scan_config_for_dataset(dataset: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = DATASET_CONFIGS[dataset]
    payload = {
        "scan_config": {
            "max_bins": 120,
            "max_issues": 5000,
            "preview_limit": 20,
            "enable_time_series_shift": False,
            "enable_cross_column_consistency": True,
            "consistency_rules": [
                {
                    "name": config.consistency_rule_name,
                    "type": "lte",
                    "left_col": config.consistency_left,
                    "right_col": config.consistency_right,
                }
            ],
            "enable_duplicate_record": True,
            "duplicate_subset": list(config.duplicate_subset),
            "auto_pair_constraints": False,
        }
    }
    if overrides:
        payload["scan_config"].update(overrides)
    return _scan_config_from_payload(payload)


def repair_strategy() -> dict[str, Any]:
    return {
        "conflict_policy": "first_wins",
        "issue_priority": list(REPAIRABLE_TYPES),
        "missing_numeric": "median",
        "missing_categorical": "mode",
        "outlier": "clip",
        "rare_category": "mode",
        "preview_limit": 20,
    }


def generate_orders_clean(rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    categories = ["electronics", "books", "grocery", "apparel", "home", "sports"]
    methods = ["card", "wallet", "bank_transfer", "paypal"]
    statuses = ["paid", "shipped", "completed", "pending", "refunded"]
    records: list[dict[str, Any]] = []
    for idx in range(rows):
        category = categories[(idx * 7) % len(categories)]
        quantity = 1 + ((idx * 7 + 3) % 5)
        unit_price = round(8.0 + ((idx * 17 + rng.randint(0, 9)) % 700) / 10.0, 2)
        discount = [0.0, 0.05, 0.1, 0.15][(idx * 3) % 4]
        order_time = 1_700_000_000 + idx * 90
        pay_time = order_time + 30 + ((idx * 13) % 900)
        total_amount = round(quantity * unit_price * (1.0 - discount), 2)
        order_id = f"ord-{idx:07d}"
        records.append(
            {
                "row_id": f"orders-row-{idx:07d}",
                "source_row_id": order_id,
                "order_id": order_id,
                "user_id": f"user-{(idx % max(50, rows // 20)):05d}",
                "product_category": category,
                "payment_method": methods[(idx * 11) % len(methods)],
                "order_status": statuses[(idx * 13) % len(statuses)],
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "total_amount": total_amount,
                "order_time": order_time,
                "pay_time": pay_time,
            }
        )
    return pd.DataFrame(records)


def generate_user_logs_clean(rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    device_types = ["phone", "tablet", "desktop", "laptop"]
    os_values = ["Android", "iOS", "Windows", "macOS", "Linux"]
    versions = ["2.4.0", "2.4.1", "2.5.0", "2.5.1"]
    events = ["login", "view", "purchase", "logout", "login_failed"]
    records: list[dict[str, Any]] = []
    for idx in range(rows):
        event_type = events[(idx * 7) % len(events)]
        is_success = 0 if event_type == "login_failed" else 1
        base_duration = 5 + ((idx * 19 + rng.randint(0, 7)) % 900)
        session_duration = min(base_duration, 45) if event_type == "login_failed" else base_duration + 60
        login_time = 1_800_000_000 + idx * 75
        logout_time = login_time + max(1, int(session_duration))
        log_id = f"log-{idx:07d}"
        records.append(
            {
                "row_id": f"logs-row-{idx:07d}",
                "source_row_id": log_id,
                "log_id": log_id,
                "user_id": f"user-{(idx % max(60, rows // 25)):05d}",
                "device_type": device_types[(idx * 3) % len(device_types)],
                "os": os_values[(idx * 5 + 1) % len(os_values)],
                "app_version": versions[(idx * 7) % len(versions)],
                "event_type": event_type,
                "session_duration": int(session_duration),
                "bytes_sent": int(400 + ((idx * 31) % 5000)),
                "bytes_received": int(800 + ((idx * 37) % 12000)),
                "login_time": login_time,
                "logout_time": logout_time,
                "is_success": is_success,
            }
        )
    return pd.DataFrame(records)


def _truth_record(
    corrupted: pd.DataFrame,
    *,
    anomaly_id: str,
    dataset: str,
    expected_issue_type: str,
    row_index: int,
    column_name: str,
    original_value: Any,
    corrupted_value: Any,
    repairable: bool,
    duplicate_group: str = "",
    consistency_rule_name: str = "",
    seed: int,
    notes: str,
) -> dict[str, Any]:
    row = corrupted.iloc[row_index]
    source_row_id = _value_text(row["source_row_id"]) if "source_row_id" in corrupted.columns else ""
    return {
        "anomaly_id": anomaly_id,
        "dataset": dataset,
        "expected_issue_type": expected_issue_type,
        "row_index": int(row_index),
        "column_name": column_name,
        "original_value": _value_text(original_value),
        "corrupted_value": _value_text(corrupted_value),
        "repairable": bool(repairable),
        "source_row_id": source_row_id,
        "duplicate_group": duplicate_group,
        "consistency_rule_name": consistency_rule_name,
        "created_by_seed": int(seed),
        "notes": notes,
    }


def inject_anomalies(clean: pd.DataFrame, dataset: str, seed: int, injection_plan: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    corrupted = clean.copy(deep=True)
    truth: list[dict[str, Any]] = []
    used_cells: set[tuple[int, str]] = set()
    anomaly_counter = 1

    if dataset == "orders_transactions":
        missing_columns = ["product_category", "payment_method", "unit_price", "discount"]
        outlier_values = {"quantity": 999, "unit_price": 9999.0, "discount": 1.5, "total_amount": 999999.0}
        rare_columns = ["product_category", "payment_method", "order_status"]
        consistency_column = "pay_time"
        consistency_left = "order_time"
        consistency_right = "pay_time"
    elif dataset == "user_device_logs":
        missing_columns = ["device_type", "os", "app_version", "session_duration"]
        outlier_values = {"session_duration": 99999, "bytes_sent": 9999999, "bytes_received": 9999999}
        rare_columns = ["device_type", "os", "event_type"]
        consistency_column = "logout_time"
        consistency_left = "login_time"
        consistency_right = "logout_time"
    else:
        raise ValueError(f"unsupported synthetic dataset: {dataset}")

    def next_id() -> str:
        nonlocal anomaly_counter
        value = f"{dataset}-{anomaly_counter:04d}"
        anomaly_counter += 1
        return value

    def choose_rows(count: int, candidates: Iterable[int] | None = None) -> list[int]:
        pool = list(range(len(clean))) if candidates is None else list(candidates)
        if count > len(pool):
            raise ValueError(f"cannot choose {count} rows from {len(pool)} candidates for {dataset}")
        return rng.sample(pool, count)

    def choose_cell(row_index: int, columns: list[str]) -> str:
        candidates = [column for column in columns if (row_index, column) not in used_cells]
        if not candidates:
            raise ValueError(f"no available injection cells for row {row_index} in {dataset}")
        column = rng.choice(candidates)
        used_cells.add((row_index, column))
        return column

    for row_index in choose_rows(injection_plan["missing_values"]):
        column = choose_cell(row_index, missing_columns)
        original = corrupted.at[row_index, column]
        corrupted.at[row_index, column] = pd.NA
        truth.append(
            _truth_record(
                corrupted,
                anomaly_id=next_id(),
                dataset=dataset,
                expected_issue_type="missing_values",
                row_index=row_index,
                column_name=column,
                original_value=original,
                corrupted_value="",
                repairable=True,
                seed=seed,
                notes="Cell was replaced with a missing value.",
            )
        )

    outlier_columns = list(outlier_values)
    for row_index in choose_rows(injection_plan["numeric_outlier"]):
        column = choose_cell(row_index, outlier_columns)
        original = corrupted.at[row_index, column]
        new_value = outlier_values[column]
        corrupted.at[row_index, column] = new_value
        truth.append(
            _truth_record(
                corrupted,
                anomaly_id=next_id(),
                dataset=dataset,
                expected_issue_type="numeric_outlier",
                row_index=row_index,
                column_name=column,
                original_value=original,
                corrupted_value=new_value,
                repairable=True,
                seed=seed,
                notes="Numeric value was moved outside the normal synthetic range.",
            )
        )

    for offset, row_index in enumerate(choose_rows(injection_plan["rare_category"])):
        column = choose_cell(row_index, rare_columns)
        original = corrupted.at[row_index, column]
        new_value = f"__CD_RARE_{dataset.upper()}_{column.upper()}_{offset:03d}__"
        corrupted.at[row_index, column] = new_value
        truth.append(
            _truth_record(
                corrupted,
                anomaly_id=next_id(),
                dataset=dataset,
                expected_issue_type="rare_category",
                row_index=row_index,
                column_name=column,
                original_value=original,
                corrupted_value=new_value,
                repairable=True,
                seed=seed,
                notes="Categorical value was replaced with a deterministic singleton category.",
            )
        )

    config = DATASET_CONFIGS[dataset]
    consistency_candidates = [idx for idx in range(len(clean)) if (idx, consistency_column) not in used_cells]
    for row_index in choose_rows(injection_plan["cross_column_consistency"], consistency_candidates):
        used_cells.add((row_index, consistency_column))
        left_value = corrupted.at[row_index, consistency_left]
        right_value = corrupted.at[row_index, consistency_right]
        new_value = int(left_value) - 120
        corrupted.at[row_index, consistency_right] = new_value
        truth.append(
            _truth_record(
                corrupted,
                anomaly_id=next_id(),
                dataset=dataset,
                expected_issue_type="cross_column_consistency",
                row_index=row_index,
                column_name=f"{consistency_left},{consistency_right}",
                original_value=f"{left_value}<={right_value}",
                corrupted_value=f"{left_value}>{new_value}",
                repairable=False,
                consistency_rule_name=config.consistency_rule_name,
                seed=seed,
                notes=f"{consistency_right} was made smaller than {consistency_left}.",
            )
        )

    duplicate_source_rows = choose_rows(injection_plan["duplicate_record"])
    duplicate_rows = corrupted.iloc[duplicate_source_rows].copy(deep=True)
    duplicate_rows["row_id"] = [f"{dataset}-dup-{idx:05d}" for idx in range(len(duplicate_rows))]
    corrupted = pd.concat([corrupted, duplicate_rows], ignore_index=True)
    for duplicate_offset, source_row_index in enumerate(duplicate_source_rows):
        duplicate_row_index = len(clean) + duplicate_offset
        duplicate_group = _value_text(clean.at[source_row_index, "source_row_id"])
        truth.append(
            _truth_record(
                corrupted,
                anomaly_id=next_id(),
                dataset=dataset,
                expected_issue_type="duplicate_record",
                row_index=duplicate_row_index,
                column_name="source_row_id",
                original_value=clean.at[source_row_index, "row_id"],
                corrupted_value=corrupted.at[duplicate_row_index, "row_id"],
                repairable=False,
                duplicate_group=duplicate_group,
                seed=seed,
                notes="A duplicate row was appended with the same source_row_id.",
            )
        )

    return corrupted.reset_index(drop=True), pd.DataFrame(truth, columns=GROUND_TRUTH_COLUMNS)


def _convert_stroke_ground_truth(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    old_truth = pd.read_csv(source_dir / "ground_truth.csv")
    old_summary = json.loads((source_dir / "injection_summary.json").read_text(encoding="utf-8"))
    seed = int(old_summary.get("seed", DEFAULT_SEED))
    converted: list[dict[str, Any]] = []
    for raw in old_truth.sort_values("injection_id").to_dict(orient="records"):
        issue_type = str(raw["expected_issue_type"])
        column = str(raw.get("column", ""))
        source_row_id = _value_text(raw.get("source_row_id", ""))
        converted.append(
            {
                "anomaly_id": str(raw["injection_id"]).replace("m1-", "stroke-"),
                "dataset": "stroke",
                "expected_issue_type": issue_type,
                "row_index": int(raw["row_index"]),
                "column_name": column,
                "original_value": _value_text(raw.get("original_value", "")),
                "corrupted_value": _value_text(raw.get("corrupted_value", "")),
                "repairable": _to_bool(raw.get("repairable", False)),
                "source_row_id": source_row_id,
                "duplicate_group": source_row_id if issue_type == "duplicate_record" else "",
                "consistency_rule_name": "record_start_before_end" if issue_type == "cross_column_consistency" else "",
                "created_by_seed": seed,
                "notes": str(raw.get("notes", "")),
            }
        )
    _write_csv(output_dir / "ground_truth.csv", pd.DataFrame(converted, columns=GROUND_TRUTH_COLUMNS))
    return {
        "dataset": "stroke",
        "source": _display_path(source_dir),
        "seed": seed,
        "ground_truth_rows": len(converted),
        "injection_counts_by_type": old_summary.get("injection_counts_by_type", {}),
        "notes": [
            "Stroke cross-dataset artifacts are standardized copies of data/experiments/m1_stroke.",
            "The original M1/M2/M3 directories are not modified.",
        ],
    }


def generate_dataset(dataset: str, output_dir: Path, synthetic_rows: int, seed: int, total_injections: int) -> dict[str, Any]:
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_seed = seed + DATASET_SEED_OFFSETS[dataset]

    if dataset == "stroke":
        if not (M1_STROKE_DIR / "clean.csv").exists():
            raise FileNotFoundError(f"missing M1 stroke data: {M1_STROKE_DIR}")
        shutil.copy2(M1_STROKE_DIR / "clean.csv", dataset_dir / "clean.csv")
        shutil.copy2(M1_STROKE_DIR / "corrupted.csv", dataset_dir / "corrupted.csv")
        summary = _convert_stroke_ground_truth(M1_STROKE_DIR, dataset_dir)
        clean = pd.read_csv(dataset_dir / "clean.csv")
        corrupted = pd.read_csv(dataset_dir / "corrupted.csv")
        summary.update(
            {
                "clean_rows": int(len(clean)),
                "clean_columns": int(len(clean.columns)),
                "corrupted_rows": int(len(corrupted)),
                "corrupted_columns": int(len(corrupted.columns)),
                "scan_config": scan_config_for_dataset(dataset),
            }
        )
        _write_json(dataset_dir / "injection_summary.json", summary)
        return summary

    if dataset == "orders_transactions":
        clean = generate_orders_clean(synthetic_rows, dataset_seed)
    elif dataset == "user_device_logs":
        clean = generate_user_logs_clean(synthetic_rows, dataset_seed)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    plan = scaled_injection_plan(total_injections)
    corrupted, ground_truth = inject_anomalies(clean, dataset, dataset_seed, plan)
    _write_csv(dataset_dir / "clean.csv", clean)
    _write_csv(dataset_dir / "corrupted.csv", corrupted)
    _write_csv(dataset_dir / "ground_truth.csv", ground_truth)

    summary = {
        "dataset": dataset,
        "seed": dataset_seed,
        "clean_rows": int(len(clean)),
        "clean_columns": int(len(clean.columns)),
        "corrupted_rows": int(len(corrupted)),
        "corrupted_columns": int(len(corrupted.columns)),
        "ground_truth_rows": int(len(ground_truth)),
        "injection_counts_by_type": {key: int(value) for key, value in plan.items()},
        "repairable_ground_truth_rows": int(ground_truth["repairable"].astype(bool).sum()),
        "non_repairable_ground_truth_rows": int((~ground_truth["repairable"].astype(bool)).sum()),
        "scan_config": scan_config_for_dataset(dataset),
        "notes": [
            "Synthetic clean data and injected anomalies are deterministic for the recorded seed.",
            "The 30-row Auto Agent sample CSVs are retained separately and are not used as ground truth inputs.",
        ],
    }
    _write_json(dataset_dir / "injection_summary.json", summary)
    return summary


def run_generate(output_dir: Path, datasets: list[str], synthetic_rows: int, seed: int, total_injections: int) -> list[dict[str, Any]]:
    summaries = [generate_dataset(dataset, output_dir, synthetic_rows, seed, total_injections) for dataset in datasets]
    _write_json(output_dir / "generation_summary.json", {"datasets": summaries})
    return summaries


def _load_dataset_inputs(output_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = output_dir / dataset
    required = [dataset_dir / "clean.csv", dataset_dir / "corrupted.csv", dataset_dir / "ground_truth.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"dataset artifacts missing; run --generate first: {missing}")
    return pd.read_csv(required[0]), pd.read_csv(required[1]), pd.read_csv(required[2])


def _scan_internal(corrupted: pd.DataFrame, dataset: str, overrides: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_pd = _load_dataframe_module("CrossDatasetValidation")
    scan_config = scan_config_for_dataset(dataset, overrides)
    issues = _detect_issues_for_frame(corrupted, frame_pd, scan_config=scan_config)
    return issues, scan_config


def _mask_positions(mask: Any) -> list[int]:
    return [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]


def _metric_counts(ground_truth_keys: set[str], prediction_keys: set[str]) -> dict[str, Any]:
    tp = len(ground_truth_keys & prediction_keys)
    fp = len(prediction_keys - ground_truth_keys)
    fn = len(ground_truth_keys - prediction_keys)
    precision_raw = tp / len(prediction_keys) if prediction_keys else 0.0
    recall_raw = tp / len(ground_truth_keys) if ground_truth_keys else 0.0
    f1_raw = 2 * precision_raw * recall_raw / (precision_raw + recall_raw) if precision_raw + recall_raw else 0.0
    return {
        "gt": len(ground_truth_keys),
        "pred": len(prediction_keys),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision_raw, 6),
        "recall": round(recall_raw, 6),
        "f1": round(f1_raw, 6),
    }


def _truth_key(record: dict[str, Any]) -> str:
    issue_type = str(record["expected_issue_type"])
    if issue_type in {"missing_values", "numeric_outlier", "rare_category"}:
        return f"{issue_type}|row={int(record['row_index'])}|col={record['column_name']}"
    if issue_type == "cross_column_consistency":
        rule = _value_text(record.get("consistency_rule_name") or record.get("column_name") or "")
        return f"{issue_type}|row={int(record['row_index'])}|rule={rule}"
    if issue_type == "duplicate_record":
        group = _value_text(record.get("duplicate_group") or record.get("source_row_id") or "")
        return f"{issue_type}|group={group}"
    raise ValueError(f"unsupported ground-truth issue type: {issue_type}")


def _prediction_key(issue_type: str, *, row_index: int | None = None, column: str = "", group: str = "", rule: str = "") -> str:
    if issue_type in {"missing_values", "numeric_outlier", "rare_category"}:
        return f"{issue_type}|row={int(row_index or 0)}|col={column}"
    if issue_type == "cross_column_consistency":
        return f"{issue_type}|row={int(row_index or 0)}|rule={rule}"
    if issue_type == "duplicate_record":
        return f"{issue_type}|group={group}"
    raise ValueError(f"unsupported prediction issue type: {issue_type}")


def build_truth_events(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    events = {issue_type: set() for issue_type in ISSUE_TYPES}
    for record in ground_truth.to_dict(orient="records"):
        issue_type = str(record["expected_issue_type"])
        if issue_type in events:
            events[issue_type].add(_truth_key(record))
    return events


def build_prediction_events(corrupted: pd.DataFrame, issues: list[dict[str, Any]]) -> dict[str, set[str]]:
    predictions = {issue_type: set() for issue_type in ISSUE_TYPES}
    for issue in issues:
        issue_type = str(issue.get("issue_type", ""))
        if issue_type not in predictions:
            continue
        column = str(issue.get("column", ""))
        positions = _mask_positions(issue["mask"])
        if issue_type in {"missing_values", "numeric_outlier", "rare_category"}:
            for row_index in positions:
                predictions[issue_type].add(_prediction_key(issue_type, row_index=row_index, column=column))
        elif issue_type == "cross_column_consistency":
            detail = issue.get("detail", {})
            rule = str(detail.get("rule_name") or "")
            for row_index in positions:
                predictions[issue_type].add(_prediction_key(issue_type, row_index=row_index, rule=rule))
        elif issue_type == "duplicate_record":
            group_column = "source_row_id" if "source_row_id" in corrupted.columns else column
            for row_index in positions:
                group = _value_text(corrupted.at[row_index, group_column])
                predictions[issue_type].add(_prediction_key(issue_type, group=group))
    return predictions


def detection_metric_rows(dataset: str, truth_events: dict[str, set[str]], prediction_events: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_truth: set[str] = set()
    all_predictions: set[str] = set()
    for issue_type in ISSUE_TYPES:
        truth_keys = truth_events[issue_type]
        prediction_keys = prediction_events[issue_type]
        all_truth.update(truth_keys)
        all_predictions.update(prediction_keys)
        rows.append({"dataset": dataset, "issue_type": issue_type, **_metric_counts(truth_keys, prediction_keys)})
    rows.append({"dataset": dataset, "issue_type": "Overall", **_metric_counts(all_truth, all_predictions)})
    return rows


def run_detect(output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        _, corrupted, ground_truth = _load_dataset_inputs(output_dir, dataset)
        issues, scan_config = _scan_internal(corrupted, dataset)
        truth_events = build_truth_events(ground_truth)
        prediction_events = build_prediction_events(corrupted, issues)
        rows = detection_metric_rows(dataset, truth_events, prediction_events)
        dataset_dir = output_dir / dataset
        _write_rows(dataset_dir / "detection_metrics.csv", rows, DETECTION_COLUMNS)
        _write_json(
            dataset_dir / "scan_summary.json",
            {
                "dataset": dataset,
                "scan_config": scan_config,
                "issue_count": len(issues),
                "issue_type_counts": {
                    issue_type: sum(1 for issue in issues if str(issue.get("issue_type")) == issue_type)
                    for issue_type in ISSUE_TYPES
                },
            },
        )
        summary_rows.extend(rows)
    _write_rows(output_dir / "summary_detection_metrics.csv", summary_rows, DETECTION_COLUMNS)
    return summary_rows


def _repairable_truth_records(ground_truth: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in ground_truth.to_dict(orient="records"):
        issue_type = str(record["expected_issue_type"])
        if issue_type in REPAIRABLE_TYPES and _to_bool(record.get("repairable", False)):
            rows.append(record)
    return rows


def _selected_repair_issue_ids(issues: list[dict[str, Any]]) -> list[str]:
    issue_ids: list[str] = []
    for issue in issues:
        if str(issue.get("issue_type")) not in REPAIRABLE_TYPES:
            continue
        issue_id = str(issue.get("issue_id") or "")
        if issue_id and issue_id not in issue_ids:
            issue_ids.append(issue_id)
    return issue_ids


def _selected_issue_cell_types(issues: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    mapping: dict[tuple[int, str], str] = {}
    for issue in issues:
        issue_type = str(issue.get("issue_type", ""))
        if issue_type not in REPAIRABLE_TYPES:
            continue
        column = str(issue.get("column", ""))
        for row_index in _mask_positions(issue["mask"]):
            mapping.setdefault((int(row_index), column), issue_type)
    return mapping


def _evaluate_repair_rows(
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[list[dict[str, Any]], set[tuple[int, str]]]:
    rows: list[dict[str, Any]] = []
    repairable_cells: set[tuple[int, str]] = set()
    for record in _repairable_truth_records(ground_truth):
        issue_type = str(record["expected_issue_type"])
        row_index = int(record["row_index"])
        column = str(record["column_name"])
        if column not in corrupted.columns or column not in repaired.columns:
            continue
        before = corrupted.at[row_index, column]
        after = repaired.at[row_index, column]
        original = record["original_value"]
        before_num = _number_or_none(before)
        after_num = _number_or_none(after)
        original_num = _number_or_none(original)
        exact = _values_equal(after, original)
        improved = False
        if issue_type == "numeric_outlier" and before_num is not None and after_num is not None and original_num is not None:
            improved = abs(after_num - original_num) < abs(before_num - original_num)
        improved_or_exact = exact or improved if issue_type == "numeric_outlier" else exact
        repairable_cells.add((row_index, column))
        rows.append(
            {
                "issue_type": issue_type,
                "row_index": row_index,
                "column": column,
                "changed": _cell_changed(before, after),
                "exact": exact,
                "improved_or_exact": improved_or_exact,
                "before": _value_text(before),
                "after": _value_text(after),
                "original": _value_text(original),
            }
        )
    return rows, repairable_cells


def _changed_cells(
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    repairable_cells: set[tuple[int, str]],
    issue_cell_types: dict[tuple[int, str], str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    changed: list[dict[str, Any]] = []
    non_gt_by_type = {issue_type: 0 for issue_type in REPAIRABLE_TYPES}
    non_gt_by_type["unknown"] = 0
    comparable_rows = min(len(corrupted), len(repaired))
    for row_index in range(comparable_rows):
        for column in corrupted.columns:
            if column not in repaired.columns:
                continue
            before = corrupted.at[row_index, column]
            after = repaired.at[row_index, column]
            if not _cell_changed(before, after):
                continue
            key = (row_index, str(column))
            issue_type = issue_cell_types.get(key, "unknown")
            is_gt_cell = key in repairable_cells
            changed.append(
                {
                    "row_index": row_index,
                    "column": str(column),
                    "before": _value_text(before),
                    "after": _value_text(after),
                    "is_ground_truth_repairable_cell": is_gt_cell,
                    "attributed_issue_type": issue_type,
                }
            )
            if not is_gt_cell:
                non_gt_by_type[issue_type] = non_gt_by_type.get(issue_type, 0) + 1
    return changed, non_gt_by_type


def repair_metric_rows(
    dataset: str,
    repair_rows: list[dict[str, Any]],
    non_gt_by_type: dict[str, int],
    skipped_non_repairable_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue_type in REPAIRABLE_TYPES:
        scoped = [row for row in repair_rows if row["issue_type"] == issue_type]
        exact = sum(1 for row in scoped if row["exact"])
        improved_or_exact = sum(1 for row in scoped if row["improved_or_exact"])
        changed = sum(1 for row in scoped if row["changed"])
        rows.append(
            {
                "dataset": dataset,
                "issue_type": issue_type,
                "repairable_gt": len(scoped),
                "changed": changed,
                "exact": exact,
                "improved_or_exact": improved_or_exact,
                "exact_rate": _rate(exact, len(scoped)),
                "improved_or_exact_rate": _rate(improved_or_exact, len(scoped)),
                "non_gt_modified": int(non_gt_by_type.get(issue_type, 0)),
                "skipped_non_repairable_count": 0,
            }
        )

    total = len(repair_rows)
    exact_total = sum(1 for row in repair_rows if row["exact"])
    improved_total = sum(1 for row in repair_rows if row["improved_or_exact"])
    changed_total = sum(1 for row in repair_rows if row["changed"])
    rows.append(
        {
            "dataset": dataset,
            "issue_type": "Overall",
            "repairable_gt": total,
            "changed": changed_total,
            "exact": exact_total,
            "improved_or_exact": improved_total,
            "exact_rate": _rate(exact_total, total),
            "improved_or_exact_rate": _rate(improved_total, total),
            "non_gt_modified": sum(int(value) for value in non_gt_by_type.values()),
            "skipped_non_repairable_count": int(skipped_non_repairable_count),
        }
    )
    return rows


def side_effect_rows(
    dataset: str,
    ground_truth: pd.DataFrame,
    non_gt_by_type: dict[str, int],
    changed_cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue_type in REPAIRABLE_TYPES:
        rows.append(
            {
                "dataset": dataset,
                "metric": "non_gt_modified",
                "issue_type": issue_type,
                "count": int(non_gt_by_type.get(issue_type, 0)),
                "notes": "Changed cells outside repairable ground truth attributed to this repair issue type.",
            }
        )
    unknown = int(non_gt_by_type.get("unknown", 0))
    if unknown:
        rows.append(
            {
                "dataset": dataset,
                "metric": "non_gt_modified",
                "issue_type": "unknown",
                "count": unknown,
                "notes": "Changed cells outside repairable ground truth that could not be attributed to a selected issue mask.",
            }
        )
    for issue_type in NON_REPAIRABLE_TYPES:
        count = int(
            sum(
                1
                for record in ground_truth.to_dict(orient="records")
                if str(record["expected_issue_type"]) == issue_type and not _to_bool(record.get("repairable", False))
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "review_only_skipped",
                "issue_type": issue_type,
                "count": count,
                "notes": "Skipped / review-only; not counted as automatic repair failure.",
            }
        )
    rows.append(
        {
            "dataset": dataset,
            "metric": "changed_cells_observed",
            "issue_type": "Overall",
            "count": len(changed_cells),
            "notes": "All observed changed cells between corrupted and repaired CSV.",
        }
    )
    return rows


def run_repair(output_dir: Path, datasets: list[str], repair_work_dir: Path) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        _, corrupted, ground_truth = _load_dataset_inputs(output_dir, dataset)
        dataset_dir = output_dir / dataset
        corrupted_csv = dataset_dir / "corrupted.csv"
        issues, scan_config = _scan_internal(corrupted, dataset)
        issue_ids = _selected_repair_issue_ids(issues)
        repair_work = repair_work_dir / "repair_outputs" / dataset
        repair_work.mkdir(parents=True, exist_ok=True)
        output_csv = repair_work / "repaired.csv"
        repair_result = action_repair_batch(
            {
                "csv_path": str(corrupted_csv),
                "issue_ids": issue_ids,
                "scan_config": scan_config,
                "repair_strategy": repair_strategy(),
                "plan_only": False,
                "write_output": True,
                "enable_rollback": False,
                "output_csv": str(output_csv),
            }
        )
        repaired = pd.read_csv(output_csv)
        repair_rows, repairable_cells = _evaluate_repair_rows(corrupted, repaired, ground_truth)
        changed_cells, non_gt_by_type = _changed_cells(corrupted, repaired, repairable_cells, _selected_issue_cell_types(issues))
        skipped_non_repairable_count = int((~ground_truth["repairable"].map(_to_bool)).sum())
        rows = repair_metric_rows(dataset, repair_rows, non_gt_by_type, skipped_non_repairable_count)
        side_rows = side_effect_rows(dataset, ground_truth, non_gt_by_type, changed_cells)
        _write_rows(dataset_dir / "repair_metrics.csv", rows, REPAIR_COLUMNS)
        _write_rows(dataset_dir / "side_effect_summary.csv", side_rows, ["dataset", "metric", "issue_type", "count", "notes"])
        _write_json(
            dataset_dir / "repair_run_summary.json",
            {
                "dataset": dataset,
                "selected_issue_ids": issue_ids,
                "repair_batch": {
                    "selected_issue_count": repair_result.get("selected_issue_count"),
                    "applied_issue_count": repair_result.get("applied_issue_count"),
                    "total_cells_modified": repair_result.get("total_cells_modified"),
                    "comparison": repair_result.get("comparison"),
                    "skipped_issues": repair_result.get("skipped_issues"),
                },
                "repaired_csv": _display_path(output_csv),
            },
        )
        summary_rows.extend(rows)
    _write_rows(output_dir / "summary_repair_metrics.csv", summary_rows, REPAIR_COLUMNS)
    return summary_rows


def run_threshold_sensitivity(
    output_dir: Path,
    datasets: list[str],
    threshold_configs: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        _, corrupted, ground_truth = _load_dataset_inputs(output_dir, dataset)
        truth_events = build_truth_events(ground_truth)
        numeric_truth = truth_events["numeric_outlier"]
        for iqr_factor, robust_z in threshold_configs:
            try:
                issues, _ = _scan_internal(
                    corrupted,
                    dataset,
                    {"numeric_iqr_factor": float(iqr_factor), "robust_z_threshold": float(robust_z)},
                )
                predictions = build_prediction_events(corrupted, issues)["numeric_outlier"]
                metric = _metric_counts(numeric_truth, predictions)
                rows.append(
                    {
                        "dataset": dataset,
                        "iqr_factor": float(iqr_factor),
                        "robust_z_threshold": float(robust_z),
                        **metric,
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "iqr_factor": float(iqr_factor),
                        "robust_z_threshold": float(robust_z),
                        "gt": "N/A",
                        "pred": "N/A",
                        "tp": "N/A",
                        "fp": "N/A",
                        "fn": "N/A",
                        "precision": "N/A",
                        "recall": "N/A",
                        "f1": "N/A",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    _write_rows(output_dir / "threshold_sensitivity_numeric_outlier.csv", rows, THRESHOLD_COLUMNS)
    return rows


def _public_scan(csv_path: Path, dataset: str) -> dict[str, Any]:
    return action_scan_file({"csv_path": str(csv_path), "scan_config": scan_config_for_dataset(dataset)})


def _detected_item_count(scan_result: dict[str, Any]) -> int:
    return int(sum(int(issue.get("count", 0)) for issue in scan_result.get("issues", [])))


def run_scale(output_dir: Path, scale_work_dir: Path, scale_rows: list[int], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scale_dir = scale_work_dir / "scale_inputs"
    scale_dir.mkdir(parents=True, exist_ok=True)
    for row_count in scale_rows:
        dataset = "orders_transactions"
        dataset_seed = seed + 900 + int(row_count)
        clean = generate_orders_clean(row_count, dataset_seed)
        injection_total = 200 if row_count >= 10000 else 100
        corrupted, _ = inject_anomalies(clean, dataset, dataset_seed, scaled_injection_plan(injection_total))
        corrupted_csv = scale_dir / f"{dataset}_{row_count}_corrupted.csv"
        repaired_csv = scale_dir / f"{dataset}_{row_count}_repaired.csv"
        _write_csv(corrupted_csv, corrupted)

        scan_start = time.perf_counter()
        scan_result = _public_scan(corrupted_csv, dataset)
        scan_time = time.perf_counter() - scan_start
        issue_ids = [
            str(issue.get("issue_id"))
            for issue in scan_result.get("issues", [])
            if str(issue.get("issue_type")) in REPAIRABLE_TYPES and str(issue.get("issue_id") or "")
        ]

        repair_start = time.perf_counter()
        repair_result = action_repair_batch(
            {
                "csv_path": str(corrupted_csv),
                "issue_ids": issue_ids,
                "scan_config": scan_config_for_dataset(dataset),
                "repair_strategy": repair_strategy(),
                "plan_only": False,
                "write_output": True,
                "enable_rollback": False,
                "output_csv": str(repaired_csv),
            }
        )
        repair_time = time.perf_counter() - repair_start
        rows.append(
            {
                "dataset_name": f"{dataset}_scale",
                "rows": int(row_count),
                "columns": int(corrupted.shape[1]),
                "scan_time_seconds": round(scan_time, 6),
                "repair_time_seconds": round(repair_time, 6),
                "detected_issue_count": _detected_item_count(scan_result),
                "changed_cell_count": int(repair_result.get("total_cells_modified", 0)),
                "output_file_size_mb": round(repaired_csv.stat().st_size / (1024 * 1024), 6) if repaired_csv.exists() else 0.0,
            }
        )
    _write_rows(output_dir / "summary_scale_metrics.csv", rows, SCALE_COLUMNS)
    return rows


def _parse_csv_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_threshold_configs(raw_values: list[str] | None) -> list[tuple[float, float]]:
    if not raw_values:
        return list(DEFAULT_THRESHOLD_CONFIGS)
    configs: list[tuple[float, float]] = []
    for raw in raw_values:
        for chunk in str(raw).split(","):
            text = chunk.strip()
            if not text:
                continue
            if ":" not in text:
                raise ValueError("--threshold-config values must use iqr:robust_z format")
            left, right = text.split(":", 1)
            configs.append((float(left), float(right)))
    return configs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-dataset validation experiments.")
    parser.add_argument("--all", action="store_true", help="Run generation, detection, repair, threshold, and scale stages.")
    parser.add_argument("--generate", action="store_true", help="Generate or copy clean/corrupted/ground-truth CSVs.")
    parser.add_argument("--detect", action="store_true", help="Run detection evaluation.")
    parser.add_argument("--repair", action="store_true", help="Run repair evaluation.")
    parser.add_argument("--threshold-sensitivity", action="store_true", help="Run numeric_outlier threshold sensitivity.")
    parser.add_argument("--scale", action="store_true", help="Run scale/performance testing.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Cross-dataset artifact output directory.")
    parser.add_argument("--scale-work-dir", default=str(DEFAULT_SCALE_WORK_DIR), help="Ignored workspace for scale and repair outputs.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--synthetic-rows", type=int, default=DEFAULT_SYNTHETIC_ROWS, help="Rows for synthetic clean datasets.")
    parser.add_argument("--injections", type=int, default=DEFAULT_TOTAL_INJECTIONS, help="Total anomalies to inject per synthetic dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base deterministic seed.")
    parser.add_argument("--scale-rows", default=",".join(str(item) for item in DEFAULT_SCALE_ROWS), help="Comma-separated scale row counts.")
    parser.add_argument(
        "--threshold-config",
        action="append",
        help="Threshold config in iqr:robust_z format. Repeat or comma-separate. Defaults to 3x3 full grid.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve_path(args.output_dir)
    scale_work_dir = _resolve_path(args.scale_work_dir)
    datasets = _parse_csv_list(args.datasets, DEFAULT_DATASETS)
    unknown = [dataset for dataset in datasets if dataset not in DATASET_CONFIGS]
    if unknown:
        raise SystemExit(f"unsupported dataset names: {unknown}")

    run_all = args.all or not any([args.generate, args.detect, args.repair, args.threshold_sensitivity, args.scale])
    stages = {
        "generate": run_all or args.generate,
        "detect": run_all or args.detect,
        "repair": run_all or args.repair,
        "threshold_sensitivity": run_all or args.threshold_sensitivity,
        "scale": run_all or args.scale,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"output_dir": _display_path(output_dir), "stages": stages}

    if stages["generate"]:
        results["generation"] = run_generate(output_dir, datasets, args.synthetic_rows, args.seed, args.injections)
    if stages["detect"]:
        results["detection_rows"] = len(run_detect(output_dir, datasets))
    if stages["repair"]:
        results["repair_rows"] = len(run_repair(output_dir, datasets, scale_work_dir))
    if stages["threshold_sensitivity"]:
        configs = _parse_threshold_configs(args.threshold_config)
        results["threshold_rows"] = len(run_threshold_sensitivity(output_dir, datasets, configs))
    if stages["scale"]:
        scale_rows = _parse_int_list(args.scale_rows, DEFAULT_SCALE_ROWS)
        results["scale_rows"] = len(run_scale(output_dir, scale_work_dir, scale_rows, args.seed))

    _write_json(output_dir / "run_summary.json", results)
    print(json.dumps(_to_builtin(results), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
