"""Run large labeled validation experiments for orders_transactions.

This harness bridges the controlled ground-truth experiments and the large
scale stability runs. It keeps the core engine unchanged: generation, memory
sampling, scoring, and paper-ready summaries live here, while scan/repair still
reuse the existing Python Engine logic.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
PYTHON_ENGINE_DIR = PROJECT_ROOT / "appshell" / "core" / "python_engine"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "large_labeled_validation"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "large_labeled_validation_20260531"
DEFAULT_SEED = 20260531
DEFAULT_1M_ROWS = 1_000_000
DEFAULT_10M_ROWS = 10_000_000
SOURCE_DATASET = "orders_transactions"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PYTHON_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_ENGINE_DIR))

import run_cross_dataset_validation as cd  # noqa: E402
from engine_core import _detect_issues_for_frame, _load_dataframe_module, _to_builtin, action_repair_batch  # noqa: E402


ORDER_COLUMNS = [
    "row_id",
    "source_row_id",
    "order_id",
    "user_id",
    "product_category",
    "payment_method",
    "order_status",
    "quantity",
    "unit_price",
    "discount",
    "total_amount",
    "order_time",
    "pay_time",
]
RUNTIME_COLUMNS = [
    "dataset",
    "stage",
    "rows",
    "columns",
    "wall_seconds",
    "engine_duration_ms",
    "peak_working_set_mb",
    "peak_private_memory_mb",
    "output_size_mb",
    "status",
    "notes",
]
DEFAULT_INJECTION_PLAN = dict(cd.DEFAULT_INJECTION_PLAN)


@dataclass(frozen=True)
class InjectionSpec:
    seq: int
    anomaly_id: str
    issue_type: str
    row_index: int
    column: str
    new_value: Any = None


@dataclass(frozen=True)
class DuplicateSpec:
    seq: int
    anomaly_id: str
    source_row_index: int
    duplicate_offset: int


class ProcessMemorySampler:
    """Sample current-process memory without adding a dependency."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = float(interval_seconds)
        self.peak_working_set_bytes: int | None = None
        self.peak_private_bytes: int | None = None
        self.note = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ProcessMemorySampler":
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample_once()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            working_set, private_bytes = _current_process_memory_bytes()
        except Exception as exc:  # pragma: no cover - platform fallback
            self.note = f"memory sampling unavailable: {exc}"
            return
        if working_set is not None:
            self.peak_working_set_bytes = max(self.peak_working_set_bytes or 0, int(working_set))
        if private_bytes is not None:
            self.peak_private_bytes = max(self.peak_private_bytes or 0, int(private_bytes))

    def peak_working_set_mb(self) -> float | str:
        return _bytes_to_mb(self.peak_working_set_bytes)

    def peak_private_memory_mb(self) -> float | str:
        return _bytes_to_mb(self.peak_private_bytes)


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _current_process_memory_bytes() -> tuple[int | None, int | None]:
    if os.name == "nt":
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi.dll")
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        ok = get_process_memory_info(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)

    try:
        import resource
    except ImportError as exc:  # pragma: no cover - Windows is the expected env
        raise RuntimeError("resource module unavailable") from exc

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KB; macOS reports bytes. The project is Windows-first, so
    # this fallback intentionally stays conservative.
    rss_bytes = int(usage.ru_maxrss) * 1024
    return rss_bytes, None


def _bytes_to_mb(value: int | None) -> float | str:
    if value is None:
        return "N/A"
    return round(float(value) / (1024.0 * 1024.0), 6)


def _display_path(path: str | Path) -> str:
    return cd._display_path(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    cd._write_rows(path, rows, columns)


def _file_size_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024.0 * 1024.0), 6)


def _rate(numerator: int, denominator: int) -> float:
    return cd._rate(numerator, denominator)


def _next_anomaly_id(seq: int) -> str:
    return f"{SOURCE_DATASET}-{seq:04d}"


def _choose_rows(rng: random.Random, row_count: int, count: int, excluded: set[int] | None = None) -> list[int]:
    if count <= 0:
        return []
    if excluded:
        if count > row_count - len(excluded):
            raise ValueError(f"cannot choose {count} rows from {row_count - len(excluded)} available rows")
        rows: set[int] = set()
        while len(rows) < count:
            candidate = rng.randrange(row_count)
            if candidate not in excluded:
                rows.add(candidate)
        return list(rows)
    return rng.sample(range(row_count), count)


def build_injection_specs(
    row_count: int,
    seed: int,
    injection_plan: dict[str, int] | None = None,
) -> tuple[dict[int, list[InjectionSpec]], list[DuplicateSpec]]:
    plan = dict(injection_plan or DEFAULT_INJECTION_PLAN)
    rng = random.Random(seed)
    used_cells: set[tuple[int, str]] = set()
    specs_by_row: dict[int, list[InjectionSpec]] = {}
    duplicate_specs: list[DuplicateSpec] = []
    seq = 1

    missing_columns = ["product_category", "payment_method", "unit_price", "discount"]
    outlier_values = {"quantity": 999, "unit_price": 9999.0, "discount": 1.5, "total_amount": 999999.0}
    rare_columns = ["product_category", "payment_method", "order_status"]

    def choose_cell(row_index: int, columns: list[str]) -> str:
        candidates = [column for column in columns if (row_index, column) not in used_cells]
        if not candidates:
            raise ValueError(f"no available injection cells for row {row_index}")
        column = rng.choice(candidates)
        used_cells.add((row_index, column))
        return column

    def add_spec(row_index: int, issue_type: str, column: str, new_value: Any = None) -> None:
        nonlocal seq
        spec = InjectionSpec(
            seq=seq,
            anomaly_id=_next_anomaly_id(seq),
            issue_type=issue_type,
            row_index=int(row_index),
            column=column,
            new_value=new_value,
        )
        specs_by_row.setdefault(int(row_index), []).append(spec)
        seq += 1

    for row_index in _choose_rows(rng, row_count, int(plan["missing_values"])):
        add_spec(row_index, "missing_values", choose_cell(row_index, missing_columns))

    for row_index in _choose_rows(rng, row_count, int(plan["numeric_outlier"])):
        column = choose_cell(row_index, list(outlier_values))
        add_spec(row_index, "numeric_outlier", column, outlier_values[column])

    for offset, row_index in enumerate(_choose_rows(rng, row_count, int(plan["rare_category"]))):
        column = choose_cell(row_index, rare_columns)
        add_spec(row_index, "rare_category", column, f"__LLV_RARE_{column.upper()}_{offset:03d}__")

    pay_time_used_rows = {row for row, column in used_cells if column == "pay_time"}
    for row_index in _choose_rows(rng, row_count, int(plan["cross_column_consistency"]), excluded=pay_time_used_rows):
        used_cells.add((row_index, "pay_time"))
        add_spec(row_index, "cross_column_consistency", "pay_time")

    for duplicate_offset, source_row_index in enumerate(_choose_rows(rng, row_count, int(plan["duplicate_record"]))):
        duplicate_specs.append(
            DuplicateSpec(
                seq=seq,
                anomaly_id=_next_anomaly_id(seq),
                source_row_index=int(source_row_index),
                duplicate_offset=int(duplicate_offset),
            )
        )
        seq += 1

    return specs_by_row, duplicate_specs


def _base_order_row(idx: int, rng: random.Random, row_count: int) -> dict[str, Any]:
    categories = ["electronics", "books", "grocery", "apparel", "home", "sports"]
    methods = ["card", "wallet", "bank_transfer", "paypal"]
    statuses = ["paid", "shipped", "completed", "pending", "refunded"]
    category = categories[(idx * 7) % len(categories)]
    quantity = 1 + ((idx * 7 + 3) % 5)
    unit_price = round(8.0 + ((idx * 17 + rng.randint(0, 9)) % 700) / 10.0, 2)
    discount = [0.0, 0.05, 0.1, 0.15][(idx * 3) % 4]
    order_time = 1_700_000_000 + idx * 90
    pay_time = order_time + 30 + ((idx * 13) % 900)
    total_amount = round(quantity * unit_price * (1.0 - discount), 2)
    order_id = f"ord-{idx:07d}"
    return {
        "row_id": f"orders-row-{idx:07d}",
        "source_row_id": order_id,
        "order_id": order_id,
        "user_id": f"user-{(idx % max(50, row_count // 20)):05d}",
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


def _truth_record_for_spec(row: dict[str, Any], spec: InjectionSpec, seed: int) -> dict[str, Any]:
    def record(
        *,
        column_name: str,
        original_value: Any,
        corrupted_value: Any,
        repairable: bool,
        consistency_rule_name: str = "",
        notes: str,
    ) -> dict[str, Any]:
        return {
            "anomaly_id": spec.anomaly_id,
            "dataset": SOURCE_DATASET,
            "expected_issue_type": spec.issue_type,
            "row_index": int(spec.row_index),
            "column_name": column_name,
            "original_value": cd._value_text(original_value),
            "corrupted_value": cd._value_text(corrupted_value),
            "repairable": bool(repairable),
            "source_row_id": cd._value_text(row.get("source_row_id", "")),
            "duplicate_group": "",
            "consistency_rule_name": consistency_rule_name,
            "created_by_seed": int(seed),
            "notes": notes,
        }

    if spec.issue_type == "missing_values":
        original = row[spec.column]
        row[spec.column] = ""
        return record(
            column_name=spec.column,
            original_value=original,
            corrupted_value="",
            repairable=True,
            notes="Cell was replaced with a missing value in the large labeled validation run.",
        )

    if spec.issue_type == "numeric_outlier":
        original = row[spec.column]
        row[spec.column] = spec.new_value
        return record(
            column_name=spec.column,
            original_value=original,
            corrupted_value=spec.new_value,
            repairable=True,
            notes="Numeric value was moved outside the normal synthetic range.",
        )

    if spec.issue_type == "rare_category":
        original = row[spec.column]
        row[spec.column] = spec.new_value
        return record(
            column_name=spec.column,
            original_value=original,
            corrupted_value=spec.new_value,
            repairable=True,
            notes="Categorical value was replaced with a deterministic singleton category.",
        )

    if spec.issue_type == "cross_column_consistency":
        left_value = int(row["order_time"])
        right_value = int(row["pay_time"])
        new_value = left_value - 120
        row["pay_time"] = new_value
        return record(
            column_name="order_time,pay_time",
            original_value=f"{left_value}<={right_value}",
            corrupted_value=f"{left_value}>{new_value}",
            repairable=False,
            consistency_rule_name=cd.DATASET_CONFIGS[SOURCE_DATASET].consistency_rule_name,
            notes="pay_time was made smaller than order_time.",
        )

    raise ValueError(f"unsupported injection type: {spec.issue_type}")


def generate_streaming_orders_dataset(
    *,
    dataset_name: str,
    row_count: int,
    seed: int,
    dataset_work_dir: Path,
    injection_plan: dict[str, int] | None = None,
) -> dict[str, Any]:
    plan = dict(injection_plan or DEFAULT_INJECTION_PLAN)
    dataset_work_dir.mkdir(parents=True, exist_ok=True)
    specs_by_row, duplicate_specs = build_injection_specs(row_count, seed, plan)
    duplicate_source_rows = {item.source_row_index for item in duplicate_specs}
    duplicate_rows: dict[int, dict[str, Any]] = {}
    truth_by_seq: dict[int, dict[str, Any]] = {}
    rng = random.Random(seed)

    corrupted_csv = dataset_work_dir / "corrupted.csv"
    ground_truth_csv = dataset_work_dir / "ground_truth.csv"

    with corrupted_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORDER_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for idx in range(row_count):
            row = _base_order_row(idx, rng, row_count)
            for spec in sorted(specs_by_row.get(idx, []), key=lambda item: item.seq):
                truth_by_seq[spec.seq] = _truth_record_for_spec(row, spec, seed)
            writer.writerow(row)
            if idx in duplicate_source_rows:
                duplicate_rows[idx] = dict(row)

        for spec in sorted(duplicate_specs, key=lambda item: item.seq):
            if spec.source_row_index not in duplicate_rows:
                raise RuntimeError(f"duplicate source row was not generated: {spec.source_row_index}")
            duplicate_row = dict(duplicate_rows[spec.source_row_index])
            source_row_id = duplicate_row["source_row_id"]
            original_row_id = duplicate_row["row_id"]
            duplicate_row["row_id"] = f"{SOURCE_DATASET}-dup-{spec.duplicate_offset:05d}"
            duplicate_row_index = row_count + spec.duplicate_offset
            writer.writerow(duplicate_row)
            truth_by_seq[spec.seq] = {
                "anomaly_id": spec.anomaly_id,
                "dataset": SOURCE_DATASET,
                "expected_issue_type": "duplicate_record",
                "row_index": int(duplicate_row_index),
                "column_name": "source_row_id",
                "original_value": cd._value_text(original_row_id),
                "corrupted_value": cd._value_text(duplicate_row["row_id"]),
                "repairable": False,
                "source_row_id": cd._value_text(source_row_id),
                "duplicate_group": cd._value_text(source_row_id),
                "consistency_rule_name": "",
                "created_by_seed": int(seed),
                "notes": "A duplicate row was appended with the same source_row_id.",
            }

    truth_rows = [truth_by_seq[key] for key in sorted(truth_by_seq)]
    ground_truth = pd.DataFrame(truth_rows, columns=cd.GROUND_TRUTH_COLUMNS)
    cd._write_csv(ground_truth_csv, ground_truth)

    summary = {
        "dataset": dataset_name,
        "source_dataset": SOURCE_DATASET,
        "seed": int(seed),
        "clean_rows": int(row_count),
        "clean_columns": len(ORDER_COLUMNS),
        "corrupted_rows": int(row_count + int(plan["duplicate_record"])),
        "corrupted_columns": len(ORDER_COLUMNS),
        "ground_truth_rows": int(len(ground_truth)),
        "injection_counts_by_type": {key: int(value) for key, value in plan.items()},
        "repairable_ground_truth_rows": int(ground_truth["repairable"].map(cd._to_bool).sum()),
        "non_repairable_ground_truth_rows": int((~ground_truth["repairable"].map(cd._to_bool)).sum()),
        "corrupted_csv": _display_path(corrupted_csv),
        "ground_truth_csv": _display_path(ground_truth_csv),
        "corrupted_file_size_bytes": int(corrupted_csv.stat().st_size),
        "scan_config": cd.scan_config_for_dataset(SOURCE_DATASET),
        "notes": [
            "Large labeled validation uses 100 injected anomalies to match the controlled baseline proportions.",
            "Duplicate records append new rows, so corrupted_rows = clean_rows + duplicate_record count.",
        ],
    }
    _write_json(dataset_work_dir / "injection_summary.json", summary)
    return summary


def _timed_stage(
    *,
    dataset_name: str,
    stage: str,
    rows: int,
    columns: int,
    output_path: Path | None,
    func: Callable[[], Any],
    notes: str = "",
) -> tuple[Any, dict[str, Any]]:
    gc.collect()
    sampler = ProcessMemorySampler()
    start = time.perf_counter()
    status = "ok"
    try:
        with sampler:
            result = func()
    except Exception:
        status = "failed"
        raise
    finally:
        wall_seconds = time.perf_counter() - start
    runtime_row = {
        "dataset": dataset_name,
        "stage": stage,
        "rows": int(rows),
        "columns": int(columns),
        "wall_seconds": round(wall_seconds, 6),
        "engine_duration_ms": int(round(wall_seconds * 1000.0)),
        "peak_working_set_mb": sampler.peak_working_set_mb(),
        "peak_private_memory_mb": sampler.peak_private_memory_mb(),
        "output_size_mb": _file_size_mb(output_path),
        "status": status,
        "notes": notes or sampler.note,
    }
    return result, runtime_row


def _issue_type_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    return {
        issue_type: sum(1 for issue in issues if str(issue.get("issue_type")) == issue_type)
        for issue_type in cd.ISSUE_TYPES
    }


def run_detection_stage(
    *,
    dataset_name: str,
    dataset_work_dir: Path,
    dataset_artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[int, str], str], dict[str, Any]]:
    corrupted_csv = dataset_work_dir / "corrupted.csv"
    ground_truth_csv = dataset_work_dir / "ground_truth.csv"
    ground_truth = pd.read_csv(ground_truth_csv)
    frame_pd = _load_dataframe_module("Large labeled validation scan")
    scan_config = cd.scan_config_for_dataset(SOURCE_DATASET)

    def do_scan() -> dict[str, Any]:
        corrupted = pd.read_csv(corrupted_csv)
        issues = _detect_issues_for_frame(corrupted, frame_pd, scan_config=scan_config)
        truth_events = cd.build_truth_events(ground_truth)
        prediction_events = cd.build_prediction_events(corrupted, issues)
        metric_rows = cd.detection_metric_rows(dataset_name, truth_events, prediction_events)
        selected_issue_ids = cd._selected_repair_issue_ids(issues)
        issue_cell_types = cd._selected_issue_cell_types(issues)
        result = {
            "rows": int(corrupted.shape[0]),
            "columns": int(corrupted.shape[1]),
            "issues": issues,
            "metric_rows": metric_rows,
            "selected_issue_ids": selected_issue_ids,
            "issue_cell_types": issue_cell_types,
        }
        return result

    scan_result, runtime_row = _timed_stage(
        dataset_name=dataset_name,
        stage="scan_detect",
        rows=0,
        columns=len(ORDER_COLUMNS),
        output_path=None,
        func=do_scan,
        notes="in-process full scan with ground-truth matching",
    )
    runtime_row["rows"] = int(scan_result["rows"])
    runtime_row["columns"] = int(scan_result["columns"])

    issues = scan_result["issues"]
    scan_summary = {
        "dataset": dataset_name,
        "source_dataset": SOURCE_DATASET,
        "csv_path": _display_path(corrupted_csv),
        "rows": int(scan_result["rows"]),
        "columns": int(scan_result["columns"]),
        "issue_count": int(len(issues)),
        "detected_item_count": int(sum(int(issue.get("count", 0)) for issue in issues)),
        "issue_type_counts": _issue_type_counts(issues),
        "selected_repair_issue_ids": list(scan_result["selected_issue_ids"]),
        "scan_config": scan_config,
        "runtime": runtime_row,
    }
    dataset_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(dataset_artifact_dir / "detection_metrics.csv", scan_result["metric_rows"], cd.DETECTION_COLUMNS)
    _write_json(dataset_artifact_dir / "scan_summary.json", scan_summary)

    # Do not keep full boolean masks alive after the stage. The returned
    # issue_cell_types mapping is compact enough for 1M repair side-effect
    # attribution and is not used by the 10M scan-only run.
    selected_issue_ids = {"issue_ids": list(scan_result["selected_issue_ids"])}
    issue_cell_types = dict(scan_result["issue_cell_types"])
    del scan_result
    del issues
    gc.collect()
    return scan_summary["runtime"], selected_issue_ids, issue_cell_types, {
        "metric_rows_path": dataset_artifact_dir / "detection_metrics.csv",
        "scan_summary": scan_summary,
    }


def run_repair_stage(
    *,
    dataset_name: str,
    dataset_work_dir: Path,
    dataset_artifact_dir: Path,
    selected_issue_ids: list[str],
    issue_cell_types: dict[tuple[int, str], str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    corrupted_csv = dataset_work_dir / "corrupted.csv"
    ground_truth_csv = dataset_work_dir / "ground_truth.csv"
    repaired_csv = dataset_work_dir / "repaired.csv"
    scan_config = cd.scan_config_for_dataset(SOURCE_DATASET)

    def do_repair() -> dict[str, Any]:
        return action_repair_batch(
            {
                "csv_path": str(corrupted_csv),
                "issue_ids": selected_issue_ids,
                "scan_config": scan_config,
                "repair_strategy": cd.repair_strategy(),
                "plan_only": False,
                "write_output": True,
                "enable_rollback": True,
                "output_csv": str(repaired_csv),
            }
        )

    repair_result, runtime_row = _timed_stage(
        dataset_name=dataset_name,
        stage="repair_evaluate",
        rows=0,
        columns=len(ORDER_COLUMNS),
        output_path=repaired_csv,
        func=do_repair,
        notes="repair_batch write_output=true with rollback manifest enabled",
    )

    corrupted = pd.read_csv(corrupted_csv)
    repaired = pd.read_csv(repaired_csv)
    ground_truth = pd.read_csv(ground_truth_csv)
    runtime_row["rows"] = int(corrupted.shape[0])
    runtime_row["columns"] = int(corrupted.shape[1])
    repair_rows, repairable_cells = cd._evaluate_repair_rows(corrupted, repaired, ground_truth)
    changed_cells, non_gt_by_type = cd._changed_cells(corrupted, repaired, repairable_cells, issue_cell_types)
    skipped_non_repairable_count = int((~ground_truth["repairable"].map(cd._to_bool)).sum())
    metric_rows = cd.repair_metric_rows(dataset_name, repair_rows, non_gt_by_type, skipped_non_repairable_count)
    side_rows = cd.side_effect_rows(dataset_name, ground_truth, non_gt_by_type, changed_cells)

    rollback = repair_result.get("rollback") if isinstance(repair_result.get("rollback"), dict) else {}
    repair_summary = {
        "dataset": dataset_name,
        "source_dataset": SOURCE_DATASET,
        "selected_issue_ids": selected_issue_ids,
        "repair_batch": {
            "selected_issue_count": repair_result.get("selected_issue_count"),
            "applied_issue_count": repair_result.get("applied_issue_count"),
            "total_cells_modified": repair_result.get("total_cells_modified"),
            "comparison": repair_result.get("comparison"),
            "skipped_issues": repair_result.get("skipped_issues"),
            "write_strategy_used": repair_result.get("write_strategy_used"),
            "streaming_replaced_cell_count": repair_result.get("streaming_replaced_cell_count"),
        },
        "repaired_csv": _display_path(repaired_csv),
        "repaired_file_size_bytes": int(repaired_csv.stat().st_size) if repaired_csv.exists() else 0,
        "rollback_manifest": rollback.get("manifest_path") or "",
        "rollback_manifest_generated": bool(rollback.get("manifest_path")),
        "runtime": runtime_row,
    }

    _write_rows(dataset_artifact_dir / "repair_metrics.csv", metric_rows, cd.REPAIR_COLUMNS)
    _write_rows(dataset_artifact_dir / "side_effect_summary.csv", side_rows, ["dataset", "metric", "issue_type", "count", "notes"])
    _write_json(dataset_artifact_dir / "repair_run_summary.json", repair_summary)

    del corrupted
    del repaired
    del ground_truth
    gc.collect()
    return metric_rows, runtime_row, repair_summary


def _copy_small_summary(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def run_experiment(
    *,
    dataset_name: str,
    row_count: int,
    seed: int,
    artifact_dir: Path,
    work_dir: Path,
    run_repair: bool,
) -> dict[str, Any]:
    dataset_artifact_dir = artifact_dir / dataset_name
    dataset_work_dir = work_dir / dataset_name
    dataset_artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_work_dir.mkdir(parents=True, exist_ok=True)

    def do_generate() -> dict[str, Any]:
        return generate_streaming_orders_dataset(
            dataset_name=dataset_name,
            row_count=row_count,
            seed=seed,
            dataset_work_dir=dataset_work_dir,
            injection_plan=DEFAULT_INJECTION_PLAN,
        )

    generation_summary, generation_runtime = _timed_stage(
        dataset_name=dataset_name,
        stage="generate_labeled_csv",
        rows=row_count,
        columns=len(ORDER_COLUMNS),
        output_path=dataset_work_dir / "corrupted.csv",
        func=do_generate,
        notes="streaming deterministic generation and 100-anomaly injection",
    )
    _copy_small_summary(dataset_work_dir / "injection_summary.json", dataset_artifact_dir / "injection_summary.json")

    detection_runtime, selected_payload, issue_cell_types, detection_artifacts = run_detection_stage(
        dataset_name=dataset_name,
        dataset_work_dir=dataset_work_dir,
        dataset_artifact_dir=dataset_artifact_dir,
    )

    repair_metric_rows: list[dict[str, Any]] = []
    repair_runtime: dict[str, Any] | None = None
    repair_summary: dict[str, Any] | None = None
    if run_repair:
        repair_metric_rows, repair_runtime, repair_summary = run_repair_stage(
            dataset_name=dataset_name,
            dataset_work_dir=dataset_work_dir,
            dataset_artifact_dir=dataset_artifact_dir,
            selected_issue_ids=list(selected_payload["issue_ids"]),
            issue_cell_types=issue_cell_types,
        )
    else:
        _write_json(
            dataset_artifact_dir / "repair_run_summary.json",
            {
                "dataset": dataset_name,
                "source_dataset": SOURCE_DATASET,
                "repair_evaluated": False,
                "reason": "10M labeled run is scan/detection-only; repair accuracy at this scale is future work.",
            },
        )

    detection_metrics = pd.read_csv(detection_artifacts["metric_rows_path"]).to_dict(orient="records")
    runtime_rows = [generation_runtime, detection_runtime]
    if repair_runtime is not None:
        runtime_rows.append(repair_runtime)

    return {
        "dataset": dataset_name,
        "row_count": int(row_count),
        "corrupted_rows": int(generation_summary["corrupted_rows"]),
        "ground_truth_rows": int(generation_summary["ground_truth_rows"]),
        "repairable_ground_truth_rows": int(generation_summary["repairable_ground_truth_rows"]),
        "run_repair": bool(run_repair),
        "work_dir": _display_path(dataset_work_dir),
        "artifact_dir": _display_path(dataset_artifact_dir),
        "detection_metrics": detection_metrics,
        "repair_metrics": repair_metric_rows,
        "runtime_rows": runtime_rows,
        "scan_summary": detection_artifacts["scan_summary"],
        "repair_summary": repair_summary,
    }


def _dataset_names_for_args(args: argparse.Namespace) -> list[str]:
    requested = str(args.run or "both").strip().lower()
    if requested == "both":
        return ["orders_transactions_1m_labeled", "orders_transactions_10m_labeled"]
    if requested in {"1m", "orders_transactions_1m_labeled"}:
        return ["orders_transactions_1m_labeled"]
    if requested in {"10m", "orders_transactions_10m_labeled"}:
        return ["orders_transactions_10m_labeled"]
    raise ValueError("--run must be one of both, 1m, 10m")


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.output_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    experiments: list[dict[str, Any]] = []
    all_detection_rows: list[dict[str, Any]] = []
    all_repair_rows: list[dict[str, Any]] = []
    all_runtime_rows: list[dict[str, Any]] = []

    for dataset_name in _dataset_names_for_args(args):
        if dataset_name.endswith("1m_labeled"):
            row_count = int(args.rows_1m)
            seed = int(args.seed) + 1
            repair = True
        else:
            row_count = int(args.rows_10m)
            seed = int(args.seed) + 10
            repair = False
        experiment = run_experiment(
            dataset_name=dataset_name,
            row_count=row_count,
            seed=seed,
            artifact_dir=artifact_dir,
            work_dir=work_dir,
            run_repair=repair,
        )
        experiments.append(experiment)
        all_detection_rows.extend(experiment["detection_metrics"])
        all_repair_rows.extend(experiment["repair_metrics"])
        all_runtime_rows.extend(experiment["runtime_rows"])

    _write_rows(artifact_dir / "summary_detection_metrics.csv", all_detection_rows, cd.DETECTION_COLUMNS)
    _write_rows(artifact_dir / "summary_repair_metrics.csv", all_repair_rows, cd.REPAIR_COLUMNS)
    _write_rows(artifact_dir / "summary_runtime_memory.csv", all_runtime_rows, RUNTIME_COLUMNS)

    run_summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "large labeled validation bridge between controlled accuracy and large-scale stability experiments",
        "artifact_dir": _display_path(artifact_dir),
        "work_dir": _display_path(work_dir),
        "source_dataset": SOURCE_DATASET,
        "injection_plan": DEFAULT_INJECTION_PLAN,
        "assumptions": [
            "Each labeled scale uses 100 injected anomalies to match the controlled baseline proportions.",
            "10M repair accuracy is not evaluated in this run; only labeled detection scan metrics are reported.",
            "Peak memory is current-process working set/private memory sampled during the stage.",
        ],
        "experiments": experiments,
    }
    _write_json(artifact_dir / "run_summary.json", run_summary)
    return run_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large labeled orders_transactions validation experiments.")
    parser.add_argument("--run", default="both", help="Experiment to run: both, 1m, or 10m.")
    parser.add_argument("--rows-1m", type=int, default=DEFAULT_1M_ROWS, help="Base rows for the 1M labeled experiment.")
    parser.add_argument("--rows-10m", type=int, default=DEFAULT_10M_ROWS, help="Base rows for the 10M labeled experiment.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base deterministic seed.")
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Tracked artifact summary directory.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Ignored work directory for large CSV files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(_to_builtin({"output_dir": summary["artifact_dir"], "work_dir": summary["work_dir"]}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
