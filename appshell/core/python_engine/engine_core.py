"""Engine algorithm layer (independent from CLI transport)."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from engine_protocol import ErrorCode, KnownEngineError

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _require(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Missing required field: {key}",
            details={"field": key},
        )
    return value


def _resolve_input_csv(csv_path: str) -> Path:
    raw = Path(csv_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    cwd_candidate = raw.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / raw).resolve()
    return project_candidate


def _resolve_output_dir(output_dir: str | None) -> Path:
    if not output_dir:
        return (PROJECT_ROOT / "data" / "processed").resolve()

    raw = Path(output_dir).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _resolve_existing_dir(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas values to transport-safe Python builtins."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, (np.integer,)):
        return int(value)
    if np is not None and isinstance(value, (np.floating,)):
        return round(float(value), 12)
    if isinstance(value, float):
        return round(value, 12)
    if pd is not None and isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "f1",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1_weighted",
        "precision_weighted",
        "recall_weighted",
        "f1_anomaly",
        "precision_anomaly",
        "recall_anomaly",
        "decision_threshold",
        "threshold_optimization",
    ):
        if key in metrics:
            summary[key] = metrics[key]

    summary["confusion_matrix"] = metrics.get("confusion_matrix")
    summary["feature_importance"] = metrics.get("feature_importance", {})
    if "roc_curve" in metrics:
        summary["roc_curve"] = {
            "fpr": metrics["roc_curve"].get("fpr"),
            "tpr": metrics["roc_curve"].get("tpr"),
        }
    return _to_builtin(summary)


def _validate_train_target(df: Any, target_col: str, train_pd: Any) -> None:
    target = df[target_col]
    missing_count = int(target.isna().sum())
    non_null_target = target.dropna()
    non_null_rows = int(non_null_target.shape[0])
    unique_count = int(non_null_target.nunique(dropna=True))

    is_numeric = bool(train_pd.api.types.is_numeric_dtype(non_null_target))
    is_float_target = bool(train_pd.api.types.is_float_dtype(non_null_target))
    unique_ratio = float(unique_count) / float(non_null_rows) if non_null_rows else 0.0
    looks_continuous = (is_float_target and unique_count > 20) or (
        is_numeric and unique_count > 20 and unique_ratio > 0.2
    )
    if looks_continuous:
        raise KnownEngineError(
            code=ErrorCode.UNSUPPORTED_TARGET_TYPE,
            message=f"Target column appears continuous and is unsupported: {target_col}",
            details={
                "target_col": target_col,
                "unique_count": unique_count,
                "row_count": non_null_rows,
                "unique_ratio": round(unique_ratio, 6),
                "missing_count": missing_count,
                "reason": "continuous numeric targets are not supported by the current classification pipeline",
                "suggestion": "Choose a categorical label (for example: stroke, heart_disease) or switch to a regression workflow.",
            },
        )

    if missing_count > 0:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Target column contains missing values: {target_col}",
            details={
                "target_col": target_col,
                "missing_count": missing_count,
                "row_count": int(df.shape[0]),
                "reason": "target column has NaN values",
                "suggestion": "Fill or drop rows with missing target values before training.",
            },
        )

    if unique_count < 2:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Target column must contain at least 2 classes: {target_col}",
            details={
                "target_col": target_col,
                "unique_count": unique_count,
                "reason": "target has fewer than two distinct classes",
            },
        )

    class_counts = non_null_target.value_counts(dropna=False)
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count < 2:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Target column has classes with fewer than 2 rows: {target_col}",
            details={
                "target_col": target_col,
                "min_class_count": min_class_count,
                "reason": "stratified split requires at least 2 samples per class",
                "suggestion": "Merge rare classes or provide more rows for low-frequency classes.",
            },
        )


def _load_training_modules() -> tuple[Any, Any, Any]:
    try:
        import pandas as train_pd  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise KnownEngineError(
            code=ErrorCode.MISSING_DEPENDENCY,
            message="Training dependency missing: pandas",
            details={"reason": str(exc)},
        ) from exc

    try:
        from src.training_core import process_and_train, save_system_state  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise KnownEngineError(
            code=ErrorCode.TRAINING_MODULE_IMPORT_FAILED,
            message="Failed to import training modules",
            details={"reason": str(exc)},
        ) from exc

    return train_pd, process_and_train, save_system_state


def _load_repair_modules() -> tuple[Any, Any, Any]:
    try:
        import pandas as repair_pd  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise KnownEngineError(
            code=ErrorCode.MISSING_DEPENDENCY,
            message="Repair dependency missing: pandas",
            details={"reason": str(exc)},
        ) from exc

    try:
        from src.repair_core import repair_anomaly_sample  # type: ignore
        from src.training_core import load_system_state, predict_with_threshold  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise KnownEngineError(
            code=ErrorCode.REPAIR_MODULE_IMPORT_FAILED,
            message="Failed to import repair modules",
            details={"reason": str(exc)},
        ) from exc

    return load_system_state, predict_with_threshold, repair_anomaly_sample


def _to_positive_int(payload: dict[str, Any], key: str, default: int, minimum: int = 1, maximum: int = 1000) -> int:
    raw = payload.get(key, default)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be an integer",
            details={"field": key, "value": raw},
        ) from exc
    if value < minimum or value > maximum:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be between {minimum} and {maximum}",
            details={"field": key, "value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def _to_int(payload: dict[str, Any], key: str, default: int, minimum: int = 0, maximum: int = 10_000_000) -> int:
    raw = payload.get(key, default)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be an integer",
            details={"field": key, "value": raw},
        ) from exc
    if value < minimum or value > maximum:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be between {minimum} and {maximum}",
            details={"field": key, "value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def action_health(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": "python-anomaly-engine",
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "actions": ["health", "train", "repair"],
    }


def action_train(payload: dict[str, Any]) -> dict[str, Any]:
    train_pd, process_and_train, save_system_state = _load_training_modules()

    csv_path = _require(payload, "csv_path")
    target_col = str(_require(payload, "target_col"))
    output_dir = _resolve_output_dir(payload.get("output_dir"))

    csv_file = _resolve_input_csv(str(csv_path))
    if not csv_file.exists():
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    try:
        df = train_pd.read_csv(csv_file)
    except Exception as exc:  # pragma: no cover - serialization guard
        raise KnownEngineError(
            code=ErrorCode.CSV_READ_FAILED,
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc

    if target_col not in df.columns:
        raise KnownEngineError(
            code=ErrorCode.INVALID_TARGET_COLUMN,
            message=f"Target column not found: {target_col}",
            details={"available_columns": list(df.columns)},
        )

    _validate_train_target(df, target_col, train_pd)

    try:
        model, x_test, normal_data, metrics, feature_names = process_and_train(df, target_col)
        os.makedirs(output_dir, exist_ok=True)
        save_system_state(model, x_test, normal_data, feature_names, save_dir=output_dir)
    except KnownEngineError:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard
        raise KnownEngineError(
            code=ErrorCode.TRAINING_FAILED,
            message="Model training failed",
            details={"reason": str(exc)},
        ) from exc

    return {
        "artifacts": {
            "output_dir": str(output_dir),
            "model": str(output_dir / "model_lgb.pkl"),
            "test_data": str(output_dir / "test_data.pkl"),
            "normal_data": str(output_dir / "normal_data.pkl"),
            "config": str(output_dir / "config.pkl"),
        },
        "data_profile": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_col": target_col,
        },
        "metrics": _metric_summary(metrics),
    }


def action_repair(payload: dict[str, Any]) -> dict[str, Any]:
    load_system_state, predict_with_threshold, repair_anomaly_sample = _load_repair_modules()

    model_dir_text = str(_require(payload, "model_dir"))
    model_dir = _resolve_existing_dir(model_dir_text)
    if not model_dir.exists() or not model_dir.is_dir():
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Model directory does not exist: {model_dir}",
            details={"model_dir": str(model_dir)},
        )

    required_files = ["model_lgb.pkl", "test_data.pkl", "normal_data.pkl"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message="Model directory is missing required artifacts",
            details={"model_dir": str(model_dir), "missing_files": missing},
        )

    try:
        model, x_test, normal_data = load_system_state(model_dir)
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.MODEL_STATE_LOAD_FAILED,
            message="Failed to load model state artifacts",
            details={"model_dir": str(model_dir), "reason": str(exc)},
        ) from exc

    sample_index = _to_int(payload, "sample_index", default=0, minimum=0)
    if sample_index >= int(x_test.shape[0]):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="sample_index is out of range",
            details={
                "sample_index": sample_index,
                "min_index": 0,
                "max_index": max(0, int(x_test.shape[0]) - 1),
                "rows": int(x_test.shape[0]),
            },
        )

    dry_run = bool(payload.get("dry_run", False))
    if dry_run:
        max_changes = 0
    else:
        max_changes = _to_int(payload, "max_changes", default=3, minimum=1, maximum=20)
    k_neighbors = _to_positive_int(payload, "k_neighbors", default=9, minimum=3, maximum=200)

    immutable_raw = payload.get("immutable_columns", [])
    if immutable_raw is None:
        immutable_columns: list[str] = []
    elif isinstance(immutable_raw, (list, tuple, set)):
        immutable_columns = [str(item).strip() for item in immutable_raw if str(item).strip()]
    else:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field immutable_columns must be a string list",
            details={"field": "immutable_columns"},
        )

    numeric_bounds_raw = payload.get("numeric_bounds", {})
    if numeric_bounds_raw is None:
        numeric_bounds: dict[str, dict[str, Any]] = {}
    elif isinstance(numeric_bounds_raw, dict):
        numeric_bounds = {}
        for col, bound in numeric_bounds_raw.items():
            if isinstance(bound, dict):
                numeric_bounds[str(col)] = {
                    "min": bound.get("min"),
                    "max": bound.get("max"),
                }
    else:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field numeric_bounds must be an object",
            details={"field": "numeric_bounds"},
        )

    sample = x_test.iloc[[sample_index]].copy()
    before_pred_arr, before_prob_arr = predict_with_threshold(model, sample)
    before_pred = int(before_pred_arr[0])
    before_score = float(before_prob_arr[0])

    try:
        repair_bundle = repair_anomaly_sample(
            model=model,
            sample=sample,
            normal_data=normal_data,
            max_changes=max_changes,
            k_neighbors=k_neighbors,
            immutable_columns=immutable_columns,
            numeric_bounds=numeric_bounds,
        )
    except KnownEngineError:
        raise
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.REPAIR_FAILED,
            message="Repair search failed",
            details={"reason": str(exc)},
        ) from exc

    after_pred_arr, after_prob_arr = predict_with_threshold(model, repair_bundle.repaired_sample)
    after_pred = int(after_pred_arr[0])
    after_score = float(after_prob_arr[0])

    artifacts: dict[str, str] = {}
    output_dir_raw = payload.get("output_dir")
    if (not dry_run) and output_dir_raw is not None and str(output_dir_raw).strip():
        output_dir = _resolve_output_dir(str(output_dir_raw))
        os.makedirs(output_dir, exist_ok=True)
        repaired_csv = output_dir / f"repair_sample_{sample_index}.csv"
        report_json = output_dir / f"repair_sample_{sample_index}.json"
        repair_bundle.repaired_sample.to_csv(repaired_csv, index=False)
        report_json.write_text(
            json.dumps(
                {
                    "sample_index": sample_index,
                    "summary": _to_builtin(repair_bundle.summary),
                    "changes": _to_builtin(repair_bundle.changes),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts = {
            "output_dir": str(output_dir),
            "repaired_sample_csv": str(repaired_csv),
            "repair_report_json": str(report_json),
        }

    result: dict[str, Any] = {
        "model_dir": str(model_dir),
        "sample_index": sample_index,
        "dry_run": dry_run,
        "repair_summary": _to_builtin(repair_bundle.summary),
        "repair_changes": _to_builtin(repair_bundle.changes),
        "original_sample": _to_builtin(sample.iloc[0].to_dict()),
        "repaired_sample": _to_builtin(repair_bundle.repaired_sample.iloc[0].to_dict()),
        "before": {
            "pred": before_pred,
            "score": round(before_score, 12),
        },
        "after": {
            "pred": after_pred,
            "score": round(after_score, 12),
        },
        "data_profile": {
            "rows": int(x_test.shape[0]),
            "columns": int(x_test.shape[1]),
            "sample_index": sample_index,
        },
    }
    if artifacts:
        result["artifacts"] = artifacts

    return result
