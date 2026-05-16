"""Engine algorithm layer (independent from CLI transport)."""

from __future__ import annotations

import importlib
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from action_catalog import public_action_names
from engine_protocol import ErrorCode, KnownEngineError
from engine_logging import log_event

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

_SYSTEM_STATE_CACHE: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
_MODEL_IMPORTANCE_CACHE: dict[tuple[Any, ...], list[float] | None] = {}
DEFAULT_GOWER_AUTO_MAX_CANDIDATES = 512
DEFAULT_GOWER_FULL_SCAN_THRESHOLD = 5_000
DEFAULT_MISSFOREST_MAX_TRAIN_ROWS = 5_000
DEFAULT_MISSFOREST_RANDOM_STATE = 42
DEFAULT_MISSFOREST_MAX_ITER = 5
DEFAULT_MISSFOREST_CONVERGENCE_TOLERANCE = 0.001


def _emit_stage_progress(
    action: str,
    stage: str,
    phase: str,
    progress: int,
    message: str,
    **fields: Any,
) -> None:
    payload = {
        "action": str(action or "").strip(),
        "stage": str(stage or "").strip(),
        "phase": str(phase or "").strip(),
        "progress": int(max(0, min(100, int(progress)))),
        "message": str(message or "").strip(),
    }
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        payload[key] = value
    try:
        level = "warning" if payload["phase"] == "error" else "info"
        log_event(level, "stage_progress", **payload)
    except Exception:
        return


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


def _runtime_dependency_snapshot() -> dict[str, dict[str, Any]]:
    specs = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("lightgbm", "lightgbm"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
    ]

    snapshot: dict[str, dict[str, Any]] = {}
    for label, module_name in specs:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise KnownEngineError(
                code=ErrorCode.MISSING_DEPENDENCY,
                message=f"Runtime dependency check failed: {label}",
                details={
                    "dependency": label,
                    "module": module_name,
                    "reason": str(exc),
                },
            ) from exc

        version = getattr(module, "__version__", None)
        if not isinstance(version, str) or not version.strip():
            raise KnownEngineError(
                code=ErrorCode.MISSING_DEPENDENCY,
                message=f"Runtime dependency version is unavailable: {label}",
                details={
                    "dependency": label,
                    "module": module_name,
                    "reason": "__version__ is missing or empty",
                },
            )

        snapshot[label] = {
            "status": "ok",
            "module": module_name,
            "version": version.strip(),
        }
    return snapshot


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas values to transport-safe Python builtins."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if np is not None and isinstance(value, (np.integer,)):
        return int(value)
    if np is not None and isinstance(value, (np.floating,)):
        fv = float(value)
        if not math.isfinite(fv):
            return None
        return round(fv, 12)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    if pd is not None and isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "task_type",
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
        "mae",
        "rmse",
        "r2",
        "mape",
        "prediction_confidence_mean",
        "prediction_confidence_p10",
        "prediction_confidence_p90",
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
    feature_importance = metrics.get("feature_importance", {})
    if isinstance(feature_importance, dict):
        top_features = sorted(
            ((str(name), float(score)) for name, score in feature_importance.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        summary["explain_features"] = [
            {"feature": name, "importance": round(score, 6)} for name, score in top_features
        ]
    return _to_builtin(summary)


def _resolve_train_task_type(df: Any, target_col: str, train_pd: Any, requested_mode: str = "auto") -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if mode not in {"auto", "classification", "regression"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field task_type must be auto/classification/regression",
            details={"field": "task_type", "value": requested_mode},
        )

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
            message=f"Target column must contain at least 2 distinct values: {target_col}",
            details={
                "target_col": target_col,
                "unique_count": unique_count,
                "reason": "target has fewer than two distinct values",
            },
        )

    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "regression" if looks_continuous else "classification"

    if resolved_mode == "classification" and looks_continuous:
        raise KnownEngineError(
            code=ErrorCode.UNSUPPORTED_TARGET_TYPE,
            message=f"Target column appears continuous and is unsupported for classification: {target_col}",
            details={
                "target_col": target_col,
                "unique_count": unique_count,
                "row_count": non_null_rows,
                "unique_ratio": round(unique_ratio, 6),
                "missing_count": missing_count,
                "requested_task_type": "classification",
                "suggestion": "Use task_type=regression for continuous targets.",
            },
        )

    if resolved_mode == "classification":
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
    return resolved_mode


def _validate_train_target(df: Any, target_col: str, train_pd: Any, requested_mode: str = "auto") -> str:
    resolved_mode = _resolve_train_task_type(df, target_col, train_pd, requested_mode=requested_mode)
    if resolved_mode not in {"classification", "regression"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Unsupported resolved task_type: {resolved_mode}",
            details={"resolved_task_type": resolved_mode},
        )
    return resolved_mode


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


def _to_optional_positive_int(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: int = 1,
    maximum: int = 1_000_000,
) -> int | None:
    raw = payload.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
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


def _to_bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise KnownEngineError(
        code=ErrorCode.INVALID_INPUT,
        message=f"Field {key} must be a boolean",
        details={"field": key, "value": raw},
    )


def _to_float(
    payload: dict[str, Any],
    key: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1_000_000.0,
) -> float:
    raw = payload.get(key, default)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be a number",
            details={"field": key, "value": raw},
        ) from exc
    if value < minimum or value > maximum:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be between {minimum} and {maximum}",
            details={"field": key, "value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def _to_string_list(
    payload: dict[str, Any],
    key: str,
    default: list[str] | None = None,
    allow_empty: bool = True,
) -> list[str]:
    raw = payload.get(key, default if default is not None else [])
    if raw is None:
        return []

    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must be a string list",
            details={"field": key, "value": raw},
        )

    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text or text in result:
            continue
        result.append(text)

    if (not allow_empty) and (not result):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Field {key} must include at least one value",
            details={"field": key},
        )
    return result


def _normalize_consistency_rules(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field consistency_rules must be a rule list",
            details={"field": "consistency_rules", "value": raw},
        )

    rules: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Each consistency rule must be an object",
                details={"field": "consistency_rules", "index": idx, "value": item},
            )

        rule_type = str(item.get("type", "lte")).strip().lower()
        name = str(item.get("name") or f"rule_{idx + 1}").strip()
        if not name:
            name = f"rule_{idx + 1}"

        if rule_type in {"lte", "gte", "eq"}:
            left_col = str(item.get("left_col") or "").strip()
            right_col = str(item.get("right_col") or "").strip()
            if not left_col or not right_col:
                raise KnownEngineError(
                    code=ErrorCode.INVALID_INPUT,
                    message="Consistency rule requires left_col and right_col",
                    details={"field": "consistency_rules", "index": idx, "type": rule_type},
                )
            if rule_type == "gte":
                left_col, right_col = right_col, left_col
                rule_type = "lte"
            rules.append(
                {
                    "name": name,
                    "type": rule_type,
                    "left_col": left_col,
                    "right_col": right_col,
                }
            )
            continue

        if rule_type == "implies":
            if_col = str(item.get("if_col") or "").strip()
            then_col = str(item.get("then_col") or "").strip()
            if not if_col or not then_col:
                raise KnownEngineError(
                    code=ErrorCode.INVALID_INPUT,
                    message="Imply rule requires if_col and then_col",
                    details={"field": "consistency_rules", "index": idx},
                )
            rule: dict[str, Any] = {
                "name": name,
                "type": "implies",
                "if_col": if_col,
                "if_equals": item.get("if_equals"),
                "then_col": then_col,
            }
            if "then_in" in item:
                rule["then_in"] = _to_string_list(item, "then_in", default=[], allow_empty=False)
            else:
                rule["then_equals"] = item.get("then_equals")
            rules.append(rule)
            continue

        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Consistency rule type must be lte/gte/eq/implies",
            details={"field": "consistency_rules", "index": idx, "type": rule_type},
        )
    return rules


def _resolve_output_file(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _load_dataframe_module(action_label: str) -> Any:
    if pd is not None:
        return pd
    try:
        import pandas as runtime_pd  # type: ignore
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.MISSING_DEPENDENCY,
            message=f"{action_label} dependency missing: pandas",
            details={"reason": str(exc)},
        ) from exc
    return runtime_pd


def _scan_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested_raw = payload.get("scan_config")
    if nested_raw is None:
        nested: dict[str, Any] = {}
    elif isinstance(nested_raw, dict):
        nested = nested_raw
    else:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field scan_config must be an object",
            details={"field": "scan_config", "value": nested_raw},
        )

    merged = dict(nested)
    for key in (
        "max_bins",
        "max_issues",
        "numeric_iqr_factor",
        "robust_z_threshold",
        "rare_ratio_threshold",
        "rare_count_floor",
        "min_numeric_samples",
        "min_categorical_samples",
        "preview_limit",
        "enable_time_series_shift",
        "time_series_z_threshold",
        "time_series_min_points",
        "enable_cross_column_consistency",
        "consistency_rules",
        "enable_duplicate_record",
        "duplicate_subset",
        "auto_pair_constraints",
    ):
        if key in payload:
            merged[key] = payload[key]

    return {
        "max_bins": _to_positive_int(merged, "max_bins", default=120, minimum=20, maximum=360),
        "max_issues": _to_positive_int(merged, "max_issues", default=1000, minimum=10, maximum=5000),
        "numeric_iqr_factor": _to_float(merged, "numeric_iqr_factor", default=1.5, minimum=0.8, maximum=5.0),
        "robust_z_threshold": _to_float(merged, "robust_z_threshold", default=3.5, minimum=1.5, maximum=8.0),
        "rare_ratio_threshold": _to_float(merged, "rare_ratio_threshold", default=0.01, minimum=0.001, maximum=0.2),
        "rare_count_floor": _to_positive_int(merged, "rare_count_floor", default=2, minimum=1, maximum=30),
        "min_numeric_samples": _to_positive_int(merged, "min_numeric_samples", default=6, minimum=4, maximum=10000),
        "min_categorical_samples": _to_positive_int(
            merged, "min_categorical_samples", default=8, minimum=4, maximum=10000
        ),
        "preview_limit": _to_positive_int(merged, "preview_limit", default=5, minimum=1, maximum=20),
        "enable_time_series_shift": _to_bool(merged, "enable_time_series_shift", default=True),
        "time_series_z_threshold": _to_float(
            merged, "time_series_z_threshold", default=4.0, minimum=1.5, maximum=12.0
        ),
        "time_series_min_points": _to_positive_int(
            merged, "time_series_min_points", default=24, minimum=6, maximum=1_000_000
        ),
        "enable_cross_column_consistency": _to_bool(
            merged, "enable_cross_column_consistency", default=True
        ),
        "consistency_rules": _normalize_consistency_rules(merged.get("consistency_rules")),
        "enable_duplicate_record": _to_bool(merged, "enable_duplicate_record", default=True),
        "duplicate_subset": _to_string_list(merged, "duplicate_subset", default=[]),
        "auto_pair_constraints": _to_bool(merged, "auto_pair_constraints", default=True),
    }


def _severity_from_ratio(ratio: float) -> tuple[str, int]:
    if ratio >= 0.15:
        return "high", 0
    if ratio >= 0.05:
        return "medium", 1
    return "low", 2


def _severity_weight(severity: str) -> float:
    if severity == "high":
        return 1.25
    if severity == "medium":
        return 1.0
    return 0.72


def _issue_weight(issue_type: str) -> float:
    if issue_type == "numeric_outlier":
        return 1.2
    if issue_type == "time_series_shift":
        return 1.28
    if issue_type == "cross_column_consistency":
        return 1.18
    if issue_type == "missing_values":
        return 1.05
    if issue_type == "rare_category":
        return 0.9
    if issue_type == "duplicate_record":
        return 0.88
    return 1.0


def _issue_score(issue_type: str, ratio: float, severity: str) -> float:
    score = float(ratio) * 100.0 * _issue_weight(issue_type) * _severity_weight(severity)
    return round(score, 6)


def _risk_level_from_score(score: float) -> str:
    if score >= 65.0:
        return "high"
    if score >= 28.0:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def _index_to_builtin(index_value: Any) -> Any:
    try:
        return int(index_value)
    except Exception:
        return str(index_value)


def _preview_hits(series: Any, mask: Any, limit: int) -> list[dict[str, Any]]:
    hits = series.loc[mask]
    previews: list[dict[str, Any]] = []
    for idx, value in hits.head(limit).items():
        previews.append({"row": _index_to_builtin(idx), "value": _to_builtin(value)})
    return previews


def _issue_confidence(ratio: float, severity: str, signal_strength: float = 1.0) -> float:
    base = 0.35 + min(0.5, max(0.0, float(ratio) * 4.0))
    if severity == "high":
        base += 0.24
    elif severity == "medium":
        base += 0.12
    else:
        base += 0.05
    scaled = base * max(0.55, min(1.45, float(signal_strength)))
    return round(max(0.05, min(0.99, scaled)), 6)


_OUTLIER_RISK_RANK = {
    "mild": 0,
    "strong": 1,
    "extreme": 2,
}


def _finite_float_or_none(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        return None
    return resolved


def _rounded_float_or_none(value: Any, digits: int = 6) -> float | None:
    resolved = _finite_float_or_none(value)
    if resolved is None:
        return None
    return round(resolved, digits)


def _row_outlier_risk_level(
    *,
    relative_distance_to_bound: float,
    robust_z_value: float | None,
    iqr_hit: bool,
    robust_hit: bool,
    robust_z_threshold: float,
) -> str:
    robust_extreme = robust_z_value is not None and robust_z_value >= robust_z_threshold * 2.0
    if relative_distance_to_bound >= 1.0 or robust_extreme or (
        iqr_hit and robust_hit and relative_distance_to_bound >= 0.5
    ):
        return "extreme"
    if relative_distance_to_bound >= 0.2 or robust_hit or (iqr_hit and robust_hit):
        return "strong"
    return "mild"


def _dominant_outlier_risk_level(risk_counts: dict[str, int]) -> str:
    if sum(int(risk_counts.get(level, 0)) for level in ("mild", "strong", "extreme")) <= 0:
        return "mild"
    return max(
        ("mild", "strong", "extreme"),
        key=lambda level: (int(risk_counts.get(level, 0)), _OUTLIER_RISK_RANK[level]),
    )


def _outlier_policy_reason(issue_risk_level: str, auto_repair_eligible: bool) -> str:
    if issue_risk_level == "mild":
        return "mild_outlier_prompt_only"
    if auto_repair_eligible:
        return "strong_or_extreme_outlier_auto_candidate_requires_validation"
    return "mixed_outlier_mild_or_tie_prompt_only"


def _numeric_outlier_risk_metadata(
    series: Any,
    numeric_series: Any,
    outlier_mask: Any,
    iqr_mask: Any,
    robust_mask: Any,
    robust_z: Any | None,
    *,
    lower: float,
    upper: float,
    robust_z_threshold: float,
    preview_limit: int,
) -> dict[str, Any]:
    risk_counts = {"mild": 0, "strong": 0, "extreme": 0}
    row_evidence_preview: list[dict[str, Any]] = []
    max_relative_distance = 0.0
    max_robust_z: float | None = None
    both_hits = 0
    bound_span = max(abs(float(upper) - float(lower)), 1e-12)
    positions = [idx for idx, flag in enumerate(outlier_mask.tolist()) if bool(flag)]

    for pos in positions:
        value = _finite_float_or_none(numeric_series.iloc[pos])
        if value is None:
            continue

        if value < lower:
            bound_side = "lower"
            nearest_bound = lower
            absolute_distance = lower - value
        elif value > upper:
            bound_side = "upper"
            nearest_bound = upper
            absolute_distance = value - upper
        else:
            lower_distance = abs(value - lower)
            upper_distance = abs(value - upper)
            bound_side = "inside_bounds"
            nearest_bound = lower if lower_distance <= upper_distance else upper
            absolute_distance = 0.0

        relative_distance = absolute_distance / bound_span
        robust_z_value = None
        if robust_z is not None:
            robust_z_value = _finite_float_or_none(robust_z.iloc[pos])
        iqr_hit = bool(iqr_mask.iloc[pos])
        robust_hit = bool(robust_mask.iloc[pos])
        if iqr_hit and robust_hit:
            both_hits += 1
        row_risk_level = _row_outlier_risk_level(
            relative_distance_to_bound=relative_distance,
            robust_z_value=robust_z_value,
            iqr_hit=iqr_hit,
            robust_hit=robust_hit,
            robust_z_threshold=robust_z_threshold,
        )
        risk_counts[row_risk_level] += 1
        max_relative_distance = max(max_relative_distance, relative_distance)
        if robust_z_value is not None:
            max_robust_z = robust_z_value if max_robust_z is None else max(max_robust_z, robust_z_value)

        if len(row_evidence_preview) < preview_limit:
            row_evidence_preview.append(
                {
                    "row": _index_to_builtin(series.index[pos]),
                    "value": _to_builtin(series.iloc[pos]),
                    "iqr_hit": iqr_hit,
                    "robust_z_hit": robust_hit,
                    "bound_side": bound_side,
                    "nearest_bound": round(float(nearest_bound), 12),
                    "absolute_distance_to_bound": round(float(absolute_distance), 6),
                    "relative_distance_to_bound": round(float(relative_distance), 6),
                    "robust_z_value": _rounded_float_or_none(robust_z_value),
                    "row_risk_level": row_risk_level,
                }
            )

    issue_risk_level = max(
        (level for level, count in risk_counts.items() if count > 0),
        key=lambda level: _OUTLIER_RISK_RANK[level],
        default="mild",
    )
    auto_repair_eligible = (
        issue_risk_level in {"strong", "extreme"}
        and int(risk_counts["strong"]) + int(risk_counts["extreme"]) > int(risk_counts["mild"])
    )

    return {
        "outlier_risk_level": issue_risk_level,
        "auto_repair_eligible": bool(auto_repair_eligible),
        "outlier_policy_reason": _outlier_policy_reason(issue_risk_level, auto_repair_eligible),
        "outlier_evidence": {
            "risk_counts": risk_counts,
            "dominant_row_risk_level": _dominant_outlier_risk_level(risk_counts),
            "max_relative_distance_to_bound": round(float(max_relative_distance), 6),
            "max_robust_z": _rounded_float_or_none(max_robust_z),
            "iqr_hits": int(iqr_mask.sum()),
            "robust_hits": int(robust_mask.sum()),
            "both_iqr_and_robust_hits": int(both_hits),
            "row_evidence_preview": row_evidence_preview,
        },
    }


def _preview_shift_hits(series: Any, delta: Any, zscore: Any, mask: Any, limit: int) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    positions = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]
    for pos in positions[:limit]:
        row_index = _index_to_builtin(series.index[pos])
        current_value = series.iloc[pos]
        prev_value = series.iloc[pos - 1] if pos > 0 else None
        delta_value = delta.iloc[pos]
        z_value = zscore.iloc[pos]
        previews.append(
            {
                "row": row_index,
                "previous": _to_builtin(prev_value),
                "current": _to_builtin(current_value),
                "delta": _to_builtin(delta_value),
                "shift_z": _to_builtin(z_value),
            }
        )
    return previews


def _preview_consistency_hits(df: Any, mask: Any, columns: list[str], limit: int) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    positions = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]
    for pos in positions[:limit]:
        row_values: dict[str, Any] = {}
        for column in columns:
            row_values[column] = _to_builtin(df[column].iloc[pos])
        previews.append({"row": _index_to_builtin(df.index[pos]), "values": row_values})
    return previews


def _preview_duplicate_hits(df: Any, mask: Any, subset_columns: list[str], limit: int) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    positions = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]
    for pos in positions[:limit]:
        row_values: dict[str, Any] = {}
        for column in subset_columns:
            row_values[column] = _to_builtin(df[column].iloc[pos])
        previews.append({"row": _index_to_builtin(df.index[pos]), "signature": row_values})
    return previews


def _normalize_rule_name(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in value).strip("_") or "rule"


def _auto_pair_consistency_rules(columns: list[str]) -> list[dict[str, Any]]:
    lower_map: dict[str, str] = {}
    for col in columns:
        key = str(col).strip().lower()
        if key and key not in lower_map:
            lower_map[key] = str(col)

    pair_set: set[tuple[str, str, str]] = set()
    for col in columns:
        col_text = str(col)
        lower = col_text.strip().lower()
        candidates: list[tuple[str, str]] = []
        if lower.endswith("_min"):
            candidates.append((col_text, lower[:-4] + "_max"))
        if lower.startswith("min_"):
            candidates.append((col_text, "max_" + lower[4:]))
        if lower.endswith("_start"):
            candidates.append((col_text, lower[:-6] + "_end"))
        if lower.startswith("start_"):
            candidates.append((col_text, "end_" + lower[6:]))

        for left_col, right_key in candidates:
            right_col = lower_map.get(right_key)
            if not right_col or right_col == left_col:
                continue
            pair_set.add((left_col, right_col, "lte"))

    rules: list[dict[str, Any]] = []
    for left_col, right_col, rule_type in sorted(pair_set):
        rules.append(
            {
                "name": f"auto_{left_col}_lte_{right_col}",
                "type": rule_type,
                "left_col": left_col,
                "right_col": right_col,
            }
        )
    return rules


def _build_consistency_rules(columns: list[str], scan_config: dict[str, Any]) -> list[dict[str, Any]]:
    custom_rules = list(scan_config.get("consistency_rules", []))
    auto_rules: list[dict[str, Any]] = []
    if bool(scan_config.get("auto_pair_constraints", True)):
        auto_rules = _auto_pair_consistency_rules(columns)

    seen_keys: set[tuple[str, str, str, str]] = set()
    merged_rules: list[dict[str, Any]] = []
    for rule in custom_rules + auto_rules:
        rule_type = str(rule.get("type", "")).strip().lower()
        key = (
            rule_type,
            str(rule.get("left_col", "")).strip(),
            str(rule.get("right_col", "")).strip(),
            str(rule.get("name", "")).strip(),
        )
        if rule_type == "implies":
            key = (
                rule_type,
                str(rule.get("if_col", "")).strip(),
                str(rule.get("then_col", "")).strip(),
                str(rule.get("name", "")).strip(),
            )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_rules.append(rule)
    return merged_rules


def _evaluate_consistency_rule(df: Any, frame_pd: Any, rule: dict[str, Any]) -> tuple[Any, dict[str, Any], float, str] | None:
    rule_type = str(rule.get("type", "")).strip().lower()
    name = str(rule.get("name") or "rule").strip() or "rule"

    if rule_type in {"lte", "eq"}:
        left_col = str(rule.get("left_col") or "").strip()
        right_col = str(rule.get("right_col") or "").strip()
        if left_col not in df.columns or right_col not in df.columns:
            return None

        left_series = df[left_col]
        right_series = df[right_col]
        valid_mask = left_series.notna() & right_series.notna()

        if rule_type == "lte":
            left_num = frame_pd.to_numeric(left_series, errors="coerce")
            right_num = frame_pd.to_numeric(right_series, errors="coerce")
            valid_mask = valid_mask & left_num.notna() & right_num.notna()
            violation_mask = valid_mask & (left_num > right_num)
            margin_series = (left_num - right_num).where(violation_mask)
            peak_margin = float(margin_series.max()) if int(violation_mask.sum()) > 0 else 0.0
            signal_strength = 1.0
            if peak_margin > 0:
                baseline = float(abs(right_num[valid_mask]).median()) if int(valid_mask.sum()) > 0 else 0.0
                signal_strength = 1.0 + min(0.45, peak_margin / max(1.0, baseline))
            detail = {
                "rule_name": name,
                "rule_type": "lte",
                "left_col": left_col,
                "right_col": right_col,
                "operator": "<=",
                "valid_rows": int(valid_mask.sum()),
                "peak_margin": round(peak_margin, 6),
            }
            return violation_mask, detail, signal_strength, left_col

        violation_mask = valid_mask & (left_series != right_series)
        detail = {
            "rule_name": name,
            "rule_type": "eq",
            "left_col": left_col,
            "right_col": right_col,
            "operator": "==",
            "valid_rows": int(valid_mask.sum()),
        }
        return violation_mask, detail, 1.0, left_col

    if rule_type == "implies":
        if_col = str(rule.get("if_col") or "").strip()
        then_col = str(rule.get("then_col") or "").strip()
        if if_col not in df.columns or then_col not in df.columns:
            return None

        if_series = df[if_col]
        then_series = df[then_col]
        if_value = rule.get("if_equals")
        cond_mask = if_series.notna() if if_value is None else (if_series == if_value)

        if "then_in" in rule:
            allowed = {str(v) for v in list(rule.get("then_in", []))}
            violation_mask = cond_mask & (~then_series.astype(str).isin(allowed))
            detail = {
                "rule_name": name,
                "rule_type": "implies",
                "if_col": if_col,
                "if_equals": _to_builtin(if_value),
                "then_col": then_col,
                "then_in": sorted(list(allowed)),
                "valid_rows": int(cond_mask.sum()),
            }
        else:
            then_value = rule.get("then_equals")
            violation_mask = cond_mask & (then_series != then_value)
            detail = {
                "rule_name": name,
                "rule_type": "implies",
                "if_col": if_col,
                "if_equals": _to_builtin(if_value),
                "then_col": then_col,
                "then_equals": _to_builtin(then_value),
                "valid_rows": int(cond_mask.sum()),
            }
        return violation_mask, detail, 1.0, then_col

    return None


def _build_bin_layout(row_count: int, max_bins: int) -> tuple[int, int]:
    if row_count <= 0:
        return 0, 1
    bin_count = max(1, min(max_bins, row_count))
    bin_size = int(math.ceil(float(row_count) / float(bin_count)))
    return bin_count, max(1, bin_size)


def _mask_to_bin_counts(mask: Any, row_count: int, max_bins: int) -> tuple[list[int], int]:
    bin_count, bin_size = _build_bin_layout(row_count=row_count, max_bins=max_bins)
    if bin_count <= 0:
        return [], bin_size

    counts: list[int] = []
    for idx in range(bin_count):
        start = idx * bin_size
        if start >= row_count:
            break
        end = min(row_count, (idx + 1) * bin_size)
        window = mask.iloc[start:end]
        counts.append(int(window.sum()))
    return counts, bin_size


def _bin_counts_to_heat(bin_counts: list[int], bin_size: int) -> list[float]:
    if bin_size <= 0:
        return [0.0 for _ in bin_counts]
    return [round(min(1.0, float(count) / float(bin_size)), 4) for count in bin_counts]


def _bin_counts_to_segments(bin_counts: list[int], row_count: int, bin_size: int, max_segments: int = 24) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    idx = 0
    while idx < len(bin_counts):
        if bin_counts[idx] <= 0:
            idx += 1
            continue

        segment_start = idx
        segment_count = 0
        segment_peak = 0
        while idx < len(bin_counts) and bin_counts[idx] > 0:
            value = int(bin_counts[idx])
            segment_count += value
            if value > segment_peak:
                segment_peak = value
            idx += 1

        start_row = segment_start * bin_size
        if start_row >= row_count:
            break
        end_row = min(row_count, idx * bin_size) - 1
        span_rows = max(1, end_row - start_row + 1)
        density = float(segment_count) / float(span_rows)
        segments.append(
            {
                "bin_start": int(segment_start),
                "bin_end": int(max(segment_start, idx - 1)),
                "start_row": int(start_row),
                "end_row": int(end_row),
                "count": int(segment_count),
                "peak_count_per_bin": int(segment_peak),
                "density": round(density, 6),
            }
        )
        if len(segments) >= max_segments:
            break
    return segments


def _build_issue_id(base: str, used_ids: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}#{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _detect_issues_for_frame(df: Any, frame_pd: Any, scan_config: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    row_count = int(df.shape[0])
    max_issues = int(scan_config["max_issues"])
    min_numeric_samples = int(scan_config["min_numeric_samples"])
    min_categorical_samples = int(scan_config["min_categorical_samples"])
    preview_limit = int(scan_config["preview_limit"])
    iqr_factor = float(scan_config["numeric_iqr_factor"])
    robust_z_threshold = float(scan_config["robust_z_threshold"])
    rare_ratio_threshold = float(scan_config["rare_ratio_threshold"])
    rare_count_floor = int(scan_config["rare_count_floor"])
    enable_time_series_shift = bool(scan_config.get("enable_time_series_shift", True))
    time_series_z_threshold = float(scan_config.get("time_series_z_threshold", 4.0))
    time_series_min_points = int(scan_config.get("time_series_min_points", 24))
    enable_cross_column_consistency = bool(scan_config.get("enable_cross_column_consistency", True))
    enable_duplicate_record = bool(scan_config.get("enable_duplicate_record", True))
    duplicate_subset = [col for col in list(scan_config.get("duplicate_subset", [])) if str(col).strip()]

    if row_count <= 0:
        return issues

    for raw_col in list(df.columns):
        if len(issues) >= max_issues:
            break
        column = str(raw_col)
        series = df[raw_col]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])

        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())
        if missing_count > 0:
            ratio = float(missing_count) / float(row_count)
            severity, severity_rank = _severity_from_ratio(ratio)
            issue_type = "missing_values"
            score = _issue_score(issue_type=issue_type, ratio=ratio, severity=severity)
            confidence = _issue_confidence(ratio=ratio, severity=severity)
            issues.append(
                {
                    "issue_id": _build_issue_id(f"{column}::missing_values", used_ids),
                    "column": column,
                    "issue_type": issue_type,
                    "count": missing_count,
                    "ratio": ratio,
                    "issue_score": score,
                    "severity": severity,
                    "severity_rank": severity_rank,
                    "confidence": confidence,
                    "explain_features": [
                        {"name": "missing_ratio", "value": round(ratio, 6)},
                        {"name": "missing_count", "value": missing_count},
                    ],
                    "mask": missing_mask,
                    "detail": {
                        "missing_count": missing_count,
                        "row_count": row_count,
                        "preview": _preview_hits(series, missing_mask, limit=preview_limit),
                    },
                    "repair_rule": {
                        "strategy": "fill_missing",
                    },
                }
            )

        if len(issues) >= max_issues or non_null_count < min_numeric_samples:
            continue

        is_numeric = bool(frame_pd.api.types.is_numeric_dtype(series))
        if is_numeric:
            numeric_series = frame_pd.to_numeric(series, errors="coerce")
            numeric_non_null = numeric_series.dropna()
            q1 = float(numeric_non_null.quantile(0.25))
            q3 = float(numeric_non_null.quantile(0.75))
            iqr = float(q3 - q1)
            iqr_mask = frame_pd.Series(False, index=series.index, dtype=bool)
            robust_mask = frame_pd.Series(False, index=series.index, dtype=bool)
            robust_z = None
            lower = float(q1)
            upper = float(q3)

            if iqr > 0.0:
                lower = float(q1 - iqr_factor * iqr)
                upper = float(q3 + iqr_factor * iqr)
                iqr_mask = numeric_series.notna() & ((numeric_series < lower) | (numeric_series > upper))

            median = float(numeric_non_null.median())
            mad = float((numeric_non_null - median).abs().median())
            if mad > 0.0:
                robust_z = (numeric_series - median).abs() / (1.4826 * mad)
                robust_mask = numeric_series.notna() & (robust_z > robust_z_threshold)
                if iqr <= 0.0:
                    lower = float(median - robust_z_threshold * 1.4826 * mad)
                    upper = float(median + robust_z_threshold * 1.4826 * mad)

            outlier_mask = iqr_mask | robust_mask

            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                ratio = float(outlier_count) / float(row_count)
                severity, severity_rank = _severity_from_ratio(ratio)
                issue_type = "numeric_outlier"
                score = _issue_score(issue_type=issue_type, ratio=ratio, severity=severity)
                dominant_hits = max(int(iqr_mask.sum()), int(robust_mask.sum()))
                signal_strength = 1.0 + min(0.35, float(dominant_hits) / max(1.0, float(outlier_count)))
                confidence = _issue_confidence(ratio=ratio, severity=severity, signal_strength=signal_strength)
                outlier_risk_metadata = _numeric_outlier_risk_metadata(
                    series,
                    numeric_series,
                    outlier_mask,
                    iqr_mask,
                    robust_mask,
                    robust_z,
                    lower=lower,
                    upper=upper,
                    robust_z_threshold=robust_z_threshold,
                    preview_limit=preview_limit,
                )
                issues.append(
                    {
                        "issue_id": _build_issue_id(f"{column}::numeric_outlier", used_ids),
                        "column": column,
                        "issue_type": issue_type,
                        "count": outlier_count,
                        "ratio": ratio,
                        "issue_score": score,
                        "severity": severity,
                        "severity_rank": severity_rank,
                        "confidence": confidence,
                        **outlier_risk_metadata,
                        "explain_features": [
                            {"name": "outlier_ratio", "value": round(ratio, 6)},
                            {"name": "iqr_hits", "value": int(iqr_mask.sum())},
                            {"name": "robust_hits", "value": int(robust_mask.sum())},
                        ],
                        "mask": outlier_mask,
                        "detail": {
                            "lower_bound": round(lower, 12),
                            "upper_bound": round(upper, 12),
                            "outlier_count": outlier_count,
                            "iqr_factor": iqr_factor,
                            "robust_z_threshold": robust_z_threshold,
                            "iqr_hits": int(iqr_mask.sum()),
                            "robust_hits": int(robust_mask.sum()),
                            "preview": _preview_hits(series, outlier_mask, limit=preview_limit),
                        },
                        "repair_rule": {
                            "strategy": "clip",
                            "lower_bound": lower,
                            "upper_bound": upper,
                        },
                    }
                )
            if (
                enable_time_series_shift
                and len(issues) < max_issues
                and non_null_count >= time_series_min_points
            ):
                delta = numeric_series.diff()
                valid_shift = numeric_series.notna() & numeric_series.shift(1).notna() & delta.notna()
                delta_non_null = delta.loc[valid_shift]
                if int(delta_non_null.shape[0]) >= max(4, time_series_min_points - 1):
                    median_delta = float(delta_non_null.median())
                    mad_delta = float((delta_non_null - median_delta).abs().median())
                    if mad_delta > 0.0:
                        shift_z = (delta - median_delta).abs() / (1.4826 * mad_delta)
                    else:
                        std_delta = float(delta_non_null.std(ddof=0))
                        if std_delta > 0.0:
                            shift_z = (delta - median_delta).abs() / std_delta
                        else:
                            shift_z = frame_pd.Series(0.0, index=series.index, dtype=float)
                    shift_z = shift_z.fillna(0.0)
                    shift_mask = valid_shift & (shift_z > time_series_z_threshold)
                    shift_count = int(shift_mask.sum())
                    if shift_count > 0:
                        ratio = float(shift_count) / float(row_count)
                        severity, severity_rank = _severity_from_ratio(ratio)
                        issue_type = "time_series_shift"
                        score = _issue_score(issue_type=issue_type, ratio=ratio, severity=severity)
                        peak_shift_z = float(shift_z[shift_mask].max())
                        signal_strength = 1.0 + min(
                            0.45,
                            max(0.0, peak_shift_z / max(1e-6, time_series_z_threshold) - 1.0),
                        )
                        confidence = _issue_confidence(
                            ratio=ratio,
                            severity=severity,
                            signal_strength=signal_strength,
                        )
                        issues.append(
                            {
                                "issue_id": _build_issue_id(f"{column}::time_series_shift", used_ids),
                                "column": column,
                                "issue_type": issue_type,
                                "count": shift_count,
                                "ratio": ratio,
                                "issue_score": score,
                                "severity": severity,
                                "severity_rank": severity_rank,
                                "confidence": confidence,
                                "explain_features": [
                                    {"name": "shift_ratio", "value": round(ratio, 6)},
                                    {"name": "peak_shift_z", "value": round(peak_shift_z, 6)},
                                    {"name": "z_threshold", "value": round(time_series_z_threshold, 6)},
                                ],
                                "mask": shift_mask,
                                "detail": {
                                    "shift_count": shift_count,
                                    "z_threshold": time_series_z_threshold,
                                    "peak_shift_z": round(peak_shift_z, 6),
                                    "median_delta": round(median_delta, 6),
                                    "preview": _preview_shift_hits(
                                        numeric_series,
                                        delta,
                                        shift_z,
                                        shift_mask,
                                        limit=preview_limit,
                                    ),
                                },
                                "repair_rule": {
                                    "strategy": "manual_review",
                                },
                            }
                        )
            continue

        if non_null_count < min_categorical_samples:
            continue
        value_counts = non_null.value_counts(dropna=True)
        distinct_count = int(value_counts.shape[0])
        if distinct_count < 3:
            continue

        rare_threshold = max(rare_count_floor, int(math.ceil(non_null_count * rare_ratio_threshold)))
        rare_values = [val for val, count in value_counts.items() if int(count) <= rare_threshold]
        if not rare_values or len(rare_values) >= distinct_count:
            continue

        rare_mask = series.notna() & series.isin(rare_values)
        rare_count = int(rare_mask.sum())
        if rare_count <= 0:
            continue

        common_counts = value_counts[~value_counts.index.isin(rare_values)]
        if common_counts.empty:
            replacement = value_counts.index[0]
        else:
            replacement = common_counts.index[0]

        ratio = float(rare_count) / float(row_count)
        severity, severity_rank = _severity_from_ratio(ratio)
        issue_type = "rare_category"
        score = _issue_score(issue_type=issue_type, ratio=ratio, severity=severity)
        confidence = _issue_confidence(ratio=ratio, severity=severity, signal_strength=0.95)
        issues.append(
            {
                "issue_id": _build_issue_id(f"{column}::rare_category", used_ids),
                "column": column,
                "issue_type": issue_type,
                "count": rare_count,
                "ratio": ratio,
                "issue_score": score,
                "severity": severity,
                "severity_rank": severity_rank,
                "confidence": confidence,
                "explain_features": [
                    {"name": "rare_ratio", "value": round(ratio, 6)},
                    {"name": "rare_threshold", "value": rare_threshold},
                    {"name": "distinct_count", "value": distinct_count},
                ],
                "mask": rare_mask,
                "detail": {
                    "rare_count": rare_count,
                    "rare_threshold": rare_threshold,
                    "rare_values_preview": [str(v) for v in rare_values[:10]],
                    "rare_ratio_threshold": rare_ratio_threshold,
                    "preview": _preview_hits(series, rare_mask, limit=preview_limit),
                },
                "repair_rule": {
                    "strategy": "replace_rare",
                    "rare_values": list(rare_values),
                    "replacement_value": replacement,
                },
            }
        )

    if enable_cross_column_consistency and len(issues) < max_issues:
        consistency_rules = _build_consistency_rules([str(col) for col in list(df.columns)], scan_config)
        for rule in consistency_rules:
            if len(issues) >= max_issues:
                break
            evaluated = _evaluate_consistency_rule(df, frame_pd, rule)
            if evaluated is None:
                continue
            violation_mask, rule_detail, signal_strength, primary_column = evaluated
            violation_count = int(violation_mask.sum())
            if violation_count <= 0:
                continue

            ratio = float(violation_count) / float(row_count)
            severity, severity_rank = _severity_from_ratio(ratio)
            issue_type = "cross_column_consistency"
            score = _issue_score(issue_type=issue_type, ratio=ratio, severity=severity)
            confidence = _issue_confidence(ratio=ratio, severity=severity, signal_strength=signal_strength)

            rule_name = _normalize_rule_name(str(rule_detail.get("rule_name", "rule")))
            preview_columns: list[str] = []
            if "left_col" in rule_detail and "right_col" in rule_detail:
                preview_columns = [str(rule_detail["left_col"]), str(rule_detail["right_col"])]
            elif "if_col" in rule_detail and "then_col" in rule_detail:
                preview_columns = [str(rule_detail["if_col"]), str(rule_detail["then_col"])]
            rule_detail["violated_count"] = violation_count
            rule_detail["row_count"] = row_count
            rule_detail["preview"] = _preview_consistency_hits(df, violation_mask, preview_columns, limit=preview_limit)

            issues.append(
                {
                    "issue_id": _build_issue_id(
                        f"{primary_column}::cross_column_consistency::{rule_name}",
                        used_ids,
                    ),
                    "column": primary_column,
                    "issue_type": issue_type,
                    "count": violation_count,
                    "ratio": ratio,
                    "issue_score": score,
                    "severity": severity,
                    "severity_rank": severity_rank,
                    "confidence": confidence,
                    "explain_features": [
                        {"name": "rule", "value": str(rule_detail.get("rule_name", "rule"))},
                        {"name": "violation_ratio", "value": round(ratio, 6)},
                        {"name": "valid_rows", "value": int(rule_detail.get("valid_rows", 0))},
                    ],
                    "mask": violation_mask,
                    "detail": rule_detail,
                    "repair_rule": {
                        "strategy": "manual_review",
                        "rule_name": str(rule_detail.get("rule_name", "rule")),
                    },
                }
            )

    if enable_duplicate_record and len(issues) < max_issues:
        subset_columns = [col for col in duplicate_subset if col in df.columns]
        if not subset_columns:
            subset_columns = [str(col) for col in list(df.columns)]
        if subset_columns:
            duplicate_mask = df.duplicated(subset=subset_columns, keep=False)
            duplicate_count = int(duplicate_mask.sum())
            if duplicate_count > 0:
                duplicate_rows = df.loc[duplicate_mask, subset_columns]
                group_sizes = duplicate_rows.value_counts(dropna=False) if not duplicate_rows.empty else frame_pd.Series(dtype=int)
                duplicate_groups = int(group_sizes.shape[0]) if not group_sizes.empty else 0
                largest_group = int(group_sizes.max()) if not group_sizes.empty else 0
                ratio = float(duplicate_count) / float(row_count)
                severity, severity_rank = _severity_from_ratio(ratio)
                issue_type = "duplicate_record"
                score = _issue_score(issue_type=issue_type, ratio=ratio, severity=severity)
                signal_strength = 1.0 + min(0.4, float(max(0, largest_group - 1)) / 10.0)
                confidence = _issue_confidence(ratio=ratio, severity=severity, signal_strength=signal_strength)
                issues.append(
                    {
                        "issue_id": _build_issue_id("row::duplicate_record", used_ids),
                        "column": subset_columns[0],
                        "issue_type": issue_type,
                        "count": duplicate_count,
                        "ratio": ratio,
                        "issue_score": score,
                        "severity": severity,
                        "severity_rank": severity_rank,
                        "confidence": confidence,
                        "explain_features": [
                            {"name": "duplicate_ratio", "value": round(ratio, 6)},
                            {"name": "duplicate_groups", "value": duplicate_groups},
                            {"name": "largest_group_size", "value": largest_group},
                        ],
                        "mask": duplicate_mask,
                        "detail": {
                            "duplicate_count": duplicate_count,
                            "duplicate_groups": duplicate_groups,
                            "largest_group_size": largest_group,
                            "subset_columns": subset_columns,
                            "preview": _preview_duplicate_hits(
                                df,
                                duplicate_mask,
                                subset_columns,
                                limit=preview_limit,
                            ),
                        },
                        "repair_rule": {
                            "strategy": "manual_review",
                        },
                    }
                )

    issues.sort(
        key=lambda item: (
            -float(item.get("issue_score", 0.0)),
            int(item["severity_rank"]),
            -int(item["count"]),
            str(item["issue_id"]),
        )
    )
    return issues[:max_issues]


def _repair_strategy_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("repair_strategy")
    if raw is None:
        source: dict[str, Any] = {}
    elif isinstance(raw, dict):
        source = raw
    else:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy must be an object",
            details={"field": "repair_strategy", "value": raw},
        )

    conflict_policy = str(source.get("conflict_policy", "first_wins")).strip().lower()
    if conflict_policy not in {"first_wins", "last_wins", "skip_conflict"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy.conflict_policy must be first_wins/last_wins/skip_conflict",
            details={"field": "repair_strategy.conflict_policy", "value": conflict_policy},
        )

    issue_priority_raw = source.get("issue_priority", ["missing_values", "numeric_outlier", "rare_category"])
    if not isinstance(issue_priority_raw, (list, tuple)):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy.issue_priority must be a list",
            details={"field": "repair_strategy.issue_priority", "value": issue_priority_raw},
        )
    issue_priority: list[str] = []
    for item in issue_priority_raw:
        name = str(item).strip().lower()
        if not name or name in issue_priority:
            continue
        issue_priority.append(name)
    for fallback in ("missing_values", "numeric_outlier", "rare_category"):
        if fallback not in issue_priority:
            issue_priority.append(fallback)

    missing_numeric = str(source.get("missing_numeric", "median")).strip().lower()
    if missing_numeric not in {"median", "mean", "constant"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy.missing_numeric must be median/mean/constant",
            details={"field": "repair_strategy.missing_numeric", "value": missing_numeric},
        )
    missing_categorical = str(source.get("missing_categorical", "mode")).strip().lower()
    if missing_categorical not in {"mode", "constant"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy.missing_categorical must be mode/constant",
            details={"field": "repair_strategy.missing_categorical", "value": missing_categorical},
        )
    outlier_strategy = str(source.get("outlier", "clip")).strip().lower()
    if outlier_strategy not in {"clip", "skip"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy.outlier must be clip/skip",
            details={"field": "repair_strategy.outlier", "value": outlier_strategy},
        )
    rare_strategy = str(source.get("rare_category", "mode")).strip().lower()
    if rare_strategy not in {"mode", "constant"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field repair_strategy.rare_category must be mode/constant",
            details={"field": "repair_strategy.rare_category", "value": rare_strategy},
        )

    return {
        "conflict_policy": conflict_policy,
        "issue_priority": issue_priority,
        "missing_numeric": missing_numeric,
        "missing_categorical": missing_categorical,
        "missing_constant_value": source.get("missing_constant_value", "UNKNOWN"),
        "outlier": outlier_strategy,
        "rare_category": rare_strategy,
        "rare_constant_value": source.get("rare_constant_value", "OTHER"),
        "preview_limit": _to_positive_int(source, "preview_limit", default=5, minimum=1, maximum=50),
    }


def _column_dependencies_from_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    raw = payload.get("column_dependencies")
    if raw is None:
        return {}

    deps: dict[str, list[str]] = {}

    def _normalize_dep_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            result: list[str] = []
            for item in value:
                text = str(item).strip()
                if text and text not in result:
                    result.append(text)
            return result
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="column_dependencies must contain string list values",
            details={"field": "column_dependencies", "value": value},
        )

    if isinstance(raw, dict):
        for key, value in raw.items():
            col = str(key).strip()
            if not col:
                continue
            dep_list = [item for item in _normalize_dep_list(value) if item != col]
            deps[col] = dep_list
        return deps

    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, dict):
                raise KnownEngineError(
                    code=ErrorCode.INVALID_INPUT,
                    message="column_dependencies list items must be objects",
                    details={"field": "column_dependencies", "value": item},
                )
            col = str(item.get("column") or "").strip()
            if not col:
                continue
            dep_list = [name for name in _normalize_dep_list(item.get("depends_on")) if name != col]
            deps[col] = dep_list
        return deps

    raise KnownEngineError(
        code=ErrorCode.INVALID_INPUT,
        message="Field column_dependencies must be an object or list",
        details={"field": "column_dependencies", "value": raw},
    )


def _topological_column_order(columns: list[str], dependencies: dict[str, list[str]]) -> tuple[list[str], set[str]]:
    selected = [col for col in columns if col]
    selected_set = set(selected)
    indegree: dict[str, int] = {}
    graph: dict[str, list[str]] = {}

    for col in selected:
        if col in indegree:
            continue
        indegree[col] = 0
        graph[col] = []

    for col in selected:
        for dep in dependencies.get(col, []):
            if dep not in selected_set:
                continue
            graph.setdefault(dep, [])
            graph[dep].append(col)
            indegree[col] = indegree.get(col, 0) + 1

    queue = [col for col in selected if indegree.get(col, 0) == 0]
    ordered: list[str] = []
    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for nxt in graph.get(current, []):
            indegree[nxt] = indegree.get(nxt, 0) - 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    cycle_columns = {col for col in selected if col not in set(ordered)}
    if cycle_columns:
        for col in selected:
            if col in cycle_columns:
                ordered.append(col)
    return ordered, cycle_columns


def _issue_mask_from_rule(series: Any, issue_type: str, rule: dict[str, Any], frame_pd: Any) -> Any:
    if issue_type == "missing_values":
        return series.isna()
    if issue_type == "numeric_outlier":
        lower_raw = rule.get("lower_bound")
        upper_raw = rule.get("upper_bound")
        if lower_raw is None or upper_raw is None:
            return frame_pd.Series(False, index=series.index, dtype=bool)
        try:
            lower = float(lower_raw)
            upper = float(upper_raw)
        except Exception:
            return frame_pd.Series(False, index=series.index, dtype=bool)
        return series.notna() & ((series < lower) | (series > upper))
    if issue_type == "rare_category":
        rare_values = list(rule.get("rare_values", []))
        if not rare_values:
            return frame_pd.Series(False, index=series.index, dtype=bool)
        return series.notna() & series.isin(rare_values)
    return frame_pd.Series(False, index=series.index, dtype=bool)


def _mode_or_fallback(series: Any, fallback: Any) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return fallback
    mode_values = non_null.mode(dropna=True)
    if mode_values.empty:
        return non_null.iloc[0]
    return mode_values.iloc[0]


def _replacement_for_missing(series: Any, frame_pd: Any, strategy: dict[str, Any]) -> Any:
    if frame_pd.api.types.is_numeric_dtype(series):
        method = str(strategy.get("missing_numeric", "median"))
        non_null = series.dropna()
        if non_null.empty:
            return strategy.get("missing_constant_value", 0.0)
        if method == "mean":
            return float(non_null.mean())
        if method == "constant":
            return strategy.get("missing_constant_value", 0.0)
        return float(non_null.median())

    method = str(strategy.get("missing_categorical", "mode"))
    if method == "constant":
        return strategy.get("missing_constant_value", "UNKNOWN")
    return _mode_or_fallback(series, "UNKNOWN")


def _replacement_for_rare(series: Any, rare_values: list[Any], strategy: dict[str, Any], rule: dict[str, Any]) -> Any:
    method = str(strategy.get("rare_category", "mode"))
    if method == "constant":
        return strategy.get("rare_constant_value", "OTHER")

    non_null = series.dropna()
    if non_null.empty:
        return strategy.get("rare_constant_value", "OTHER")
    if rare_values:
        filtered = non_null[~non_null.isin(list(rare_values))]
    else:
        filtered = non_null
    if filtered.empty:
        fallback = rule.get("replacement_value", strategy.get("rare_constant_value", "OTHER"))
        return fallback
    mode_values = filtered.mode(dropna=True)
    if mode_values.empty:
        return filtered.iloc[0]
    return mode_values.iloc[0]


def _issue_type_counter(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        issue_type = str(issue.get("issue_type", "unknown"))
        counts[issue_type] = counts.get(issue_type, 0) + 1
    return counts


def _issue_counter_by_column(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        column = str(issue.get("column", ""))
        if not column:
            continue
        counts[column] = counts.get(column, 0) + 1
    return counts


def _rollback_manifest_to_target(manifest: dict[str, Any], restore_target: str, custom_target: str | None = None) -> Path:
    if restore_target == "source_csv":
        return Path(str(manifest.get("source_csv") or "")).expanduser().resolve()
    if restore_target == "output_csv":
        output_csv = str(manifest.get("output_csv") or "").strip()
        if not output_csv:
            raise KnownEngineError(
                code=ErrorCode.ROLLBACK_FAILED,
                message="Rollback manifest does not contain output_csv",
                details={"restore_target": "output_csv"},
            )
        return Path(output_csv).expanduser().resolve()
    if restore_target == "custom":
        if not custom_target:
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Field target_csv is required when restore_target=custom",
                details={"field": "target_csv"},
            )
        return _resolve_output_file(custom_target)
    raise KnownEngineError(
        code=ErrorCode.INVALID_INPUT,
        message="Field restore_target must be source_csv/output_csv/custom",
        details={"field": "restore_target", "value": restore_target},
    )


def _build_manifest(
    *,
    manifest_version: int,
    source_tool_id: str,
    rollback_id: str,
    source_csv: Path,
    output_csv: Path,
    backup_csv: Path,
    selected_issue_ids: list[str],
    issue_source_map: dict[str, Any] | None = None,
    execution_steps: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": manifest_version,
        "source_tool_id": source_tool_id,
        "rollback_id": rollback_id,
        "created_at": int(time.time()),
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "backup_csv": str(backup_csv),
        "selected_issue_ids": list(selected_issue_ids),
        "issue_source_map": _to_builtin(issue_source_map or {}),
        "execution_steps": _to_builtin(execution_steps or []),
    }
    if extra:
        manifest.update(_to_builtin(extra))
    return manifest


def _create_rollback_artifacts(
    *,
    source_tool_id: str,
    csv_file: Path,
    output_path: Path,
    selected_issue_ids: list[str],
    issue_source_map: dict[str, Any] | None,
    execution_steps: list[dict[str, Any]] | None,
    payload: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    enable_rollback = _to_bool(payload, "enable_rollback", default=True)
    if not enable_rollback:
        return None, None

    rollback_dir_raw = str(payload.get("rollback_dir") or "").strip()
    if rollback_dir_raw:
        rollback_dir = _resolve_output_dir(rollback_dir_raw)
    else:
        rollback_dir = output_path.parent / ".rollback"
    os.makedirs(rollback_dir, exist_ok=True)

    rollback_id = f"rb-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    backup_csv = rollback_dir / f"{rollback_id}.{csv_file.name}.bak.csv"
    manifest_path = rollback_dir / f"{rollback_id}.json"
    shutil.copy2(csv_file, backup_csv)

    manifest = _build_manifest(
        manifest_version=2,
        source_tool_id=source_tool_id,
        rollback_id=rollback_id,
        source_csv=csv_file,
        output_csv=output_path,
        backup_csv=backup_csv,
        selected_issue_ids=selected_issue_ids,
        issue_source_map=issue_source_map,
        execution_steps=execution_steps,
        extra=extra,
    )
    manifest_path.write_text(json.dumps(_to_builtin(manifest), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    rollback_info = {
        "rollback_id": rollback_id,
        "manifest_path": str(manifest_path),
        "backup_csv": str(backup_csv),
        "restore_action": "rollback_repair_batch",
        "manifest_version": 2,
        "source_tool_id": source_tool_id,
    }
    return manifest, rollback_info


def _resolve_output_path(csv_file: Path, payload: dict[str, Any], *, default_suffix: str) -> Path:
    output_csv_raw = str(payload.get("output_csv") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if output_csv_raw:
        return _resolve_output_file(output_csv_raw)
    if output_dir_raw:
        output_dir = _resolve_output_dir(output_dir_raw)
        return output_dir / f"{csv_file.stem}{default_suffix}"
    return csv_file.with_name(f"{csv_file.stem}{default_suffix}")


def _csv_fingerprint(csv_file: Path) -> dict[str, Any]:
    resolved = csv_file.expanduser().resolve()
    stat = resolved.stat()
    return {
        "csv_path": str(resolved),
        "csv_size": int(stat.st_size),
        "csv_mtime_unix_nano": int(stat.st_mtime_ns),
    }


def _precomputed_issues_from_payload(
    payload: dict[str, Any],
    csv_file: Path,
    *,
    plan_only: bool,
) -> tuple[list[dict[str, Any]] | None, bool]:
    if not plan_only:
        return None, False
    raw_issues = payload.get("precomputed_issues")
    raw_meta = payload.get("precomputed_issues_meta")
    if not isinstance(raw_issues, list) or not isinstance(raw_meta, dict):
        return None, False

    try:
        current = _csv_fingerprint(csv_file)
        meta_path = Path(str(raw_meta.get("csv_path") or "")).expanduser().resolve()
    except Exception:
        return None, False

    if str(meta_path) != str(current["csv_path"]):
        return None, False
    try:
        if int(raw_meta.get("csv_size")) != int(current["csv_size"]):
            return None, False
        if int(raw_meta.get("csv_mtime_unix_nano")) != int(current["csv_mtime_unix_nano"]):
            return None, False
    except Exception:
        return None, False

    issues: list[dict[str, Any]] = []
    for item in raw_issues:
        if isinstance(item, dict):
            issues.append(dict(item))
    return issues, True


def _model_state_cache_key(model_dir: Path) -> tuple[Any, ...]:
    resolved = model_dir.expanduser().resolve()
    parts: list[Any] = [str(resolved)]
    for name in ("model_lgb.pkl", "normal_data.pkl", "config.pkl", "test_data.pkl"):
        path = resolved / name
        try:
            stat = path.stat()
            parts.extend([name, int(stat.st_size), int(stat.st_mtime_ns)])
        except OSError:
            parts.extend([name, -1, -1])
    return tuple(parts)


def _cached_load_system_state(model_dir: Path) -> tuple[Any, Any, Any]:
    key = _model_state_cache_key(model_dir)
    cached = _SYSTEM_STATE_CACHE.get(key)
    if cached is not None:
        return cached
    from src.training_core import load_system_state  # type: ignore

    state = load_system_state(model_dir)
    _SYSTEM_STATE_CACHE[key] = state
    return state


def _model_importance_weights(model_dir: Path, feature_columns: list[str]) -> list[float] | None:
    if not feature_columns:
        return None
    try:
        state_key = _model_state_cache_key(model_dir)
    except Exception:
        return None
    cache_key = (*state_key, tuple(str(column) for column in feature_columns))
    if cache_key in _MODEL_IMPORTANCE_CACHE:
        cached = _MODEL_IMPORTANCE_CACHE[cache_key]
        return list(cached) if cached is not None else None

    try:
        model, _, normal_data = _cached_load_system_state(model_dir)
    except Exception:
        return None

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        _MODEL_IMPORTANCE_CACHE[cache_key] = None
        return None
    values = list(importances)
    normal_columns = [str(column) for column in list(normal_data.columns)]
    if len(values) != len(normal_columns):
        _MODEL_IMPORTANCE_CACHE[cache_key] = None
        return None

    weight_map = {
        normal_columns[idx]: max(0.0, float(values[idx]))
        for idx in range(len(normal_columns))
    }
    weights = [float(weight_map.get(column, 0.0)) for column in feature_columns]
    if any(weight > 0.0 for weight in weights):
        _MODEL_IMPORTANCE_CACHE[cache_key] = list(weights)
        return weights
    _MODEL_IMPORTANCE_CACHE[cache_key] = None
    return None


def _gower_strategy_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("gower_strategy", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field gower_strategy must be an object",
            details={"field": "gower_strategy"},
        )

    strategy = {
        "k_neighbors": _to_positive_int(raw, "k_neighbors", default=5, minimum=1, maximum=200),
        "weight_mode": str(raw.get("weight_mode", "uniform") or "uniform").strip().lower(),
        "feature_weights": raw.get("feature_weights"),
        "preview_limit": _to_positive_int(raw, "preview_limit", default=5, minimum=1, maximum=50),
        "candidate_policy": str(raw.get("candidate_policy", "auto") or "auto").strip().lower(),
        "auto_max_candidates": _to_positive_int(
            raw,
            "auto_max_candidates",
            default=DEFAULT_GOWER_AUTO_MAX_CANDIDATES,
            minimum=32,
            maximum=1_000_000,
        ),
        "full_scan_threshold": _to_positive_int(
            raw,
            "full_scan_threshold",
            default=DEFAULT_GOWER_FULL_SCAN_THRESHOLD,
            minimum=1,
            maximum=10_000_000,
        ),
    }
    if "max_candidates" in raw:
        strategy["max_candidates"] = _to_optional_positive_int(raw, "max_candidates", minimum=1, maximum=1_000_000)
    elif "sample_size" in raw:
        strategy["max_candidates"] = _to_optional_positive_int(raw, "sample_size", minimum=1, maximum=1_000_000)
    else:
        strategy["max_candidates"] = None
    if strategy["candidate_policy"] not in {"auto", "sample", "full"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="gower_strategy.candidate_policy must be auto/sample/full",
            details={"field": "gower_strategy.candidate_policy", "value": strategy["candidate_policy"]},
        )
    if strategy["weight_mode"] not in {"uniform", "model_importance", "custom"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="gower_strategy.weight_mode must be uniform/model_importance/custom",
            details={"field": "gower_strategy.weight_mode", "value": strategy["weight_mode"]},
        )
    return strategy


def _resolve_gower_feature_weights(
    feature_columns: list[str],
    strategy: dict[str, Any],
    model_dir: Path | None,
) -> tuple[list[float] | None, str]:
    weight_mode = str(strategy.get("weight_mode", "uniform"))
    if weight_mode == "uniform":
        return None, "uniform"

    if weight_mode == "custom":
        raw = strategy.get("feature_weights")
        if raw is None:
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="gower_strategy.feature_weights is required when weight_mode=custom",
                details={"field": "gower_strategy.feature_weights"},
            )
        weights: list[float] = []
        if isinstance(raw, dict):
            for column in feature_columns:
                try:
                    weights.append(max(0.0, float(raw.get(column, 0.0))))
                except Exception as exc:
                    raise KnownEngineError(
                        code=ErrorCode.INVALID_INPUT,
                        message="Custom gower feature weights must be numeric",
                        details={"field": "gower_strategy.feature_weights", "column": column},
                    ) from exc
        elif isinstance(raw, (list, tuple)):
            if len(raw) != len(feature_columns):
                raise KnownEngineError(
                    code=ErrorCode.INVALID_INPUT,
                    message="Custom gower feature weights length must match feature columns",
                    details={
                        "field": "gower_strategy.feature_weights",
                        "expected": len(feature_columns),
                        "actual": len(raw),
                    },
                )
            for idx, item in enumerate(raw):
                try:
                    weights.append(max(0.0, float(item)))
                except Exception as exc:
                    raise KnownEngineError(
                        code=ErrorCode.INVALID_INPUT,
                        message="Custom gower feature weights must be numeric",
                        details={"field": "gower_strategy.feature_weights", "index": idx},
                    ) from exc
        else:
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="gower_strategy.feature_weights must be an object or list",
                details={"field": "gower_strategy.feature_weights"},
            )
        if not any(weight > 0.0 for weight in weights):
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Custom gower feature weights must contain at least one positive value",
                details={"field": "gower_strategy.feature_weights"},
            )
        return weights, "custom"

    if model_dir is not None and model_dir.exists():
        weights = _model_importance_weights(model_dir, feature_columns)
        if weights is not None:
            return weights, "model_importance"
    return None, "uniform"


def _stable_gower_sample_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _limit_gower_candidate_rows(
    candidate_rows: Any,
    max_candidates: int | None,
    *,
    issue_id: str,
    column: str,
    row_pos: int,
    candidate_policy: str = "auto",
    auto_max_candidates: int = DEFAULT_GOWER_AUTO_MAX_CANDIDATES,
    full_scan_threshold: int = DEFAULT_GOWER_FULL_SCAN_THRESHOLD,
) -> tuple[Any, int, int, bool, str]:
    pool_size = int(len(candidate_rows))
    selection_mode = "full"
    effective_limit: int | None = None
    if max_candidates is not None:
        effective_limit = int(max_candidates)
        selection_mode = "explicit_max_candidates"
    else:
        policy = str(candidate_policy or "auto").strip().lower()
        if policy == "full":
            effective_limit = None
            selection_mode = "full"
        elif policy == "sample":
            effective_limit = int(auto_max_candidates)
            selection_mode = "sample"
        else:
            if pool_size <= int(full_scan_threshold):
                effective_limit = None
                selection_mode = "auto_full"
            else:
                effective_limit = int(auto_max_candidates)
                selection_mode = "auto_sample"

    if effective_limit is None or pool_size <= effective_limit:
        return candidate_rows, pool_size, pool_size, False, selection_mode
    sample_size = int(effective_limit)
    seed = _stable_gower_sample_seed(issue_id, column, row_pos, pool_size, sample_size)
    sampled = candidate_rows.sample(n=sample_size, random_state=seed).sort_index()
    return sampled, pool_size, sample_size, True, selection_mode


def _missforest_strategy_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("missforest_strategy", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field missforest_strategy must be an object",
            details={"field": "missforest_strategy"},
        )

    max_features = str(raw.get("max_features", "sqrt") or "sqrt").strip().lower()
    if max_features == "none":
        resolved_max_features: str | None = None
    elif max_features in {"sqrt", "log2"}:
        resolved_max_features = max_features
    else:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="missforest_strategy.max_features must be sqrt/log2/none",
            details={"field": "missforest_strategy.max_features", "value": max_features},
        )

    algorithm_mode = str(raw.get("algorithm_mode", "iterative") or "iterative").strip().lower()
    if algorithm_mode not in {"iterative", "single_pass"}:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="missforest_strategy.algorithm_mode must be iterative/single_pass",
            details={"field": "missforest_strategy.algorithm_mode", "value": algorithm_mode},
        )

    return {
        "algorithm_mode": algorithm_mode,
        "max_iter": _to_positive_int(
            raw,
            "max_iter",
            default=DEFAULT_MISSFOREST_MAX_ITER,
            minimum=1,
            maximum=50,
        ),
        "convergence_tolerance": _to_float(
            raw,
            "convergence_tolerance",
            default=DEFAULT_MISSFOREST_CONVERGENCE_TOLERANCE,
            minimum=0.0,
            maximum=1.0,
        ),
        "n_estimators": _to_positive_int(raw, "n_estimators", default=40, minimum=5, maximum=300),
        "max_depth": _to_optional_positive_int(raw, "max_depth", minimum=1, maximum=100),
        "min_training_rows": _to_positive_int(raw, "min_training_rows", default=8, minimum=2, maximum=100_000),
        "max_train_rows": _to_positive_int(
            raw,
            "max_train_rows",
            default=DEFAULT_MISSFOREST_MAX_TRAIN_ROWS,
            minimum=32,
            maximum=1_000_000,
        ),
        "random_state": _to_int(
            raw,
            "random_state",
            default=DEFAULT_MISSFOREST_RANDOM_STATE,
            minimum=0,
            maximum=2_147_483_647,
        ),
        "max_features": resolved_max_features,
        "preview_limit": _to_positive_int(raw, "preview_limit", default=5, minimum=1, maximum=50),
    }


def _limit_missforest_training_rows(train_rows: Any, strategy: dict[str, Any], *, issue_id: str, column: str) -> tuple[Any, int, int, bool]:
    pool_size = int(len(train_rows))
    max_train_rows = int(strategy["max_train_rows"])
    if pool_size <= max_train_rows:
        return train_rows, pool_size, pool_size, False
    seed = _stable_gower_sample_seed("missforest", issue_id, column, pool_size, max_train_rows, strategy["random_state"])
    sampled = train_rows.sample(n=max_train_rows, random_state=seed).sort_index()
    return sampled, pool_size, max_train_rows, True


def _missforest_feature_matrices(
    frame_pd: Any,
    original_df: Any,
    feature_columns: list[str],
    train_rows: Any,
    predict_rows: Any,
) -> tuple[Any, Any]:
    if not feature_columns:
        train_features = frame_pd.DataFrame({"__constant__": [1.0] * int(len(train_rows))})
        predict_features = frame_pd.DataFrame({"__constant__": [1.0] * int(len(predict_rows))})
        return train_features, predict_features

    train_features = train_rows[feature_columns].copy()
    predict_features = predict_rows[feature_columns].copy()
    combined = frame_pd.concat([train_features, predict_features], axis=0, ignore_index=True)
    for column in feature_columns:
        source_series = original_df[column]
        if frame_pd.api.types.is_numeric_dtype(source_series):
            values = frame_pd.to_numeric(combined[column], errors="coerce")
            train_values = frame_pd.to_numeric(train_features[column], errors="coerce")
            if train_values.notna().any():
                fill_value = float(train_values.median())
            else:
                fill_value = 0.0
            combined[column] = values.fillna(fill_value)
        else:
            combined[column] = combined[column].astype("object").where(combined[column].notna(), "__MISSING__")
            combined[column] = combined[column].astype(str)

    encoded = frame_pd.get_dummies(combined, dummy_na=False)
    if encoded.shape[1] == 0:
        encoded = frame_pd.DataFrame({"__constant__": [1.0] * int(combined.shape[0])})
    encoded = encoded.astype(float)
    train_count = int(len(train_rows))
    return encoded.iloc[:train_count, :], encoded.iloc[train_count:, :]


def _missforest_regression_confidence(model: Any, x_pred: Any, train_target: Any) -> float:
    if np is None or not hasattr(model, "estimators_"):
        return 0.65
    try:
        x_values = x_pred.to_numpy() if hasattr(x_pred, "to_numpy") else x_pred
        tree_predictions = np.asarray([est.predict(x_values) for est in model.estimators_], dtype=float)
        if tree_predictions.size == 0:
            return 0.65
        prediction_spread = float(np.nanmean(np.nanstd(tree_predictions, axis=0)))
        target_spread = float(train_target.std()) if hasattr(train_target, "std") else 0.0
        scale = max(abs(target_spread), 1e-9)
        confidence = 1.0 - min(0.85, prediction_spread / scale)
        return round(max(0.1, min(0.95, confidence)), 6)
    except Exception:
        return 0.65


def _missforest_classifier_confidence(model: Any, x_pred: Any) -> float:
    try:
        probabilities = model.predict_proba(x_pred)
    except Exception:
        return 0.65
    if np is None:
        return 0.65
    try:
        max_probabilities = np.max(np.asarray(probabilities, dtype=float), axis=1)
        return round(max(0.1, min(0.99, float(np.nanmean(max_probabilities)))), 6)
    except Exception:
        return 0.65


def _missforest_predict_issue(
    *,
    frame_pd: Any,
    original_df: Any,
    issue: dict[str, Any],
    mask: Any,
    positions: list[int],
    strategy: dict[str, Any],
    random_forest_regressor: Any,
    random_forest_classifier: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issue_id = str(issue["issue_id"])
    column = str(issue["column"])
    series = original_df[column]
    healthy_mask = (~mask) & series.notna()
    is_numeric = bool(frame_pd.api.types.is_numeric_dtype(series))
    if is_numeric:
        numeric_series = frame_pd.to_numeric(series, errors="coerce")
        healthy_mask = healthy_mask & numeric_series.notna()

    train_rows = original_df.loc[healthy_mask]
    train_rows, train_pool_size, train_sample_size, train_sampled = _limit_missforest_training_rows(
        train_rows,
        strategy,
        issue_id=issue_id,
        column=column,
    )
    if int(len(train_rows)) < int(strategy["min_training_rows"]):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Not enough healthy rows for MissForest repair",
            details={
                "issue_id": issue_id,
                "column": column,
                "train_rows": int(len(train_rows)),
                "min_training_rows": int(strategy["min_training_rows"]),
            },
        )

    predict_rows = original_df.iloc[positions]
    feature_columns = [str(name) for name in list(original_df.columns) if str(name) != column]
    x_train, x_pred = _missforest_feature_matrices(frame_pd, original_df, feature_columns, train_rows, predict_rows)
    if is_numeric:
        y_train = frame_pd.to_numeric(train_rows[column], errors="coerce")
        valid_mask = y_train.notna()
        if int(valid_mask.sum()) < int(strategy["min_training_rows"]):
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Not enough numeric training targets for MissForest repair",
                details={"issue_id": issue_id, "column": column},
            )
        valid_positions = [idx for idx, flag in enumerate(valid_mask.tolist()) if bool(flag)]
        y_valid = y_train.loc[valid_mask]
        model = random_forest_regressor(
            n_estimators=int(strategy["n_estimators"]),
            max_depth=strategy["max_depth"],
            max_features=strategy["max_features"],
            random_state=int(strategy["random_state"]),
            n_jobs=1,
        )
        x_train_valid = x_train.iloc[valid_positions, :].to_numpy()
        x_pred_values = x_pred.to_numpy()
        model.fit(x_train_valid, y_valid)
        predictions = list(model.predict(x_pred_values))
        confidence = _missforest_regression_confidence(model, x_pred_values, y_valid)
        model_type = "random_forest_regressor"
    else:
        y_train = train_rows[column].astype(str)
        if int(y_train.nunique(dropna=True)) <= 0:
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Not enough categorical training targets for MissForest repair",
                details={"issue_id": issue_id, "column": column},
            )
        model = random_forest_classifier(
            n_estimators=int(strategy["n_estimators"]),
            max_depth=strategy["max_depth"],
            max_features=strategy["max_features"],
            random_state=int(strategy["random_state"]),
            n_jobs=1,
        )
        x_train_values = x_train.to_numpy()
        x_pred_values = x_pred.to_numpy()
        model.fit(x_train_values, y_train)
        predictions = list(model.predict(x_pred_values))
        confidence = _missforest_classifier_confidence(model, x_pred_values)
        model_type = "random_forest_classifier"

    changes: list[dict[str, Any]] = []
    col_pos = int(list(original_df.columns).index(column))
    for idx, pos in enumerate(positions):
        before_value = series.iat[pos]
        after_value = predictions[idx]
        before_is_nan = bool(frame_pd.isna(before_value))
        after_is_nan = bool(frame_pd.isna(after_value))
        if (before_is_nan and after_is_nan) or (
            (not before_is_nan) and (not after_is_nan) and before_value == after_value
        ):
            continue
        changes.append(
            {
                "issue_id": issue_id,
                "column": column,
                "row_pos": int(pos),
                "row": _index_to_builtin(original_df.index[pos]),
                "col_pos": col_pos,
                "before": before_value,
                "after": after_value,
            }
        )

    evidence = {
        "issue_id": issue_id,
        "column": column,
        "issue_type": str(issue["issue_type"]),
        "algorithm_mode": str(strategy["algorithm_mode"]),
        "iterations_run": 1,
        "converged": True,
        "convergence_delta": 0.0,
        "model_type": model_type,
        "feature_count": int(x_train.shape[1]),
        "train_pool_size": train_pool_size,
        "train_sample_size": train_sample_size,
        "train_sampled": bool(train_sampled),
        "target_cell_count": int(len(positions)),
        "candidate_confidence": confidence,
        "n_estimators": int(strategy["n_estimators"]),
        "max_train_rows": int(strategy["max_train_rows"]),
        "max_iter": int(strategy["max_iter"]),
        "convergence_tolerance": float(strategy["convergence_tolerance"]),
    }
    return changes, evidence


def _limit_missforest_training_positions(
    frame_pd: Any,
    positions: list[int],
    strategy: dict[str, Any],
    *,
    issue_id: str,
    column: str,
) -> tuple[list[int], int, int, bool]:
    normalized = [int(pos) for pos in positions]
    pool_size = int(len(normalized))
    max_train_rows = int(strategy["max_train_rows"])
    if pool_size <= max_train_rows:
        return normalized, pool_size, pool_size, False
    seed = _stable_gower_sample_seed(
        "missforest_iterative",
        issue_id,
        column,
        pool_size,
        max_train_rows,
        strategy["random_state"],
    )
    sampled = frame_pd.Series(normalized).sample(n=max_train_rows, random_state=seed).sort_values()
    return [int(pos) for pos in sampled.tolist()], pool_size, max_train_rows, True


def _missforest_initialize_working_frame(frame_pd: Any, original_df: Any, target_masks_by_column: dict[str, Any]) -> Any:
    working_df = original_df.copy(deep=True)
    for column, mask in target_masks_by_column.items():
        if column not in working_df.columns:
            continue
        if frame_pd.api.types.is_numeric_dtype(original_df[column]):
            working_df.loc[mask, column] = np.nan if np is not None else None
        else:
            working_df.loc[mask, column] = None

    for column in list(working_df.columns):
        if frame_pd.api.types.is_numeric_dtype(original_df[column]):
            values = frame_pd.to_numeric(working_df[column], errors="coerce")
            non_null = values.dropna()
            fill_value = float(non_null.median()) if int(non_null.shape[0]) > 0 else 0.0
            working_df[column] = values.fillna(fill_value)
            continue

        values = working_df[column].astype("object")
        non_null = values.dropna()
        if int(non_null.shape[0]) > 0:
            mode_values = non_null.mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else non_null.iloc[0]
        else:
            fill_value = "__MISSING__"
        working_df[column] = values.where(values.notna(), fill_value)
    return working_df


def _missforest_convergence_delta(
    frame_pd: Any,
    original_df: Any,
    previous_df: Any,
    current_df: Any,
    target_positions_by_column: dict[str, list[int]],
) -> float:
    deltas: list[float] = []
    for column, positions in target_positions_by_column.items():
        is_numeric = bool(frame_pd.api.types.is_numeric_dtype(original_df[column]))
        for pos in positions:
            before = previous_df[column].iat[int(pos)]
            after = current_df[column].iat[int(pos)]
            if is_numeric:
                before_value = _finite_float_or_none(before)
                after_value = _finite_float_or_none(after)
                if before_value is None or after_value is None:
                    deltas.append(0.0 if str(before) == str(after) else 1.0)
                    continue
                deltas.append(abs(after_value - before_value) / max(1.0, abs(before_value)))
            else:
                deltas.append(0.0 if str(before) == str(after) else 1.0)
    if not deltas:
        return 0.0
    return round(sum(deltas) / float(len(deltas)), 6)


def _missforest_predict_iterative(
    *,
    frame_pd: Any,
    original_df: Any,
    selected_issues: list[dict[str, Any]],
    issue_masks: dict[str, Any],
    issue_positions: dict[str, list[int]],
    strategy: dict[str, Any],
    random_forest_regressor: Any,
    random_forest_classifier: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    target_masks_by_column: dict[str, Any] = {}
    target_positions_by_column: dict[str, list[int]] = {}
    issue_by_id = {str(issue["issue_id"]): issue for issue in selected_issues}
    skipped: list[dict[str, Any]] = []

    for issue in selected_issues:
        issue_id = str(issue["issue_id"])
        column = str(issue["column"])
        mask = issue_masks.get(issue_id)
        if mask is None:
            continue
        if column not in target_masks_by_column:
            target_masks_by_column[column] = mask.copy()
        else:
            target_masks_by_column[column] = target_masks_by_column[column] | mask

    for column, mask in target_masks_by_column.items():
        target_positions_by_column[column] = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]

    working_df = _missforest_initialize_working_frame(frame_pd, original_df, target_masks_by_column)
    row_count = max(1, int(original_df.shape[0]))
    target_columns = sorted(
        target_positions_by_column.keys(),
        key=lambda column: (float(len(target_positions_by_column[column])) / float(row_count), str(column)),
    )
    if not target_columns:
        return {}, {}, skipped

    column_meta: dict[str, dict[str, Any]] = {}
    skipped_columns: dict[str, dict[str, Any]] = {}
    convergence_delta = 0.0
    converged = False
    iterations_run = 0
    feature_columns_by_target = {
        column: [str(name) for name in list(original_df.columns) if str(name) != column] for column in target_columns
    }

    for iteration_idx in range(int(strategy["max_iter"])):
        previous_df = working_df.copy(deep=True)
        for column in target_columns:
            if column in skipped_columns:
                continue
            positions = target_positions_by_column.get(column, [])
            if not positions:
                continue
            target_mask = target_masks_by_column[column]
            series = original_df[column]
            is_numeric = bool(frame_pd.api.types.is_numeric_dtype(series))
            train_mask = (~target_mask) & series.notna()
            if is_numeric:
                numeric_series = frame_pd.to_numeric(series, errors="coerce")
                train_mask = train_mask & numeric_series.notna()
            train_positions = [idx for idx, flag in enumerate(train_mask.tolist()) if bool(flag)]
            sample_positions, train_pool_size, train_sample_size, train_sampled = _limit_missforest_training_positions(
                frame_pd,
                train_positions,
                strategy,
                issue_id=f"{column}::iterative",
                column=column,
            )
            if int(len(sample_positions)) < int(strategy["min_training_rows"]):
                skipped_columns[column] = {
                    "reason": "missforest_training_unavailable",
                    "column": column,
                    "train_rows": int(len(sample_positions)),
                    "min_training_rows": int(strategy["min_training_rows"]),
                }
                continue

            train_rows = working_df.iloc[sample_positions]
            predict_rows = working_df.iloc[positions]
            x_train, x_pred = _missforest_feature_matrices(
                frame_pd,
                working_df,
                feature_columns_by_target[column],
                train_rows,
                predict_rows,
            )
            model_random_state = (int(strategy["random_state"]) + iteration_idx) % 2_147_483_647
            if is_numeric:
                y_train = frame_pd.to_numeric(original_df.iloc[sample_positions][column], errors="coerce")
                valid_mask = y_train.notna()
                if int(valid_mask.sum()) < int(strategy["min_training_rows"]):
                    skipped_columns[column] = {
                        "reason": "missforest_training_unavailable",
                        "column": column,
                        "train_rows": int(valid_mask.sum()),
                        "min_training_rows": int(strategy["min_training_rows"]),
                    }
                    continue
                valid_positions = [idx for idx, flag in enumerate(valid_mask.tolist()) if bool(flag)]
                y_valid = y_train.loc[valid_mask]
                model = random_forest_regressor(
                    n_estimators=int(strategy["n_estimators"]),
                    max_depth=strategy["max_depth"],
                    max_features=strategy["max_features"],
                    random_state=model_random_state,
                    n_jobs=1,
                )
                x_train_valid = x_train.iloc[valid_positions, :].to_numpy()
                x_pred_values = x_pred.to_numpy()
                model.fit(x_train_valid, y_valid)
                predictions = list(model.predict(x_pred_values))
                confidence = _missforest_regression_confidence(model, x_pred_values, y_valid)
                model_type = "random_forest_regressor"
            else:
                y_train = original_df.iloc[sample_positions][column].astype(str)
                if int(y_train.nunique(dropna=True)) <= 0:
                    skipped_columns[column] = {
                        "reason": "missforest_training_unavailable",
                        "column": column,
                        "train_rows": int(len(y_train)),
                        "min_training_rows": int(strategy["min_training_rows"]),
                    }
                    continue
                model = random_forest_classifier(
                    n_estimators=int(strategy["n_estimators"]),
                    max_depth=strategy["max_depth"],
                    max_features=strategy["max_features"],
                    random_state=model_random_state,
                    n_jobs=1,
                )
                x_train_values = x_train.to_numpy()
                x_pred_values = x_pred.to_numpy()
                model.fit(x_train_values, y_train)
                predictions = list(model.predict(x_pred_values))
                confidence = _missforest_classifier_confidence(model, x_pred_values)
                model_type = "random_forest_classifier"

            col_pos = int(list(working_df.columns).index(column))
            for idx, pos in enumerate(positions):
                working_df.iat[int(pos), col_pos] = predictions[idx]
            column_meta[column] = {
                "algorithm_mode": "iterative",
                "model_type": model_type,
                "feature_count": int(x_train.shape[1]),
                "train_pool_size": int(train_pool_size),
                "train_sample_size": int(train_sample_size),
                "train_sampled": bool(train_sampled),
                "candidate_confidence": confidence,
                "n_estimators": int(strategy["n_estimators"]),
                "max_train_rows": int(strategy["max_train_rows"]),
            }

        iterations_run = iteration_idx + 1
        convergence_delta = _missforest_convergence_delta(
            frame_pd,
            original_df,
            previous_df,
            working_df,
            {
                column: positions
                for column, positions in target_positions_by_column.items()
                if column not in skipped_columns
            },
        )
        if convergence_delta <= float(strategy["convergence_tolerance"]):
            converged = True
            break

    changes_by_issue: dict[str, list[dict[str, Any]]] = {}
    issue_evidence: dict[str, dict[str, Any]] = {}
    for issue_id, issue in issue_by_id.items():
        column = str(issue["column"])
        if column in skipped_columns:
            details = dict(skipped_columns[column])
            details["issue_id"] = issue_id
            details["issue_type"] = str(issue["issue_type"])
            skipped.append(details)
            continue
        if column not in column_meta:
            skipped.append(
                {
                    "issue_id": issue_id,
                    "reason": "missforest_training_unavailable",
                    "issue_type": str(issue["issue_type"]),
                    "column": column,
                }
            )
            continue

        issue_changes: list[dict[str, Any]] = []
        col_pos = int(list(original_df.columns).index(column))
        for pos in issue_positions.get(issue_id, []):
            before_value = original_df[column].iat[int(pos)]
            after_value = working_df[column].iat[int(pos)]
            before_is_nan = bool(frame_pd.isna(before_value))
            after_is_nan = bool(frame_pd.isna(after_value))
            if (before_is_nan and after_is_nan) or (
                (not before_is_nan) and (not after_is_nan) and before_value == after_value
            ):
                continue
            issue_changes.append(
                {
                    "issue_id": issue_id,
                    "column": column,
                    "row_pos": int(pos),
                    "row": _index_to_builtin(original_df.index[int(pos)]),
                    "col_pos": col_pos,
                    "before": before_value,
                    "after": after_value,
                }
            )
        if not issue_changes:
            continue

        evidence = dict(column_meta[column])
        evidence.update(
            {
                "issue_id": issue_id,
                "column": column,
                "issue_type": str(issue["issue_type"]),
                "iterations_run": int(iterations_run),
                "converged": bool(converged),
                "convergence_delta": float(convergence_delta),
                "target_cell_count": int(len(issue_positions.get(issue_id, []))),
                "max_iter": int(strategy["max_iter"]),
                "convergence_tolerance": float(strategy["convergence_tolerance"]),
            }
        )
        changes_by_issue[issue_id] = issue_changes
        issue_evidence[issue_id] = evidence

    return changes_by_issue, issue_evidence, skipped


def action_health(_: dict[str, Any]) -> dict[str, Any]:
    dependencies = _runtime_dependency_snapshot()
    return {
        "engine": "python-anomaly-engine",
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "actions": public_action_names(),
        "dependencies": dependencies,
    }


def action_scan_file(payload: dict[str, Any]) -> dict[str, Any]:
    _emit_stage_progress("scan_file", "validate_input", "start", 2, "开始校验扫描参数")
    frame_pd = _load_dataframe_module("Scan")
    csv_path = _require(payload, "csv_path")
    scan_config = _scan_config_from_payload(payload)
    max_bins = int(scan_config["max_bins"])

    csv_file = _resolve_input_csv(str(csv_path))
    if not csv_file.exists():
        _emit_stage_progress(
            "scan_file",
            "validate_input",
            "error",
            100,
            "输入文件不存在",
            file=str(csv_file),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    _emit_stage_progress("scan_file", "load_csv", "start", 14, "开始读取待扫描文件", file=str(csv_file))
    try:
        df = frame_pd.read_csv(csv_file)
    except Exception as exc:
        _emit_stage_progress(
            "scan_file",
            "load_csv",
            "error",
            100,
            "读取待扫描文件失败",
            file=str(csv_file),
            error_code=ErrorCode.CSV_READ_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.CSV_READ_FAILED,
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "scan_file",
        "load_csv",
        "complete",
        28,
        "待扫描文件读取完成",
        file=str(csv_file),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
    )

    _emit_stage_progress("scan_file", "scan_columns", "start", 42, "开始扫描列级异常", file=str(csv_file))
    try:
        issues_internal = _detect_issues_for_frame(df, frame_pd, scan_config=scan_config)
    except KnownEngineError:
        raise
    except Exception as exc:
        _emit_stage_progress(
            "scan_file",
            "scan_columns",
            "error",
            100,
            "扫描异常失败",
            file=str(csv_file),
            error_code=ErrorCode.SCAN_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.SCAN_FAILED,
            message="File scan failed",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "scan_file",
        "scan_columns",
        "complete",
        78,
        "列级异常扫描完成",
        file=str(csv_file),
        issue_count=int(len(issues_internal)),
    )

    row_count = int(df.shape[0])
    issue_type_counts = _issue_type_counter(issues_internal)
    column_names = [str(col) for col in list(df.columns)]
    column_masks: dict[str, Any] = {col: frame_pd.Series(False, index=df.index, dtype=bool) for col in column_names}
    issue_counts_by_column: dict[str, int] = {col: 0 for col in column_names}
    issue_score_by_column: dict[str, float] = {col: 0.0 for col in column_names}
    issue_types_by_column: dict[str, list[str]] = {col: [] for col in column_names}

    issues: list[dict[str, Any]] = []
    for issue in issues_internal:
        column = str(issue["column"])
        mask = issue["mask"]
        if column not in column_masks:
            column_masks[column] = frame_pd.Series(False, index=df.index, dtype=bool)
        column_masks[column] = column_masks[column] | mask
        issue_counts_by_column[column] = issue_counts_by_column.get(column, 0) + 1
        issue_score_by_column[column] = float(issue_score_by_column.get(column, 0.0)) + float(issue["issue_score"])
        issue_types_by_column.setdefault(column, [])
        if issue["issue_type"] not in issue_types_by_column[column]:
            issue_types_by_column[column].append(str(issue["issue_type"]))

        bin_counts, bin_size = _mask_to_bin_counts(mask, row_count=row_count, max_bins=max_bins)
        heat_bins = _bin_counts_to_heat(bin_counts, bin_size=bin_size)
        segments = _bin_counts_to_segments(bin_counts, row_count=row_count, bin_size=bin_size, max_segments=24)
        public_issue = {
            "issue_id": issue["issue_id"],
            "column": column,
            "issue_type": issue["issue_type"],
            "severity": issue["severity"],
            "issue_score": round(float(issue["issue_score"]), 6),
            "confidence": round(float(issue.get("confidence", 0.0)), 6),
            "count": int(issue["count"]),
            "ratio": round(float(issue["ratio"]), 6),
            "risk_level": _risk_level_from_score(float(issue["issue_score"])),
            "explain_features": issue.get("explain_features", []),
            "bins": [1 if count > 0 else 0 for count in bin_counts],
            "heat_bins": heat_bins,
            "segments": segments,
            "detail": issue["detail"],
            "repair_supported": True,
        }
        if issue["issue_type"] == "numeric_outlier":
            for key in ("outlier_risk_level", "auto_repair_eligible", "outlier_policy_reason", "outlier_evidence"):
                if key in issue:
                    public_issue[key] = issue[key]
        issues.append(public_issue)

    column_profiles: list[dict[str, Any]] = []
    column_thumbnails: list[dict[str, Any]] = []
    anomaly_columns: list[str] = []

    for raw_col in list(df.columns):
        column = str(raw_col)
        series = df[raw_col]
        missing_count = int(series.isna().sum())
        missing_ratio = float(missing_count) / float(row_count) if row_count > 0 else 0.0
        dtype_text = str(series.dtype)
        is_numeric = bool(frame_pd.api.types.is_numeric_dtype(series))

        column_profiles.append(
            {
                "column": column,
                "dtype": dtype_text,
                "is_numeric": is_numeric,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 6),
                "issue_count": issue_counts_by_column.get(column, 0),
                "issue_types": list(issue_types_by_column.get(column, [])),
            }
        )

        union_mask = column_masks[column]
        anomaly_points = int(union_mask.sum())
        bin_counts, bin_size = _mask_to_bin_counts(union_mask, row_count=row_count, max_bins=max_bins)
        heat_bins = _bin_counts_to_heat(bin_counts, bin_size=bin_size)
        binary_bins = [1 if count > 0 else 0 for count in bin_counts]
        anomalous_bins = int(sum(binary_bins))
        anomaly_ratio = float(anomaly_points) / float(row_count) if row_count > 0 else 0.0
        risk_score = min(100.0, float(issue_score_by_column.get(column, 0.0)) + anomaly_ratio * 40.0)
        risk_level = _risk_level_from_score(risk_score)
        hot_segments = _bin_counts_to_segments(bin_counts, row_count=row_count, bin_size=bin_size, max_segments=16)
        if anomaly_points > 0:
            anomaly_columns.append(column)

        if column_profiles:
            column_profiles[-1]["anomaly_points"] = anomaly_points
            column_profiles[-1]["anomaly_ratio"] = round(anomaly_ratio, 6)
            column_profiles[-1]["risk_score"] = round(risk_score, 6)
            column_profiles[-1]["risk_level"] = risk_level

        column_thumbnails.append(
            {
                "column": column,
                "dtype": dtype_text,
                "issue_count": issue_counts_by_column.get(column, 0),
                "anomaly_points": anomaly_points,
                "anomaly_ratio": round(anomaly_ratio, 6),
                "risk_score": round(risk_score, 6),
                "risk_level": risk_level,
                "total_bins": len(binary_bins),
                "anomalous_bins": anomalous_bins,
                "bins": binary_bins,
                "heat_bins": heat_bins,
                "hot_segments": hot_segments,
            }
        )

    issues.sort(
        key=lambda item: (
            -float(item.get("issue_score", 0.0)),
            str(item.get("column", "")),
            str(item.get("issue_id", "")),
        )
    )
    column_thumbnails.sort(
        key=lambda item: (
            -float(item.get("risk_score", 0.0)),
            str(item.get("column", "")),
        )
    )
    numeric_outlier_risk_counts = {"mild": 0, "strong": 0, "extreme": 0}
    numeric_outlier_auto_eligible_count = 0
    for issue in issues:
        if issue.get("issue_type") != "numeric_outlier":
            continue
        risk_level = str(issue.get("outlier_risk_level") or "mild")
        if risk_level not in numeric_outlier_risk_counts:
            numeric_outlier_risk_counts[risk_level] = 0
        numeric_outlier_risk_counts[risk_level] += 1
        if bool(issue.get("auto_repair_eligible")):
            numeric_outlier_auto_eligible_count += 1

    _emit_stage_progress(
        "scan_file",
        "build_summary",
        "complete",
        100,
        "扫描结果已汇总",
        file=str(csv_file),
        issue_count=int(len(issues)),
        anomaly_column_count=int(len(anomaly_columns)),
    )

    return _to_builtin(
        {
            "csv_path": str(csv_file),
            "scan_config": scan_config,
            "data_profile": {
                "rows": row_count,
                "columns": int(df.shape[1]),
            },
            "column_profiles": column_profiles,
            "column_thumbnails": column_thumbnails,
            "issues": issues,
            "issue_count": len(issues),
            "anomaly_columns": anomaly_columns,
            "scan_summary": {
                "anomaly_column_count": len(anomaly_columns),
                "high_risk_columns": [item["column"] for item in column_thumbnails if item.get("risk_level") == "high"],
                "medium_risk_columns": [
                    item["column"] for item in column_thumbnails if item.get("risk_level") == "medium"
                ],
                "total_issues": len(issues),
                "issue_type_counts": issue_type_counts,
                "numeric_outlier_risk_counts": numeric_outlier_risk_counts,
                "numeric_outlier_auto_eligible_count": numeric_outlier_auto_eligible_count,
            },
        }
    )


def action_train(payload: dict[str, Any]) -> dict[str, Any]:
    _emit_stage_progress("train", "validate_input", "start", 2, "开始校验训练参数")
    train_pd, process_and_train, save_system_state = _load_training_modules()

    csv_path = _require(payload, "csv_path")
    target_col = str(_require(payload, "target_col"))
    requested_task_type = str(payload.get("task_type", "auto") or "auto").strip().lower()
    output_dir = _resolve_output_dir(payload.get("output_dir"))

    csv_file = _resolve_input_csv(str(csv_path))
    if not csv_file.exists():
        _emit_stage_progress(
            "train",
            "validate_input",
            "error",
            100,
            "输入文件不存在",
            file=str(csv_file),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    _emit_stage_progress("train", "load_csv", "start", 12, "开始读取训练数据", file=str(csv_file))
    try:
        df = train_pd.read_csv(csv_file)
    except Exception as exc:  # pragma: no cover - serialization guard
        _emit_stage_progress(
            "train",
            "load_csv",
            "error",
            100,
            "读取训练数据失败",
            file=str(csv_file),
            error_code=ErrorCode.CSV_READ_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.CSV_READ_FAILED,
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "train",
        "load_csv",
        "complete",
        26,
        "训练数据读取完成",
        file=str(csv_file),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
    )

    if target_col not in df.columns:
        _emit_stage_progress(
            "train",
            "validate_input",
            "error",
            100,
            "目标列不存在",
            file=str(csv_file),
            column=target_col,
            error_code=ErrorCode.INVALID_TARGET_COLUMN,
        )
        raise KnownEngineError(
            code=ErrorCode.INVALID_TARGET_COLUMN,
            message=f"Target column not found: {target_col}",
            details={"available_columns": list(df.columns)},
        )

    try:
        resolved_task_type = _validate_train_target(df, target_col, train_pd, requested_mode=requested_task_type)
    except KnownEngineError as exc:
        details = exc.details if isinstance(exc.details, dict) else {}
        _emit_stage_progress(
            "train",
            "validate_input",
            "error",
            100,
            "训练参数校验失败",
            file=str(csv_file),
            column=str(details.get("target_col") or target_col),
            error_code=exc.code,
            reason=str(exc.message),
        )
        raise
    _emit_stage_progress(
        "train",
        "validate_input",
        "complete",
        34,
        "训练参数校验通过",
        file=str(csv_file),
        column=target_col,
        task_type=resolved_task_type,
    )

    _emit_stage_progress(
        "train",
        "train_model",
        "start",
        46,
        "开始训练模型",
        column=target_col,
        task_type=resolved_task_type,
    )
    try:
        model, x_test, normal_data, metrics, feature_names = process_and_train(
            df,
            target_col,
            task_type=resolved_task_type,
        )
        _emit_stage_progress(
            "train",
            "train_model",
            "complete",
            78,
            "模型训练完成",
            column=target_col,
            task_type=resolved_task_type,
        )
        _emit_stage_progress(
            "train",
            "save_artifacts",
            "start",
            86,
            "开始写入模型产物",
            file=str(output_dir),
        )
        os.makedirs(output_dir, exist_ok=True)
        save_system_state(model, x_test, normal_data, feature_names, save_dir=output_dir)
        _emit_stage_progress(
            "train",
            "save_artifacts",
            "complete",
            96,
            "模型产物写入完成",
            file=str(output_dir),
        )
    except KnownEngineError:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard
        _emit_stage_progress(
            "train",
            "train_model",
            "error",
            100,
            "训练流程失败",
            column=target_col,
            file=str(csv_file),
            error_code=ErrorCode.TRAINING_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.TRAINING_FAILED,
            message="Model training failed",
            details={"reason": str(exc)},
        ) from exc

    _emit_stage_progress(
        "train",
        "complete",
        "complete",
        100,
        "训练任务完成",
        file=str(output_dir),
    )

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
            "task_type": resolved_task_type,
            "requested_task_type": requested_task_type,
        },
        "metrics": _metric_summary(metrics),
    }


def action_repair(payload: dict[str, Any]) -> dict[str, Any]:
    _emit_stage_progress("repair", "validate_input", "start", 2, "开始校验修复参数")
    load_system_state, predict_with_threshold, repair_anomaly_sample = _load_repair_modules()

    model_dir_text = str(_require(payload, "model_dir"))
    model_dir = _resolve_existing_dir(model_dir_text)
    if not model_dir.exists() or not model_dir.is_dir():
        _emit_stage_progress(
            "repair",
            "validate_input",
            "error",
            100,
            "模型目录不存在",
            file=str(model_dir),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Model directory does not exist: {model_dir}",
            details={"model_dir": str(model_dir)},
        )

    required_files = ["model_lgb.pkl", "test_data.pkl", "normal_data.pkl"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        _emit_stage_progress(
            "repair",
            "validate_input",
            "error",
            100,
            "模型目录缺少必要产物",
            file=str(model_dir),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message="Model directory is missing required artifacts",
            details={"model_dir": str(model_dir), "missing_files": missing},
        )

    _emit_stage_progress("repair", "load_model", "start", 16, "开始加载模型状态", file=str(model_dir))
    try:
        model, x_test, normal_data = load_system_state(model_dir)
    except Exception as exc:
        _emit_stage_progress(
            "repair",
            "load_model",
            "error",
            100,
            "加载模型状态失败",
            file=str(model_dir),
            error_code=ErrorCode.MODEL_STATE_LOAD_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.MODEL_STATE_LOAD_FAILED,
            message="Failed to load model state artifacts",
            details={"model_dir": str(model_dir), "reason": str(exc)},
        ) from exc
    _emit_stage_progress("repair", "load_model", "complete", 34, "模型状态加载完成", file=str(model_dir))

    model_task_type = str(getattr(model, "task_type", "classification")).strip().lower()
    if model_task_type != "classification":
        _emit_stage_progress(
            "repair",
            "validate_input",
            "error",
            100,
            "当前模型类型不支持单样本修复",
            file=str(model_dir),
            error_code=ErrorCode.UNSUPPORTED_TARGET_TYPE,
        )
        raise KnownEngineError(
            code=ErrorCode.UNSUPPORTED_TARGET_TYPE,
            message=f"Repair action supports classification models only: {model_task_type}",
            details={
                "model_task_type": model_task_type,
                "model_dir": str(model_dir),
                "suggestion": "Use action=repair_batch for rule-based CSV repair or train with classification target.",
            },
        )

    sample_index = _to_int(payload, "sample_index", default=0, minimum=0)
    if sample_index >= int(x_test.shape[0]):
        _emit_stage_progress(
            "repair",
            "validate_input",
            "error",
            100,
            "样本索引超出范围",
            file=str(model_dir),
            error_code=ErrorCode.INVALID_INPUT,
        )
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

    _emit_stage_progress(
        "repair",
        "repair_search",
        "start",
        54,
        "开始执行单样本修复搜索",
        file=str(model_dir),
        sample_index=sample_index,
    )
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
        _emit_stage_progress(
            "repair",
            "repair_search",
            "error",
            100,
            "修复搜索失败",
            file=str(model_dir),
            error_code=ErrorCode.REPAIR_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.REPAIR_FAILED,
            message="Repair search failed",
            details={"reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "repair",
        "repair_search",
        "complete",
        78,
        "单样本修复搜索完成",
        file=str(model_dir),
        sample_index=sample_index,
        changed_fields=int(len(repair_bundle.changes)),
    )

    after_pred_arr, after_prob_arr = predict_with_threshold(model, repair_bundle.repaired_sample)
    after_pred = int(after_pred_arr[0])
    after_score = float(after_prob_arr[0])

    artifacts: dict[str, str] = {}
    output_dir_raw = payload.get("output_dir")
    if (not dry_run) and output_dir_raw is not None and str(output_dir_raw).strip():
        _emit_stage_progress("repair", "write_output", "start", 86, "开始写出修复结果", file=str(output_dir_raw))
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
        _emit_stage_progress("repair", "write_output", "complete", 96, "修复结果写出完成", file=str(output_dir))

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
            "task_type": model_task_type,
        },
    }
    if artifacts:
        result["artifacts"] = artifacts

    _emit_stage_progress(
        "repair",
        "complete",
        "complete",
        100,
        "单样本修复任务完成",
        file=str(model_dir),
        sample_index=sample_index,
    )

    return result


def action_repair_batch(payload: dict[str, Any]) -> dict[str, Any]:
    _emit_stage_progress("repair_batch", "validate_input", "start", 2, "开始校验批量修复参数")
    frame_pd = _load_dataframe_module("Repair batch")
    csv_path = _require(payload, "csv_path")
    scan_config = _scan_config_from_payload(payload)
    write_output_requested = _to_bool(payload, "write_output", default=True)
    plan_only = _to_bool(payload, "plan_only", default=False)
    write_output = write_output_requested and (not plan_only)
    enable_rollback = _to_bool(payload, "enable_rollback", default=True)
    repair_strategy = _repair_strategy_from_payload(payload)
    column_dependencies = _column_dependencies_from_payload(payload)

    raw_issue_ids = payload.get("issue_ids", [])
    if raw_issue_ids is None:
        raw_issue_ids = []
    if not isinstance(raw_issue_ids, (list, tuple, set)):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field issue_ids must be a string list",
            details={"field": "issue_ids", "value": raw_issue_ids},
        )

    selected_issue_ids: list[str] = []
    seen_issue_ids: set[str] = set()
    for raw in raw_issue_ids:
        issue_id = str(raw).strip()
        if not issue_id or issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue_id)
        selected_issue_ids.append(issue_id)

    csv_file = _resolve_input_csv(str(csv_path))
    if not csv_file.exists():
        _emit_stage_progress(
            "repair_batch",
            "validate_input",
            "error",
            100,
            "输入文件不存在",
            file=str(csv_file),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    _emit_stage_progress("repair_batch", "load_csv", "start", 12, "开始读取待修复文件", file=str(csv_file))
    try:
        df = frame_pd.read_csv(csv_file)
    except Exception as exc:
        _emit_stage_progress(
            "repair_batch",
            "load_csv",
            "error",
            100,
            "读取待修复文件失败",
            file=str(csv_file),
            error_code=ErrorCode.CSV_READ_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.CSV_READ_FAILED,
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "repair_batch",
        "load_csv",
        "complete",
        22,
        "待修复文件读取完成",
        file=str(csv_file),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
    )

    _emit_stage_progress("repair_batch", "scan_columns", "start", 34, "开始识别可修复问题", file=str(csv_file))
    try:
        precomputed_issues, precomputed_issues_used = _precomputed_issues_from_payload(
            payload,
            csv_file,
            plan_only=plan_only,
        )
        if precomputed_issues_used and precomputed_issues is not None:
            issues_internal = precomputed_issues
        else:
            issues_internal = _detect_issues_for_frame(df, frame_pd, scan_config=scan_config)
        _emit_stage_progress(
            "repair_batch",
            "scan_columns",
            "complete",
            48,
            "可修复问题识别完成",
            file=str(csv_file),
            issue_count=int(len(issues_internal)),
            precomputed_issues_used=bool(precomputed_issues_used),
        )
        issue_map = {str(item["issue_id"]): item for item in issues_internal}

        original_df = df.copy(deep=True)
        repaired_df = df.copy(deep=True)
        applied_repairs: list[dict[str, Any]] = []
        skipped_issues: list[dict[str, Any]] = []
        skipped_ids: set[str] = set()
        issue_accept_rows: dict[str, set[int]] = {}
        issue_conflict_rows: dict[str, set[int]] = {}
        issue_replacement_preview: dict[str, Any] = {}
        issue_raw_counts: dict[str, int] = {}
        conflict_events: list[dict[str, Any]] = []

        def add_skip(issue_id: str, reason: str, extra: dict[str, Any] | None = None) -> None:
            if issue_id in skipped_ids:
                return
            row: dict[str, Any] = {"issue_id": issue_id, "reason": reason}
            if extra:
                row.update(extra)
            skipped_issues.append(row)
            skipped_ids.add(issue_id)

        selected_issues: list[dict[str, Any]] = []
        selected_columns: list[str] = []
        selected_columns_seen: set[str] = set()

        for issue_id in selected_issue_ids:
            issue = issue_map.get(issue_id)
            if issue is None:
                add_skip(issue_id, "issue_not_found")
                continue

            column = str(issue["column"])
            if column not in repaired_df.columns:
                add_skip(issue_id, "column_not_found")
                continue
            selected_issues.append(issue)
            if column not in selected_columns_seen:
                selected_columns_seen.add(column)
                selected_columns.append(column)

        column_order, cycle_columns = _topological_column_order(selected_columns, column_dependencies)
        column_rank = {col: idx for idx, col in enumerate(column_order)}
        issue_priority = {name: idx for idx, name in enumerate(repair_strategy["issue_priority"])}
        selected_issues.sort(
            key=lambda item: (
                int(column_rank.get(str(item["column"]), 10_000)),
                int(issue_priority.get(str(item["issue_type"]), 10_000)),
                -float(item.get("issue_score", 0.0)),
                str(item.get("issue_id", "")),
            )
        )

        remaining_by_column: dict[str, int] = {}
        for issue in selected_issues:
            col = str(issue["column"])
            remaining_by_column[col] = remaining_by_column.get(col, 0) + 1
        completed_columns: set[str] = set()

        column_position = {str(col): idx for idx, col in enumerate(repaired_df.columns)}
        conflict_policy = str(repair_strategy["conflict_policy"])
        preview_limit = int(repair_strategy["preview_limit"])
        cell_plan: dict[tuple[int, str], dict[str, Any]] = {}

        _emit_stage_progress(
            "repair_batch",
            "apply_repairs",
            "start",
            56,
            "开始应用修复计划",
            file=str(csv_file),
            issue_count=int(len(selected_issues)),
        )

        for issue in selected_issues:
            issue_id = str(issue["issue_id"])
            if issue_id in skipped_ids:
                col = str(issue["column"])
                remaining_by_column[col] = max(0, remaining_by_column.get(col, 1) - 1)
                if remaining_by_column.get(col, 0) <= 0:
                    completed_columns.add(col)
                continue

            column = str(issue["column"])
            issue_type = str(issue["issue_type"])
            if issue_type not in {"missing_values", "numeric_outlier", "rare_category"}:
                add_skip(
                    issue_id,
                    "unsupported_issue_type",
                    {"issue_type": issue_type, "column": column},
                )
                remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
                if remaining_by_column.get(column, 0) <= 0:
                    completed_columns.add(column)
                continue
            rule = issue.get("repair_rule", {})
            deps = column_dependencies.get(column, [])
            missing_dep_cols = [dep for dep in deps if dep not in df.columns]
            if missing_dep_cols:
                add_skip(
                    issue_id,
                    "dependency_column_not_found",
                    {"column": column, "depends_on": missing_dep_cols, "issue_type": issue_type},
                )
                remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
                if remaining_by_column.get(column, 0) <= 0:
                    completed_columns.add(column)
                continue
            if column in cycle_columns:
                add_skip(
                    issue_id,
                    "dependency_cycle",
                    {"column": column, "depends_on": deps, "issue_type": issue_type},
                )
                remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
                if remaining_by_column.get(column, 0) <= 0:
                    completed_columns.add(column)
                continue
            unresolved = [dep for dep in deps if dep in remaining_by_column and dep not in completed_columns]
            if unresolved:
                add_skip(
                    issue_id,
                    "dependency_unresolved",
                    {"column": column, "depends_on": unresolved, "issue_type": issue_type},
                )
                remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
                if remaining_by_column.get(column, 0) <= 0:
                    completed_columns.add(column)
                continue

            series_original = original_df[column]
            mask = _issue_mask_from_rule(series_original, issue_type, rule, frame_pd)
            positions = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]
            issue_raw_counts[issue_id] = len(positions)

            if issue_type == "numeric_outlier" and str(repair_strategy["outlier"]) == "skip":
                add_skip(
                    issue_id,
                    "strategy_disabled",
                    {"column": column, "issue_type": issue_type, "strategy": "outlier=skip"},
                )
                remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
                if remaining_by_column.get(column, 0) <= 0:
                    completed_columns.add(column)
                continue

            if not positions:
                add_skip(issue_id, "no_rows_matched", {"column": column, "issue_type": issue_type})
                remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
                if remaining_by_column.get(column, 0) <= 0:
                    completed_columns.add(column)
                continue

            replacement_preview: Any = None
            if issue_type == "missing_values":
                replacement_value = _replacement_for_missing(series_original, frame_pd, repair_strategy)
                replacement_preview = replacement_value
            elif issue_type == "numeric_outlier":
                lower = float(rule.get("lower_bound"))
                upper = float(rule.get("upper_bound"))
                replacement_preview = {"lower_bound": round(lower, 12), "upper_bound": round(upper, 12)}
            elif issue_type == "rare_category":
                rare_values = list(rule.get("rare_values", []))
                replacement_value = _replacement_for_rare(series_original, rare_values, repair_strategy, rule)
                replacement_preview = replacement_value

            issue_replacement_preview[issue_id] = replacement_preview
            issue_accept_rows.setdefault(issue_id, set())
            issue_conflict_rows.setdefault(issue_id, set())
            col_pos = int(column_position[column])

            for pos in positions:
                before_value = series_original.iat[pos]
                after_value = before_value

                if issue_type == "missing_values":
                    after_value = replacement_preview
                elif issue_type == "numeric_outlier":
                    lower = float(rule.get("lower_bound"))
                    upper = float(rule.get("upper_bound"))
                    try:
                        val = float(before_value)
                    except Exception:
                        continue
                    after_value = min(upper, max(lower, val))
                elif issue_type == "rare_category":
                    after_value = replacement_preview

                before_is_nan = bool(frame_pd.isna(before_value))
                after_is_nan = bool(frame_pd.isna(after_value))
                if (before_is_nan and after_is_nan) or ((not before_is_nan) and (not after_is_nan) and before_value == after_value):
                    continue

                key = (pos, column)
                proposal = {
                    "issue_id": issue_id,
                    "column": column,
                    "row_pos": pos,
                    "row": _index_to_builtin(original_df.index[pos]),
                    "col_pos": col_pos,
                    "before": before_value,
                    "after": after_value,
                }
                existing = cell_plan.get(key)
                if existing is None:
                    cell_plan[key] = proposal
                    issue_accept_rows[issue_id].add(pos)
                    continue

                existing_issue_id = str(existing["issue_id"])
                issue_conflict_rows.setdefault(issue_id, set()).add(pos)
                if len(conflict_events) < 120:
                    conflict_events.append(
                        {
                            "row": _index_to_builtin(original_df.index[pos]),
                            "column": column,
                            "existing_issue_id": existing_issue_id,
                            "incoming_issue_id": issue_id,
                            "resolution": conflict_policy,
                        }
                    )

                if conflict_policy == "last_wins":
                    issue_accept_rows.setdefault(existing_issue_id, set()).discard(pos)
                    issue_conflict_rows.setdefault(existing_issue_id, set()).add(pos)
                    cell_plan[key] = proposal
                    issue_accept_rows[issue_id].add(pos)

            remaining_by_column[column] = max(0, remaining_by_column.get(column, 1) - 1)
            if remaining_by_column.get(column, 0) <= 0:
                completed_columns.add(column)

        for proposal in cell_plan.values():
            repaired_df.iat[int(proposal["row_pos"]), int(proposal["col_pos"])] = proposal["after"]

        post_issues_internal = _detect_issues_for_frame(repaired_df, frame_pd, scan_config=scan_config)
        _emit_stage_progress(
            "repair_batch",
            "apply_repairs",
            "complete",
            78,
            "修复计划应用完成",
            file=str(csv_file),
            changed_cells=int(len(cell_plan)),
        )
        before_issue_type_counts = _issue_type_counter(issues_internal)
        after_issue_type_counts = _issue_type_counter(post_issues_internal)
        before_issue_column_counts = _issue_counter_by_column(issues_internal)
        after_issue_column_counts = _issue_counter_by_column(post_issues_internal)

        changes_by_issue: dict[str, list[dict[str, Any]]] = {}
        for proposal in cell_plan.values():
            issue_id = str(proposal["issue_id"])
            changes_by_issue.setdefault(issue_id, []).append(proposal)

        for issue_id in selected_issue_ids:
            issue = issue_map.get(issue_id)
            if issue is None:
                continue
            if issue_id in skipped_ids:
                continue

            issue_type = str(issue.get("issue_type"))
            column = str(issue.get("column"))
            issue_changes = sorted(changes_by_issue.get(issue_id, []), key=lambda item: int(item["row_pos"]))
            rows_touched = len(issue_changes)

            if rows_touched <= 0:
                reason = "conflict_all_skipped" if issue_conflict_rows.get(issue_id) else "no_rows_matched"
                add_skip(
                    issue_id,
                    reason,
                    {"column": column, "issue_type": issue_type, "conflict_rows": len(issue_conflict_rows.get(issue_id, set()))},
                )
                continue

            rule = issue.get("repair_rule", {})
            before_count = int(issue.get("count", issue_raw_counts.get(issue_id, rows_touched)))
            after_mask = _issue_mask_from_rule(repaired_df[column], issue_type, rule, frame_pd)
            after_count = int(after_mask.sum())
            resolved_count = max(0, before_count - after_count)
            cells_preview = [
                {
                    "row": _index_to_builtin(change["row"]),
                    "before": _to_builtin(change["before"]),
                    "after": _to_builtin(change["after"]),
                }
                for change in issue_changes[:preview_limit]
            ]

            applied_repairs.append(
                {
                    "issue_id": issue_id,
                    "column": column,
                    "issue_type": issue_type,
                    "rows_touched": rows_touched,
                    "replacement_preview": issue_replacement_preview.get(issue_id),
                    "before_count": before_count,
                    "after_count": after_count,
                    "resolved_count": resolved_count,
                    "conflict_rows_skipped": len(issue_conflict_rows.get(issue_id, set())),
                    "cells_preview": cells_preview,
                    "strategy": {
                        "conflict_policy": conflict_policy,
                        "missing_numeric": repair_strategy["missing_numeric"],
                        "missing_categorical": repair_strategy["missing_categorical"],
                        "outlier": repair_strategy["outlier"],
                        "rare_category": repair_strategy["rare_category"],
                    },
                }
            )
    except KnownEngineError as exc:
        details = exc.details if isinstance(exc.details, dict) else {}
        _emit_stage_progress(
            "repair_batch",
            "apply_repairs",
            "error",
            100,
            "批量修复失败",
            file=str(details.get("csv_path") or csv_file),
            column=str(details.get("column") or ""),
            rule=str(details.get("rule") or details.get("rule_name") or ""),
            error_code=exc.code,
            reason=str(exc.message),
        )
        raise
    except Exception as exc:
        _emit_stage_progress(
            "repair_batch",
            "apply_repairs",
            "error",
            100,
            "批量修复失败",
            file=str(csv_file),
            error_code=ErrorCode.REPAIR_BATCH_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.REPAIR_BATCH_FAILED,
            message="Batch repair failed",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc

    output_csv: str | None = None
    rollback_info: dict[str, Any] | None = None
    if write_output:
        _emit_stage_progress("repair_batch", "write_output", "start", 86, "开始写出修复结果", file=str(csv_file))
        output_csv_raw = str(payload.get("output_csv") or "").strip()
        output_dir_raw = str(payload.get("output_dir") or "").strip()
        if output_csv_raw:
            output_path = _resolve_output_file(output_csv_raw)
        elif output_dir_raw:
            output_dir = _resolve_output_dir(output_dir_raw)
            output_path = output_dir / f"{csv_file.stem}.repaired.csv"
        else:
            output_path = csv_file.with_name(f"{csv_file.stem}.repaired.csv")

        os.makedirs(output_path.parent, exist_ok=True)

        if enable_rollback:
            _, rollback_info = _create_rollback_artifacts(
                source_tool_id="engine.repair_batch",
                csv_file=csv_file,
                output_path=output_path,
                selected_issue_ids=selected_issue_ids,
                issue_source_map={issue_id: "rule" for issue_id in selected_issue_ids},
                execution_steps=[
                    {
                        "step": 1,
                        "tool_id": "engine.repair_batch",
                        "source": "rule",
                    }
                ],
                payload=payload,
                extra={
                    "scan_config": scan_config,
                    "repair_strategy": repair_strategy,
                },
            )

        repaired_df.to_csv(output_path, index=False)
        output_csv = str(output_path)
        _emit_stage_progress("repair_batch", "write_output", "complete", 96, "修复结果写出完成", file=str(output_csv))

    total_cells_modified = int(len(cell_plan))
    before_issue_count = int(len(issues_internal))
    after_issue_count = int(len(post_issues_internal))
    changed_cells_preview = [
        {
            "row": _index_to_builtin(change["row"]),
            "column": str(change["column"]),
            "issue_id": str(change["issue_id"]),
            "before": _to_builtin(change["before"]),
            "after": _to_builtin(change["after"]),
        }
        for change in sorted(cell_plan.values(), key=lambda item: (int(item["row_pos"]), str(item["column"])))[:preview_limit]
    ]
    _emit_stage_progress(
        "repair_batch",
        "complete",
        "complete",
        100,
        "批量修复任务完成",
        file=str(output_csv or csv_file),
        selected_issue_count=int(len(selected_issue_ids)),
        applied_issue_count=int(len(applied_repairs)),
    )
    return _to_builtin(
        {
            "csv_path": str(csv_file),
            "scan_config": scan_config,
            "repair_strategy": repair_strategy,
            "column_dependencies": column_dependencies,
            "plan_only": plan_only,
            "execution_mode": "plan_only" if plan_only else "apply",
            "write_output_requested": write_output_requested,
            "write_output": write_output,
            "precomputed_issues_used": bool(precomputed_issues_used),
            "output_csv": output_csv,
            "selected_issue_count": len(selected_issue_ids),
            "applied_issue_count": len(applied_repairs),
            "total_cells_modified": total_cells_modified,
            "selected_issue_ids": selected_issue_ids,
            "applied_repairs": applied_repairs,
            "skipped_issues": skipped_issues,
            "scan_issue_count": before_issue_count,
            "conflict_summary": {
                "policy": conflict_policy,
                "total_conflicts": len(conflict_events),
                "events_preview": conflict_events[:preview_limit],
            },
            "comparison": {
                "before_issue_count": before_issue_count,
                "after_issue_count": after_issue_count,
                "resolved_issue_count": max(0, before_issue_count - after_issue_count),
                "before_issue_type_counts": before_issue_type_counts,
                "after_issue_type_counts": after_issue_type_counts,
                "before_column_issue_counts": before_issue_column_counts,
                "after_column_issue_counts": after_issue_column_counts,
                "changed_cell_count": total_cells_modified,
                "changed_cells_preview": changed_cells_preview,
            },
            "rollback": rollback_info,
        }
    )


def action_repair_with_gower(payload: dict[str, Any]) -> dict[str, Any]:
    _emit_stage_progress("repair_with_gower", "validate_input", "start", 2, "开始校验 Gower 修复参数")
    frame_pd = _load_dataframe_module("Repair with gower")
    csv_path = _require(payload, "csv_path")
    scan_config = _scan_config_from_payload(payload)
    column_dependencies = _column_dependencies_from_payload(payload)
    gower_strategy = _gower_strategy_from_payload(payload)
    write_output_requested = _to_bool(payload, "write_output", default=True)
    plan_only = _to_bool(payload, "plan_only", default=False)
    write_output = write_output_requested and (not plan_only)

    raw_issue_ids = payload.get("issue_ids", [])
    if raw_issue_ids is None:
        raw_issue_ids = []
    if not isinstance(raw_issue_ids, (list, tuple, set)):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field issue_ids must be a string list",
            details={"field": "issue_ids", "value": raw_issue_ids},
        )

    selected_issue_ids: list[str] = []
    seen_issue_ids: set[str] = set()
    for raw in raw_issue_ids:
        issue_id = str(raw).strip()
        if not issue_id or issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue_id)
        selected_issue_ids.append(issue_id)

    model_dir: Path | None = None
    model_dir_text = str(payload.get("model_dir") or "").strip()
    if model_dir_text:
        candidate_model_dir = _resolve_existing_dir(model_dir_text)
        if candidate_model_dir.exists() and candidate_model_dir.is_dir():
            model_dir = candidate_model_dir

    try:
        from src.repair_module import suggest_replacement_from_neighbors  # type: ignore
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.REPAIR_MODULE_IMPORT_FAILED,
            message="Failed to import Gower repair modules",
            details={"reason": str(exc)},
        ) from exc

    csv_file = _resolve_input_csv(str(csv_path))
    if not csv_file.exists():
        _emit_stage_progress(
            "repair_with_gower",
            "validate_input",
            "error",
            100,
            "输入文件不存在",
            file=str(csv_file),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    _emit_stage_progress("repair_with_gower", "load_csv", "start", 12, "开始读取待修复文件", file=str(csv_file))
    try:
        df = frame_pd.read_csv(csv_file)
    except Exception as exc:
        _emit_stage_progress(
            "repair_with_gower",
            "load_csv",
            "error",
            100,
            "读取待修复文件失败",
            file=str(csv_file),
            error_code=ErrorCode.CSV_READ_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.CSV_READ_FAILED,
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "repair_with_gower",
        "load_csv",
        "complete",
        22,
        "待修复文件读取完成",
        file=str(csv_file),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
    )

    _emit_stage_progress("repair_with_gower", "scan_columns", "start", 34, "开始识别 Gower 可修复问题", file=str(csv_file))
    try:
        precomputed_issues, precomputed_issues_used = _precomputed_issues_from_payload(
            payload,
            csv_file,
            plan_only=plan_only,
        )
        if precomputed_issues_used and precomputed_issues is not None:
            issues_internal = precomputed_issues
        else:
            issues_internal = _detect_issues_for_frame(df, frame_pd, scan_config=scan_config)
        _emit_stage_progress(
            "repair_with_gower",
            "scan_columns",
            "complete",
            46,
            "Gower 可修复问题识别完成",
            file=str(csv_file),
            issue_count=int(len(issues_internal)),
            precomputed_issues_used=bool(precomputed_issues_used),
        )
        issue_map = {str(item["issue_id"]): item for item in issues_internal}
        original_df = df.copy(deep=True)
        repaired_df = df.copy(deep=True)
        preview_limit = int(gower_strategy["preview_limit"])
        supported_issue_types = {"missing_values", "numeric_outlier", "rare_category"}
        skipped_issues: list[dict[str, Any]] = []
        skipped_ids: set[str] = set()
        selected_issues: list[dict[str, Any]] = []
        cell_plan: dict[tuple[int, str], dict[str, Any]] = {}
        changes_by_issue: dict[str, list[dict[str, Any]]] = {}
        issue_evidence: dict[str, dict[str, Any]] = {}

        def add_skip(issue_id: str, reason: str, extra: dict[str, Any] | None = None) -> None:
            if issue_id in skipped_ids:
                return
            row: dict[str, Any] = {"issue_id": issue_id, "reason": reason}
            if extra:
                row.update(extra)
            skipped_issues.append(row)
            skipped_ids.add(issue_id)

        for issue_id in selected_issue_ids:
            issue = issue_map.get(issue_id)
            if issue is None:
                add_skip(issue_id, "issue_not_found")
                continue
            column = str(issue["column"])
            if column not in repaired_df.columns:
                add_skip(issue_id, "column_not_found")
                continue
            selected_issues.append(issue)

        selected_issues.sort(
            key=lambda item: (
                -float(item.get("issue_score", 0.0)),
                str(item.get("column", "")),
                str(item.get("issue_id", "")),
            )
        )

        _emit_stage_progress(
            "repair_with_gower",
            "repair_search",
            "start",
            56,
            "开始执行 Gower 邻居检索",
            file=str(csv_file),
            issue_count=int(len(selected_issues)),
        )

        for issue in selected_issues:
            issue_id = str(issue["issue_id"])
            column = str(issue["column"])
            issue_type = str(issue["issue_type"])
            if issue_type not in supported_issue_types:
                add_skip(
                    issue_id,
                    "unsupported_issue_type",
                    {"issue_type": issue_type, "column": column},
                )
                continue

            rule = issue.get("repair_rule", {})
            mask = _issue_mask_from_rule(original_df[column], issue_type, rule, frame_pd)
            positions = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]
            if not positions:
                add_skip(issue_id, "no_rows_matched", {"issue_type": issue_type, "column": column})
                continue

            healthy_mask = ~mask
            if issue_type == "missing_values":
                healthy_mask = healthy_mask & original_df[column].notna()
            elif issue_type == "numeric_outlier":
                numeric_series = frame_pd.to_numeric(original_df[column], errors="coerce")
                healthy_mask = healthy_mask & numeric_series.notna()
            elif issue_type == "rare_category":
                healthy_mask = healthy_mask & original_df[column].notna()

            candidate_pool = original_df.loc[healthy_mask]
            if candidate_pool.empty:
                add_skip(issue_id, "no_healthy_neighbors", {"issue_type": issue_type, "column": column})
                continue

            feature_columns = [str(name) for name in repaired_df.columns if str(name) != column]
            feature_weights, effective_weight_mode = _resolve_gower_feature_weights(
                feature_columns,
                gower_strategy,
                model_dir,
            )

            issue_changes: list[dict[str, Any]] = []
            mean_distances: list[float] = []
            confidences: list[float] = []
            replacement_values: list[Any] = []
            preview_rows: list[dict[str, Any]] = []
            neighbor_count = 0
            candidate_pool_sizes: list[int] = []
            candidate_sample_sizes: list[int] = []
            candidate_limit_applied = False
            candidate_selection_modes: list[str] = []

            for pos in positions:
                row_label = original_df.index[pos]
                if row_label in candidate_pool.index:
                    candidate_rows = candidate_pool.drop(index=row_label, errors="ignore")
                    if candidate_rows.empty:
                        candidate_rows = candidate_pool
                else:
                    candidate_rows = candidate_pool
                candidate_rows, pool_size, sample_size, limit_applied, selection_mode = _limit_gower_candidate_rows(
                    candidate_rows,
                    gower_strategy.get("max_candidates"),
                    issue_id=issue_id,
                    column=column,
                    row_pos=int(pos),
                    candidate_policy=str(gower_strategy["candidate_policy"]),
                    auto_max_candidates=int(gower_strategy["auto_max_candidates"]),
                    full_scan_threshold=int(gower_strategy["full_scan_threshold"]),
                )
                candidate_pool_sizes.append(pool_size)
                candidate_sample_sizes.append(sample_size)
                candidate_limit_applied = candidate_limit_applied or limit_applied
                if selection_mode not in candidate_selection_modes:
                    candidate_selection_modes.append(selection_mode)
                try:
                    suggestion = suggest_replacement_from_neighbors(
                        candidate_rows,
                        original_df.iloc[[pos]],
                        column,
                        feature_columns=feature_columns,
                        k_neighbors=int(gower_strategy["k_neighbors"]),
                        feature_weights=feature_weights,
                        preview_limit=preview_limit,
                    )
                except Exception:
                    continue

                before_value = original_df[column].iat[pos]
                after_value = suggestion.replacement_value
                before_is_nan = bool(frame_pd.isna(before_value))
                after_is_nan = bool(frame_pd.isna(after_value))
                if (before_is_nan and after_is_nan) or (
                    (not before_is_nan) and (not after_is_nan) and before_value == after_value
                ):
                    continue

                proposal = {
                    "issue_id": issue_id,
                    "column": column,
                    "row_pos": pos,
                    "row": _index_to_builtin(original_df.index[pos]),
                    "col_pos": int(list(repaired_df.columns).index(column)),
                    "before": before_value,
                    "after": after_value,
                }
                cell_plan[(pos, column)] = proposal
                issue_changes.append(proposal)
                changes_by_issue.setdefault(issue_id, []).append(proposal)
                mean_distances.append(float(suggestion.distance_summary["mean"]))
                confidences.append(float(suggestion.confidence))
                replacement_values.append(after_value)
                neighbor_count = max(neighbor_count, int(suggestion.neighbor_count))
                for item in suggestion.neighbor_rows_preview:
                    if len(preview_rows) >= preview_limit:
                        break
                    preview_rows.append(item)

            if not issue_changes:
                add_skip(issue_id, "no_healthy_neighbors", {"issue_type": issue_type, "column": column})
                continue

            for proposal in issue_changes:
                repaired_df.iat[int(proposal["row_pos"]), int(proposal["col_pos"])] = proposal["after"]

            unique_replacements: list[Any] = []
            for item in replacement_values:
                builtin_item = _to_builtin(item)
                if builtin_item in unique_replacements:
                    continue
                unique_replacements.append(builtin_item)

            replacement_value: Any
            if len(unique_replacements) == 1:
                replacement_value = unique_replacements[0]
            else:
                replacement_value = unique_replacements[:preview_limit]

            distance_summary = {
                "min": round(min(mean_distances), 6),
                "max": round(max(mean_distances), 6),
                "mean": round(sum(mean_distances) / float(len(mean_distances)), 6),
                "median": round(float(frame_pd.Series(mean_distances).median()), 6),
            }
            issue_evidence[issue_id] = {
                "issue_id": issue_id,
                "column": column,
                "issue_type": issue_type,
                "neighbor_count": neighbor_count,
                "neighbor_rows_preview": preview_rows[:preview_limit],
                "distance_summary": distance_summary,
                "replacement_value": replacement_value,
                "candidate_confidence": round(sum(confidences) / float(len(confidences)), 6),
                "weight_mode": effective_weight_mode,
                "candidate_pool_size": max(candidate_pool_sizes) if candidate_pool_sizes else 0,
                "candidate_sample_size": max(candidate_sample_sizes) if candidate_sample_sizes else 0,
                "candidate_limit_applied": bool(candidate_limit_applied),
                "candidate_policy": str(gower_strategy["candidate_policy"]),
                "candidate_selection_mode": candidate_selection_modes[0] if len(candidate_selection_modes) == 1 else candidate_selection_modes,
                "auto_max_candidates": int(gower_strategy["auto_max_candidates"]),
                "full_scan_threshold": int(gower_strategy["full_scan_threshold"]),
            }

        _emit_stage_progress(
            "repair_with_gower",
            "repair_search",
            "complete",
            76,
            "Gower 邻居检索完成",
            file=str(csv_file),
            changed_cells=int(len(cell_plan)),
        )

        post_issues_internal = _detect_issues_for_frame(repaired_df, frame_pd, scan_config=scan_config)
        before_issue_type_counts = _issue_type_counter(issues_internal)
        after_issue_type_counts = _issue_type_counter(post_issues_internal)
        before_issue_column_counts = _issue_counter_by_column(issues_internal)
        after_issue_column_counts = _issue_counter_by_column(post_issues_internal)
        applied_repairs: list[dict[str, Any]] = []
        neighbor_evidence: list[dict[str, Any]] = []

        for issue_id in selected_issue_ids:
            issue = issue_map.get(issue_id)
            if issue is None or issue_id in skipped_ids:
                continue
            issue_type = str(issue.get("issue_type"))
            column = str(issue.get("column"))
            issue_changes = sorted(changes_by_issue.get(issue_id, []), key=lambda item: int(item["row_pos"]))
            if not issue_changes:
                continue
            rule = issue.get("repair_rule", {})
            before_count = int(issue.get("count", len(issue_changes)))
            after_mask = _issue_mask_from_rule(repaired_df[column], issue_type, rule, frame_pd)
            after_count = int(after_mask.sum())
            resolved_count = max(0, before_count - after_count)
            evidence = issue_evidence.get(issue_id, {})
            applied_repairs.append(
                {
                    "issue_id": issue_id,
                    "column": column,
                    "issue_type": issue_type,
                    "rows_touched": len(issue_changes),
                    "replacement_preview": evidence.get("replacement_value"),
                    "before_count": before_count,
                    "after_count": after_count,
                    "resolved_count": resolved_count,
                    "candidate_confidence": evidence.get("candidate_confidence", 0.0),
                    "cells_preview": [
                        {
                            "row": _index_to_builtin(change["row"]),
                            "before": _to_builtin(change["before"]),
                            "after": _to_builtin(change["after"]),
                        }
                        for change in issue_changes[:preview_limit]
                    ],
                    "strategy": {
                        "tool_id": "engine.repair_with_gower",
                        "weight_mode": evidence.get("weight_mode", "uniform"),
                        "k_neighbors": int(gower_strategy["k_neighbors"]),
                    },
                }
            )
            if evidence:
                neighbor_evidence.append(evidence)
    except KnownEngineError:
        raise
    except Exception as exc:
        _emit_stage_progress(
            "repair_with_gower",
            "repair_search",
            "error",
            100,
            "Gower 修复失败",
            file=str(csv_file),
            error_code=ErrorCode.REPAIR_BATCH_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.REPAIR_BATCH_FAILED,
            message="Gower repair failed",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc

    output_csv: str | None = None
    rollback_info: dict[str, Any] | None = None
    if write_output:
        _emit_stage_progress("repair_with_gower", "write_output", "start", 84, "开始写出 Gower 修复结果", file=str(csv_file))
        output_path = _resolve_output_path(csv_file, payload, default_suffix=".repaired.gower.csv")
        os.makedirs(output_path.parent, exist_ok=True)
        if _to_bool(payload, "enable_rollback", default=True):
            _, rollback_info = _create_rollback_artifacts(
                source_tool_id="engine.repair_with_gower",
                csv_file=csv_file,
                output_path=output_path,
                selected_issue_ids=selected_issue_ids,
                issue_source_map={issue_id: "gower" for issue_id in selected_issue_ids},
                execution_steps=[
                    {
                        "step": 1,
                        "tool_id": "engine.repair_with_gower",
                        "source": "gower",
                    }
                ],
                payload=payload,
                extra={
                    "scan_config": scan_config,
                    "gower_strategy": gower_strategy,
                    "column_dependencies": column_dependencies,
                    "model_dir": str(model_dir) if model_dir is not None else None,
                },
            )
        repaired_df.to_csv(output_path, index=False)
        output_csv = str(output_path)
        _emit_stage_progress("repair_with_gower", "write_output", "complete", 96, "Gower 修复结果写出完成", file=str(output_csv))

    total_cells_modified = int(len(cell_plan))
    before_issue_count = int(len(issues_internal))
    after_issue_count = int(len(post_issues_internal))
    changed_cells_preview = [
        {
            "row": _index_to_builtin(change["row"]),
            "column": str(change["column"]),
            "issue_id": str(change["issue_id"]),
            "before": _to_builtin(change["before"]),
            "after": _to_builtin(change["after"]),
        }
        for change in sorted(cell_plan.values(), key=lambda item: (int(item["row_pos"]), str(item["column"])))[:preview_limit]
    ]
    _emit_stage_progress(
        "repair_with_gower",
        "complete",
        "complete",
        100,
        "Gower 修复任务完成",
        file=str(output_csv or csv_file),
        selected_issue_count=int(len(selected_issue_ids)),
        applied_issue_count=int(len(applied_repairs)),
    )
    return _to_builtin(
        {
            "csv_path": str(csv_file),
            "scan_config": scan_config,
            "column_dependencies": column_dependencies,
            "gower_strategy": {
                "k_neighbors": int(gower_strategy["k_neighbors"]),
                "weight_mode": str(gower_strategy["weight_mode"]),
                "preview_limit": int(gower_strategy["preview_limit"]),
                "max_candidates": gower_strategy.get("max_candidates"),
                "candidate_policy": str(gower_strategy["candidate_policy"]),
                "auto_max_candidates": int(gower_strategy["auto_max_candidates"]),
                "full_scan_threshold": int(gower_strategy["full_scan_threshold"]),
            },
            "plan_only": plan_only,
            "execution_mode": "plan_only" if plan_only else "apply",
            "write_output": write_output,
            "precomputed_issues_used": bool(precomputed_issues_used),
            "output_csv": output_csv,
            "selected_issue_ids": selected_issue_ids,
            "selected_issue_count": len(selected_issue_ids),
            "applied_issue_count": len(applied_repairs),
            "total_cells_modified": total_cells_modified,
            "applied_repairs": applied_repairs,
            "skipped_issues": skipped_issues,
            "neighbor_evidence": neighbor_evidence,
            "comparison": {
                "before_issue_count": before_issue_count,
                "after_issue_count": after_issue_count,
                "resolved_issue_count": max(0, before_issue_count - after_issue_count),
                "before_issue_type_counts": before_issue_type_counts,
                "after_issue_type_counts": after_issue_type_counts,
                "before_column_issue_counts": before_issue_column_counts,
                "after_column_issue_counts": after_issue_column_counts,
                "changed_cell_count": total_cells_modified,
                "changed_cells_preview": changed_cells_preview,
            },
            "rollback": rollback_info,
        }
    )


def action_repair_with_missforest(payload: dict[str, Any]) -> dict[str, Any]:
    _emit_stage_progress(
        "repair_with_missforest",
        "validate_input",
        "start",
        2,
        "开始校验 MissForest 修复参数",
    )
    frame_pd = _load_dataframe_module("Repair with MissForest")
    csv_path = _require(payload, "csv_path")
    scan_config = _scan_config_from_payload(payload)
    missforest_strategy = _missforest_strategy_from_payload(payload)
    write_output_requested = _to_bool(payload, "write_output", default=True)
    plan_only = _to_bool(payload, "plan_only", default=False)
    write_output = write_output_requested and (not plan_only)

    raw_issue_ids = payload.get("issue_ids", [])
    if raw_issue_ids is None:
        raw_issue_ids = []
    if not isinstance(raw_issue_ids, (list, tuple, set)):
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Field issue_ids must be a string list",
            details={"field": "issue_ids", "value": raw_issue_ids},
        )

    selected_issue_ids: list[str] = []
    seen_issue_ids: set[str] = set()
    for raw in raw_issue_ids:
        issue_id = str(raw).strip()
        if not issue_id or issue_id in seen_issue_ids:
            continue
        seen_issue_ids.add(issue_id)
        selected_issue_ids.append(issue_id)

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.MISSING_DEPENDENCY,
            message="MissForest repair dependency missing: scikit-learn",
            details={"reason": str(exc)},
        ) from exc

    csv_file = _resolve_input_csv(str(csv_path))
    if not csv_file.exists():
        _emit_stage_progress(
            "repair_with_missforest",
            "validate_input",
            "error",
            100,
            "输入文件不存在",
            file=str(csv_file),
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    _emit_stage_progress("repair_with_missforest", "load_csv", "start", 12, "开始读取待修复文件", file=str(csv_file))
    try:
        df = frame_pd.read_csv(csv_file)
    except Exception as exc:
        _emit_stage_progress(
            "repair_with_missforest",
            "load_csv",
            "error",
            100,
            "读取待修复文件失败",
            file=str(csv_file),
            error_code=ErrorCode.CSV_READ_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.CSV_READ_FAILED,
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc
    _emit_stage_progress(
        "repair_with_missforest",
        "load_csv",
        "complete",
        22,
        "待修复文件读取完成",
        file=str(csv_file),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
    )

    _emit_stage_progress(
        "repair_with_missforest",
        "scan_columns",
        "start",
        34,
        "开始识别 MissForest 可修复问题",
        file=str(csv_file),
    )
    try:
        precomputed_issues, precomputed_issues_used = _precomputed_issues_from_payload(
            payload,
            csv_file,
            plan_only=plan_only,
        )
        if precomputed_issues_used and precomputed_issues is not None:
            issues_internal = precomputed_issues
        else:
            issues_internal = _detect_issues_for_frame(df, frame_pd, scan_config=scan_config)
        _emit_stage_progress(
            "repair_with_missforest",
            "scan_columns",
            "complete",
            46,
            "MissForest 可修复问题识别完成",
            file=str(csv_file),
            issue_count=int(len(issues_internal)),
            precomputed_issues_used=bool(precomputed_issues_used),
        )
        issue_map = {str(item["issue_id"]): item for item in issues_internal}
        original_df = df.copy(deep=True)
        repaired_df = df.copy(deep=True)
        supported_issue_types = {"missing_values", "numeric_outlier", "rare_category"}
        preview_limit = int(missforest_strategy["preview_limit"])
        skipped_issues: list[dict[str, Any]] = []
        skipped_ids: set[str] = set()
        selected_issues: list[dict[str, Any]] = []
        cell_plan: dict[tuple[int, str], dict[str, Any]] = {}
        changes_by_issue: dict[str, list[dict[str, Any]]] = {}
        issue_evidence: dict[str, dict[str, Any]] = {}

        def add_skip(issue_id: str, reason: str, extra: dict[str, Any] | None = None) -> None:
            if issue_id in skipped_ids:
                return
            row: dict[str, Any] = {"issue_id": issue_id, "reason": reason}
            if extra:
                row.update(extra)
            skipped_issues.append(row)
            skipped_ids.add(issue_id)

        for issue_id in selected_issue_ids:
            issue = issue_map.get(issue_id)
            if issue is None:
                add_skip(issue_id, "issue_not_found")
                continue
            column = str(issue["column"])
            if column not in repaired_df.columns:
                add_skip(issue_id, "column_not_found")
                continue
            selected_issues.append(issue)

        selected_issues.sort(
            key=lambda item: (
                -float(item.get("issue_score", 0.0)),
                str(item.get("column", "")),
                str(item.get("issue_id", "")),
            )
        )
        _emit_stage_progress(
            "repair_with_missforest",
            "repair_search",
            "start",
            56,
            "开始执行 MissForest 候选修复",
            file=str(csv_file),
            issue_count=int(len(selected_issues)),
        )

        issue_masks: dict[str, Any] = {}
        issue_positions: dict[str, list[int]] = {}
        active_issues: list[dict[str, Any]] = []
        for issue in selected_issues:
            issue_id = str(issue["issue_id"])
            column = str(issue["column"])
            issue_type = str(issue["issue_type"])
            if issue_type not in supported_issue_types:
                add_skip(
                    issue_id,
                    "unsupported_issue_type",
                    {"issue_type": issue_type, "column": column},
                )
                continue

            rule = issue.get("repair_rule", {})
            mask = _issue_mask_from_rule(original_df[column], issue_type, rule, frame_pd)
            positions = [idx for idx, flag in enumerate(mask.tolist()) if bool(flag)]
            if not positions:
                add_skip(issue_id, "no_rows_matched", {"issue_type": issue_type, "column": column})
                continue
            issue_masks[issue_id] = mask
            issue_positions[issue_id] = positions
            active_issues.append(issue)

        if str(missforest_strategy["algorithm_mode"]) == "iterative":
            iterative_changes, iterative_evidence, iterative_skips = _missforest_predict_iterative(
                frame_pd=frame_pd,
                original_df=original_df,
                selected_issues=active_issues,
                issue_masks=issue_masks,
                issue_positions=issue_positions,
                strategy=missforest_strategy,
                random_forest_regressor=RandomForestRegressor,
                random_forest_classifier=RandomForestClassifier,
            )
            for skipped in iterative_skips:
                issue_id = str(skipped.get("issue_id") or "")
                if not issue_id:
                    continue
                reason = str(skipped.get("reason") or "missforest_training_unavailable")
                extra = dict(skipped)
                extra.pop("issue_id", None)
                extra.pop("reason", None)
                add_skip(issue_id, reason, extra)
            for issue in active_issues:
                issue_id = str(issue["issue_id"])
                column = str(issue["column"])
                issue_type = str(issue["issue_type"])
                issue_changes = iterative_changes.get(issue_id, [])
                if not issue_changes:
                    if issue_id not in skipped_ids:
                        add_skip(issue_id, "no_prediction_change", {"issue_type": issue_type, "column": column})
                    continue
                for proposal in issue_changes:
                    cell_plan[(int(proposal["row_pos"]), column)] = proposal
                    changes_by_issue.setdefault(issue_id, []).append(proposal)
                    repaired_df.iat[int(proposal["row_pos"]), int(proposal["col_pos"])] = proposal["after"]
                issue_evidence[issue_id] = iterative_evidence.get(issue_id, {})
        else:
            for issue in active_issues:
                issue_id = str(issue["issue_id"])
                column = str(issue["column"])
                issue_type = str(issue["issue_type"])
                try:
                    issue_changes, evidence = _missforest_predict_issue(
                        frame_pd=frame_pd,
                        original_df=original_df,
                        issue=issue,
                        mask=issue_masks[issue_id],
                        positions=issue_positions[issue_id],
                        strategy=missforest_strategy,
                        random_forest_regressor=RandomForestRegressor,
                        random_forest_classifier=RandomForestClassifier,
                    )
                except KnownEngineError as exc:
                    details = dict(exc.details) if isinstance(exc.details, dict) else {}
                    details.update({"issue_type": issue_type, "column": column})
                    add_skip(issue_id, "missforest_training_unavailable", details)
                    continue
                except Exception as exc:
                    add_skip(
                        issue_id,
                        "missforest_prediction_failed",
                        {"issue_type": issue_type, "column": column, "reason": str(exc)},
                    )
                    continue

                if not issue_changes:
                    add_skip(issue_id, "no_prediction_change", {"issue_type": issue_type, "column": column})
                    continue

                for proposal in issue_changes:
                    cell_plan[(int(proposal["row_pos"]), column)] = proposal
                    changes_by_issue.setdefault(issue_id, []).append(proposal)
                    repaired_df.iat[int(proposal["row_pos"]), int(proposal["col_pos"])] = proposal["after"]
                issue_evidence[issue_id] = evidence

        _emit_stage_progress(
            "repair_with_missforest",
            "repair_search",
            "complete",
            76,
            "MissForest 候选修复完成",
            file=str(csv_file),
            changed_cells=int(len(cell_plan)),
        )

        post_issues_internal = _detect_issues_for_frame(repaired_df, frame_pd, scan_config=scan_config)
        before_issue_type_counts = _issue_type_counter(issues_internal)
        after_issue_type_counts = _issue_type_counter(post_issues_internal)
        before_issue_column_counts = _issue_counter_by_column(issues_internal)
        after_issue_column_counts = _issue_counter_by_column(post_issues_internal)
        applied_repairs: list[dict[str, Any]] = []
        model_evidence: list[dict[str, Any]] = []

        for issue_id in selected_issue_ids:
            issue = issue_map.get(issue_id)
            if issue is None or issue_id in skipped_ids:
                continue
            issue_type = str(issue.get("issue_type"))
            column = str(issue.get("column"))
            issue_changes = sorted(changes_by_issue.get(issue_id, []), key=lambda item: int(item["row_pos"]))
            if not issue_changes:
                continue
            rule = issue.get("repair_rule", {})
            before_count = int(issue.get("count", len(issue_changes)))
            after_mask = _issue_mask_from_rule(repaired_df[column], issue_type, rule, frame_pd)
            after_count = int(after_mask.sum())
            resolved_count = max(0, before_count - after_count)
            evidence = issue_evidence.get(issue_id, {})
            replacements: list[Any] = []
            for change in issue_changes:
                value = _to_builtin(change["after"])
                if value not in replacements:
                    replacements.append(value)
                if len(replacements) >= preview_limit:
                    break
            applied_repairs.append(
                {
                    "issue_id": issue_id,
                    "column": column,
                    "issue_type": issue_type,
                    "rows_touched": len(issue_changes),
                    "replacement_preview": replacements[0] if len(replacements) == 1 else replacements,
                    "before_count": before_count,
                    "after_count": after_count,
                    "resolved_count": resolved_count,
                    "candidate_confidence": evidence.get("candidate_confidence", 0.0),
                    "cells_preview": [
                        {
                            "row": _index_to_builtin(change["row"]),
                            "before": _to_builtin(change["before"]),
                            "after": _to_builtin(change["after"]),
                        }
                        for change in issue_changes[:preview_limit]
                    ],
                    "strategy": {
                        "tool_id": "engine.repair_with_missforest",
                        "model_type": evidence.get("model_type", "random_forest"),
                        "n_estimators": int(missforest_strategy["n_estimators"]),
                    },
                }
            )
            if evidence:
                model_evidence.append(evidence)
    except KnownEngineError:
        raise
    except Exception as exc:
        _emit_stage_progress(
            "repair_with_missforest",
            "repair_search",
            "error",
            100,
            "MissForest 修复失败",
            file=str(csv_file),
            error_code=ErrorCode.REPAIR_BATCH_FAILED,
            reason=str(exc),
        )
        raise KnownEngineError(
            code=ErrorCode.REPAIR_BATCH_FAILED,
            message="MissForest repair failed",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc

    output_csv: str | None = None
    rollback_info: dict[str, Any] | None = None
    if write_output:
        _emit_stage_progress(
            "repair_with_missforest",
            "write_output",
            "start",
            84,
            "开始写出 MissForest 修复结果",
            file=str(csv_file),
        )
        output_path = _resolve_output_path(csv_file, payload, default_suffix=".repaired.missforest.csv")
        os.makedirs(output_path.parent, exist_ok=True)
        if _to_bool(payload, "enable_rollback", default=True):
            _, rollback_info = _create_rollback_artifacts(
                source_tool_id="engine.repair_with_missforest",
                csv_file=csv_file,
                output_path=output_path,
                selected_issue_ids=selected_issue_ids,
                issue_source_map={issue_id: "missforest" for issue_id in selected_issue_ids},
                execution_steps=[
                    {
                        "step": 1,
                        "tool_id": "engine.repair_with_missforest",
                        "source": "missforest",
                    }
                ],
                payload=payload,
                extra={
                    "scan_config": scan_config,
                    "missforest_strategy": missforest_strategy,
                },
            )
        repaired_df.to_csv(output_path, index=False)
        output_csv = str(output_path)
        _emit_stage_progress(
            "repair_with_missforest",
            "write_output",
            "complete",
            96,
            "MissForest 修复结果写出完成",
            file=str(output_csv),
        )

    total_cells_modified = int(len(cell_plan))
    before_issue_count = int(len(issues_internal))
    after_issue_count = int(len(post_issues_internal))
    changed_cells_preview = [
        {
            "row": _index_to_builtin(change["row"]),
            "column": str(change["column"]),
            "issue_id": str(change["issue_id"]),
            "before": _to_builtin(change["before"]),
            "after": _to_builtin(change["after"]),
        }
        for change in sorted(cell_plan.values(), key=lambda item: (int(item["row_pos"]), str(item["column"])))[:preview_limit]
    ]
    _emit_stage_progress(
        "repair_with_missforest",
        "complete",
        "complete",
        100,
        "MissForest 修复任务完成",
        file=str(output_csv or csv_file),
        selected_issue_count=int(len(selected_issue_ids)),
        applied_issue_count=int(len(applied_repairs)),
    )
    return _to_builtin(
        {
            "csv_path": str(csv_file),
            "scan_config": scan_config,
            "missforest_strategy": {
                "algorithm_mode": str(missforest_strategy["algorithm_mode"]),
                "max_iter": int(missforest_strategy["max_iter"]),
                "convergence_tolerance": float(missforest_strategy["convergence_tolerance"]),
                "n_estimators": int(missforest_strategy["n_estimators"]),
                "max_depth": missforest_strategy["max_depth"],
                "min_training_rows": int(missforest_strategy["min_training_rows"]),
                "max_train_rows": int(missforest_strategy["max_train_rows"]),
                "random_state": int(missforest_strategy["random_state"]),
                "max_features": missforest_strategy["max_features"],
                "preview_limit": int(missforest_strategy["preview_limit"]),
            },
            "plan_only": plan_only,
            "execution_mode": "plan_only" if plan_only else "apply",
            "write_output": write_output,
            "precomputed_issues_used": bool(precomputed_issues_used),
            "output_csv": output_csv,
            "selected_issue_ids": selected_issue_ids,
            "selected_issue_count": len(selected_issue_ids),
            "applied_issue_count": len(applied_repairs),
            "total_cells_modified": total_cells_modified,
            "applied_repairs": applied_repairs,
            "skipped_issues": skipped_issues,
            "model_evidence": model_evidence,
            "comparison": {
                "before_issue_count": before_issue_count,
                "after_issue_count": after_issue_count,
                "resolved_issue_count": max(0, before_issue_count - after_issue_count),
                "before_issue_type_counts": before_issue_type_counts,
                "after_issue_type_counts": after_issue_type_counts,
                "before_column_issue_counts": before_issue_column_counts,
                "after_column_issue_counts": after_issue_column_counts,
                "changed_cell_count": total_cells_modified,
                "changed_cells_preview": changed_cells_preview,
            },
            "rollback": rollback_info,
        }
    )


def action_rollback_repair_batch(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path_raw = str(_require(payload, "manifest_path"))
    manifest_path = _resolve_output_file(manifest_path_raw)
    if not manifest_path.exists():
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Rollback manifest does not exist: {manifest_path}",
            details={"manifest_path": str(manifest_path)},
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.ROLLBACK_FAILED,
            message="Failed to read rollback manifest",
            details={"manifest_path": str(manifest_path), "reason": str(exc)},
        ) from exc

    if not isinstance(manifest, dict):
        raise KnownEngineError(
            code=ErrorCode.ROLLBACK_FAILED,
            message="Rollback manifest must be an object",
            details={"manifest_path": str(manifest_path)},
        )

    backup_csv = Path(str(manifest.get("backup_csv") or "")).expanduser().resolve()
    if not backup_csv.exists():
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Rollback backup does not exist: {backup_csv}",
            details={"backup_csv": str(backup_csv), "manifest_path": str(manifest_path)},
        )

    restore_target = str(payload.get("restore_target") or "source_csv").strip().lower()
    target_csv_raw = str(payload.get("target_csv") or "").strip()
    target_path = _rollback_manifest_to_target(manifest, restore_target, custom_target=target_csv_raw)

    try:
        os.makedirs(target_path.parent, exist_ok=True)
        shutil.copy2(backup_csv, target_path)
    except Exception as exc:
        raise KnownEngineError(
            code=ErrorCode.ROLLBACK_FAILED,
            message="Rollback copy failed",
            details={"backup_csv": str(backup_csv), "target_csv": str(target_path), "reason": str(exc)},
        ) from exc

    return _to_builtin(
        {
            "rollback_id": manifest.get("rollback_id"),
            "manifest_path": str(manifest_path),
            "manifest_version": manifest.get("manifest_version", 1),
            "source_tool_id": manifest.get("source_tool_id", "engine.repair_batch"),
            "backup_csv": str(backup_csv),
            "restored_to": str(target_path),
            "source_csv": manifest.get("source_csv"),
            "output_csv": manifest.get("output_csv"),
            "issue_source_map": manifest.get("issue_source_map", {}),
        }
    )
