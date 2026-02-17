"""Service layer for handling engine actions."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from engine_protocol import KnownEngineError

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
            code="INVALID_INPUT",
            message=f"Missing required field: {key}",
            details={"field": key},
        )
    return value


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas values to JSON-safe builtins recursively."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, (np.integer,)):
        return int(value)
    if np is not None and isinstance(value, (np.floating,)):
        return float(value)
    if pd is not None and isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return only transport-friendly metric fields for API responses."""
    summary = {
        "f1": metrics.get("f1", 0.0),
        "auc": metrics.get("auc", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "feature_importance": metrics.get("feature_importance", {}),
    }
    if "roc_curve" in metrics:
        summary["roc_curve"] = {
            "fpr": metrics["roc_curve"].get("fpr"),
            "tpr": metrics["roc_curve"].get("tpr"),
        }
    return _to_builtin(summary)


def _load_training_modules() -> tuple[Any, Any, Any]:
    try:
        import pandas as train_pd  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise KnownEngineError(
            code="MISSING_DEPENDENCY",
            message="Training dependency missing: pandas",
            details={"reason": str(exc)},
        ) from exc

    try:
        from src.utils import process_and_train, save_system_state  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise KnownEngineError(
            code="TRAINING_MODULE_IMPORT_FAILED",
            message="Failed to import training modules",
            details={"reason": str(exc)},
        ) from exc

    return train_pd, process_and_train, save_system_state


def action_health(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": "python-anomaly-engine",
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "actions": ["health", "train"],
    }


def action_train(payload: dict[str, Any]) -> dict[str, Any]:
    train_pd, process_and_train, save_system_state = _load_training_modules()

    csv_path = _require(payload, "csv_path")
    target_col = _require(payload, "target_col")
    output_dir = payload.get("output_dir") or str(PROJECT_ROOT / "data" / "processed")

    csv_file = Path(csv_path).expanduser().resolve()
    if not csv_file.exists():
        raise KnownEngineError(
            code="FILE_NOT_FOUND",
            message=f"Input CSV does not exist: {csv_file}",
            details={"csv_path": str(csv_file)},
        )

    try:
        df = train_pd.read_csv(csv_file)
    except Exception as exc:  # pragma: no cover - serialization guard
        raise KnownEngineError(
            code="CSV_READ_FAILED",
            message="Failed to read CSV",
            details={"csv_path": str(csv_file), "reason": str(exc)},
        ) from exc

    if target_col not in df.columns:
        raise KnownEngineError(
            code="INVALID_TARGET_COLUMN",
            message=f"Target column not found: {target_col}",
            details={"available_columns": list(df.columns)},
        )

    try:
        model, x_test, normal_data, metrics, feature_names = process_and_train(df, target_col)
        os.makedirs(output_dir, exist_ok=True)
        save_system_state(model, x_test, normal_data, feature_names, save_dir=output_dir)
    except KnownEngineError:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard
        raise KnownEngineError(
            code="TRAINING_FAILED",
            message="Model training failed",
            details={"reason": str(exc)},
        ) from exc

    return {
        "artifacts": {
            "output_dir": output_dir,
            "model": str(Path(output_dir) / "model_lgb.pkl"),
            "test_data": str(Path(output_dir) / "test_data.pkl"),
            "normal_data": str(Path(output_dir) / "normal_data.pkl"),
            "config": str(Path(output_dir) / "config.pkl"),
        },
        "data_profile": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_col": target_col,
        },
        "metrics": _metric_summary(metrics),
    }


def handle_action(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    registry = {
        "health": action_health,
        "train": action_train,
    }
    fn = registry.get(action)
    if fn is None:
        raise KnownEngineError(
            code="UNKNOWN_ACTION",
            message=f"Unsupported action: {action}",
            details={"supported_actions": list(registry.keys())},
        )
    return fn(payload)
