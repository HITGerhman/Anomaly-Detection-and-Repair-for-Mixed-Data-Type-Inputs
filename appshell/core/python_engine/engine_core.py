"""Engine algorithm layer (independent from CLI transport)."""

from __future__ import annotations

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

    try:
        model, x_test, normal_data, metrics, feature_names = process_and_train(df, str(target_col))
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
            "target_col": str(target_col),
        },
        "metrics": _metric_summary(metrics),
    }

