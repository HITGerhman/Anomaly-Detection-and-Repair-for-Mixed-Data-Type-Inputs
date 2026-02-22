"""Repair logic for encoded mixed-type anomaly samples.

This module is independent from UI/runtime layers and focuses on:
- constrained candidate generation from healthy neighbors
- greedy search for minimal edits that reduce anomaly score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .training_core import predict_with_threshold


@dataclass(frozen=True)
class RepairBundle:
    repaired_sample: pd.DataFrame
    summary: dict[str, Any]
    changes: list[dict[str, Any]]


def _is_discrete(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        return True
    if pd.api.types.is_integer_dtype(dtype):
        return int(series.nunique(dropna=True)) <= 12
    return False


def _feature_weights(model: Any, feature_names: list[str]) -> dict[str, float]:
    raw = np.asarray(getattr(model, "feature_importances_", np.array([])), dtype=float)
    if raw.size != len(feature_names):
        raw = np.ones(len(feature_names), dtype=float)
    raw = np.clip(raw, 0.0, None)
    if np.all(raw == 0):
        raw = np.ones(len(feature_names), dtype=float)
    raw = raw + 1e-6
    raw = raw / raw.sum()
    return {name: float(raw[idx]) for idx, name in enumerate(feature_names)}


def _numeric_scales(normal_data: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for col in feature_names:
        series = normal_data[col]
        if _is_discrete(series):
            scales[col] = 1.0
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        span = float(numeric.quantile(0.95) - numeric.quantile(0.05))
        if not np.isfinite(span) or span <= 0:
            span = float(numeric.max() - numeric.min())
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        scales[col] = span
    return scales


def _distance_to_normals(
    sample_row: pd.Series,
    normal_data: pd.DataFrame,
    feature_names: list[str],
    weights: dict[str, float],
    scales: dict[str, float],
) -> np.ndarray:
    total = np.zeros(len(normal_data), dtype=float)
    total_weight = 0.0

    for col in feature_names:
        weight = float(weights.get(col, 0.0))
        if weight <= 0:
            continue
        total_weight += weight

        series = normal_data[col]
        sample_val = sample_row[col]
        if _is_discrete(series):
            sample_token = str(sample_val)
            comp = (series.astype(str).to_numpy() != sample_token).astype(float)
            total += weight * comp
            continue

        sample_num = pd.to_numeric(pd.Series([sample_val]), errors="coerce").iloc[0]
        if pd.isna(sample_num):
            total += weight
            continue
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        comp = np.abs(numeric - float(sample_num)) / max(scales.get(col, 1.0), 1e-12)
        comp = np.nan_to_num(comp, nan=1.0, posinf=1.0, neginf=1.0)
        total += weight * comp

    if total_weight <= 0:
        return total
    return total / total_weight


def _cast_like_column(candidate: Any, col_series: pd.Series, fallback: Any) -> Any:
    if pd.isna(candidate):
        return fallback
    dtype = col_series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        categories = list(col_series.cat.categories)
        if candidate in categories:
            return candidate
        return fallback
    if pd.api.types.is_bool_dtype(dtype):
        return bool(candidate)
    if pd.api.types.is_integer_dtype(dtype):
        return int(round(float(candidate)))
    if pd.api.types.is_float_dtype(dtype):
        return float(candidate)
    return candidate


def _clip_numeric(candidate: Any, bounds: dict[str, Any]) -> Any:
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        return candidate

    low = bounds.get("min")
    high = bounds.get("max")
    if low is not None:
        try:
            value = max(value, float(low))
        except (TypeError, ValueError):
            pass
    if high is not None:
        try:
            value = min(value, float(high))
        except (TypeError, ValueError):
            pass
    return value


def _same_value(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def _candidate_for_feature(
    col: str,
    current_row: pd.Series,
    neighbors: pd.DataFrame,
    normal_data: pd.DataFrame,
    numeric_bounds: dict[str, dict[str, Any]],
) -> Any:
    current_val = current_row[col]
    col_neighbors = neighbors[col].dropna()
    if col_neighbors.empty:
        return current_val

    reference_col = normal_data[col]
    if _is_discrete(reference_col):
        mode_series = col_neighbors.mode(dropna=True)
        if mode_series.empty:
            return current_val
        candidate = mode_series.iloc[0]
    else:
        candidate = float(pd.to_numeric(col_neighbors, errors="coerce").median())
        candidate = _clip_numeric(candidate, numeric_bounds.get(col, {}))

    return _cast_like_column(candidate, current_row.to_frame().T[col], current_val)


def _rank_features(feature_names: list[str], weights: dict[str, float], immutable: set[str]) -> list[str]:
    ranked = sorted(feature_names, key=lambda name: float(weights.get(name, 0.0)), reverse=True)
    return [name for name in ranked if name not in immutable]


def repair_anomaly_sample(
    model: Any,
    sample: pd.DataFrame,
    normal_data: pd.DataFrame,
    *,
    max_changes: int = 3,
    k_neighbors: int = 9,
    immutable_columns: list[str] | None = None,
    numeric_bounds: dict[str, dict[str, Any]] | None = None,
    min_improvement: float = 1e-4,
) -> RepairBundle:
    if sample.shape[0] != 1:
        raise ValueError("sample must contain exactly one row")
    if normal_data.empty:
        raise ValueError("normal_data is empty")

    feature_names = list(sample.columns)
    immutable = {str(name).strip() for name in (immutable_columns or []) if str(name).strip()}
    numeric_bounds = numeric_bounds or {}

    for col in feature_names:
        if col not in normal_data.columns:
            raise ValueError(f"normal_data missing required feature column: {col}")

    max_changes = max(0, int(max_changes))
    max_changes = min(max_changes, len(feature_names))
    k_neighbors = max(3, int(k_neighbors))

    before_pred_arr, before_prob_arr = predict_with_threshold(model, sample)
    before_pred = int(before_pred_arr[0])
    before_score = float(before_prob_arr[0])

    current = sample.copy()
    changes: list[dict[str, Any]] = []

    if before_pred == 1:
        weights = _feature_weights(model, feature_names)
        scales = _numeric_scales(normal_data, feature_names)
        ranked_features = _rank_features(feature_names, weights, immutable)

        for _ in range(max_changes):
            current_pred_arr, current_prob_arr = predict_with_threshold(model, current)
            current_pred = int(current_pred_arr[0])
            current_score = float(current_prob_arr[0])
            if current_pred == 0:
                break

            sample_row = current.iloc[0]
            distances = _distance_to_normals(
                sample_row=sample_row,
                normal_data=normal_data,
                feature_names=feature_names,
                weights=weights,
                scales=scales,
            )
            nearest_idx = np.argsort(distances)[: min(k_neighbors, len(distances))]
            neighbors = normal_data.iloc[nearest_idx]

            best: dict[str, Any] | None = None
            best_score = (-1, float("-inf"), float("-inf"))

            changed_set = {item["feature"] for item in changes}
            for feature in ranked_features:
                if feature in changed_set:
                    continue

                candidate = _candidate_for_feature(
                    feature,
                    current_row=sample_row,
                    neighbors=neighbors,
                    normal_data=normal_data,
                    numeric_bounds=numeric_bounds,
                )
                old_value = sample_row[feature]
                if _same_value(old_value, candidate):
                    continue

                trial = current.copy()
                trial.at[trial.index[0], feature] = candidate
                pred_arr, prob_arr = predict_with_threshold(model, trial)
                new_pred = int(pred_arr[0])
                new_score = float(prob_arr[0])
                improvement = current_score - new_score
                if new_pred == 1 and improvement < min_improvement:
                    continue

                rank = (1 if new_pred == 0 else 0, improvement, float(weights.get(feature, 0.0)))
                if rank > best_score:
                    best_score = rank
                    best = {
                        "feature": feature,
                        "before": old_value,
                        "after": candidate,
                        "before_score": current_score,
                        "after_score": new_score,
                        "after_pred": new_pred,
                        "score_delta": improvement,
                    }

            if best is None:
                break

            current.at[current.index[0], best["feature"]] = best["after"]
            changes.append(
                {
                    "feature": best["feature"],
                    "before": best["before"],
                    "after": best["after"],
                    "before_score": best["before_score"],
                    "after_score": best["after_score"],
                    "score_delta": best["score_delta"],
                    "reason": "replace with healthy-neighbor consensus candidate",
                }
            )

            if best["after_pred"] == 0:
                break

    after_pred_arr, after_prob_arr = predict_with_threshold(model, current)
    after_pred = int(after_pred_arr[0])
    after_score = float(after_prob_arr[0])

    is_dry_run = max_changes == 0

    if before_pred == 0:
        status = "already_normal"
        success = True
    elif after_pred == 0:
        status = "repaired"
        success = True
    elif is_dry_run:
        status = "anomaly_detected"
        success = False
    elif after_score < before_score:
        status = "partially_repaired"
        success = False
    else:
        status = "not_repaired"
        success = False

    summary = {
        "status": status,
        "success": success,
        "before_pred": before_pred,
        "after_pred": after_pred,
        "before_score": before_score,
        "after_score": after_score,
        "score_reduction": before_score - after_score,
        "applied_changes": len(changes),
        "dry_run": is_dry_run,
    }

    return RepairBundle(repaired_sample=current, summary=summary, changes=changes)
