from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gower
import numpy as np
import pandas as pd


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return round(value, 12)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, 12)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _categorical_flags(frame: pd.DataFrame) -> list[bool]:
    return [
        bool(pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype))
        for dtype in frame.dtypes
    ]


def _prepare_gower_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        series = prepared[column]
        if pd.api.types.is_categorical_dtype(series):
            categories = series.cat.add_categories(["__MISSING__"])
            prepared[column] = categories.fillna("__MISSING__").cat.codes
            continue
        if pd.api.types.is_object_dtype(series):
            prepared[column] = series.fillna("__MISSING__").astype(str)
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) > 0:
            fill_value = float(numeric.median())
        else:
            fill_value = 0.0
        prepared[column] = numeric.fillna(fill_value)
    return prepared


def _effective_feature_columns(
    normal_data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns:
        resolved = [str(column) for column in feature_columns if str(column) in normal_data.columns]
    else:
        resolved = [str(column) for column in normal_data.columns if str(column) != target_column]
    if not resolved:
        resolved = [str(column) for column in normal_data.columns if str(column) != target_column]
    return resolved


def _normalize_weights(feature_columns: list[str], feature_weights: Any) -> list[float] | None:
    if feature_weights is None:
        return None

    if isinstance(feature_weights, dict):
        weights: list[float] = []
        for column in feature_columns:
            raw = feature_weights.get(column, 0.0)
            try:
                weights.append(max(0.0, float(raw)))
            except Exception:
                weights.append(0.0)
        if any(weight > 0.0 for weight in weights):
            return weights
        return None

    if isinstance(feature_weights, (list, tuple, np.ndarray)):
        values = list(feature_weights)
        if len(values) != len(feature_columns):
            return None
        weights = []
        for raw in values:
            try:
                weights.append(max(0.0, float(raw)))
            except Exception:
                weights.append(0.0)
        if any(weight > 0.0 for weight in weights):
            return weights
    return None


@dataclass
class NeighborSuggestion:
    replacement_value: Any
    confidence: float
    neighbor_count: int
    neighbor_rows_preview: list[dict[str, Any]]
    distance_summary: dict[str, float]
    nearest_indices: list[int]


def suggest_replacement_from_neighbors(
    normal_data: pd.DataFrame,
    anomaly_sample: pd.DataFrame,
    target_column: str,
    *,
    feature_columns: list[str] | None = None,
    k_neighbors: int = 5,
    feature_weights: Any = None,
    preview_limit: int = 5,
) -> NeighborSuggestion:
    if target_column not in normal_data.columns:
        raise ValueError(f"target column not found in normal_data: {target_column}")
    if target_column not in anomaly_sample.columns:
        raise ValueError(f"target column not found in anomaly_sample: {target_column}")
    if anomaly_sample.empty:
        raise ValueError("anomaly_sample must contain at least one row")
    if normal_data.empty:
        raise ValueError("normal_data must contain at least one row")

    resolved_feature_columns = _effective_feature_columns(normal_data, target_column, feature_columns)
    candidate_frame = normal_data.loc[:, resolved_feature_columns].copy()
    query_frame = anomaly_sample.iloc[[0]].loc[:, resolved_feature_columns].copy()

    if candidate_frame.empty:
        raise ValueError("no feature columns available for gower retrieval")

    cat_features = _categorical_flags(candidate_frame)
    prepared_candidates = _prepare_gower_matrix(candidate_frame)
    prepared_query = _prepare_gower_matrix(query_frame)
    weights = _normalize_weights(resolved_feature_columns, feature_weights)
    weight_array = np.asarray(weights, dtype=float) if weights is not None else None

    distances = gower.gower_matrix(
        prepared_query,
        prepared_candidates,
        weight=weight_array,
        cat_features=np.asarray(cat_features, dtype=bool),
    )[0]
    ordered = np.argsort(distances)
    k = max(1, min(int(k_neighbors), len(ordered)))
    nearest_indices = ordered[:k].tolist()

    neighbors = normal_data.iloc[nearest_indices].copy()
    target_values = neighbors[target_column].dropna()
    if target_values.empty:
        raise ValueError(f"nearest neighbors do not contain valid values for {target_column}")

    if pd.api.types.is_numeric_dtype(target_values):
        replacement_value = float(target_values.median())
        if float(replacement_value).is_integer():
            replacement_value = int(replacement_value)
        else:
            replacement_value = round(replacement_value, 6)
    else:
        modes = target_values.mode(dropna=True)
        replacement_value = modes.iloc[0] if not modes.empty else target_values.iloc[0]

    nearest_distances = np.asarray(distances[nearest_indices], dtype=float)
    mean_distance = float(np.mean(nearest_distances)) if nearest_distances.size else 1.0
    confidence = max(0.0, min(1.0, 1.0 - mean_distance))

    preview = [
        _to_builtin(record)
        for record in neighbors.head(max(1, int(preview_limit))).to_dict(orient="records")
    ]
    return NeighborSuggestion(
        replacement_value=_to_builtin(replacement_value),
        confidence=round(confidence, 6),
        neighbor_count=int(len(nearest_indices)),
        neighbor_rows_preview=preview,
        distance_summary={
            "min": round(float(np.min(nearest_distances)), 6),
            "max": round(float(np.max(nearest_distances)), 6),
            "mean": round(mean_distance, 6),
            "median": round(float(np.median(nearest_distances)), 6),
        },
        nearest_indices=[int(idx) for idx in nearest_indices],
    )


class AnomalyRepairer:
    def __init__(self, normal_data: pd.DataFrame, feature_weights: Any = None):
        self.normal_data = normal_data.copy()
        self.feature_weights = feature_weights

    def generate_repair_suggestion(
        self,
        anomaly_sample: pd.DataFrame,
        feature_to_fix: str,
        k: int = 5,
        *,
        preview_limit: int = 5,
        feature_columns: list[str] | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        suggestion = suggest_replacement_from_neighbors(
            self.normal_data,
            anomaly_sample,
            feature_to_fix,
            feature_columns=feature_columns,
            k_neighbors=k,
            feature_weights=self.feature_weights,
            preview_limit=preview_limit,
        )
        neighbors = self.normal_data.iloc[suggestion.nearest_indices].copy()
        return (
            {
                "Suggested Value": suggestion.replacement_value,
                "Repair Logic": f"Found {suggestion.neighbor_count} nearest healthy neighbors using Gower distance.",
                "Neighbors": neighbors,
                "Distance Summary": suggestion.distance_summary,
                "Candidate Confidence": suggestion.confidence,
            },
            neighbors,
        )
