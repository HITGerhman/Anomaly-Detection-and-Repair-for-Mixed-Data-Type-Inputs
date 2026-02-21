"""Core training logic shared by Streamlit UI and engine CLI.

This module intentionally avoids any UI concerns and exposes deterministic,
callable functions that can be used by service/CLI layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import MODEL_CONFIG, PATHS


@dataclass(frozen=True)
class TrainingBundle:
    model: Any
    x_test: pd.DataFrame
    normal_data: pd.DataFrame
    metrics: dict[str, Any]
    feature_names: list[str]


def _build_classifier(random_state: int, class_weight: Any = "balanced") -> lgb.LGBMClassifier:
    base_params: dict[str, Any] = {
        "random_state": random_state,
        "verbose": -1,
        "n_jobs": 1,
    }
    if class_weight is not None:
        base_params["class_weight"] = class_weight
    try:
        return lgb.LGBMClassifier(
            **base_params,
            deterministic=True,
            force_col_wise=True,
        )
    except TypeError:
        # Backward compatibility with LightGBM versions that do not support
        # deterministic/force_col_wise in the sklearn wrapper.
        return lgb.LGBMClassifier(**base_params)


def _encode_features(features: pd.DataFrame) -> pd.DataFrame:
    encoded = features.copy()
    categorical_cols = encoded.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col].astype(str))
        encoded[col] = encoded[col].astype("category")

    return encoded


def get_decision_threshold(model: Any, default: float = 0.5) -> float:
    raw = getattr(model, "decision_threshold", default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    return max(0.0, min(1.0, value))


def predict_with_threshold(model: Any, features: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict(features))
        return pred, pred.astype(float)

    prob_matrix = np.asarray(model.predict_proba(features))
    # Multi-class path: keep model default argmax behavior.
    if prob_matrix.ndim != 2 or prob_matrix.shape[1] != 2:
        pred = np.asarray(model.predict(features))
        return pred, pred.astype(float)

    y_prob = prob_matrix[:, 1].astype(float)
    threshold = get_decision_threshold(model)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def _select_binary_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    default_threshold: float,
    beta: float,
    min_precision: float,
) -> tuple[float, dict[str, float]]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    if y_true_arr.size == 0:
        return default_threshold, {
            "beta": beta,
            "min_precision": min_precision,
            "selected_fbeta": 0.0,
            "selected_recall": 0.0,
            "selected_precision": 0.0,
        }

    quantiles = np.linspace(0.02, 0.98, 49)
    candidates = np.unique(
        np.concatenate(
            [
                np.array([default_threshold, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float),
                np.quantile(y_prob_arr, quantiles),
            ]
        )
    )
    candidates = candidates[(candidates >= 0.0) & (candidates <= 1.0)]
    if candidates.size == 0:
        candidates = np.array([default_threshold], dtype=float)

    best_threshold = float(default_threshold)
    best_tuple = (-1.0, -1.0, -1.0, float("-inf"))
    best_meta = {
        "beta": float(beta),
        "min_precision": float(min_precision),
        "selected_fbeta": 0.0,
        "selected_recall": 0.0,
        "selected_precision": 0.0,
    }

    for threshold in candidates:
        y_pred = (y_prob_arr >= threshold).astype(int)
        precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
        if precision < min_precision:
            continue
        recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
        fbeta = float(fbeta_score(y_true_arr, y_pred, beta=beta, zero_division=0))
        score_tuple = (fbeta, recall, precision, -abs(float(threshold) - default_threshold))
        if score_tuple > best_tuple:
            best_tuple = score_tuple
            best_threshold = float(threshold)
            best_meta["selected_fbeta"] = fbeta
            best_meta["selected_recall"] = recall
            best_meta["selected_precision"] = precision

    if best_tuple[0] < 0:
        default_pred = (y_prob_arr >= default_threshold).astype(int)
        best_meta["selected_fbeta"] = float(
            fbeta_score(y_true_arr, default_pred, beta=beta, zero_division=0)
        )
        best_meta["selected_recall"] = float(recall_score(y_true_arr, default_pred, zero_division=0))
        best_meta["selected_precision"] = float(
            precision_score(y_true_arr, default_pred, zero_division=0)
        )

    return best_threshold, best_meta


def _compute_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_names: list[str],
    feature_importance: np.ndarray,
    decision_threshold: float,
    threshold_meta: dict[str, float] | None = None,
) -> dict[str, Any]:
    is_binary = len(np.unique(y_test)) == 2
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

    precision = weighted_precision
    recall = weighted_recall
    f1 = weighted_f1
    precision_anomaly = weighted_precision
    recall_anomaly = weighted_recall
    f1_anomaly = weighted_f1

    if is_binary:
        precision_anomaly = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        recall_anomaly = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1_anomaly = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        # Keep top-level keys anomaly-focused for binary tasks.
        precision = precision_anomaly
        recall = recall_anomaly
        f1 = f1_anomaly

    metrics: dict[str, Any] = {
        "f1": f1,
        "auc": float(roc_auc_score(y_test, y_prob)) if is_binary else 0.0,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1_weighted": weighted_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_anomaly": f1_anomaly,
        "precision_anomaly": precision_anomaly,
        "recall_anomaly": recall_anomaly,
        "decision_threshold": float(decision_threshold),
        "y_test": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importance": dict(zip(feature_names, feature_importance)),
    }
    if threshold_meta:
        metrics["threshold_optimization"] = threshold_meta

    if is_binary:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}

    return metrics


def train_model(df: pd.DataFrame, target_col: str) -> TrainingBundle:
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    feature_names = [name for name in df.columns if name != target_col]
    features = df.loc[:, feature_names].copy()
    target = df[target_col]

    encoded_features = _encode_features(features)
    random_state = int(MODEL_CONFIG.get("random_state", 42))
    test_size = float(MODEL_CONFIG.get("test_size", 0.2))
    stratify = target if target.nunique() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        encoded_features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    class_weight = MODEL_CONFIG.get("class_weight", "balanced")
    default_threshold = float(MODEL_CONFIG.get("decision_threshold", 0.5))
    threshold_beta = float(MODEL_CONFIG.get("threshold_beta", 2.0))
    threshold_min_precision = float(MODEL_CONFIG.get("threshold_min_precision", 0.5))
    threshold_validation_size = float(MODEL_CONFIG.get("threshold_validation_size", 0.25))
    if threshold_validation_size <= 0.0 or threshold_validation_size >= 0.5:
        threshold_validation_size = 0.25

    decision_threshold = default_threshold
    threshold_meta: dict[str, float] | None = None
    is_binary = target.nunique() == 2

    if is_binary:
        x_fit = x_train
        y_fit = y_train
        x_val = None
        y_val = None
        try:
            x_fit, x_val, y_fit, y_val = train_test_split(
                x_train,
                y_train,
                test_size=threshold_validation_size,
                random_state=random_state,
                stratify=y_train,
            )
        except ValueError:
            # Rare small-data path: fallback to train-set threshold search.
            x_val = None
            y_val = None

        threshold_model = _build_classifier(random_state=random_state, class_weight=class_weight)
        threshold_model.fit(x_fit, y_fit)

        threshold_source_x = x_val if x_val is not None else x_fit
        threshold_source_y = y_val if y_val is not None else y_fit
        threshold_prob = np.asarray(threshold_model.predict_proba(threshold_source_x))[:, 1]
        decision_threshold, threshold_meta = _select_binary_threshold(
            y_true=threshold_source_y,
            y_prob=threshold_prob,
            default_threshold=default_threshold,
            beta=threshold_beta,
            min_precision=threshold_min_precision,
        )
        threshold_meta["threshold_rows"] = float(len(threshold_source_y))

    model = _build_classifier(random_state=random_state, class_weight=class_weight)
    model.fit(x_train, y_train)
    if is_binary:
        model.decision_threshold = decision_threshold

    y_pred, y_prob = predict_with_threshold(model, x_test)

    metrics = _compute_metrics(
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_names=feature_names,
        feature_importance=model.feature_importances_,
        decision_threshold=get_decision_threshold(model),
        threshold_meta=threshold_meta,
    )

    normal_data = x_train[y_train == 0].copy()
    if normal_data.empty:
        normal_data = x_train.copy()
    return TrainingBundle(
        model=model,
        x_test=x_test,
        normal_data=normal_data,
        metrics=metrics,
        feature_names=feature_names,
    )


def process_and_train(df: pd.DataFrame, target_col: str) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    bundle = train_model(df, target_col)
    return bundle.model, bundle.x_test, bundle.normal_data, bundle.metrics, bundle.feature_names


def save_system_state(
    model: Any,
    x_test: pd.DataFrame,
    normal_data: pd.DataFrame,
    feature_names: list[str],
    save_dir: str | Path | None = None,
) -> None:
    target_dir = Path(save_dir or PATHS["data_processed"]).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, target_dir / "model_lgb.pkl")
    joblib.dump(x_test, target_dir / "test_data.pkl")
    joblib.dump(normal_data, target_dir / "normal_data.pkl")
    joblib.dump({"feature_names": feature_names}, target_dir / "config.pkl")


def load_system_state(load_dir: str | Path | None = None) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    source_dir = Path(load_dir or PATHS["data_processed"]).resolve()
    model = joblib.load(source_dir / "model_lgb.pkl")
    x_test = joblib.load(source_dir / "test_data.pkl")
    normal_data = joblib.load(source_dir / "normal_data.pkl")
    return model, x_test, normal_data
