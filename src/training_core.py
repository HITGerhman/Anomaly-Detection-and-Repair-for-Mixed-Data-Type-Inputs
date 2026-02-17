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


def _build_classifier(random_state: int) -> lgb.LGBMClassifier:
    base_params: dict[str, Any] = {
        "random_state": random_state,
        "verbose": -1,
        "n_jobs": 1,
    }
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


def _compute_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_names: list[str],
    feature_importance: np.ndarray,
) -> dict[str, Any]:
    is_binary = len(np.unique(y_test)) == 2

    metrics: dict[str, Any] = {
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "auc": float(roc_auc_score(y_test, y_prob)) if is_binary else 0.0,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
            if is_binary
            else precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
            if is_binary
            else recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "y_test": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importance": dict(zip(feature_names, feature_importance)),
    }

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

    model = _build_classifier(random_state=random_state)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = _compute_metrics(
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_names=feature_names,
        feature_importance=model.feature_importances_,
    )

    normal_data = x_train[y_train == 0].copy()
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
