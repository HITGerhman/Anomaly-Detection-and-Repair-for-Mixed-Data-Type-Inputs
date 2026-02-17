from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training_core import load_system_state, save_system_state, train_model


def _make_dataset(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(20260217)
    age = rng.integers(18, 85, size=rows)
    hypertension = rng.integers(0, 2, size=rows)
    heart_disease = rng.integers(0, 2, size=rows)
    avg_glucose_level = rng.normal(120, 30, size=rows).clip(55, 260)
    bmi = rng.normal(30, 6, size=rows).clip(15, 55)
    gender = rng.choice(["Male", "Female"], size=rows)
    smoking_status = rng.choice(["never", "smokes", "formerly"], size=rows)

    raw_risk = (
        (age > 60).astype(int)
        + hypertension
        + heart_disease
        + (avg_glucose_level > 145).astype(int)
        + (smoking_status == "smokes").astype(int)
    )
    stroke = (raw_risk >= 3).astype(int)

    return pd.DataFrame(
        {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
            "stroke": stroke,
        }
    )


def _bundle_signature(bundle: object) -> dict[str, object]:
    metrics = bundle.metrics  # type: ignore[attr-defined]
    return {
        "f1": round(float(metrics["f1"]), 12),
        "auc": round(float(metrics["auc"]), 12),
        "accuracy": round(float(metrics["accuracy"]), 12),
        "precision": round(float(metrics["precision"]), 12),
        "recall": round(float(metrics["recall"]), 12),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "feature_importance": {k: int(v) for k, v in metrics["feature_importance"].items()},
    }


def test_train_model_is_deterministic_across_10_runs() -> None:
    df = _make_dataset()

    reference_bundle = train_model(df, "stroke")
    reference_signature = _bundle_signature(reference_bundle)
    reference_x_test = reference_bundle.x_test
    reference_normal_data = reference_bundle.normal_data

    for _ in range(10):
        bundle = train_model(df, "stroke")
        assert _bundle_signature(bundle) == reference_signature
        assert bundle.x_test.equals(reference_x_test)
        assert bundle.normal_data.equals(reference_normal_data)


def test_train_model_invalid_target_column_raises() -> None:
    df = _make_dataset()
    try:
        train_model(df, "not_exists")
    except ValueError as exc:
        assert "target column not found" in str(exc)
        return
    raise AssertionError("expected ValueError for invalid target column")


def test_save_and_load_system_state_round_trip(tmp_path: Path) -> None:
    df = _make_dataset()
    bundle = train_model(df, "stroke")

    save_system_state(
        model=bundle.model,
        x_test=bundle.x_test,
        normal_data=bundle.normal_data,
        feature_names=bundle.feature_names,
        save_dir=tmp_path,
    )

    assert (tmp_path / "model_lgb.pkl").exists()
    assert (tmp_path / "test_data.pkl").exists()
    assert (tmp_path / "normal_data.pkl").exists()
    assert (tmp_path / "config.pkl").exists()

    loaded_model, loaded_x_test, loaded_normal_data = load_system_state(tmp_path)
    assert loaded_x_test.equals(bundle.x_test)
    assert loaded_normal_data.equals(bundle.normal_data)

    sample = bundle.x_test.iloc[[0]]
    assert np.array_equal(loaded_model.predict(sample), bundle.model.predict(sample))

