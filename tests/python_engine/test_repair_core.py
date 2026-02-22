from __future__ import annotations

import numpy as np
import pandas as pd

from src.repair_core import repair_anomaly_sample
from src.training_core import predict_with_threshold, train_model


def _make_repairable_dataset(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(20260221)
    normal_rows = rows - 48
    anomaly_rows = 48

    normal = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=normal_rows),
            "age": rng.integers(18, 55, size=normal_rows),
            "hypertension": rng.integers(0, 2, size=normal_rows),
            "avg_glucose_level": rng.normal(95, 12, size=normal_rows).clip(60, 135),
            "smoking_status": rng.choice(["never", "formerly"], size=normal_rows),
            "stroke": np.zeros(normal_rows, dtype=int),
        }
    )
    anomaly = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=anomaly_rows),
            "age": rng.integers(62, 88, size=anomaly_rows),
            "hypertension": rng.integers(1, 2, size=anomaly_rows),
            "avg_glucose_level": rng.normal(185, 16, size=anomaly_rows).clip(140, 260),
            "smoking_status": rng.choice(["smokes", "formerly"], size=anomaly_rows),
            "stroke": np.ones(anomaly_rows, dtype=int),
        }
    )
    return pd.concat([normal, anomaly], ignore_index=True)


def _pick_anomaly_sample(bundle: object) -> pd.DataFrame:
    x_test = bundle.x_test  # type: ignore[attr-defined]
    model = bundle.model  # type: ignore[attr-defined]
    pred, prob = predict_with_threshold(model, x_test)
    idx = np.where(pred == 1)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(prob))])
    return x_test.iloc[[int(idx[0])]].copy()


def test_repair_anomaly_sample_reduces_or_keeps_score() -> None:
    df = _make_repairable_dataset()
    bundle = train_model(df, "stroke")
    sample = _pick_anomaly_sample(bundle)

    result = repair_anomaly_sample(
        model=bundle.model,
        sample=sample,
        normal_data=bundle.normal_data,
        max_changes=3,
        k_neighbors=9,
    )

    summary = result.summary
    assert "before_score" in summary
    assert "after_score" in summary
    assert float(summary["after_score"]) <= float(summary["before_score"]) + 1e-12
    assert int(summary["applied_changes"]) <= 3
    assert result.repaired_sample.shape == sample.shape


def test_repair_respects_immutable_columns() -> None:
    df = _make_repairable_dataset()
    bundle = train_model(df, "stroke")
    sample = _pick_anomaly_sample(bundle)

    result = repair_anomaly_sample(
        model=bundle.model,
        sample=sample,
        normal_data=bundle.normal_data,
        max_changes=3,
        k_neighbors=9,
        immutable_columns=["age", "avg_glucose_level"],
    )

    touched = {str(item["feature"]) for item in result.changes}
    assert "age" not in touched
    assert "avg_glucose_level" not in touched
