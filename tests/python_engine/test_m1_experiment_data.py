from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = PROJECT_ROOT / "scripts" / "generate_m1_experiment_data.py"

EXPECTED_FILES = {
    "clean.csv",
    "corrupted.csv",
    "ground_truth.csv",
    "injection_summary.json",
    "README.md",
}

GROUND_TRUTH_COLUMNS = [
    "injection_id",
    "anomaly_type",
    "expected_issue_type",
    "row_id",
    "source_row_id",
    "row_index",
    "column",
    "original_value",
    "corrupted_value",
    "repairable",
    "notes",
]


def _run_generator(output_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--output-dir",
            str(output_dir),
            "--seed",
            "20260503",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"milestone": "M1"' in proc.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m1_generator_creates_expected_files_and_truth(tmp_path: Path) -> None:
    output_dir = tmp_path / "m1"
    _run_generator(output_dir)

    assert EXPECTED_FILES.issubset({item.name for item in output_dir.iterdir()})

    clean = pd.read_csv(output_dir / "clean.csv")
    corrupted = pd.read_csv(output_dir / "corrupted.csv")
    ground_truth = pd.read_csv(output_dir / "ground_truth.csv")
    summary = json.loads((output_dir / "injection_summary.json").read_text(encoding="utf-8"))

    assert list(ground_truth.columns) == GROUND_TRUTH_COLUMNS
    assert not clean.empty
    assert not ground_truth.empty
    assert clean["source_row_id"].is_unique
    assert len(corrupted) > len(clean)
    assert int(summary["clean_rows"]) == len(clean)
    assert int(summary["corrupted_rows"]) == len(corrupted)
    assert int(summary["ground_truth_rows"]) == len(ground_truth)

    actual_counts = ground_truth["anomaly_type"].value_counts().sort_index().to_dict()
    assert summary["injection_counts_by_type"] == {str(k): int(v) for k, v in actual_counts.items()}
    assert set(actual_counts) == {
        "cross_column_consistency",
        "duplicate_record",
        "missing_values",
        "numeric_outlier",
        "rare_category",
    }
    assert (clean["record_start_day"] <= clean["record_end_day"]).all()
    assert (corrupted["source_row_id"].duplicated()).any()


def test_m1_generator_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_generator(first)
    _run_generator(second)

    for filename in EXPECTED_FILES:
        assert _sha256(first / filename) == _sha256(second / filename)
