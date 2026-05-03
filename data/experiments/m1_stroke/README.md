# M1 Stroke Experiment Data

This directory contains the M1 experiment data generated from the stroke dataset.

## Files

- `clean.csv`: conservative clean subset used as the reference table.
- `corrupted.csv`: clean subset plus deterministic injected anomalies.
- `ground_truth.csv`: row/cell-level injection records.
- `injection_summary.json`: machine-readable generation summary.

## Generation

```powershell
.\.venv-win\Scripts\python.exe scripts\generate_m1_experiment_data.py --output-dir data\experiments\m1_stroke --seed 20260503
```

## Injection Types

{
  "cross_column_consistency": 16,
  "duplicate_record": 12,
  "missing_values": 30,
  "numeric_outlier": 24,
  "rare_category": 18
}

M1 only creates controlled data and ground truth. Detection metrics belong to M2, and repair metrics belong to M3.
