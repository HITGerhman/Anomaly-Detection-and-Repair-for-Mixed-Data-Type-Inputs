# Large-scale Labeled Validation, 2026-05-31

This note records the supplementary labeled scale experiment for the thesis
project. It bridges the earlier controlled accuracy baseline and the 10M-row
stability validation: the datasets are still synthetic and controlled, but the
scan is evaluated against ground truth at 1M and 10M scale.

## Purpose

The previous 10M stability run proved that the AppShell workflow could complete
large CSV scan, repair write, validation, and rollback-compatible output on the
tested machine. It did not include ground-truth labels. This experiment adds
known injected anomalies to larger `orders_transactions` CSV files and measures
whether those injected anomalies are detected at scale.

The safe claim is narrow:

> On generated `orders_transactions` data, the current scanner recalled all 100
> injected anomalies at both 1M and 10M scale. Numeric outlier precision dropped
> sharply because the default threshold also flagged many naturally high
> `total_amount` values.

## Dataset and Injection

Both labeled scale datasets use the same anomaly composition as the controlled
baseline:

| Type | Count |
|---|---:|
| `missing_values` | 30 |
| `numeric_outlier` | 24 |
| `rare_category` | 18 |
| `duplicate_record` | 12 |
| `cross_column_consistency` | 16 |
| **Total** | **100** |

| Dataset | Base rows | Corrupted rows | Ground truth | Repair evaluation |
|---|---:|---:|---:|---|
| `orders_transactions_1m_labeled` | 1,000,000 | 1,000,012 | 100 | Full repair metrics |
| `orders_transactions_10m_labeled` | 10,000,000 | 10,000,012 | 100 | Detection-only |

The 12 extra corrupted rows come from appended duplicate records. Large CSV
files and full ground-truth CSVs are stored under:

```text
outputs/large_labeled_validation_20260531/
```

Paper-facing summaries are stored under:

```text
artifacts/experiments/large_labeled_validation/
```

## Detection Results

| Dataset | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `orders_transactions_1m_labeled` | 100 | 7,680 | 100 | 7,580 | 0 | 0.013021 | 1.000000 | 0.025707 |
| `orders_transactions_10m_labeled` | 100 | 75,864 | 100 | 75,764 | 0 | 0.001318 | 1.000000 | 0.002633 |

Issue-level pattern:

- `missing_values`, `rare_category`, `duplicate_record`, and
  `cross_column_consistency` reached precision, recall, and F1 of `1.000000` at
  both scales.
- `numeric_outlier` reached recall `1.000000` at both scales, but precision was
  low: `0.003156` at 1M and `0.000317` at 10M.
- The false positives are dominated by naturally high `total_amount` values in
  the deterministic generated data. This is useful evidence for the thesis
  limitation: numeric thresholds need domain tuning.

## Repair Results for 1M

Only the 1M labeled run evaluates repair accuracy. The 10M run is intentionally
scan-only to avoid turning the supplementary experiment into a high-risk,
high-I/O repair accuracy claim.

| Type | Repairable GT | Changed | Exact | Improved/Exact | Exact Rate | Improved/Exact Rate | Non-GT Modified |
|---|---:|---:|---:|---:|---:|---:|---:|
| `missing_values` | 30 | 30 | 5 | 5 | 0.166667 | 0.166667 | 0 |
| `numeric_outlier` | 24 | 24 | 0 | 24 | 0.000000 | 1.000000 | 7,580 |
| `rare_category` | 18 | 18 | 2 | 2 | 0.111111 | 0.111111 | 0 |
| **Overall** | **72** | **72** | **7** | **31** | **0.097222** | **0.430556** | **7,580** |

The 1M repair run generated rollback metadata:

```text
outputs/large_labeled_validation_20260531/orders_transactions_1m_labeled/.rollback/rb-1780222699871-b4f91f18.json
```

The repair result used streaming output and modified 7,652 cells. Of those,
7,580 were outside repairable ground truth and came from numeric outlier false
positives. This should be presented as a side-effect risk, not as a repair
success.

## Runtime and Memory

| Dataset | Stage | Rows | Runtime | Peak working set | Peak private memory | Output size |
|---|---|---:|---:|---:|---:|---:|
| 1M | Generate labeled CSV | 1,000,000 | 8.289 s | 80.805 MB | 538.672 MB | 112.055 MB |
| 1M | Full scan + GT matching | 1,000,012 | 11.481 s | 494.695 MB | 966.125 MB | N/A |
| 1M | Repair + GT evaluation | 1,000,012 | 17.074 s | 523.504 MB | 1007.016 MB | 113.007 MB |
| 10M | Generate labeled CSV | 10,000,000 | 213.854 s | 203.512 MB | 661.512 MB | 1128.159 MB |
| 10M | Full scan + GT matching | 10,000,012 | 147.100 s | 4931.457 MB | 5598.250 MB | N/A |

Peak memory is current-process working set/private memory sampled during the
stage. It should not be interpreted as a formal algorithmic memory-complexity
proof.

## Thesis-ready Wording

The original thesis separates controlled accuracy evaluation from large-scale
stability validation. To connect those two parts, I added a supplementary
labeled large-scale experiment on generated `orders_transactions` data. The
experiment injected the same 100 known anomalies used in the controlled
baseline proportions and evaluated whether the scanner could still identify
them at 1M and 10M scale. The system recalled all injected anomalies in both
runs, while numeric outlier precision dropped substantially because the default
threshold also flagged many naturally high generated transaction amounts. This
supports the system's large-scale recall evidence and also highlights the need
for domain-specific numeric threshold tuning.

For repair, the 1M labeled run completed controlled `repair_batch` execution,
generated a repaired CSV and rollback manifest, and evaluated repairable
ground-truth cells. The 10M labeled run was kept scan-only; repair accuracy at
that scale is left as future work.

## Reproduction

```powershell
.\.venv-win\Scripts\python.exe .\scripts\run_large_labeled_validation.py --run both --output-dir .\artifacts\experiments\large_labeled_validation --work-dir .\outputs\large_labeled_validation_20260531
```

Targeted smoke/regression test:

```powershell
.\.venv-win\Scripts\python.exe -m pytest tests\python_engine\test_cross_dataset_validation.py tests\python_engine\test_large_labeled_validation.py -q
```
