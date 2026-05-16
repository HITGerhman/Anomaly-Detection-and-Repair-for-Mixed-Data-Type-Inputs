# Cross-Dataset Validation Experiments

This document describes the supplementary cross-dataset validation pipeline for
the thesis project **Anomaly Detection and Repair for Mixed Data Type Inputs**.
The original stroke M1/M2/M3 experiments remain the main controlled experiment.
This pipeline adds reproducible validation across multiple mixed-type CSV
scenarios without changing the Python Engine core algorithms.

## Purpose

The earlier thesis evidence was concentrated on one stroke dataset. The
cross-dataset pipeline strengthens the experimental story by checking whether
the same scan, controlled repair, manual-review boundary, and side-effect
metrics can be reproduced on several mixed tabular data domains.

This is still a validation experiment, not a production certification. The
results do not prove that the system works for every real business dataset, and
they do not prove that all anomalies can be safely repaired automatically.

## Datasets

The pipeline writes all paper-level outputs to:

```text
artifacts/experiments/cross_dataset/
```

It currently supports three datasets.

| Dataset | Source | Rows used by default | Notes |
|---|---|---:|---|
| `stroke` | Standardized copy of `data/experiments/m1_stroke/` | Existing M1 size | M1/M2/M3 artifacts are not modified. |
| `orders_transactions` | Deterministic synthetic dataset | 5000 clean rows | Existing 30-row Auto Agent sample is retained but not used as GT input. |
| `user_device_logs` | Deterministic synthetic dataset | 5000 clean rows | Existing 30-row Auto Agent sample is retained but not used as GT input. |

## Field Structure

`stroke` reuses the M1 columns and standardized experiment helpers, including
`row_id`, `source_row_id`, `record_start_day`, and `record_end_day`.

`orders_transactions` contains mixed identifier, category, numeric, and time
fields:

```text
row_id, source_row_id, order_id, user_id, product_category, payment_method,
order_status, quantity, unit_price, discount, total_amount, order_time, pay_time
```

The main consistency rule is:

```text
paid_pay_time_not_before_order_time: order_time <= pay_time
```

`total_amount` is generated from `quantity * unit_price * (1 - discount)` and is
available as a derived numeric field for domain realism.

`user_device_logs` contains mixed identifier, category, numeric, boolean-like,
and time fields:

```text
row_id, source_row_id, log_id, user_id, device_type, os, app_version,
event_type, session_duration, bytes_sent, bytes_received, login_time,
logout_time, is_success
```

The main consistency rule is:

```text
logout_not_before_login: login_time <= logout_time
```

## Anomaly Injection

Each dataset receives the same five issue types by default:

| Issue type | Default count |
|---|---:|
| `missing_values` | 30 |
| `numeric_outlier` | 24 |
| `rare_category` | 18 |
| `duplicate_record` | 12 |
| `cross_column_consistency` | 16 |
| Total | 100 |

For larger generated datasets the total injection count can be increased with
`--injections`; the script keeps approximately the same proportions.

All random choices are deterministic. The seed is written to each dataset's
`injection_summary.json`, and the ground-truth rows include `created_by_seed`.

## Ground Truth

Every dataset writes:

```text
clean.csv
corrupted.csv
ground_truth.csv
injection_summary.json
```

The ground-truth schema is:

```text
anomaly_id,dataset,expected_issue_type,row_index,column_name,original_value,
corrupted_value,repairable,source_row_id,duplicate_group,
consistency_rule_name,created_by_seed,notes
```

Repairability is intentionally conservative:

| Issue type | Repairability |
|---|---|
| `missing_values` | repairable |
| `numeric_outlier` | repairable |
| `rare_category` | repairable |
| `duplicate_record` | review-only |
| `cross_column_consistency` | review-only |

## Detection Metrics

The script reuses the existing Python Engine scan path. For normal runs it uses
the same detection implementation behind `scan_file` and `_detect_issues_for_frame`.
`time_series_shift` is disabled because it is outside the current thesis scope.

Per dataset and per issue type, the pipeline computes:

```text
GT, Pred, TP, FP, FN, Precision, Recall, F1
```

Matching rules:

| Issue type | Matching key |
|---|---|
| `missing_values` | issue type + row index + column |
| `numeric_outlier` | issue type + row index + column |
| `rare_category` | issue type + row index + column |
| `duplicate_record` | issue type + duplicate/source row identity |
| `cross_column_consistency` | issue type + row index + rule |

Outputs:

```text
artifacts/experiments/cross_dataset/<dataset>/detection_metrics.csv
artifacts/experiments/cross_dataset/summary_detection_metrics.csv
```

## Repair Metrics

The repair stage calls the existing `repair_batch` behavior through the Python
Engine action. It selects only:

```text
missing_values, numeric_outlier, rare_category
```

`duplicate_record` and `cross_column_consistency` are counted as skipped
review-only issues. They are not counted as repair failures.

Per dataset and repairable issue type, the pipeline computes:

```text
repairable_gt, changed, exact, improved_or_exact, exact_rate,
improved_or_exact_rate, non_gt_modified, skipped_non_repairable_count
```

Exact restoration means that the repaired value equals the original clean value
recorded in ground truth. For `numeric_outlier`, improved-or-exact also counts a
repair when the repaired value is numerically closer to the original value than
the corrupted value was. For `missing_values` and `rare_category`, there is no
reliable partial-improvement definition, so improved-or-exact equals exact.

Outputs:

```text
artifacts/experiments/cross_dataset/<dataset>/repair_metrics.csv
artifacts/experiments/cross_dataset/<dataset>/side_effect_summary.csv
artifacts/experiments/cross_dataset/summary_repair_metrics.csv
```

## Non-GT Modified Cells

`non_gt_modified` counts cells changed by repair that are not part of the
repairable ground truth for that issue type. It is a side-effect metric, not a
success metric. In these experiments it mostly exposes the effect of
`numeric_outlier` false positives: when the scanner predicts extra numeric
outliers, `repair_batch` may also modify those extra cells.

## Manual Review Boundary

`duplicate_record` and `cross_column_consistency` remain review-only because the
correct repair often depends on business semantics. For example, duplicate rows
may represent duplicate ingestion, legitimate repeated events, or several rows
that should be merged differently. Cross-column inconsistencies may require
choosing which field is authoritative. The system therefore detects and reports
these issues but does not silently rewrite them in the repair evaluation.

## Numeric Outlier Threshold Sensitivity

The current numeric outlier scanner is intentionally sensitive, which gives high
recall but can create false positives. The sensitivity experiment reuses the
existing Engine threshold configuration entry points:

```text
numeric_iqr_factor
robust_z_threshold
```

The default grid is:

```text
IQR factor: 1.5, 2.0, 3.0
robust z threshold: 3.5, 4.5, 5.0
```

Output:

```text
artifacts/experiments/cross_dataset/threshold_sensitivity_numeric_outlier.csv
```

The 2026-05-13 run shows the expected pattern: stricter thresholds reduce
numeric false positives on `stroke` and `orders_transactions`, but overly strict
settings can reduce recall on `stroke`.

## Scale Test

The scale test is a system throughput check, not an accuracy experiment. It uses
generated `orders_transactions`-style mixed CSVs with default row counts:

```text
5000, 10000, 50000, 100000
```

It records:

```text
dataset_name, rows, columns, scan_time_seconds, repair_time_seconds,
detected_issue_count, changed_cell_count, output_file_size_mb
```

Temporary large CSV inputs and repaired files are written under the ignored
directory:

```text
outputs/cross_dataset_validation/
```

Peak memory is not reported. The script avoids adding a dependency just for
memory measurement, and platform-specific memory sampling can be misleading in
this lightweight thesis workflow.

Output:

```text
artifacts/experiments/cross_dataset/summary_scale_metrics.csv
```

## Commands

Run the complete validation pipeline:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --all
```

Generate datasets only:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --generate
```

Run detection only:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --detect
```

Run repair only:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --repair
```

Run threshold sensitivity only:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --threshold-sensitivity
```

Run scale test only:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_cross_dataset_validation.py --scale
```

Small local test command:

```powershell
.\.venv-win\Scripts\python.exe -m pytest tests\python_engine\test_cross_dataset_validation.py -q
```

## Paper Usage

Use these files as the direct table sources:

| Paper table | CSV source |
|---|---|
| Cross-dataset detection metrics summary | `artifacts/experiments/cross_dataset/summary_detection_metrics.csv` |
| Cross-dataset repair metrics summary | `artifacts/experiments/cross_dataset/summary_repair_metrics.csv` |
| Numeric outlier threshold sensitivity summary | `artifacts/experiments/cross_dataset/threshold_sensitivity_numeric_outlier.csv` |
| Extended sample and scale testing | `artifacts/experiments/cross_dataset/summary_scale_metrics.csv` |

Recommended discussion points:

- `missing_values` and `rare_category` are stable across these controlled
  datasets, although exact restoration remains limited.
- `numeric_outlier` achieves high recall but may produce false positives,
  especially on `stroke` and `orders_transactions` with the default threshold.
- Stricter numeric thresholds can reduce false positives, but the best threshold
  is dataset-dependent.
- Improved-or-exact is a more honest way to describe conservative numeric repair
  than exact restoration alone.
- `non_gt_modified` reveals repair side effects and should not be counted as a
  repair success.
- `duplicate_record` and `cross_column_consistency` remain manual-review issues.

## Limitations

These experiments improve reproducibility and broaden evidence beyond a single
stroke dataset, but they do not prove production readiness. They also do not
prove that the system generalizes to every real domain, handles all schema
designs, or can recover all original values. More real-domain datasets, domain
rules, threshold tuning, and front-end/deployment validation remain future work.
