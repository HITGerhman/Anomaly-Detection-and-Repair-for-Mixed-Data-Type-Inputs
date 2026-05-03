# M2 Stroke Detection Evaluation

This directory contains the M2 detection evaluation based on the M1 stroke experiment data.

## Inputs

- M1 directory: `data\experiments\m1_stroke`
- Ground truth rows: 100
- Corrupted rows: 4240
- Scored anomaly types: `missing_values`, `numeric_outlier`, `rare_category`, `duplicate_record`, `cross_column_consistency`

## Scoring Policy

- Missing values, numeric outliers, and rare categories are matched by anomaly type, row index, and column.
- Cross-column consistency is matched by anomaly type and row index.
- Duplicate records are matched by anomaly type and `source_row_id` group; marked row counts are reported separately.
- `time_series_shift` is disabled for M2 scoring because M1 did not inject that anomaly type.

## Metrics

| Type | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `missing_values` | 30 | 30 | 30 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| `numeric_outlier` | 24 | 146 | 24 | 122 | 0 | 0.164384 | 1.000000 | 0.282353 |
| `rare_category` | 18 | 18 | 18 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| `duplicate_record` | 12 | 12 | 12 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| `cross_column_consistency` | 16 | 16 | 16 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| **Overall** | 100 | 222 | 100 | 122 | 0 | 0.450450 | 1.000000 | 0.621118 |

## Notes

- M2 evaluates detection only. Repair quality is intentionally left for M3.
- False positives and false negatives are listed in `detection_matches.json`.
- Numeric outlier precision may be lower when the current detector also flags natural high-end values in the corrupted dataset. M2 records this behavior without tuning detector thresholds.
