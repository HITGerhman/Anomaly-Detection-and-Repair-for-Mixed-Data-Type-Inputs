# M3 Stroke Repair Evaluation

This directory contains the M3 repair evaluation based on the M1 stroke experiment data and M2 detection output.

## Inputs

- M1 directory: `data/experiments/m1_stroke`
- M2 directory: `data/experiments/m2_stroke_detection`
- Repaired data: `repaired.csv`
- Primary denominator: repairable M1 ground truth rows only

## Scoring Policy

- The primary repair success rate uses the 72 M1 rows marked `repairable=True`.
- `duplicate_record` and `cross_column_consistency` are reported as skipped/manual-review items because the current rule-based batch repair does not auto-repair them.
- Missing values, numeric outliers, and rare categories are evaluated by row index and column.
- Numeric repairs report before/after absolute error when both values are numeric.
- Extra changed cells outside repairable ground truth are counted as side effects, not successes.

## Repair Metrics

| Type | GT | Changed | Exact | Improved/Exact | Exact Rate | Improved/Exact Rate |
|---|---:|---:|---:|---:|---:|---:|
| `missing_values` | 30 | 30 | 7 | 7 | 0.233333 | 0.233333 |
| `numeric_outlier` | 24 | 24 | 0 | 24 | 0.000000 | 1.000000 |
| `rare_category` | 18 | 18 | 10 | 10 | 0.555556 | 0.555556 |
| **Overall** | 72 | 72 | 17 | 41 | 0.236111 | 0.569444 |

## Before/After Scan Summary

- Before issue count: 12
- After issue count: 4
- Resolved issue count: 8
- Total cells modified by `repair_batch`: 194
- Non-ground-truth cells modified: 122

## Skipped Manual-Review Items

- `cross_column_consistency`: 16
- `duplicate_record`: 12

## Notes

- M3 evaluates repair quality only. It does not tune detection thresholds or repair algorithms.
- The current numeric outlier detector produced false positives in M2; when those issue IDs are repaired, their resulting cell changes are recorded as side effects.
- Detailed per-row repair outcomes and side effects are listed in `repair_details.json`.
