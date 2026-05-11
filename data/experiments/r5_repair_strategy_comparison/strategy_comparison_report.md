# R5 Repair Strategy Comparison

This report compares rule-only, gower-only, and hybrid repair strategies on the same M1 corrupted CSV.

## Inputs

- Clean CSV: `data/experiments/m1_stroke/clean.csv`
- Corrupted CSV: `data/experiments/m1_stroke/corrupted.csv`
- Ground truth CSV: `data/experiments/m1_stroke/ground_truth.csv`
- Repairable issue IDs selected: 10

## Scoring Policy

- The primary denominator is M1 `ground_truth.csv` rows where `repairable=True` and the type is one of `missing_values, numeric_outlier, rare_category`.
- `exact_restored_count` requires the repaired value to match `original_value`.
- `improved_or_exact_count` also counts numeric repairs that reduce absolute error versus the corrupted value.
- Changed cells outside repairable ground-truth cells are counted as side effects.
- Hybrid is a deterministic Auto Agent approximation; it does not run a full Go CLI session.

## Metrics

| Strategy | Status | Before | After | Resolved | Modified Cells | Exact | Exact Rate | Improved/Exact | Improved/Exact Rate | Non-GT Modified | Skipped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rule-only | ok | 12 | 4 | 8 | 194 | 17 | 0.236111 | 41 | 0.569444 | 122 | 0 |
| gower-only | ok | 12 | 3 | 9 | 194 | 14 | 0.194444 | 38 | 0.527778 | 122 | 0 |
| hybrid | ok | 12 | 3 | 9 | 194 | 14 | 0.194444 | 38 | 0.527778 | 122 | 0 |

## Notes

- `rule-only`: rule-only uses engine.repair_batch with the shared R5 scan config.
- `gower-only`: gower-only uses engine.repair_with_gower with k=5 and max_candidates=512.
- `hybrid`: hybrid is a deterministic Auto Agent approximation, not a full CLI session.
- `hybrid`: issue source selection mirrors mock_planner.go: resolved count, confidence, rows touched, then rule.
