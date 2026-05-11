# R6 Gower K Sensitivity

This report evaluates `repair_with_gower` with several `k_neighbors` values on the same M1 corrupted CSV.

## Inputs

- Clean CSV: `data/experiments/m1_stroke/clean.csv`
- Corrupted CSV: `data/experiments/m1_stroke/corrupted.csv`
- Ground truth CSV: `data/experiments/m1_stroke/ground_truth.csv`
- Repairable issue IDs selected: 10

## Metrics

| K | Status | Before | After | Resolved | Modified Cells | Exact | Exact Rate | Improved/Exact | Improved/Exact Rate | Non-GT Modified | Mean Confidence |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | ok | 12 | 3 | 9 | 194 | 11 | 0.152778 | 35 | 0.486111 | 122 | 0.864886 |
| 5 | ok | 12 | 3 | 9 | 194 | 14 | 0.194444 | 38 | 0.527778 | 122 | 0.856140 |
| 7 | ok | 12 | 3 | 9 | 194 | 11 | 0.152778 | 35 | 0.486111 | 122 | 0.849530 |
| 9 | ok | 12 | 3 | 9 | 194 | 14 | 0.194444 | 37 | 0.513889 | 122 | 0.844094 |
| 15 | ok | 12 | 3 | 9 | 194 | 15 | 0.208333 | 38 | 0.527778 | 122 | 0.831685 |

## Default K Assessment

- Continue default `K=5`: **Yes**
- Reason: K=5 is within 2 percentage points of the best improved/exact rate, does not add meaningful side effects, and resolves nearly as many issues as the best K.

## Interpretation

- If K is too small, a single neighbor can dominate the candidate value, so the repair is more sensitive to local noise and may be less stable.
- If K is too large, less similar rows enter the neighbor set, pushing numeric medians and categorical modes toward global population behavior and weakening local similarity.
- This project should not use `sqrt(n)` directly because Gower KNN here is not a standard KNN classifier. It is used to generate repair candidates for mixed-type data, where local similarity matters more than broad voting coverage. On a dataset with several thousand rows, `sqrt(n)` would make K much larger than the local neighborhood needed for repair.

## Notes

- `K=3`: repair_with_gower executed with k_neighbors=3.
- `K=5`: repair_with_gower executed with k_neighbors=5.
- `K=7`: repair_with_gower executed with k_neighbors=7.
- `K=9`: repair_with_gower executed with k_neighbors=9.
- `K=15`: repair_with_gower executed with k_neighbors=15.
