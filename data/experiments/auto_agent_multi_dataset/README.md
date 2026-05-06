# Auto Agent Multi-Dataset Samples

Small controlled mixed-type CSVs for the Auto Agent multi-dataset benchmark.

## Datasets

- `orders_transactions/corrupted.csv`: order and transaction records.
- `user_device_logs/corrupted.csv`: user profile and device log records.
- `../m1_stroke/corrupted.csv`: retained existing healthcare mixed-type dataset.

Each CSV includes numeric columns, categorical columns, missing values, numeric outliers, rare categories, duplicate records, and cross-column consistency issues.

Benchmark outputs should stay under `outputs/` and should not be committed.
