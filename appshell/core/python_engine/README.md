# Python Engine

This engine wraps the algorithm layer behind a stable JSON protocol.

## Layer split

- Algorithm layer: `src/training_core.py` and `engine_core.py`
- Service routing: `engine_service.py`
- CLI/transport layer: `engine_main.py`

## Protocol files

- Input template: `appshell/core/python_engine/input.json`
- Output template: `appshell/core/python_engine/output.json`

## Request shape

```json
{
  "task_id": "train-001",
  "action": "train",
  "payload": {
    "csv_path": "data/raw/healthcare-dataset-stroke-data.csv",
    "target_col": "stroke",
    "task_type": "auto",
    "output_dir": "outputs/results/template_train"
  }
}
```

Repair request example:

```json
{
  "task_id": "repair-001",
  "action": "repair",
  "payload": {
    "model_dir": "outputs/results/wails_mvp",
    "sample_index": 0,
    "max_changes": 3,
    "k_neighbors": 9,
    "output_dir": "outputs/results/wails_repair"
  }
}
```

Scan request example:

```json
{
  "task_id": "scan-001",
  "action": "scan_file",
  "payload": {
    "csv_path": "data/raw/healthcare-dataset-stroke-data.csv",
    "max_bins": 120,
    "max_issues": 1000,
    "numeric_iqr_factor": 1.5,
    "robust_z_threshold": 3.5,
    "rare_ratio_threshold": 0.01,
    "rare_count_floor": 2,
    "enable_time_series_shift": true,
    "time_series_z_threshold": 4.0,
    "time_series_min_points": 24,
    "enable_cross_column_consistency": true,
    "consistency_rules": [
      {
        "name": "start_before_end",
        "type": "lte",
        "left_col": "start_day",
        "right_col": "end_day"
      }
    ],
    "enable_duplicate_record": true,
    "duplicate_subset": ["id", "start_day", "end_day"]
  }
}
```

Batch repair request example:

```json
{
  "task_id": "repair-batch-001",
  "action": "repair_batch",
  "payload": {
    "csv_path": "data/raw/healthcare-dataset-stroke-data.csv",
    "issue_ids": [
      "bmi::missing_values",
      "age::numeric_outlier"
    ],
    "plan_only": false,
    "write_output": true,
    "output_dir": "outputs/results/repair_batch",
    "enable_rollback": true,
    "rollback_dir": "outputs/results/repair_batch/.rollback",
    "repair_strategy": {
      "conflict_policy": "first_wins",
      "missing_numeric": "median",
      "missing_categorical": "mode",
      "outlier": "clip",
      "rare_category": "mode"
    },
    "column_dependencies": {
      "bmi": ["age"]
    }
  }
}
```

Rollback request example:

```json
{
  "task_id": "rollback-001",
  "action": "rollback_repair_batch",
  "payload": {
    "manifest_path": "outputs/results/repair_batch/.rollback/rb-1700000000000-abcd1234.json",
    "restore_target": "source_csv"
  }
}
```

## Response shape

```json
{
  "task_id": "train-001",
  "status": "ok",
  "result": {},
  "error": null,
  "timestamp": "2026-01-01T00:00:00+00:00",
  "duration_ms": 1234
}
```

## Error codes

- `INVALID_JSON`
- `INVALID_INPUT`
- `UNKNOWN_ACTION`
- `FILE_NOT_FOUND`
- `CSV_READ_FAILED`
- `INVALID_TARGET_COLUMN`
- `UNSUPPORTED_TARGET_TYPE`
- `MISSING_DEPENDENCY`
- `TRAINING_MODULE_IMPORT_FAILED`
- `TRAINING_FAILED`
- `REPAIR_MODULE_IMPORT_FAILED`
- `MODEL_STATE_LOAD_FAILED`
- `REPAIR_FAILED`
- `SCAN_FAILED`
- `REPAIR_BATCH_FAILED`
- `ROLLBACK_FAILED`
- `INTERNAL_ERROR`

## Logging

- Structured JSON logs are written to `stderr`.
- Optional file logging: set env `ENGINE_LOG_FILE=/path/to/engine.log`.

## Actions

- `health`: returns engine metadata.
- `train`: trains model and saves artifacts.
- `repair`: loads saved model artifacts and searches constrained minimal edits for one sample.
- `scan_file`: scans one CSV and returns coarse anomaly issue catalog + column thumbnails.
- `repair_batch`: repairs selected issues in one run with strategy config, dependency handling, real before/after comparison, and optional rollback manifest generation.
- `rollback_repair_batch`: restores CSV from rollback manifest backup.

### `scan_file` result highlights

- `scan_config`: effective scan parameters after default + override merge.
- `issues`: now includes `confidence` and `explain_features`.
- New issue types: `time_series_shift`, `cross_column_consistency`, `duplicate_record`.
- `column_thumbnails`: per-column coarse heat bins (`bins` + `heat_bins`) and merged `hot_segments`.
- `issues`: issue list sorted by `issue_score` desc, each with severity/risk and preview rows.
- `scan_summary`: aggregated overview for high-risk/medium-risk columns, issue totals, and `issue_type_counts`.

## Run

```bash
echo '{"task_id":"h1","action":"health","payload":{}}' | python appshell/core/python_engine/engine_main.py
python appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/input.json --output appshell/core/python_engine/output.json
```
