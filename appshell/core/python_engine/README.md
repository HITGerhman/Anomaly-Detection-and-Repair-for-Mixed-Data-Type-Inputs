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
    "output_dir": "outputs/results/template_train"
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
- `INTERNAL_ERROR`

## Logging

- Structured JSON logs are written to `stderr`.
- Optional file logging: set env `ENGINE_LOG_FILE=/path/to/engine.log`.

## Actions

- `health`: returns engine metadata.
- `train`: trains model and saves artifacts.

## Run

```bash
echo '{"task_id":"h1","action":"health","payload":{}}' | python appshell/core/python_engine/engine_main.py
python appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/input.json --output appshell/core/python_engine/output.json
```
