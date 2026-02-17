# Python Engine

This engine wraps the current algorithm modules behind a stable JSON protocol.

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

## Actions

- `health`: returns engine metadata.
- `train`: trains model via `src/utils.py` and saves artifacts.

## Run

```bash
echo '{"task_id":"h1","action":"health","payload":{}}' | python3 appshell/core/python_engine/engine_main.py
pip install -r appshell/core/python_engine/requirements.txt
python3 appshell/core/python_engine/engine_main.py --input appshell/core/python_engine/sample_train_request.json
```
