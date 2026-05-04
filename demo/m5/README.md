# M5 Demo Requests

This directory contains fixed JSON requests for the defense demo.

Run from the repository root with PowerShell.

## 1. Scan

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input demo\m5\scan_request.json
```

Expected highlights:

- `status` is `ok`.
- `result.issue_count` is non-zero.
- `result.scan_summary.issue_type_counts` includes missing values, numeric outliers, rare categories, duplicate records, and cross-column consistency.

## 2. Repair

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input demo\m5\repair_request.json
```

Expected highlights:

- `status` is `ok`.
- `result.selected_issue_count` is `10`.
- `result.applied_issue_count` is `10`.
- `result.output_csv` points to `outputs/demo/m5/repair/corrupted.repaired.csv`.
- `result.rollback.manifest_path` points to a generated rollback manifest.

## 3. Rollback

Copy `result.rollback.manifest_path` from the repair output into `rollback_request.template.json`.

For a safe live demo, keep:

```json
"restore_target": "output_csv"
```

This restores only the demo output file and does not modify the M1 experiment CSV.

Then run the filled request:

```powershell
.\.venv-win\Scripts\python.exe appshell\core\python_engine\engine_main.py --input <filled rollback request path>
```

## Notes

- M5 does not require Node/npm or a Wails build.
- `outputs/demo/m5/` is a runtime output directory and is not intended to be committed.
- If live commands fail, use `data/experiments/m2_stroke_detection/README.md` and `data/experiments/m3_stroke_repair/README.md` as the fallback evidence.
