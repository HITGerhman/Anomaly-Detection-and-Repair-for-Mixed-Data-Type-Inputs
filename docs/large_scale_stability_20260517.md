# Large-scale Stability Validation, 2026-05-17

This note is the paper-facing reference for the 500k / 1M / 10M mixed-type CSV
stability run. It is intentionally conservative: the results show that the
current local system can complete large CSV scan, preview, controlled repair,
post-validation, and rollback-compatible output on the tested machine. They do
not prove unlimited scalability or production readiness.

## 1. Experiment Purpose

The experiment validates whether the AppShell pipeline remains usable when the
input table grows from medium scale to ten million rows:

- `scan_file` can read and profile a large mixed-type CSV.
- plan-only previews can avoid unnecessary output writes.
- rule repair, Gower-KNN repair, and iterative MissForest repair can run with
large-file protections.
- real auto execution can write repaired CSV output and produce rollback
metadata.
- post-execute Validation Gate can distinguish full validation from
affected-column incremental validation.

The test source for the 10M run was:

```text
outputs/ten_million_probe/scale_inputs/orders_transactions_10000000_corrupted_stream.csv
```

The consolidated result source is:

```text
outputs/stability_reprobe_20260517/reprobe_summary.json
```

## 2. End-to-end Auto Runs

These runs use the auto agent path: scan, preview candidate repairs, execute the
selected repair path, rescan the repaired output, apply Validation Gate, and
write rollback metadata.

| Run | Rows checked after repair | Output size | Total time | Repair time | Validation time | Write strategy | Post validation | Verdict |
|---|---:|---:|---:|---:|---:|---|---|---|
| `medium_auto_500k` | 500,024 | 59,253,174 bytes | 42.761 s | 15.728 s | 7.561 s | `pandas_full` | scoped precheck + full scan | `warn` accepted |
| `auto_1m_streaming` | 1,000,024 | 118,500,360 bytes | 62.255 s | 24.705 s | 13.652 s | `streaming` | scoped precheck + full scan | `warn` accepted |
| `auto_10m_streaming_incremental_retry` | 10,000,024 | 1,192,963,894 bytes | 584.805 s | 371.740 s | 27.824 s | `streaming` | affected-column incremental estimate | `warn` accepted |

Important details:

- The 500k file was below the 64 MiB streaming threshold, so `pandas_full` was
  expected.
- The 1M and 10M runs both used streaming writes.
- The 10M run used a hybrid execution path: Gower repaired 45 cells and
  MissForest repaired 15 cells.
- The 10M output kept rollback protection:
  `outputs/stability_reprobe_20260517/auto_10m_streaming_incremental_retry/.rollback/rb-1778980239452-hybrid.json`.
- The 10M rollback source backup existed and was about 1.18 GB.

## 3. Engine Probe Results

The engine probes isolate the Python engine operations from the full auto
session. They are useful in the thesis for explaining where the speedup came
from.

| Dataset | Rows | `scan_file` | `repair_batch` plan-only | Gower single issue | MissForest single issue |
|---|---:|---:|---:|---:|---:|
| 1M | 1,000,024 | 10.182 s | 2.311 s | 4.461 s | 3.880 s |
| 10M | 10,000,024 | 129.359 s | 27.674 s | 31.180 s | 29.720 s |

10M probe evidence:

- `scan_file` detected 18 issues across 13 columns.
- schema cache identified 3 low-cardinality category columns and 4 ID-like
  columns.
- `repair_batch plan_only` used `comparison_mode=lightweight`,
  `comparison_exact=false`, `post_scan_performed=false`, and
  `precomputed_issues_used=true`.
- Gower used `candidate_prefilter_policy=auto_bucket` with
  `prefilter_columns=["product_category","payment_method"]`, reducing the
  candidate pool to about 833k rows before sampling 512 candidates.
- MissForest reduced encoded feature count from the old high-cardinality path
  to 21 features, used `compact_working_frame=true`, and limited the working
  frame to 5,012 rows.

## 4. Before/After Optimization Comparison

| 10M operation | Earlier baseline | 2026-05-17 result | Speedup |
|---|---:|---:|---:|
| `repair_batch plan_only` | 146.192 s | 27.674 s | 5.28x |
| Gower single issue | 93.140 s | 31.180 s | 2.99x |
| MissForest single issue | 189.083 s | 29.720 s | 6.36x |

Interpretation for the paper:

- The biggest repair-preview improvement came from reusing precomputed scan
  issues and using lightweight comparison for plan-only mode.
- Gower improved because high-cardinality and ID-like columns were excluded and
  candidate rows were bucket-prefiltered before sampling.
- MissForest improved because the iterative working frame became compact and
  encoded features were capped.

## 5. Validation and Rollback Boundary

The current safety model is validation-first and rollback-first:

1. The planner only compares plan-only previews.
2. The runtime layer performs the actual CSV write.
3. The repaired output is rescanned.
4. Validation Gate accepts, warns, rejects, or recommends rollback.
5. Rejected written outputs are copied to a rejected snapshot and then rolled
   back through the rollback manifest.

For small and medium outputs, the runtime can do both an affected-column scoped
precheck and a full post scan. For large outputs above 512 MiB, the runtime may
use affected-column incremental validation to avoid a second full-table scan.

The important thesis wording is:

> Affected-column validation is not presented as a full post-repair scan. It is
> marked with `post_scan_incremental_estimate`, capped at a warning verdict when
> otherwise safe, and recorded in the Validation Gate risk flags.

## 6. New Fine-grained Affected-column Reject Rule

After this update, the runtime records per-column issue counts in scan
summaries and Validation Gate compares each repaired column independently.

Rule:

```text
For each column in post_scan.affected_columns:
    if post_scan.column_issue_counts[column] > baseline_scan.column_issue_counts[column]:
        reject the output
        set risk flag affected_column_issue_count_increased
        trigger rollback for written auto-session outputs
```

This strengthens the 10M incremental path. A large repaired output is no longer
accepted merely because the total scoped issue count looks acceptable. If a
touched column becomes worse than its baseline count, the result is rejected and
rolled back.

## 7. Thesis-ready Claims

Safe claims:

- The system completed a real 10M-row auto repair run on the tested Windows
  development machine.
- The 10M run wrote a 1.19 GB repaired CSV and preserved rollback metadata.
- The large-output Validation Gate explicitly marked affected-column
  incremental validation instead of presenting it as a full scan.
- Gower and MissForest previews became substantially faster after feature
  policy, prefiltering, schema cache, and compact working-frame changes.
- The pipeline remains conservative: warnings and rollback metadata are part of
  the measured result, not hidden implementation details.

Claims to avoid:

- Do not claim production readiness.
- Do not claim the system can process arbitrary billion-row data.
- Do not claim incremental validation is equivalent to a full post scan.
- Do not claim repair recovers true original values in all cases.
- Do not claim automatic repair is safe for duplicate records or all
  cross-column consistency issues.

## 8. Suggested Thesis Paragraph

In the large-scale stability experiment, the system was evaluated on
orders-transaction CSV files up to 10,000,024 rows. The full auto session on the
10M-row file completed in 584.805 seconds and produced a 1.19 GB repaired CSV.
The run used streaming writes for the large output and preserved rollback
metadata. Because the repaired output exceeded the 512 MiB post-validation
threshold, the system used affected-column incremental validation rather than a
second full-table post scan. This condition was explicitly marked by the
`post_scan_incremental_estimate` risk flag and accepted only as a warning-level
result. To reduce the risk of local side effects, the Validation Gate also
compares affected-column issue counts against the baseline scan and rejects the
output if any repaired column has more issue items after repair than before.

## 9. Reproduction Commands

```powershell
.\.venv-win\Scripts\python.exe .\outputs\stability_reprobe_20260517\run_engine_probes.py --label 1m --csv .\outputs\scale_probe_work\scale_inputs\orders_transactions_1000000_corrupted.csv --steps scan repair_batch gower missforest --force-scan
.\.venv-win\Scripts\python.exe .\outputs\stability_reprobe_20260517\run_engine_probes.py --label 10m --csv .\outputs\ten_million_probe\scale_inputs\orders_transactions_10000000_corrupted_stream.csv --steps scan repair_batch gower missforest --force-scan
.\.venv-win\Scripts\python.exe .\appshell\core\python_engine\auto_agent_cli.py --csv .\outputs\scale_probe_work\scale_inputs\orders_transactions_500000_corrupted.csv --output-dir .\outputs\stability_reprobe_20260517\medium_auto_500k --timeout-seconds 1800
.\.venv-win\Scripts\python.exe .\appshell\core\python_engine\auto_agent_cli.py --csv .\outputs\scale_probe_work\scale_inputs\orders_transactions_1000000_corrupted.csv --output-dir .\outputs\stability_reprobe_20260517\auto_1m_streaming --timeout-seconds 3600
.\.venv-win\Scripts\python.exe .\appshell\core\python_engine\auto_agent_cli.py --csv .\outputs\ten_million_probe\scale_inputs\orders_transactions_10000000_corrupted_stream.csv --output-dir .\outputs\stability_reprobe_20260517\auto_10m_streaming_incremental_retry --timeout-seconds 3600
```
