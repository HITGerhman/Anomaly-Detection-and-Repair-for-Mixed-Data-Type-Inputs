# Windows Package Smoke Verification

- Package dir: D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs\outputs\windows-installer\installed
- Main exe: D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs\outputs\windows-installer\installed\AnomalyDetectionRepair.exe
- Engine exe: D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs\outputs\windows-installer\installed\python_engine\anomaly_engine.exe
- Sample CSV: D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs\outputs\windows-installer\installed\samples\m1_stroke_corrupted.csv
- Health status: ok
- Scan issue count: 17
- Repaired issue id: bmi::missing_values
- Repaired CSV: D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs\outputs\windows-installer\verification\installed\repair-output\m1_stroke_corrupted.repaired.csv
- Rollback manifest: D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs\outputs\windows-installer\verification\installed\repair-output\.rollback\rb-1778888732119-10a330b2.json
- GUI launch status: started
- GUI process id: 25844

Notes:
- The automated smoke test validates the packaged engine health, CSV scan, repair execution, output CSV, rollback manifest, and main GUI process launch.
- File-dialog clicking and visual result inspection remain an operator check in the running desktop UI.
