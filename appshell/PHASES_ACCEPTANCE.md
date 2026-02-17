# Phase Plan and Acceptance Criteria

## Phase 0: Baseline and contract

Goal:
- Keep old Streamlit path available.
- Add a new app shell path with a stable JSON protocol.

Tasks:
- Create `appshell/` structure.
- Define request/response schema (`task_id`, `action`, `payload`, `status`, `error`, `result`).
- Add health check action.

Acceptance:
- `health` request returns `status=ok` and engine metadata.
- Existing `app.py` still runs without code changes.

## Phase 1: Python engine packaging boundary

Goal:
- Turn algorithm code into callable engine actions.

Tasks:
- Add `train` action that calls current `src/utils.py`.
- Return serialized metric summary and artifact paths.
- Add deterministic error code mapping.

Acceptance:
- Given a CSV path and target column, engine completes with `status=ok`.
- Invalid input returns `status=error` with non-empty `error.code`.
- Output artifacts exist on disk (`model_lgb.pkl`, `test_data.pkl`, `normal_data.pkl`).

## Phase 2: Go orchestration layer

Goal:
- Move task lifecycle control from UI into Go.

Tasks:
- Implement process runner with timeout and stderr capture.
- Implement task service with `Start/Get/Cancel/List`.
- Add demo command for polling task status.

Acceptance:
- Running demo command reaches terminal state (`succeeded`/`failed`) and prints structured response.
- Timeout and cancellation produce clear task status and reason.

## Phase 3: Wails frontend MVP

Goal:
- Build the user flow `configure -> run -> inspect result`.

Tasks:
- Create UI form for csv path, target column, output dir.
- Call Go method with the same request schema.
- Render status and result JSON.

Acceptance:
- User can launch a train task and see final result from UI.
- Runtime errors are visible and not silent.

## Phase 4: Windows packaging

Goal:
- Build a distributable installer.

Tasks:
- Package Python engine executable (for example via PyInstaller).
- Bundle engine binary in Wails resources.
- Build installer with NSIS/Inno Setup.

Acceptance:
- App runs on a clean Windows machine without global Python.
- Installer creates desktop shortcut and uninstall entry.
