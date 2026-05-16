"""Stage 0 action metadata catalog for the Python engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ActionPayload = dict[str, Any]
ActionResult = dict[str, Any]
ActionHandler = Callable[[ActionPayload], ActionResult]
HandlerFactory = Callable[[], ActionHandler]


@dataclass(frozen=True)
class ActionSpec:
    action: str
    future_tool_id: str
    summary: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    side_effect: str
    artifact_keys: tuple[str, ...]
    algorithm_assets: tuple[str, ...]
    handler: HandlerFactory


def _load_action_health() -> ActionHandler:
    from engine_core import action_health

    return action_health


def _load_action_train() -> ActionHandler:
    from engine_core import action_train

    return action_train


def _load_action_repair() -> ActionHandler:
    from engine_core import action_repair

    return action_repair


def _load_action_scan_file() -> ActionHandler:
    from engine_core import action_scan_file

    return action_scan_file


def _load_action_repair_batch() -> ActionHandler:
    from engine_core import action_repair_batch

    return action_repair_batch


def _load_action_repair_with_gower() -> ActionHandler:
    from engine_core import action_repair_with_gower

    return action_repair_with_gower


def _load_action_repair_with_missforest() -> ActionHandler:
    from engine_core import action_repair_with_missforest

    return action_repair_with_missforest


def _load_action_rollback_repair_batch() -> ActionHandler:
    from engine_core import action_rollback_repair_batch

    return action_rollback_repair_batch


_ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        action="health",
        future_tool_id="engine.health",
        summary="Inspect runtime dependency availability and expose stable engine metadata.",
        required_fields=(),
        optional_fields=(),
        side_effect="none",
        artifact_keys=(),
        algorithm_assets=("runtime_dependency_snapshot",),
        handler=_load_action_health,
    ),
    ActionSpec(
        action="train",
        future_tool_id="engine.train_model",
        summary="Train the LightGBM model and persist validated model artifacts.",
        required_fields=("csv_path", "target_col"),
        optional_fields=("output_dir", "task_type"),
        side_effect="writes_model_artifacts",
        artifact_keys=("output_dir", "model", "test_data", "normal_data", "config"),
        algorithm_assets=("LightGBM", "src.training_core.train_model", "src.training_core.save_system_state"),
        handler=_load_action_train,
    ),
    ActionSpec(
        action="repair",
        future_tool_id="engine.repair_sample",
        summary="Repair a single anomaly sample with deterministic search over healthy neighbors.",
        required_fields=("model_dir",),
        optional_fields=(
            "sample_index",
            "dry_run",
            "max_changes",
            "k_neighbors",
            "output_dir",
            "immutable_columns",
            "numeric_bounds",
        ),
        side_effect="optional_output_write",
        artifact_keys=("repair_summary", "repair_changes", "repaired_sample_csv", "repair_report_json"),
        algorithm_assets=("LightGBM", "src.repair_core.repair_anomaly_sample"),
        handler=_load_action_repair,
    ),
    ActionSpec(
        action="scan_file",
        future_tool_id="engine.scan_table",
        summary="Run rule-based table scan and return issue catalog, thumbnails, and summaries.",
        required_fields=("csv_path",),
        optional_fields=(
            "max_bins",
            "max_issues",
            "numeric_iqr_factor",
            "robust_z_threshold",
            "rare_ratio_threshold",
            "rare_count_floor",
            "min_numeric_samples",
            "min_categorical_samples",
            "preview_limit",
            "enable_time_series_shift",
            "time_series_z_threshold",
            "time_series_min_points",
            "enable_cross_column_consistency",
            "consistency_rules",
            "enable_duplicate_record",
            "duplicate_subset",
            "auto_pair_constraints",
            "scan_config",
        ),
        side_effect="reads_input_only",
        artifact_keys=("issues", "column_thumbnails", "scan_summary", "data_profile"),
        algorithm_assets=("rule_scan", "engine_core._detect_issues_for_frame"),
        handler=_load_action_scan_file,
    ),
    ActionSpec(
        action="repair_batch",
        future_tool_id="engine.repair_batch",
        summary="Repair selected table issues with deterministic rules, comparison, and rollback metadata.",
        required_fields=("csv_path",),
        optional_fields=(
            "issue_ids",
            "output_csv",
            "output_dir",
            "write_output",
            "plan_only",
            "enable_rollback",
            "rollback_dir",
            "repair_strategy",
            "column_dependencies",
            "scan_config",
        ),
        side_effect="writes_repaired_output_and_rollback_metadata",
        artifact_keys=("output_csv", "rollback", "comparison", "applied_repairs", "skipped_issues"),
        algorithm_assets=("rule_scan", "src.repair_core.repair_anomaly_sample", "repair_strategy"),
        handler=_load_action_repair_batch,
    ),
    ActionSpec(
        action="repair_with_gower",
        future_tool_id="engine.repair_with_gower",
        summary="Repair selected table issues with Gower neighbor retrieval, comparison, and rollback metadata.",
        required_fields=("csv_path",),
        optional_fields=(
            "issue_ids",
            "output_csv",
            "output_dir",
            "write_output",
            "plan_only",
            "enable_rollback",
            "rollback_dir",
            "scan_config",
            "column_dependencies",
            "model_dir",
            "gower_strategy",
        ),
        side_effect="writes_repaired_output_and_rollback_metadata",
        artifact_keys=("output_csv", "rollback", "comparison", "applied_repairs", "neighbor_evidence"),
        algorithm_assets=("gower_distance", "src.repair_module.suggest_replacement_from_neighbors", "LightGBM"),
        handler=_load_action_repair_with_gower,
    ),
    ActionSpec(
        action="repair_with_missforest",
        future_tool_id="engine.repair_with_missforest",
        summary="Repair selected table issues with iterative MissForest random-forest imputation and rollback metadata.",
        required_fields=("csv_path",),
        optional_fields=(
            "issue_ids",
            "output_csv",
            "output_dir",
            "write_output",
            "plan_only",
            "enable_rollback",
            "rollback_dir",
            "scan_config",
            "missforest_strategy",
        ),
        side_effect="writes_repaired_output_and_rollback_metadata",
        artifact_keys=("output_csv", "rollback", "comparison", "applied_repairs", "model_evidence"),
        algorithm_assets=("MissForest", "RandomForestRegressor", "RandomForestClassifier"),
        handler=_load_action_repair_with_missforest,
    ),
    ActionSpec(
        action="rollback_repair_batch",
        future_tool_id="engine.rollback_batch",
        summary="Restore a previous repair target from rollback manifest and backup artifact.",
        required_fields=("manifest_path",),
        optional_fields=("restore_target", "target_csv"),
        side_effect="writes_restored_output",
        artifact_keys=("manifest_path", "backup_csv", "restored_to", "source_csv", "output_csv"),
        algorithm_assets=("rollback_manifest", "backup_copy"),
        handler=_load_action_rollback_repair_batch,
    ),
)

_ACTION_INDEX = {spec.action: spec for spec in _ACTION_SPECS}
if len(_ACTION_INDEX) != len(_ACTION_SPECS):
    raise ValueError("duplicate action names in action catalog")

_FUTURE_TOOL_IDS = {spec.future_tool_id for spec in _ACTION_SPECS}
if len(_FUTURE_TOOL_IDS) != len(_ACTION_SPECS):
    raise ValueError("duplicate future tool ids in action catalog")


def get_action_specs() -> tuple[ActionSpec, ...]:
    return _ACTION_SPECS


def get_action_spec(action: str) -> ActionSpec | None:
    return _ACTION_INDEX.get(str(action or "").strip())


def public_action_names() -> list[str]:
    return [spec.action for spec in _ACTION_SPECS]


def supported_action_names() -> list[str]:
    return sorted(public_action_names())


def build_action_registry() -> dict[str, ActionHandler]:
    return {spec.action: spec.handler() for spec in _ACTION_SPECS}
