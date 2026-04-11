from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[2] / "appshell" / "core" / "python_engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import engine_core  # type: ignore
import engine_service  # type: ignore
from action_catalog import get_action_specs, public_action_names  # type: ignore


class _FakeModule:
    def __init__(self, version: str):
        self.__version__ = version


def test_action_catalog_has_unique_actions_and_tool_ids() -> None:
    specs = get_action_specs()
    assert [spec.action for spec in specs] == [
        "health",
        "train",
        "repair",
        "scan_file",
        "repair_batch",
        "repair_with_gower",
        "rollback_repair_batch",
    ]
    assert len({spec.action for spec in specs}) == len(specs)
    assert len({spec.future_tool_id for spec in specs}) == len(specs)
    assert all(callable(spec.handler()) for spec in specs)


def test_engine_service_supported_actions_are_generated_from_catalog() -> None:
    assert engine_service.supported_actions() == sorted(public_action_names())


def test_action_health_returns_catalog_order(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "pandas": "2.2.0",
        "numpy": "1.26.4",
        "lightgbm": "4.6.0",
        "sklearn": "1.5.2",
        "joblib": "1.4.2",
    }

    def fake_import(module_name: str):
        return _FakeModule(versions[module_name])

    monkeypatch.setattr(engine_core.importlib, "import_module", fake_import)

    result = engine_core.action_health({})

    assert result["actions"] == public_action_names()
