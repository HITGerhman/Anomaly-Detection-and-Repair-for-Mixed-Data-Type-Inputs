from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[2] / "appshell" / "core" / "python_engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import engine_core  # type: ignore
from engine_protocol import ErrorCode  # type: ignore


class _FakeModule:
    def __init__(self, version: str):
        self.__version__ = version


def test_action_health_returns_dependency_versions(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert result["engine"] == "python-anomaly-engine"
    assert result["dependencies"]["numpy"]["version"] == "1.26.4"
    assert result["dependencies"]["scikit-learn"]["module"] == "sklearn"
    assert all(item["status"] == "ok" for item in result["dependencies"].values())


def test_action_health_raises_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(module_name: str):
        if module_name == "numpy":
            raise ImportError("broken numpy")
        return _FakeModule("1.0.0")

    monkeypatch.setattr(engine_core.importlib, "import_module", fake_import)

    with pytest.raises(engine_core.KnownEngineError) as exc_info:
        engine_core.action_health({})

    assert exc_info.value.code == ErrorCode.MISSING_DEPENDENCY
    assert exc_info.value.details["dependency"] == "numpy"