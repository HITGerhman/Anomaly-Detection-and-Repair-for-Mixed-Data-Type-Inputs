"""Action router for engine requests."""

from __future__ import annotations

from typing import Any, Callable

from action_catalog import build_action_registry, supported_action_names
from engine_protocol import ErrorCode, KnownEngineError


ActionHandler = Callable[[dict[str, Any]], dict[str, Any]]


_REGISTRY: dict[str, ActionHandler] = build_action_registry()


def supported_actions() -> list[str]:
    return supported_action_names()


def handle_action(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    fn = _REGISTRY.get(action)
    if fn is None:
        raise KnownEngineError(
            code=ErrorCode.UNKNOWN_ACTION,
            message=f"Unsupported action: {action}",
            details={"supported_actions": supported_actions()},
        )
    return fn(payload)
