"""Action router for engine requests."""

from __future__ import annotations

from typing import Any, Callable

from engine_core import action_health, action_train
from engine_protocol import ErrorCode, KnownEngineError


ActionHandler = Callable[[dict[str, Any]], dict[str, Any]]


_REGISTRY: dict[str, ActionHandler] = {
    "health": action_health,
    "train": action_train,
}


def supported_actions() -> list[str]:
    return sorted(_REGISTRY.keys())


def handle_action(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    fn = _REGISTRY.get(action)
    if fn is None:
        raise KnownEngineError(
            code=ErrorCode.UNKNOWN_ACTION,
            message=f"Unsupported action: {action}",
            details={"supported_actions": supported_actions()},
        )
    return fn(payload)

