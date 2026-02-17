"""JSON protocol models for the Python engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EngineError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResponse:
    task_id: str
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: EngineError | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": None
            if self.error is None
            else {
                "code": self.error.code,
                "message": self.error.message,
                "details": self.error.details,
            },
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class KnownEngineError(Exception):
    """Domain-level error that should be returned as structured JSON."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
