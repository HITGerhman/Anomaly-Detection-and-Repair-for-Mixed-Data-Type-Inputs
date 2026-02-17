"""JSON protocol and domain errors for the Python engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class ErrorCode:
    INVALID_JSON = "INVALID_JSON"
    INVALID_INPUT = "INVALID_INPUT"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    CSV_READ_FAILED = "CSV_READ_FAILED"
    INVALID_TARGET_COLUMN = "INVALID_TARGET_COLUMN"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    TRAINING_MODULE_IMPORT_FAILED = "TRAINING_MODULE_IMPORT_FAILED"
    TRAINING_FAILED = "TRAINING_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    @classmethod
    def all(cls) -> set[str]:
        return {
            cls.INVALID_JSON,
            cls.INVALID_INPUT,
            cls.UNKNOWN_ACTION,
            cls.FILE_NOT_FOUND,
            cls.CSV_READ_FAILED,
            cls.INVALID_TARGET_COLUMN,
            cls.MISSING_DEPENDENCY,
            cls.TRAINING_MODULE_IMPORT_FAILED,
            cls.TRAINING_FAILED,
            cls.INTERNAL_ERROR,
        }


@dataclass(frozen=True)
class EngineRequest:
    task_id: str
    action: str
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EngineRequest":
        if not isinstance(raw, dict):
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Request body must be a JSON object",
            )

        task_id = str(raw.get("task_id") or "no-task-id").strip()
        action = str(raw.get("action") or "").strip()
        payload = raw.get("payload", {})

        if not action:
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Missing required field: action",
                details={"field": "action"},
            )
        if not isinstance(payload, dict):
            raise KnownEngineError(
                code=ErrorCode.INVALID_INPUT,
                message="Field payload must be an object",
                details={"field": "payload"},
            )

        return cls(task_id=task_id, action=action, payload=payload)


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
    """Domain-level error returned as structured JSON."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        normalized_code = code if code in ErrorCode.all() else ErrorCode.INTERNAL_ERROR
        self.code = normalized_code
        self.message = message
        self.details = details or {}
        super().__init__(message)

