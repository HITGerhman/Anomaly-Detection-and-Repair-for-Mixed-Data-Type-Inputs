"""CLI entrypoint for Python engine.

Protocol:
- Input : JSON from stdin (or --input file)
- Output: JSON to stdout (or --output file)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from engine_logging import log_event
from engine_protocol import (
    EngineError,
    EngineRequest,
    EngineResponse,
    ErrorCode,
    KnownEngineError,
)
from engine_service import handle_action


def _read_request(input_file: str | None) -> dict[str, Any]:
    try:
        if input_file:
            content = Path(input_file).read_text(encoding="utf-8")
        else:
            content = sys.stdin.read()
    except FileNotFoundError as exc:
        raise KnownEngineError(
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"Input file not found: {input_file}",
            details={"input_file": input_file},
        ) from exc
    except OSError as exc:
        raise KnownEngineError(
            code=ErrorCode.INVALID_INPUT,
            message="Failed to read input payload",
            details={"reason": str(exc)},
        ) from exc

    if not content.strip():
        raise KnownEngineError(ErrorCode.INVALID_INPUT, "Empty request payload")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise KnownEngineError(
            code=ErrorCode.INVALID_JSON,
            message="Request is not valid JSON",
            details={"reason": str(exc)},
        ) from exc


def _write_response(resp: EngineResponse, output_file: str | None) -> None:
    body = json.dumps(resp.to_dict(), ensure_ascii=True)
    if output_file:
        Path(output_file).write_text(body + "\n", encoding="utf-8")
    else:
        sys.stdout.write(body + "\n")
        sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Python anomaly engine")
    parser.add_argument("--input", help="Path to JSON request file", default=None)
    parser.add_argument("--output", help="Path to JSON response file", default=None)
    args = parser.parse_args()

    started = time.time()
    task_id = "unknown"
    action = "unknown"

    try:
        req = EngineRequest.from_dict(_read_request(args.input))
        task_id = req.task_id
        action = req.action
        log_event("info", "engine_request_received", task_id=task_id, action=action)

        result = handle_action(req.action, req.payload)
        resp = EngineResponse(task_id=task_id, status="ok", result=result)
        log_event("info", "engine_request_succeeded", task_id=task_id, action=action)
    except KnownEngineError as exc:
        resp = EngineResponse(
            task_id=task_id,
            status="error",
            error=EngineError(code=exc.code, message=exc.message, details=exc.details),
        )
        log_event(
            "warning",
            "engine_request_failed",
            task_id=task_id,
            action=action,
            error_code=exc.code,
            error_message=exc.message,
        )
    except Exception as exc:  # pragma: no cover - emergency guard
        resp = EngineResponse(
            task_id=task_id,
            status="error",
            error=EngineError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Unhandled engine exception",
                details={"reason": str(exc)},
            ),
        )
        log_event(
            "error",
            "engine_request_crashed",
            task_id=task_id,
            action=action,
            error_code=ErrorCode.INTERNAL_ERROR,
            error_message=str(exc),
        )

    resp.duration_ms = int((time.time() - started) * 1000)
    _write_response(resp, args.output)

    # Keep exit code 0 for business-level failures to preserve structured JSON output.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

