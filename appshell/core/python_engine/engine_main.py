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

from engine_protocol import EngineError, EngineResponse, KnownEngineError
from engine_service import handle_action


def _read_request(input_file: str | None) -> dict[str, Any]:
    if input_file:
        content = Path(input_file).read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()

    if not content.strip():
        raise KnownEngineError("INVALID_INPUT", "Empty request payload")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise KnownEngineError(
            code="INVALID_JSON",
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

    try:
        req = _read_request(args.input)
        task_id = str(req.get("task_id") or "no-task-id")
        action = str(req.get("action") or "").strip()
        payload = req.get("payload") or {}
        if not action:
            raise KnownEngineError("INVALID_INPUT", "Missing required field: action")
        if not isinstance(payload, dict):
            raise KnownEngineError("INVALID_INPUT", "Field payload must be an object")

        result = handle_action(action, payload)
        resp = EngineResponse(task_id=task_id, status="ok", result=result)
    except KnownEngineError as exc:
        resp = EngineResponse(
            task_id=task_id,
            status="error",
            error=EngineError(code=exc.code, message=exc.message, details=exc.details),
        )
    except Exception as exc:  # pragma: no cover - emergency guard
        resp = EngineResponse(
            task_id=task_id,
            status="error",
            error=EngineError(
                code="INTERNAL_ERROR",
                message="Unhandled engine exception",
                details={"reason": str(exc)},
            ),
        )

    resp.duration_ms = int((time.time() - started) * 1000)
    _write_response(resp, args.output)

    # Keep exit code 0 for business-level failures to preserve structured JSON output.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
