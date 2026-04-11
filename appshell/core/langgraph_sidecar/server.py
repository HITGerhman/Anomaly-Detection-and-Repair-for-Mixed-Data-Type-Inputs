from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .graph import GRAPH_ID, invoke_explain, invoke_plan
from .llm_client import llm_health_payload
from .schemas import (
    build_health_response,
    normalize_explain_request,
    normalize_plan_request,
)


SIDECAR_VERSION = "phase_c"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    class SidecarHandler(BaseHTTPRequestHandler):
        server_version = "LangGraphSidecar/phase_c"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write_json(404, {"error": "not_found"})
                return
            payload = build_health_response(graph_id=GRAPH_ID, version=SIDECAR_VERSION, **llm_health_payload())
            self._write_json(200, payload)

        def do_POST(self) -> None:  # noqa: N802
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                decoded = json.loads(raw_body.decode("utf-8") or "{}")
                if self.path == "/v1/plan":
                    request_payload = normalize_plan_request(decoded)
                    response = invoke_plan(request_payload)
                elif self.path == "/v1/explain":
                    request_payload = normalize_explain_request(decoded)
                    response = invoke_explain(request_payload)
                else:
                    self._write_json(404, {"error": "not_found"})
                    return
            except json.JSONDecodeError as exc:
                self._write_json(400, {"error": "invalid_json", "message": str(exc)})
                return
            except ValueError as exc:
                self._write_json(400, {"error": "invalid_request", "message": str(exc)})
                return
            self._write_json(200, response)

    return ThreadingHTTPServer((host, port), SidecarHandler)


def serve(host: str, port: int) -> None:
    server = create_server(host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
