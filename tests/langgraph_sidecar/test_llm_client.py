import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from appshell.core.langgraph_sidecar.llm_client import LLMError, OpenAICompatibleConfig, invoke_json_completion, llm_health_payload


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(handler_cls):
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, port


def test_llm_health_payload_reports_fallback_without_configuration():
    payload = llm_health_payload(OpenAICompatibleConfig(base_url="", api_key="", model="", timeout_ms=1000))
    assert payload["planner_mode"] == "fallback"
    assert payload["llm_mode"] == "unavailable"


def test_llm_health_payload_requires_api_key():
    payload = llm_health_payload(OpenAICompatibleConfig(base_url="http://127.0.0.1:9999/v1", api_key="", model="gpt-test", timeout_ms=1000))
    assert payload["planner_mode"] == "fallback"
    assert payload["llm_mode"] == "unavailable"


def test_invoke_json_completion_parses_openai_compatible_response():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_POST(self):  # noqa: N802
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "strategy_label": "neighbor_similarity",
                                        "selected_candidate_id": "candidate-gower",
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread, port = _start_server(Handler)
    try:
        payload = invoke_json_completion(
            system_prompt="Return JSON only.",
            user_payload={"goal": "repair"},
            config=OpenAICompatibleConfig(
                base_url=f"http://127.0.0.1:{port}",
                api_key="test",
                model="gpt-test",
                timeout_ms=1000,
            ),
        )
        assert payload["strategy_label"] == "neighbor_similarity"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_invoke_json_completion_raises_on_http_error():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_POST(self):  # noqa: N802
            body = b'{"error":"unauthorized"}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread, port = _start_server(Handler)
    try:
        with pytest.raises(LLMError):
            invoke_json_completion(
                system_prompt="Return JSON only.",
                user_payload={"goal": "repair"},
                config=OpenAICompatibleConfig(
                    base_url=f"http://127.0.0.1:{port}",
                    api_key="test",
                    model="gpt-test",
                    timeout_ms=1000,
                ),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_invoke_json_completion_raises_on_empty_message_content():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_POST(self):  # noqa: N802
            body = json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread, port = _start_server(Handler)
    try:
        with pytest.raises(LLMError, match="empty"):
            invoke_json_completion(
                system_prompt="Return JSON only.",
                user_payload={"goal": "repair"},
                config=OpenAICompatibleConfig(
                    base_url=f"http://127.0.0.1:{port}",
                    api_key="test",
                    model="gpt-test",
                    timeout_ms=1000,
                ),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
