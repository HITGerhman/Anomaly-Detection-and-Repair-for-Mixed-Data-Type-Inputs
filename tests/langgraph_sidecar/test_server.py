import json
import socket
import threading
import urllib.request

import pytest

pytest.importorskip("langgraph")

from appshell.core.langgraph_sidecar.server import create_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_server_exposes_health_and_plan_endpoints():
    port = _free_port()
    server = create_server("127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        health = _read_json(f"http://127.0.0.1:{port}/health")
        assert health["status"] == "ok"
        assert health["service"] == "langgraph-sidecar"
        assert health["planner_mode"] == "fallback"
        assert health["llm_mode"] == "unavailable"

        plan = _read_json(
            f"http://127.0.0.1:{port}/v1/plan",
            method="POST",
            payload={
                "session_id": "session-1",
                "goal": "scan and repair",
                "scan_summary": {"total_issues": 2},
                "candidate_previews": [
                    {
                        "candidate_id": "candidate-rule",
                        "source": "rule",
                        "comparison": {"after_issue_count": 1},
                        "selected_issue_ids": ["issue-1"],
                        "tool_sequence": ["engine.repair_batch"],
                        "summary": "rule preview",
                    }
                ],
                "safety_context": {},
                "user_preferences": {},
                "output_constraints": {},
            },
        )
        assert plan["strategy_label"] in {"deterministic_rule", "neighbor_similarity", "hybrid_balanced", "fallback_langgraph_plan"}
        assert plan["selected_candidate_id"] == "candidate-rule"
        assert len(plan["short_bullets"]) <= 3

        explain = _read_json(
            f"http://127.0.0.1:{port}/v1/explain",
            method="POST",
            payload={
                "session_id": "session-1",
                "goal": "scan and repair",
                "selected_candidate": {
                    "candidate_id": "candidate-rule",
                    "source": "rule",
                    "comparison": {"after_issue_count": 1},
                    "selected_issue_ids": ["issue-1"],
                    "tool_sequence": ["engine.repair_batch"],
                    "summary": "rule preview",
                },
                "strategy_label": "deterministic_rule",
                "reason_codes": ["fallback_no_llm"],
                "risk_note": "validation first",
                "validation_preview": {"resolved_issue_count": 1},
                "safety_context": {},
                "output_constraints": {},
            },
        )
        assert explain["summary"]
        assert explain["final_message"]
        assert len(explain["short_bullets"]) <= 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_health_reflects_llm_configuration(monkeypatch):
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_API_KEY", "test-key")
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_MODEL", "gpt-test")

    port = _free_port()
    server = create_server("127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        health = _read_json(f"http://127.0.0.1:{port}/health")
        assert health["planner_mode"] == "llm"
        assert health["llm_mode"] == "configured"
        assert health["model"] == "gpt-test"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_health_falls_back_when_api_key_missing(monkeypatch):
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.delenv("APPSHELL_LANGGRAPH_LLM_API_KEY", raising=False)
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_MODEL", "gpt-test")

    port = _free_port()
    server = create_server("127.0.0.1", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        health = _read_json(f"http://127.0.0.1:{port}/health")
        assert health["planner_mode"] == "fallback"
        assert health["llm_mode"] == "unavailable"
        assert "api" not in health
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
