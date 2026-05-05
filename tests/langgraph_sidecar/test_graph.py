import pytest

pytest.importorskip("langgraph")

import appshell.core.langgraph_sidecar.graph as graph_module
from appshell.core.langgraph_sidecar.graph import EXPLAIN_GRAPH, PLAN_GRAPH, GRAPH_ID, invoke_explain, invoke_plan
from appshell.core.langgraph_sidecar.llm_client import LLMError


def _llm_plan_request() -> dict:
    return {
        "session_id": "session-llm",
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
            },
            {
                "candidate_id": "candidate-gower",
                "source": "gower",
                "comparison": {"after_issue_count": 0},
                "selected_issue_ids": ["issue-1"],
                "tool_sequence": ["engine.repair_with_gower"],
                "summary": "gower preview",
            },
        ],
        "safety_context": {
            "selected_candidate_id": "candidate-rule",
            "auto_repair_issue_ids": ["issue-1"],
            "cautious_issue_ids": ["issue-2"],
            "manual_review_issue_ids": [],
            "blocked_issue_ids": [],
        },
        "approval_context": {},
        "user_preferences": {},
        "output_constraints": {},
    }


def _enable_mock_llm(monkeypatch):
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_API_KEY", "test-key")
    monkeypatch.setenv("APPSHELL_LANGGRAPH_LLM_MODEL", "gpt-test")


def test_graph_can_compile_and_invoke():
    result = PLAN_GRAPH.invoke(
        {
            "request": {
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
            }
        }
    )
    assert result["response"]["strategy_label"]
    assert result["response"]["selected_candidate_id"] == "candidate-rule"


def test_invoke_plan_returns_short_structured_mock_result():
    result = invoke_plan(
        {
            "session_id": "session-2",
            "goal": "scan and repair",
            "scan_summary": {"total_issues": 1},
            "candidate_previews": [
                {
                    "candidate_id": "candidate-hybrid",
                    "source": "hybrid",
                    "comparison": {"after_issue_count": 0},
                    "selected_issue_ids": ["issue-1"],
                    "tool_sequence": ["engine.repair_batch", "engine.repair_with_gower"],
                    "summary": "hybrid preview",
                }
            ],
            "safety_context": {},
            "user_preferences": {},
            "output_constraints": {},
        }
    )
    assert result["strategy_label"]
    assert result["selected_candidate_id"] == "candidate-hybrid"
    assert result["reason_codes"]
    assert len(result["short_bullets"]) <= 3
    assert result["approval_needed"] is False
    assert GRAPH_ID == "phase_c_cognition_graph"


def test_invoke_explain_returns_short_structured_result():
    result = invoke_explain(
        {
            "session_id": "session-2",
            "goal": "scan and repair",
            "selected_candidate": {
                "candidate_id": "candidate-hybrid",
                "source": "hybrid",
                "comparison": {"after_issue_count": 0},
                "selected_issue_ids": ["issue-1"],
                "tool_sequence": ["engine.repair_batch", "engine.repair_with_gower"],
                "summary": "hybrid preview",
            },
            "strategy_label": "hybrid_balanced",
            "reason_codes": ["fallback_no_llm"],
            "risk_note": "validation first",
            "validation_preview": {"resolved_issue_count": 1},
            "safety_context": {},
            "output_constraints": {},
        }
    )
    assert result["summary"]
    assert result["final_message"]
    assert len(result["short_bullets"]) <= 3
    assert EXPLAIN_GRAPH is not None


def test_invoke_plan_uses_mock_llm_when_schema_is_valid(monkeypatch):
    _enable_mock_llm(monkeypatch)

    def fake_completion(*, system_prompt, user_payload, config=None):
        if "intent node" in system_prompt:
            return {"intent_label": "auto_repair", "goal_summary": "repair", "preference_tags": ["json"]}
        if "strategy node" in system_prompt:
            return {
                "strategy_label": "neighbor_similarity",
                "selected_candidate_id": "candidate-gower",
                "reason_codes": ["phase_c_llm"],
                "risk_note": "validation first",
                "intent_label": "auto_repair",
                "one_sentence_summary": "LLM selected Gower.",
                "short_bullets": ["valid candidate"],
                "approval_needed": False,
            }
        return {
            "summary": "LLM summary.",
            "final_message": "LLM final message.",
            "short_bullets": ["short"],
            "reason_codes": ["phase_c_llm", "selected_gower"],
            "risk_note": "validation first",
        }

    monkeypatch.setattr(graph_module, "invoke_json_completion", fake_completion)

    result = graph_module.invoke_plan(_llm_plan_request())
    assert result["selected_candidate_id"] == "candidate-gower"
    assert result["strategy_label"] == "neighbor_similarity"
    assert "phase_c_llm" in result["reason_codes"]


def test_invoke_plan_falls_back_when_llm_selects_unknown_candidate(monkeypatch):
    _enable_mock_llm(monkeypatch)

    def fake_completion(*, system_prompt, user_payload, config=None):
        if "strategy node" in system_prompt:
            return {
                "strategy_label": "unsafe",
                "selected_candidate_id": "candidate-missing",
                "reason_codes": ["phase_c_llm"],
            }
        if "explanation node" in system_prompt:
            return {"summary": "fallback summary", "final_message": "fallback final"}
        return {"intent_label": "auto_repair", "goal_summary": "repair", "preference_tags": ["json"]}

    monkeypatch.setattr(graph_module, "invoke_json_completion", fake_completion)

    result = graph_module.invoke_plan(_llm_plan_request())
    assert result["selected_candidate_id"] == "candidate-rule"
    assert "llm_schema_invalid" in result["reason_codes"]


def test_invoke_plan_falls_back_when_llm_request_fails(monkeypatch):
    _enable_mock_llm(monkeypatch)

    def fake_completion(*, system_prompt, user_payload, config=None):
        raise LLMError("llm content is not valid JSON")

    monkeypatch.setattr(graph_module, "invoke_json_completion", fake_completion)

    result = graph_module.invoke_plan(_llm_plan_request())
    assert result["selected_candidate_id"] == "candidate-rule"
    assert "llm_invalid_json" in result["reason_codes"]


def test_invoke_plan_enforces_approval_context_over_llm(monkeypatch):
    _enable_mock_llm(monkeypatch)

    def fake_completion(*, system_prompt, user_payload, config=None):
        if "strategy node" in system_prompt:
            return {
                "strategy_label": "neighbor_similarity",
                "selected_candidate_id": "candidate-gower",
                "reason_codes": ["phase_c_llm"],
                "approval_needed": False,
            }
        if "explanation node" in system_prompt:
            return {"summary": "LLM summary.", "final_message": "LLM final.", "reason_codes": ["phase_c_llm"]}
        return {"intent_label": "auto_repair", "goal_summary": "repair", "preference_tags": ["json"]}

    monkeypatch.setattr(graph_module, "invoke_json_completion", fake_completion)
    request = _llm_plan_request()
    request["approval_context"] = {"deterministic_required": True}

    result = graph_module.invoke_plan(request)
    assert result["approval_needed"] is True
    assert "approval_context_enforced" in result["reason_codes"]
