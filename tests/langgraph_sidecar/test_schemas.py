from appshell.core.langgraph_sidecar.schemas import (
    build_health_response,
    normalize_explain_request,
    normalize_explain_response,
    normalize_plan_request,
    normalize_plan_response,
)


def test_normalize_plan_response_adds_phase_c_fields():
    result = normalize_plan_response(
        {
            "strategy_label": "neighbor_similarity",
            "selected_candidate_id": "candidate-gower",
            "reason_codes": ["phase_c_llm"],
            "risk_note": "validation first",
            "intent_label": "auto_repair",
            "one_sentence_summary": "Selected gower.",
            "short_bullets": ["one", "two", "three", "four"],
            "approval_needed": False,
        }
    )
    assert result["intent_label"] == "auto_repair"
    assert result["risk_note"] == "validation first"
    assert len(result["short_bullets"]) == 3


def test_normalize_explain_request_and_response():
    request = normalize_explain_request(
        {
            "session_id": "session-1",
            "goal": "repair",
            "selected_candidate": {"candidate_id": "candidate-rule", "source": "rule"},
            "reason_codes": ["a", "b"],
        }
    )
    assert request["selected_candidate"]["candidate_id"] == "candidate-rule"

    response = normalize_explain_response(
        {
            "summary": "Short summary.",
            "final_message": "Short final message.",
            "short_bullets": ["one", "two", "three", "four"],
            "reason_codes": ["a"],
            "risk_note": "validation first",
        }
    )
    assert response["final_message"] == "Short final message."
    assert len(response["short_bullets"]) == 3


def test_build_health_response_includes_llm_fields():
    health = build_health_response(
        graph_id="phase_c_cognition_graph",
        version="phase_c",
        planner_mode="llm",
        llm_mode="configured",
        model="gpt-test",
    )
    assert health["planner_mode"] == "llm"
    assert health["llm_mode"] == "configured"
    assert health["model"] == "gpt-test"


def test_normalize_plan_request_preserves_candidate_preview_shape():
    payload = normalize_plan_request(
        {
            "session_id": "session-1",
            "goal": "repair",
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
        }
    )
    assert payload["candidate_previews"][0]["source"] == "rule"
