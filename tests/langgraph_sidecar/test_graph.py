import pytest

pytest.importorskip("langgraph")

from appshell.core.langgraph_sidecar.graph import EXPLAIN_GRAPH, PLAN_GRAPH, GRAPH_ID, invoke_explain, invoke_plan


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
