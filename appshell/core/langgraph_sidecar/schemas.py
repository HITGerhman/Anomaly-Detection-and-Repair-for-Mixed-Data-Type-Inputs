from __future__ import annotations

from typing import Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def build_health_response(*, graph_id: str, version: str, planner_mode: str, llm_mode: str, model: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "langgraph-sidecar",
        "planner_mode": _as_text(planner_mode) or "fallback",
        "llm_mode": _as_text(llm_mode) or "unavailable",
        "model": _as_text(model),
        "ready": True,
        "graph_id": _as_text(graph_id),
        "version": _as_text(version),
    }


def normalize_plan_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    normalized = {
        "session_id": _as_text(payload.get("session_id")),
        "goal": _as_text(payload.get("goal")),
        "scan_summary": _as_dict(payload.get("scan_summary")),
        "candidate_previews": [],
        "safety_context": _as_dict(payload.get("safety_context")),
        "approval_context": _as_dict(payload.get("approval_context")),
        "user_preferences": _as_dict(payload.get("user_preferences")),
        "output_constraints": _as_dict(payload.get("output_constraints")),
    }

    for raw_candidate in _as_list(payload.get("candidate_previews")):
        if not isinstance(raw_candidate, dict):
            continue
        normalized["candidate_previews"].append(
            {
                "candidate_id": _as_text(raw_candidate.get("candidate_id")),
                "source": _as_text(raw_candidate.get("source")),
                "comparison": _as_dict(raw_candidate.get("comparison")),
                "selected_issue_ids": [
                    _as_text(item)
                    for item in _as_list(raw_candidate.get("selected_issue_ids"))
                    if _as_text(item)
                ],
                "tool_sequence": [
                    _as_text(item)
                    for item in _as_list(raw_candidate.get("tool_sequence"))
                    if _as_text(item)
                ],
                "summary": _as_text(raw_candidate.get("summary")),
            }
        )

    return normalized


def normalize_plan_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("graph output must be a JSON object")

    bullets = _normalize_bullets(payload.get("short_bullets"))
    response = {
        "strategy_label": _as_text(payload.get("strategy_label")) or "fallback_langgraph_plan",
        "selected_candidate_id": _as_text(payload.get("selected_candidate_id")),
        "reason_codes": _normalize_reason_codes(payload.get("reason_codes")) or ["fallback_no_llm"],
        "risk_note": _as_text(payload.get("risk_note")),
        "intent_label": _as_text(payload.get("intent_label")) or "balanced_repair",
        "one_sentence_summary": _as_text(payload.get("one_sentence_summary")),
        "short_bullets": bullets,
        "approval_needed": bool(payload.get("approval_needed", False)),
    }
    if not response["one_sentence_summary"]:
        response["one_sentence_summary"] = "LangGraph kept the deterministic execution boundary and returned a short planning summary."
    return response


def normalize_explain_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    selected_candidate = _as_dict(payload.get("selected_candidate"))
    return {
        "session_id": _as_text(payload.get("session_id")),
        "goal": _as_text(payload.get("goal")),
        "selected_candidate": {
            "candidate_id": _as_text(selected_candidate.get("candidate_id")),
            "source": _as_text(selected_candidate.get("source")),
            "comparison": _as_dict(selected_candidate.get("comparison")),
            "selected_issue_ids": [
                _as_text(item)
                for item in _as_list(selected_candidate.get("selected_issue_ids"))
                if _as_text(item)
            ],
            "tool_sequence": [
                _as_text(item)
                for item in _as_list(selected_candidate.get("tool_sequence"))
                if _as_text(item)
            ],
            "summary": _as_text(selected_candidate.get("summary")),
        },
        "strategy_label": _as_text(payload.get("strategy_label")),
        "reason_codes": _normalize_reason_codes(payload.get("reason_codes")),
        "risk_note": _as_text(payload.get("risk_note")),
        "validation_preview": _as_dict(payload.get("validation_preview")),
        "safety_context": _as_dict(payload.get("safety_context")),
        "approval_context": _as_dict(payload.get("approval_context")),
        "output_constraints": _as_dict(payload.get("output_constraints")),
    }


def normalize_explain_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("graph output must be a JSON object")

    response = {
        "summary": _as_text(payload.get("summary")),
        "final_message": _as_text(payload.get("final_message")),
        "short_bullets": _normalize_bullets(payload.get("short_bullets")),
        "reason_codes": _normalize_reason_codes(payload.get("reason_codes")),
        "risk_note": _as_text(payload.get("risk_note")),
    }
    if not response["summary"] and not response["final_message"]:
        response["summary"] = "LangGraph returned a short explanation without changing deterministic execution."
    return response


def normalize_intent(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"intent_label": "balanced_repair", "goal_summary": "", "preference_tags": []}
    return {
        "intent_label": _as_text(payload.get("intent_label")) or "balanced_repair",
        "goal_summary": _as_text(payload.get("goal_summary")),
        "preference_tags": _normalize_bullets(payload.get("preference_tags")),
    }


def _normalize_reason_codes(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        text = _as_text(item)
        if text:
            out.append(text)
    return out


def _normalize_bullets(value: Any) -> list[str]:
    bullets: list[str] = []
    for item in _as_list(value)[:3]:
        text = _as_text(item)
        if text:
            bullets.append(text)
    return bullets
