from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm_client import LLMError, load_llm_config, invoke_json_completion
from .schemas import normalize_explain_request, normalize_explain_response, normalize_intent, normalize_plan_response


GRAPH_ID = "phase_c_cognition_graph"


class PlanGraphState(TypedDict, total=False):
    request: dict[str, Any]
    intent: dict[str, Any]
    strategy: dict[str, Any]
    explanation: dict[str, Any]
    response: dict[str, Any]


class ExplainGraphState(TypedDict, total=False):
    request: dict[str, Any]
    explanation: dict[str, Any]
    response: dict[str, Any]


def intent_node(state: PlanGraphState) -> PlanGraphState:
    request = dict(state.get("request") or {})
    intent = fallback_intent(request)
    if llm_enabled():
        try:
            intent = normalize_intent(
                invoke_json_completion(
                    system_prompt=(
                        "You are the intent node of a data-repair agent. "
                        "Return JSON only with keys: intent_label, goal_summary, preference_tags. "
                        "Keep goal_summary to one short sentence and preference_tags to at most 3 short items."
                    ),
                    user_payload={
                        "goal": request.get("goal", ""),
                        "scan_summary": request.get("scan_summary", {}),
                        "safety_context": request.get("safety_context", {}),
                        "approval_context": request.get("approval_context", {}),
                        "output_constraints": request.get("output_constraints", {}),
                    },
                )
            )
        except LLMError:
            pass
    return {"request": request, "intent": intent}


def strategy_node(state: PlanGraphState) -> PlanGraphState:
    request = dict(state.get("request") or {})
    intent = normalize_intent(state.get("intent"))
    strategy = fallback_strategy(request, intent)
    if llm_enabled():
        try:
            strategy = normalize_plan_response(
                invoke_json_completion(
                    system_prompt=(
                        "You are the strategy node of a data-repair agent. "
                        "Return JSON only with keys: strategy_label, selected_candidate_id, reason_codes, risk_note, intent_label, one_sentence_summary, short_bullets, approval_needed. "
                        "selected_candidate_id must be one of the provided candidate_ids. "
                        "Keep one_sentence_summary to one sentence and short_bullets to at most 3 short items."
                    ),
                    user_payload={
                        "intent": intent,
                        "goal": request.get("goal", ""),
                        "scan_summary": request.get("scan_summary", {}),
                        "candidate_previews": request.get("candidate_previews", []),
                        "safety_context": request.get("safety_context", {}),
                        "approval_context": request.get("approval_context", {}),
                        "user_preferences": request.get("user_preferences", {}),
                        "output_constraints": request.get("output_constraints", {}),
                    },
                )
            )
        except LLMError:
            pass
    strategy["intent_label"] = intent.get("intent_label", strategy.get("intent_label", "balanced_repair"))
    return {"request": request, "intent": intent, "strategy": strategy}


def explain_node(state: PlanGraphState) -> PlanGraphState:
    request = dict(state.get("request") or {})
    intent = normalize_intent(state.get("intent"))
    strategy = normalize_plan_response(state.get("strategy"))
    explanation = fallback_plan_explanation(request, intent, strategy)
    if llm_enabled():
        try:
            explanation = normalize_explain_response(
                invoke_json_completion(
                    system_prompt=(
                        "You are the explanation node of a data-repair agent. "
                        "Return JSON only with keys: summary, final_message, short_bullets, reason_codes, risk_note. "
                        "Keep summary to one sentence, final_message to at most two short sentences, and short_bullets to at most 3 items."
                    ),
                    user_payload={
                        "intent": intent,
                        "strategy": strategy,
                        "selected_candidate": _candidate_by_id(request, strategy.get("selected_candidate_id", "")),
                        "goal": request.get("goal", ""),
                        "scan_summary": request.get("scan_summary", {}),
                        "safety_context": request.get("safety_context", {}),
                        "approval_context": request.get("approval_context", {}),
                        "output_constraints": request.get("output_constraints", {}),
                    },
                )
            )
        except LLMError:
            pass

    response = normalize_plan_response(
        {
            "strategy_label": strategy.get("strategy_label", ""),
            "selected_candidate_id": strategy.get("selected_candidate_id", ""),
            "reason_codes": explanation.get("reason_codes") or strategy.get("reason_codes", []),
            "risk_note": explanation.get("risk_note") or strategy.get("risk_note", ""),
            "intent_label": strategy.get("intent_label") or intent.get("intent_label", ""),
            "one_sentence_summary": explanation.get("summary") or strategy.get("one_sentence_summary", ""),
            "short_bullets": explanation.get("short_bullets") or strategy.get("short_bullets", []),
            "approval_needed": strategy.get("approval_needed", False),
        }
    )
    return {
        "request": request,
        "intent": intent,
        "strategy": strategy,
        "explanation": explanation,
        "response": response,
    }


def explain_only_node(state: ExplainGraphState) -> ExplainGraphState:
    request = normalize_explain_request(state.get("request"))
    explanation = fallback_explain_response(request)
    if llm_enabled():
        try:
            explanation = normalize_explain_response(
                invoke_json_completion(
                    system_prompt=(
                        "You are the explanation node of a data-repair agent. "
                        "Return JSON only with keys: summary, final_message, short_bullets, reason_codes, risk_note. "
                        "Keep output concise and fully structured."
                    ),
                    user_payload=request,
                )
            )
        except LLMError:
            pass
    return {"request": request, "explanation": explanation, "response": explanation}


def build_plan_graph():
    graph = StateGraph(PlanGraphState)
    graph.add_node("intent_node", intent_node)
    graph.add_node("strategy_node", strategy_node)
    graph.add_node("explain_node", explain_node)
    graph.add_edge(START, "intent_node")
    graph.add_edge("intent_node", "strategy_node")
    graph.add_edge("strategy_node", "explain_node")
    graph.add_edge("explain_node", END)
    return graph.compile()


def build_explain_graph():
    graph = StateGraph(ExplainGraphState)
    graph.add_node("explain_only_node", explain_only_node)
    graph.add_edge(START, "explain_only_node")
    graph.add_edge("explain_only_node", END)
    return graph.compile()


PLAN_GRAPH = build_plan_graph()
EXPLAIN_GRAPH = build_explain_graph()


def invoke_plan(request: dict[str, Any]) -> dict[str, Any]:
    result = PLAN_GRAPH.invoke({"request": request})
    return normalize_plan_response(result.get("response", {}))


def invoke_explain(request: dict[str, Any]) -> dict[str, Any]:
    result = EXPLAIN_GRAPH.invoke({"request": request})
    return normalize_explain_response(result.get("response", {}))


def llm_enabled() -> bool:
    return load_llm_config().enabled


def fallback_intent(request: dict[str, Any]) -> dict[str, Any]:
    goal = str(request.get("goal", "")).strip()
    lowered = goal.lower()
    if "保守" in goal or "conservative" in lowered:
        label = "conservative_repair"
    elif "scan" in lowered and "repair" not in lowered and "修复" not in goal:
        label = "scan_only"
    elif "auto" in lowered or "修复" in goal or "repair" in lowered or "fix" in lowered:
        label = "auto_repair"
    else:
        label = "balanced_repair"
    summary = goal or "Scan and repair the uploaded CSV with the current safety policy."
    return normalize_intent(
        {
            "intent_label": label,
            "goal_summary": summary[:160],
            "preference_tags": ["short_output", "structured_json", "validation_first"],
        }
    )


def fallback_strategy(request: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    selected_id = _default_selected_candidate_id(request)
    candidate = _candidate_by_id(request, selected_id)
    source = str(candidate.get("source", "rule")).strip() or "rule"
    comparison = candidate.get("comparison", {}) if isinstance(candidate, dict) else {}
    resolved = _as_int(comparison.get("resolved_issue_count"))
    after_count = _as_int(comparison.get("after_issue_count"))
    skipped_types = _short_list(request.get("safety_context", {}).get("skipped_issue_types") if isinstance(request.get("safety_context"), dict) else [])
    approval_context = request.get("approval_context", {}) if isinstance(request.get("approval_context"), dict) else {}
    approval_needed = bool(approval_context.get("deterministic_required", False))
    if source == "hybrid":
        strategy_label = "hybrid_balanced"
    elif source == "gower":
        strategy_label = "neighbor_similarity"
    else:
        strategy_label = "deterministic_rule"
    risk_note = (
        f"Explain-only issue types remain: {', '.join(skipped_types)}."
        if skipped_types
        else "Any write will still pass through deterministic validation and rollback gates."
    )
    return normalize_plan_response(
        {
            "strategy_label": strategy_label,
            "selected_candidate_id": selected_id,
            "reason_codes": [f"selected_{source}", "fallback_no_llm"],
            "risk_note": risk_note,
            "intent_label": intent.get("intent_label", "balanced_repair"),
            "one_sentence_summary": f"Selected the {source} candidate with after_issue_count={after_count} and resolved_issue_count={resolved}.",
            "short_bullets": [
                f"Primary source: {source}.",
                f"Resolved preview issues: {resolved}.",
                "Execution remains under deterministic validation.",
            ],
            "approval_needed": approval_needed,
        }
    )


def fallback_plan_explanation(request: dict[str, Any], intent: dict[str, Any], strategy: dict[str, Any]) -> dict[str, Any]:
    candidate = _candidate_by_id(request, strategy.get("selected_candidate_id", ""))
    source = str(candidate.get("source", "rule")).strip() or "rule"
    reason_codes = strategy.get("reason_codes", ["fallback_no_llm"])
    risk_note = str(strategy.get("risk_note", "")).strip()
    summary = str(strategy.get("one_sentence_summary", "")).strip()
    final_message = (
        f"LangGraph selected the {source} path for {intent.get('intent_label', 'balanced_repair')} while keeping all writes behind the existing deterministic safety loop."
    )
    return normalize_explain_response(
        {
            "summary": summary,
            "final_message": final_message,
            "short_bullets": [
                "Intent and strategy were inferred from structured session context.",
                "The candidate choice does not bypass preview validation or rollback.",
                "Outputs stay concise and JSON-only.",
            ],
            "reason_codes": reason_codes,
            "risk_note": risk_note,
        }
    )


def fallback_explain_response(request: dict[str, Any]) -> dict[str, Any]:
    candidate = request.get("selected_candidate", {})
    source = str(candidate.get("source", "rule")).strip() or "rule"
    reason_codes = request.get("reason_codes") or ["fallback_no_llm"]
    risk_note = str(request.get("risk_note", "")).strip()
    return normalize_explain_response(
        {
            "summary": f"Selected the {source} candidate under the current deterministic safety policy.",
            "final_message": f"The {source} candidate remains subject to Go-side preview validation, execution control, rescans, and rollback.",
            "short_bullets": [
                "This explanation stays short and structured.",
                "LangGraph does not directly execute repairs.",
                "Deterministic validation remains the final authority.",
            ],
            "reason_codes": reason_codes,
            "risk_note": risk_note,
        }
    )


def _default_selected_candidate_id(request: dict[str, Any]) -> str:
    safety_context = request.get("safety_context", {})
    if isinstance(safety_context, dict):
        selected = str(safety_context.get("selected_candidate_id", "")).strip()
        if selected and _candidate_by_id(request, selected):
            return selected
    candidate_previews = request.get("candidate_previews", [])
    if isinstance(candidate_previews, list) and candidate_previews:
        first = candidate_previews[0]
        if isinstance(first, dict):
            return str(first.get("candidate_id", "")).strip()
    return ""


def _candidate_by_id(request: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for item in request.get("candidate_previews", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("candidate_id", "")).strip() == str(candidate_id).strip():
            return dict(item)
    return {}


def _short_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(item).strip() for item in value if str(item).strip()]
    return items[:3]


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return 0


def debug_state(request: dict[str, Any]) -> str:
    """Useful when manually debugging the graph in isolation."""
    return json.dumps(PLAN_GRAPH.invoke({"request": request}), ensure_ascii=True, indent=2)
