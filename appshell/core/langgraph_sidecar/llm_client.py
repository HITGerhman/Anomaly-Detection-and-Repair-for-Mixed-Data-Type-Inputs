from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


DEFAULT_TIMEOUT_MS = 4000


class LLMError(RuntimeError):
    """Raised when the configured OpenAI-compatible endpoint cannot satisfy a request."""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def load_llm_config() -> OpenAICompatibleConfig:
    timeout_raw = str(os.getenv("APPSHELL_LANGGRAPH_LLM_TIMEOUT_MS", "")).strip()
    timeout_ms = DEFAULT_TIMEOUT_MS
    if timeout_raw.isdigit():
        timeout_ms = max(int(timeout_raw), 1)
    return OpenAICompatibleConfig(
        base_url=str(os.getenv("APPSHELL_LANGGRAPH_LLM_BASE_URL", "")).strip(),
        api_key=str(os.getenv("APPSHELL_LANGGRAPH_LLM_API_KEY", "")).strip(),
        model=str(os.getenv("APPSHELL_LANGGRAPH_LLM_MODEL", "")).strip(),
        timeout_ms=timeout_ms,
    )


def llm_health_payload(config: OpenAICompatibleConfig | None = None) -> dict[str, Any]:
    config = config or load_llm_config()
    planner_mode = "llm" if config.enabled else "fallback"
    llm_mode = "configured" if config.enabled else "unavailable"
    return {
        "planner_mode": planner_mode,
        "llm_mode": llm_mode,
        "model": config.model,
    }


def invoke_json_completion(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    config: OpenAICompatibleConfig | None = None,
) -> dict[str, Any]:
    config = config or load_llm_config()
    if not config.enabled:
        raise LLMError("llm is not configured")

    body = json.dumps(
        {
            "model": config.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        },
        ensure_ascii=True,
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "appshell-langgraph-sidecar/phase_c",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    req = request.Request(config.chat_url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=max(config.timeout_ms / 1000.0, 0.1)) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - exercised in tests through message extraction
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"llm endpoint returned {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise LLMError(f"llm endpoint unavailable: {exc}") from exc
    except TimeoutError as exc:
        raise LLMError("llm request timed out") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("llm endpoint returned non-JSON payload") from exc

    content = _extract_message_content(payload)
    try:
        return _parse_json_text(content)
    except json.JSONDecodeError as exc:
        raise LLMError("llm content is not valid JSON") from exc


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("llm response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMError("llm response missing message")
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    raise LLMError("llm message content is not textual")


def _parse_json_text(content: str) -> dict[str, Any]:
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        clean = clean.replace("json", "", 1).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end >= start:
        clean = clean[start : end + 1]
    return json.loads(clean)
