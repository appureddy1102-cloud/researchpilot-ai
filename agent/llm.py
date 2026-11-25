"""
Thin, provider-agnostic LLM wrapper.

Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, Together,
Google Gemini's OpenAI-compatible URL, a local Ollama/vLLM server, ...). Change
LLM_BASE_URL + LLM_MODEL + LLM_API_KEY in `.env` - no code changes needed.

The agent only depends on the small `chat()` interface below, which also makes
it trivial to swap in a fake LLM in unit tests.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from utils.config import Settings, redact


class LLMError(Exception):
    """Any failure talking to the LLM provider (message is already redacted)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMMessage:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Format as an OpenAI assistant message so it can be re-sent."""
        d: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": t.id, "type": "function",
                 "function": {"name": t.name, "arguments": json.dumps(t.arguments)}}
                for t in self.tool_calls
            ]
        return d


def parse_json(text: str) -> dict | list | None:
    """Leniently parse JSON from an LLM reply (handles ```json fences / chatter)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMError("The 'openai' package is not installed. Run: pip install -r requirements.txt") from e
        self._client = OpenAI(
            api_key=settings.llm_api_key or "missing",
            base_url=settings.llm_base_url,
            timeout=max(settings.request_timeout, 30),
            max_retries=1,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, temperature: float = 0.2,
             max_tokens: int | None = None) -> LLMMessage:
        kwargs: dict[str, Any] = dict(model=self.settings.llm_model, messages=messages, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some providers reject response_format - retry once without it.
            if json_mode and "response_format" in str(e).lower():
                kwargs.pop("response_format", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMError(self._explain(e2)) from None
            else:
                raise LLMError(self._explain(e)) from None
        if not resp.choices:
            raise LLMError("The LLM returned an empty response.")
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args if isinstance(args, dict) else {}))
        return LLMMessage(content=msg.content or "", tool_calls=calls)

    def _explain(self, e: Exception) -> str:
        name = type(e).__name__
        text = str(e)
        low = text.lower()
        if "401" in text or "invalid api key" in low or "authentication" in name.lower():
            hint = "The LLM API key was rejected (401). Check LLM_API_KEY."
        elif "404" in text or "model" in low and "not found" in low:
            hint = "Model or endpoint not found. Check LLM_MODEL and LLM_BASE_URL."
        elif "429" in text or "rate" in low:
            hint = "Rate limit or quota exceeded at the LLM provider. Wait a moment or check billing."
        elif "timeout" in low or "timed out" in low:
            hint = "The LLM request timed out. Try again."
        elif "connection" in low:
            hint = "Could not connect to the LLM provider. Check your network and LLM_BASE_URL."
        else:
            hint = f"LLM request failed ({name})."
        return redact(f"{hint} Details: {text[:200]}", self.settings)

    def ping(self) -> tuple[bool, str]:
        """Cheap connectivity test used by the UI status indicator."""
        try:
            self.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=5)
            return True, "LLM connected"
        except LLMError as e:
            return False, str(e)
