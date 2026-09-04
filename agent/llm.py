"""Provider-neutral LLM client abstraction (Phase 4).

Deliberately dependency-free: OpenAI-compatible chat-completions endpoints
(Ollama, LM Studio, llama.cpp server, vLLM, or cloud providers) are called
with the standard library. The investigation graph only ever sees the
``LLMClient`` protocol, so the provider/model can change without touching
schemas, tools, or graph state.

The client is capability-free: it can only turn a prompt into text. It has
no tools, no filesystem paths, and no data access of its own.

API keys are never logged, serialized, or included in error messages.
"""

import json
import os
import time
import urllib.request
from typing import Protocol

from pydantic import BaseModel


class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 30.0
    seed: int | None = 42


class LLMResponse(BaseModel):
    ok: bool
    text: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    error: str | None = None


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class DisabledLLM:
    """Default client when no LLM is configured. Always fails closed."""

    provider = "disabled"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(ok=False, provider=self.provider, error="llm disabled")


class OpenAICompatibleChatLLM:
    """Thin client for any OpenAI-compatible /chat/completions endpoint."""

    provider = "openai-compatible"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=max(request.timeout_s, self.timeout_s)) as response:
                body = json.loads(response.read().decode("utf-8"))
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = body["choices"][0]["message"]["content"]
            return LLMResponse(
                ok=True, text=text, model=self.model, provider=self.provider, latency_ms=latency_ms
            )
        except Exception as exc:  # any failure becomes a safe fallback signal
            latency_ms = int((time.perf_counter() - started) * 1000)
            return LLMResponse(
                ok=False,
                model=self.model,
                provider=self.provider,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )


class ScriptedLLM:
    """Deterministic fake client for tests.

    Returns queued items in order; a queued ``Exception`` simulates an LLM
    failure (timeout, connection error, ...). Records every request so tests
    can inspect exactly what the LLM would have received.
    """

    provider = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            return LLMResponse(ok=False, provider=self.provider, error="scripted queue empty")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            return LLMResponse(
                ok=False, model="scripted", provider=self.provider, error=f"{type(item).__name__}: {item}"
            )
        return LLMResponse(ok=True, text=item, model="scripted", provider=self.provider, latency_ms=0)


def make_llm_from_env(environ: dict | None = None) -> LLMClient:
    """Build the configured client; DisabledLLM when unset (deterministic default)."""
    env = os.environ if environ is None else environ
    base_url = (env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("LLM_MODEL") or "").strip()
    if not base_url or not model:
        return DisabledLLM()
    api_key = (env.get("LLM_API_KEY") or "").strip() or None
    try:
        timeout_s = float((env.get("LLM_TIMEOUT_S") or "").strip() or 30.0)
    except ValueError:
        timeout_s = 30.0
    return OpenAICompatibleChatLLM(base_url=base_url, model=model, api_key=api_key, timeout_s=timeout_s)
