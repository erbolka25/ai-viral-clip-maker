"""Provider-neutral LLM access.

The rest of the app calls :func:`complete` and never imports a provider SDK,
so switching between Groq and Anthropic is a one-line change in ``.env``::

    LLM_PROVIDER=groq        # default
    LLM_PROVIDER=anthropic

API keys are read from ``.env`` (via python-dotenv) or the environment and are
never printed or logged. Model names live in ``src/config.py``.

Self-test (asks the model for ``{"ok": true}``)::

    python -m src.llm
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from src import config


class LLMError(RuntimeError):
    """The provider returned an error or an unusable response."""


class LLMConfigError(LLMError):
    """Missing/invalid configuration (provider name, API key, package)."""


@dataclass
class LLMResult:
    """Text plus metadata of one completion."""

    text: str
    provider: str
    model: str
    elapsed_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None  # part of output_tokens, if reported
    rate_limits: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env into os.environ (existing environment variables win)."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise LLMConfigError("python-dotenv is missing: pip install -r requirements.txt") from exc
    load_dotenv(config.ENV_FILE, override=False)


def get_provider() -> str:
    """Return the configured provider name ("groq" or "anthropic")."""
    _load_env()
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or config.LLM_DEFAULT_PROVIDER
    if provider not in config.LLM_MODELS:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(config.LLM_MODELS)}."
        )
    return provider


def _api_key(provider: str) -> str:
    """Read the provider's API key. The value is never included in messages."""
    var = config.LLM_API_KEY_VARS[provider]
    key = os.environ.get(var, "").strip().strip('"').strip("'")
    if not key:
        raise LLMConfigError(
            f"{var} is not set. Add it to {config.ENV_FILE} (see .env.example) and save the file."
        )
    return key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Send ``prompt`` to the configured LLM and return the response text.

    ``json_mode=True`` asks the provider for a single JSON object. Mention
    "JSON" and the expected shape in the prompt as well.
    """
    return complete_with_meta(
        prompt, system=system, json_mode=json_mode, max_tokens=max_tokens
    ).text


def complete_with_meta(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> LLMResult:
    """Same as :func:`complete` but also returns timing, token usage and
    rate-limit headers."""
    provider = get_provider()
    model = config.LLM_MODELS[provider]
    key = _api_key(provider)
    limit = max_tokens or config.LLM_MAX_OUTPUT_TOKENS[provider]

    call = _call_groq if provider == "groq" else _call_anthropic
    t0 = time.perf_counter()
    result = call(key, model, prompt, system, json_mode, limit)
    result.elapsed_seconds = round(time.perf_counter() - t0, 3)
    return result


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _call_groq(
    key: str, model: str, prompt: str, system: str | None, json_mode: bool, max_tokens: int
) -> LLMResult:
    try:
        import groq
    except ImportError as exc:
        raise LLMConfigError("The 'groq' package is missing: pip install -r requirements.txt") from exc

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    if model.startswith("openai/gpt-oss"):
        params["reasoning_effort"] = config.GROQ_REASONING_EFFORT

    client = groq.Groq(api_key=key)  # SDK retries 429/5xx twice by default
    try:
        raw = client.chat.completions.with_raw_response.create(**params)
    except groq.AuthenticationError as exc:
        raise LLMError("Groq rejected GROQ_API_KEY (401). Check the key in .env.") from exc
    except groq.RateLimitError as exc:
        wait = exc.response.headers.get("retry-after", "?")
        raise LLMError(f"Groq rate limit reached; retry after {wait}s. {exc.message}") from exc
    except groq.APIStatusError as exc:
        raise LLMError(f"Groq API error {exc.status_code}: {exc.message}") from exc
    except groq.APIConnectionError as exc:
        raise LLMError(f"Could not reach Groq: {exc}") from exc

    completion = raw.parse()
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise LLMError(f"Response cut off at max_tokens={max_tokens}; raise the limit.")

    usage = completion.usage
    details = getattr(usage, "completion_tokens_details", None)
    return LLMResult(
        text=choice.message.content or "",
        provider="groq",
        model=completion.model or model,
        elapsed_seconds=0.0,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        rate_limits={
            k: v for k, v in raw.headers.items()
            if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
        },
    )


def _call_anthropic(
    key: str, model: str, prompt: str, system: str | None, json_mode: bool, max_tokens: int
) -> LLMResult:
    """Claude via the official SDK. Not exercised yet (no key configured)."""
    try:
        import anthropic
    except ImportError as exc:
        raise LLMConfigError("The 'anthropic' package is missing: pip install anthropic") from exc

    if json_mode:
        rule = "Respond with a single valid JSON object only - no prose, no code fences."
        system = f"{system}\n\n{rule}" if system else rule

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        # If Claude declines on safety grounds, the API retries the request on
        # a suitable fallback model inside the same call.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }
    if system:
        params["system"] = system

    client = anthropic.Anthropic(api_key=key)
    try:
        raw = client.beta.messages.with_raw_response.create(**params)
    except anthropic.AuthenticationError as exc:
        raise LLMError("Anthropic rejected ANTHROPIC_API_KEY (401). Check the key in .env.") from exc
    except anthropic.RateLimitError as exc:
        wait = exc.response.headers.get("retry-after", "?")
        raise LLMError(f"Anthropic rate limit reached; retry after {wait}s.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(f"Could not reach Anthropic: {exc}") from exc

    message = raw.parse()
    if message.stop_reason == "refusal":
        raise LLMError("Claude declined the request (stop_reason=refusal).")
    if message.stop_reason == "max_tokens":
        raise LLMError(f"Response cut off at max_tokens={max_tokens}; raise the limit.")

    return LLMResult(
        text="".join(b.text for b in message.content if b.type == "text"),
        provider="anthropic",
        model=message.model,
        elapsed_seconds=0.0,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        rate_limits={
            k: v for k, v in raw.headers.items()
            if k.lower().startswith("anthropic-ratelimit") or k.lower() == "retry-after"
        },
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    prompt = 'Return exactly this JSON object and nothing else: {"ok": true}'
    try:
        result = complete_with_meta(prompt, json_mode=True, max_tokens=512)
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1

    try:
        parsed = json.loads(result.text)
        parsed_ok = parsed == {"ok": True}
    except json.JSONDecodeError:
        parsed, parsed_ok = None, False

    print(f"Provider:      {result.provider}")
    print(f"Model:         {result.model}")
    print(f"Raw response:  {result.text!r}")
    print(f"JSON parsed:   {parsed is not None}  (equals {{'ok': True}}: {parsed_ok})")
    print(f"Response time: {result.elapsed_seconds:.2f}s")
    print(
        f"Tokens:        input {result.input_tokens}, output {result.output_tokens}"
        f" (reasoning {result.reasoning_tokens}), total {result.total_tokens}"
    )
    print("Rate limits:")
    for k, v in sorted(result.rate_limits.items()):
        print(f"  {k}: {v}")
    return 0 if parsed_ok else 2


if __name__ == "__main__":
    sys.exit(_self_test())
