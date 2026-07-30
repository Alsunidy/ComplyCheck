"""
engine/llm_client.py

Shared OpenAI client with an automatic model fallback chain.

Every LLM call in the product (the compliance judge and the chat assistant)
goes through generate_json(), so the model choice, JSON-mode handling, retry
policy and rate-limit backoff live in exactly one place.

The fallback chain exists because a single model can become unavailable
mid-run: a per-model rate limit, a deprecated/retired model id, or an
account that lacks access to a specific tier. Rather than failing the whole
36-control audit, calls walk the chain until one model answers.
"""
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Highest-priority model first. gpt-4.1-mini is the default judge: it follows
# strict JSON schemas reliably, supports vision (needed for screenshot
# evidence), and costs a fraction of the full-size models -- which matters
# because one audit run makes ~6 batched calls plus one call per screenshot.
FALLBACK_CHAIN = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1",
]

# "Please try again in 1.5s" / "try again in 200ms"
_RETRY_SECONDS_RE = re.compile(r"try again in ([0-9.]+)(ms|s)", re.IGNORECASE)


def _parse_retry_delay(message: str) -> float | None:
    match = _RETRY_SECONDS_RE.search(message)
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "ms" else value


def _model_chain(preferred: str | None) -> list[str]:
    chain = ([preferred] if preferred else []) + FALLBACK_CHAIN
    return list(dict.fromkeys(chain))


def _is_fatal_account_error(message: str) -> bool:
    """Errors no retry or model swap can fix -- fail fast with a clear message."""
    return "insufficient_quota" in message or "invalid_api_key" in message


def generate_json(contents, system_instruction: str | None = None,
                  preferred_model: str | None = None,
                  return_model: bool = False):
    """Call OpenAI expecting a JSON object response.

    `contents` is either a plain string or a list of OpenAI content parts
    (used for screenshots: [{"type": "text", ...}, {"type": "image_url", ...}]).

    Returns the raw JSON text, or (text, model_name) when `return_model` is
    true so callers can attribute output quality to the model that actually
    served the request (fallbacks can change it mid-run).

    Raises RuntimeError when the whole chain is exhausted, or immediately on
    account-level errors (no credit / bad key) where retrying is pointless.
    """
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project "
            "root (OPENAI_API_KEY=sk-...) before running."
        )

    client = OpenAI()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": contents})

    last_error = None
    for model in _model_chain(preferred_model):
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                text = response.choices[0].message.content
                if not text:
                    raise ValueError(f"Empty response from {model}")
                return (text, model) if return_model else text
            except Exception as exc:
                message = str(exc)
                last_error = exc

                if _is_fatal_account_error(message):
                    raise RuntimeError(
                        "OpenAI rejected the request because the account has no "
                        "available quota (or the API key is invalid). Add credit at "
                        "https://platform.openai.com/settings/organization/billing "
                        f"and try again. Original error: {message[:200]}"
                    ) from exc

                # A missing/retired model or one this account can't use: no
                # point retrying it, move down the chain.
                if "model_not_found" in message or "does not exist" in message:
                    break

                delay = _parse_retry_delay(message)
                time.sleep(delay + 0.5 if delay else 2 * (attempt + 1))

    raise RuntimeError(f"All OpenAI models in the fallback chain failed: {last_error}")
