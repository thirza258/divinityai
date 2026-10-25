"""
LLM service — thin wrapper around LangChain's ChatOpenAI pointed at OpenRouter.

OpenRouter provides a unified API for many models.  We configure
ChatOpenAI with the OpenRouter base URL and let callers specify which
model to use at invocation time.
"""

import logging
import os
import random
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via environment variables)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Default model — a strong, widely-available chat model on OpenRouter.
# Examples: "openai/gpt-4o", "anthropic/claude-sonnet-4-6",
#           "google/gemini-2.5-flash", "meta-llama/llama-4-maverick"
OPENROUTER_DEFAULT_MODEL = os.getenv(
    "OPENROUTER_DEFAULT_MODEL",
    "openai/gpt-4o",
)

# Additional HTTP headers OpenRouter recognises (used by ChatOpenAI's
# default_headers parameter).
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "DivinityAI")

# Request timeout (seconds) and retry count for LLM calls.  Without a
# timeout a hung provider connection blocks the request worker forever.
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "45"))
OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "1"))

# Free fallback models when OpenRouter account credits are exhausted
FREE_OPENROUTER_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "cohere/north-mini-code:free",
    "recursive/flint-v1.0:free",
    "xingchen-agi/xing4.0-29b-a4b:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
]


def get_shuffled_free_models() -> list[str]:
    models = list(FREE_OPENROUTER_MODELS)
    random.shuffle(models)
    return models


def is_credit_exhausted_error(exc_or_status: Any, message: str = "") -> bool:
    code = None
    if isinstance(exc_or_status, int):
        code = exc_or_status
    else:
        for attr in ("status_code", "http_status", "code"):
            val = getattr(exc_or_status, attr, None)
            if isinstance(val, int):
                code = val
                break
        if code is None and hasattr(exc_or_status, "response"):
            resp = getattr(exc_or_status, "response", None)
            code = getattr(resp, "status_code", None)

    msg_str = f"{exc_or_status} {message}".lower()
    if code == 402:
        return True

    if "rate limit" in msg_str and not any(k in msg_str for k in ("credit", "payment", "balance")):
        return False

    credit_keywords = (
        "402",
        "credit",
        "payment required",
        "run out of credits",
        "insufficient balance",
        "insufficient credits",
        "insufficient_quota",
        "exceeded your current quota",
        "out of credit",
    )
    return any(kw in msg_str for kw in credit_keywords)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_llm: BaseChatModel | None = None


def get_llm(
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Return a configured :class:`ChatOpenAI` instance.

    Parameters
    ----------
    model:
        Model name in OpenRouter format, e.g. ``"openai/gpt-4o"``.
        Defaults to :data:`OPENROUTER_DEFAULT_MODEL`.
    temperature:
        Sampling temperature (0–2).  Default 0.7.
    max_tokens:
        Max tokens to generate.  ``None`` lets the model decide.
    api_key:
        OpenRouter API key.  Defaults to :data:`OPENROUTER_API_KEY`.
    base_url:
        Base URL for the OpenAI-compatible endpoint.
        Defaults to :data:`OPENROUTER_BASE_URL`.
    """
    model_name = model or OPENROUTER_DEFAULT_MODEL
    key = api_key or OPENROUTER_API_KEY
    url = base_url or OPENROUTER_BASE_URL

    if not key:
        logger.warning(
            "OPENROUTER_API_KEY is not set — LLM calls will fail until a key "
            "is provided via environment or the api_key parameter."
        )

    extra_headers: dict[str, str] = {}
    if OPENROUTER_HTTP_REFERER:
        extra_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        extra_headers["X-Title"] = OPENROUTER_APP_TITLE

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=key,
        base_url=url,
        default_headers=extra_headers if extra_headers else None,
        timeout=OPENROUTER_TIMEOUT,
        max_retries=OPENROUTER_MAX_RETRIES,
    )
    return llm


def get_cached_llm(
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Like :func:`get_llm` but caches and reuses the instance globally.

    Use this when you want a single shared client across the process.
    """
    # pylint: disable=global-statement
    global _llm
    if _llm is not None:
        return _llm

    _llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )
    return _llm


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content to a plain string.

    ``response.content`` is typed ``str | list`` — some models return a
    list of content blocks.  Callers (and the API contract) expect ``str``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    stream: bool = False,
) -> str | Any:
    """Send a prompt to the LLM and return the generated text.

    Parameters
    ----------
    prompt:
        The user message / prompt text.
    system:
        Optional system message to set behaviour / context.
    model:
        Model name override.
    temperature:
        Sampling temperature.
    max_tokens:
        Max completion tokens.
    api_key:
        API key override.
    base_url:
        Base URL override for the API endpoint.
    stream:
        If ``True``, return the async generator / stream iterator so the
        caller can consume tokens as they arrive.
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )

    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    logger.info("Generating with model=%s  temperature=%s", model or OPENROUTER_DEFAULT_MODEL, temperature)
    print(f"[llm] generating with model={model or OPENROUTER_DEFAULT_MODEL} temperature={temperature}", flush=True)

    if stream:
        try:
            return llm.stream(messages)
        except Exception as exc:
            if is_credit_exhausted_error(exc):
                logger.warning("OpenRouter credits exhausted, falling back to shuffled free models")
                for free_model in get_shuffled_free_models():
                    try:
                        fallback_llm = get_llm(
                            model=free_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_key=api_key,
                            base_url=base_url,
                        )
                        return fallback_llm.stream(messages)
                    except Exception:
                        continue
            raise

    t0 = time.time()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        if is_credit_exhausted_error(exc):
            logger.warning("OpenRouter credits exhausted (%s), falling back to shuffled free models", exc)
            for free_model in get_shuffled_free_models():
                try:
                    logger.info("Retrying generation with free fallback model: %s", free_model)
                    fallback_llm = get_llm(
                        model=free_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    response = fallback_llm.invoke(messages)
                    content = _content_to_text(response.content)
                    if content:
                        logger.info("Fallback succeeded with free model: %s", free_model)
                        return content
                except Exception as fb_exc:
                    logger.warning("Free model %s failed: %s", free_model, fb_exc)
                    continue
        raise
    elapsed = time.time() - t0
    content = _content_to_text(response.content)
    if not content:
        # finish_reason distinguishes the causes that all look alike from
        # here: a provider content filter, a length cut-off, or a reasoning-
        # only completion.  Without it an empty answer is undiagnosable.
        metadata = getattr(response, "response_metadata", {})
        logger.warning(
            "LLM returned empty content (model=%s, elapsed=%.2fs, "
            "finish_reason=%s, metadata=%s). "
            "Check API key validity, rate limits, and model availability.",
            model or OPENROUTER_DEFAULT_MODEL,
            elapsed,
            metadata.get("finish_reason"),
            metadata,
        )
        print(
            f"[llm] EMPTY content from {model or OPENROUTER_DEFAULT_MODEL} — "
            f"finish_reason={metadata.get('finish_reason')} metadata={metadata}",
            flush=True,
        )
    print(f"[llm] generation completed in {elapsed:.2f}s", flush=True)
    logger.info("generation completed in %.2fs", elapsed)
    return content


def generate_with_history(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    stream: bool = False,
) -> str | Any:
    """Send a full message history to the LLM.

    Parameters
    ----------
    messages:
        List of dicts with ``role`` and ``content`` keys, e.g.
        ``[{"role": "system", "content": "You are …"},
          {"role": "user", "content": "Hello"}]``
    model:
        Model name override.
    temperature:
        Sampling temperature.
    max_tokens:
        Max completion tokens.
    api_key:
        API key override.
    base_url:
        Base URL override for the API endpoint.
    stream:
        If ``True``, return the stream iterator.
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )

    langchain_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        else:
            langchain_messages.append(HumanMessage(content=content))

    logger.info(
        "Generating with history (%d messages)  model=%s",
        len(messages),
        model or OPENROUTER_DEFAULT_MODEL,
    )

    if stream:
        try:
            return llm.stream(langchain_messages)
        except Exception as exc:
            if is_credit_exhausted_error(exc):
                logger.warning("OpenRouter credits exhausted, falling back to shuffled free models")
                for free_model in get_shuffled_free_models():
                    try:
                        fallback_llm = get_llm(
                            model=free_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_key=api_key,
                            base_url=base_url,
                        )
                        return fallback_llm.stream(langchain_messages)
                    except Exception:
                        continue
            raise

    try:
        response = llm.invoke(langchain_messages)
        return _content_to_text(response.content)
    except Exception as exc:
        if is_credit_exhausted_error(exc):
            logger.warning("OpenRouter credits exhausted (%s), falling back to shuffled free models", exc)
            for free_model in get_shuffled_free_models():
                try:
                    logger.info("Retrying generate_with_history with free fallback model: %s", free_model)
                    fallback_llm = get_llm(
                        model=free_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    response = fallback_llm.invoke(langchain_messages)
                    content = _content_to_text(response.content)
                    if content:
                        logger.info("Fallback succeeded with free model: %s", free_model)
                        return content
                except Exception as fb_exc:
                    logger.warning("Free model %s failed: %s", free_model, fb_exc)
                    continue
        raise


