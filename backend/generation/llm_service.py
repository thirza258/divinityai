"""
LLM service — thin wrapper around LangChain's ChatOpenAI pointed at OpenRouter.

OpenRouter provides a unified API for many models.  We configure
ChatOpenAI with the OpenRouter base URL and let callers specify which
model to use at invocation time.
"""

import logging
import os
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

# ---------------------------------------------------------------------------
# DashScope (Alibaba Cloud) — Stage 1 of the two-stage pipeline
# ---------------------------------------------------------------------------

DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

DASHSCOPE_DEFAULT_MODEL = os.getenv(
    "DASHSCOPE_STAGE_MODEL",
    "deepseek-flash",
)

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


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
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
    stream:
        If ``True``, return the async generator / stream iterator so the
        caller can consume tokens as they arrive.
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )

    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    logger.info("Generating with model=%s  temperature=%s", model or OPENROUTER_DEFAULT_MODEL, temperature)

    if stream:
        return llm.stream(messages)

    response = llm.invoke(messages)
    return response.content


def generate_with_history(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: str | None = None,
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
    stream:
        If ``True``, return the stream iterator.
    """
    llm = get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
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
        return llm.stream(langchain_messages)

    response = llm.invoke(langchain_messages)
    return response.content


def generate_dashscope(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    stream: bool = False,
) -> str | Any:
    """Send a prompt to DashScope (DeepSeek Flash) — Stage 1 of the pipeline.

    Thin wrapper around :func:`generate` that targets the DashScope
    OpenAI-compatible endpoint.  Uses ``DASHSCOPE_API_KEY`` and
    ``DASHSCOPE_DEFAULT_MODEL`` by default.

    Parameters
    ----------
    prompt:
        The user message / prompt text.
    system:
        Optional system message to set behaviour / context.
    model:
        Model name override.  Defaults to ``DASHSCOPE_STAGE_MODEL`` env var
        (``deepseek-flash``).
    temperature:
        Sampling temperature.
    max_tokens:
        Max completion tokens.
    stream:
        If ``True``, return the stream iterator.
    """
    return generate(
        prompt=prompt,
        system=system,
        model=model or DASHSCOPE_DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        stream=stream,
    )
