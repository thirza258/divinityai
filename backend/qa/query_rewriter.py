"""
Query rewriting — HyDE and sub-query decomposition.

Generates richer query variants before retrieval to improve recall.
"""

import json
import logging

from django.conf import settings

from generation.llm_service import generate

logger = logging.getLogger(__name__)

HYDE_PROMPT = """\
You are a Quran and Hadith scholar.
Given the question: "{query}"
Write a short hypothetical passage (2–3 sentences) that would appear in an
authentic Hadith or Quranic tafsir that directly answers this question.
Respond in the same language as the query. Arabic terms are welcome."""

SUBQUERY_PROMPT = """\
Decompose this Islamic jurisprudence question into 2–3 atomic sub-questions,
each answerable from Quran or Hadith independently.
Return as JSON array of strings.
Query: "{query}\""""


def _get_model() -> str:
    return getattr(settings, 'OPENROUTER_CLASSIFIER_MODEL', 'google/gemini-2.5-flash')


def generate_hyde(query: str) -> str:
    """Generate a hypothetical document for HyDE retrieval."""
    try:
        return str(generate(
            prompt=HYDE_PROMPT.format(query=query),
            model=_get_model(),
            temperature=0.3,
        ))
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)
        return ""


def decompose_fiqh(query: str) -> list[str]:
    """Decompose a fiqh query into sub-questions."""
    try:
        result = generate(
            prompt=SUBQUERY_PROMPT.format(query=query),
            model=_get_model(),
            temperature=0.3,
        )
        parsed = json.loads(result)
        if isinstance(parsed, list):
            return parsed[:3]
    except (json.JSONDecodeError, TypeError, Exception) as exc:
        logger.warning("Sub-query decomposition failed: %s", exc)
    return []


def rewrite_queries(query: str, intent: str) -> dict:
    """Return dict with ``hyde`` and ``sub_queries`` lists."""
    result = {'hyde': [], 'sub_queries': []}

    # Always generate HyDE
    hyde = generate_hyde(query)
    if hyde:
        result['hyde'] = [hyde]

    # Sub-query decomposition for fiqh
    if intent == 'fiqh':
        subs = decompose_fiqh(query)
        if subs:
            result['sub_queries'] = subs

    return result