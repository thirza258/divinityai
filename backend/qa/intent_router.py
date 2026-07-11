"""
Intent router — classify incoming queries using an LLM call.

Returns structured JSON with intent type and confidence score.
"""

import json
import logging

from django.conf import settings

from generation.llm_service import generate

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """\
You are an Islamic query classifier.
Classify the query into exactly one category. Return JSON only, no explanation.

Categories:
- "quran_verse"    → user wants a specific ayah or surah reference
- "hadith"         → user wants a hadith or prophetic narration
- "fiqh"           → user wants Islamic jurisprudence guidance
- "calculation"    → user wants zakat, mirath (inheritance), prayer time math
- "off_domain"     → query is unrelated to Islam or Quran/Hadith

Query: "{query}"

Return: {{"type": "<category>", "confidence": 0.0-1.0}}"""


def classify_intent(query: str) -> dict:
    """Classify the user's query into one of five intent categories.

    Returns a dict with ``type`` and ``confidence`` keys.
    Falls back to ``{"type": "hadith", "confidence": 0.5}`` on parse failure.
    """
    classifier_model = getattr(
        settings,
        'OPENROUTER_CLASSIFIER_MODEL',
        'google/gemini-2.5-flash',
    )

    try:
        result = generate(
            prompt=INTENT_SYSTEM_PROMPT.format(query=query),
            model=classifier_model,
            temperature=0.1,
        )
        parsed = json.loads(result)
        if 'type' in parsed and 'confidence' in parsed:
            return {'type': parsed['type'], 'confidence': float(parsed['confidence'])}
        raise ValueError(f"Missing keys in response: {parsed}")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Intent classification failed: %s — falling back to 'hadith'", exc)
        return {'type': 'hadith', 'confidence': 0.5}