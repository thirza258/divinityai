"""
Intent router — classify incoming queries using an LLM call.

Returns structured JSON with intent type and confidence score.
"""

import json
import logging
import math

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
    Falls back to a low-confidence retrieval route if classification fails.
    """
    classifier_model = getattr(
        settings,
        'OPENROUTER_CLASSIFIER_MODEL',
        'google/gemini-2.5-flash',
    )

    result = None
    try:
        result = generate(
            prompt=INTENT_SYSTEM_PROMPT.format(query=query),
            model=classifier_model,
            temperature=0.1,
        )
        parsed = json.loads(result)
        if not isinstance(parsed, dict) or parsed.get('type') not in {
            'quran_verse', 'hadith', 'fiqh', 'calculation', 'off_domain',
        }:
            raise ValueError('Missing or invalid intent')
        if isinstance(parsed.get('confidence'), bool):
            raise ValueError('Invalid intent confidence')
        confidence = float(parsed['confidence'])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid intent confidence')
        return {'type': parsed['type'], 'confidence': confidence}
    except Exception as exc:
        logger.warning(
            "Intent classification failed: %s — falling back to 'hadith'. "
            "Raw response (first 200 chars): %s",
            exc,
            result[:200] if result else '<empty>',
        )
        return {'type': 'hadith', 'confidence': 0.0}
