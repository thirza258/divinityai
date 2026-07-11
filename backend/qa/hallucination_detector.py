"""
Post-generation hallucination detection.

LLM-based check that verifies every Quran/Hadith citation in the
generated answer against the actual source passages.
"""

import json
import logging

from django.conf import settings

from generation.llm_service import generate

logger = logging.getLogger(__name__)

HALLUCINATION_PROMPT = """\
Check every Quran verse reference and Hadith citation in this answer.
Compare each against the provided source passages.
Return JSON:
{{
  "hallucinated": true/false,
  "flagged_spans": [{{"text": "...", "reason": "..."}}]
}}

Answer: {answer}
Source passages: {context}"""


def detect_hallucinations(answer: str, context_chunks: list[dict]) -> dict:
    """Check if the generated answer contains hallucinated citations.

    Returns a dict with ``hallucinated`` (bool) and ``flagged_spans`` (list).
    """
    if not answer or not context_chunks:
        return {'hallucinated': False, 'flagged_spans': []}

    context_lines = []
    for chunk in context_chunks:
        meta = chunk.get('metadata', {})
        source_tag = meta.get('source_tag', chunk.get('id', ''))
        text_ar = meta.get('text_ar', '')
        context_lines.append(f"[{source_tag}] {text_ar}")

    context_str = "\n".join(context_lines)

    hallucination_model = getattr(
        settings,
        'OPENROUTER_HALLUCINATION_MODEL',
        'meta-llama/llama-3.3-70b-instruct',
    )

    try:
        result = generate(
            prompt=HALLUCINATION_PROMPT.format(answer=answer, context=context_str),
            model=hallucination_model,
            temperature=0.1,
        )
        parsed = json.loads(result)
        return {
            'hallucinated': bool(parsed.get('hallucinated', False)),
            'flagged_spans': parsed.get('flagged_spans', []),
        }
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Hallucination detection failed: %s", exc)
        return {'hallucinated': False, 'flagged_spans': []}