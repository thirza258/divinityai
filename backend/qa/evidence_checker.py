"""
Evidence sufficiency check — determines if retrieved chunks are
sufficient to answer the query before generating a response.

An insufficient or unavailable check limits the answer; it does not block it.
"""

import json
import logging

from django.conf import settings

from generation.llm_service import generate

logger = logging.getLogger(__name__)

EVIDENCE_PROMPT = """\
Given these retrieved passages from Quran and Hadith:
{retrieved_chunks}

Can these passages sufficiently answer the question: "{query}"?
Return JSON: {{"sufficient": true/false, "missing_aspect": "..." or null}}"""


def check_evidence_sufficiency(query: str, chunks: list[dict]) -> bool:
    """Check if retrieved chunks are sufficient to answer the query.

    Returns True if sufficient, False otherwise.
    """
    if not chunks:
        return False

    context_lines = []
    for chunk in chunks:
        meta = chunk.get('metadata', {})
        source_tag = meta.get('source_tag', chunk.get('id', ''))
        text_ar = (meta.get('text_ar') or '').strip()
        text_en = (meta.get('text_en') or '').strip()
        passage = f"{text_ar} | {text_en}" if text_ar or text_en else chunk.get('text', '')
        context_lines.append(f"[{source_tag}] {passage}")

    context_str = "\n".join(context_lines)

    evidence_model = getattr(
        settings,
        'OPENROUTER_EVIDENCE_CHECK_MODEL',
        'meta-llama/llama-3.3-70b-instruct',
    )

    try:
        result = generate(
            prompt=EVIDENCE_PROMPT.format(retrieved_chunks=context_str, query=query),
            model=evidence_model,
            temperature=0.1,
        )
        parsed = json.loads(result)
        if not isinstance(parsed, dict) or not isinstance(parsed.get('sufficient'), bool):
            raise ValueError('Missing or invalid evidence verdict')
        sufficient = parsed['sufficient']
        if not sufficient:
            missing = parsed.get('missing_aspect', 'unknown')
            logger.info("Evidence insufficient for query — missing: %s", missing)
        return sufficient
    except Exception as exc:
        logger.warning("Evidence check failed: %s — using limited evidence", exc)
        return False
