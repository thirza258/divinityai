"""
Evidence sufficiency check — determines if retrieved chunks are
sufficient to answer the query before generating a response.

If insufficient, the pipeline can trigger a re-retrieval loop.
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
        text_en = meta.get('text_en', '')
        context_lines.append(f"[{source_tag}] {text_en}")

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
        sufficient = parsed.get('sufficient', True)
        if not sufficient:
            missing = parsed.get('missing_aspect', 'unknown')
            logger.info("Evidence insufficient for query — missing: %s", missing)
        return bool(sufficient)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Evidence check failed: %s — assuming sufficient", exc)
        return True