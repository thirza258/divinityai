"""
Citation verification — ensures every cited verse/hadith in a generated
answer actually exists in the canonical corpus at the exact reference claimed.

Cascade (deterministic, no LLM for steps 1-3):
1. Exact string match
2. Diacritic-normalized match
3. Fuzzy match (rapidfuzz, threshold 85%)
4. Semantic LLM fallback (Phase 2, optional)
"""

import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from corpus.arabic_utils import normalize_arabic

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    status: str   # "exact" | "normalized" | "fuzzy" | "hallucinated"
    score: float


# Lookup: source_tag → canonical text
# Populated at startup from ChromaDB documents
_canonical_corpus: dict[str, str] = {}


def load_canonical_corpus(documents: list[dict]) -> None:
    """Load canonical texts keyed by source_tag.

    Each document dict should have a ``source_tag`` and either
    ``text_ar`` or ``text`` key.
    """
    global _canonical_corpus
    for doc in documents:
        tag = doc.get('source_tag', '')
        if not tag:
            meta = doc.get('metadata', {})
            tag = meta.get('source_tag', '')
        text = doc.get('text_ar') or doc.get('text', '')
        if tag and text:
            _canonical_corpus[tag] = text
    logger.info("Loaded canonical corpus with %d entries", len(_canonical_corpus))


def verify_citation(cited_text: str, source_tag: str) -> VerificationResult:
    """Verify a cited text against the canonical corpus.

    Returns a :class:`VerificationResult` with status and confidence score.
    """
    canonical = _canonical_corpus.get(source_tag)
    if canonical is None:
        logger.warning("Source tag not found in canonical corpus: %s", source_tag)
        return VerificationResult(status="hallucinated", score=0.0)

    # Step 1: Exact match
    if cited_text == canonical:
        return VerificationResult(status="exact", score=1.0)

    # Step 2: Diacritic-normalized match
    if normalize_arabic(cited_text) == normalize_arabic(canonical):
        return VerificationResult(status="normalized", score=0.95)

    # Step 3: Fuzzy match on normalized text (rapidfuzz)
    ratio = fuzz.ratio(normalize_arabic(cited_text), normalize_arabic(canonical))
    if ratio >= 85:
        return VerificationResult(status="fuzzy", score=ratio / 100.0)

    # Step 4: Semantic LLM fallback (Phase 2, optional — not implemented here)
    # The pipeline can call an LLM-based semantic check if needed.

    return VerificationResult(status="hallucinated", score=0.0)


def verify_chunks(chunks: list[dict]) -> list[dict]:
    """Verify each chunk's source tag against the canonical corpus.

    Each chunk dict should have a ``source_tag`` key (in metadata or
    top-level) and text content.  Returns the chunks with added
    ``verification_status`` and ``verification_score`` keys.

    Chunks without a source_tag get status ``"unknown"``.
    """
    for chunk in chunks:
        meta = chunk.get('metadata', {})
        source_tag = meta.get('source_tag', chunk.get('source_tag', ''))
        text = meta.get('text_ar') or chunk.get('text', '')

        if source_tag:
            result = verify_citation(text, source_tag)
            chunk['verification_status'] = result.status
            chunk['verification_score'] = result.score
        else:
            chunk['verification_status'] = 'unknown'
            chunk['verification_score'] = 0.0

    return chunks


def get_hallucinated(chunks: list[dict]) -> list[str]:
    """Return the IDs of chunks that failed verification."""
    return [c['id'] for c in chunks if c.get('verification_status') == 'hallucinated']