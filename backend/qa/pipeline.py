"""
Pipeline orchestrator — wires together all RAG stages.

Phase 1: direct retrieval → RRF → citation verify → grounded generation
Phase 2: intent → scope → rewrite → retrieve → verify → check → generate → safety
"""

import logging
import time
from pathlib import Path
from typing import Optional

from django.conf import settings

from corpus.bm25_index import BM25Index
from retrieval.hybrid_retriever import retrieve_from_all_corpora
from retrieval.citation_verifier import verify_chunks, load_canonical_corpus
from retrieval.dense_rag import QURAN_COLLECTION, HADITH_COLLECTION
from generation.llm_service import generate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded resources (loaded once per process)
# ---------------------------------------------------------------------------

_bm25_quran: Optional[BM25Index] = None
_bm25_hadith: Optional[BM25Index] = None
_canonical_loaded: bool = False


def _get_bm25_index_dir() -> Path:
    return Path(getattr(settings, 'BM25_INDEX_DIR', Path.home() / '.divinityai' / 'bm25_indexes'))


def _load_indexes() -> None:
    """Load BM25 indexes and canonical corpus at first request."""
    global _bm25_quran, _bm25_hadith, _canonical_loaded

    if _bm25_quran is not None:
        return

    index_dir = _get_bm25_index_dir()

    _bm25_quran = BM25Index(index_dir)
    try:
        _bm25_quran.load('quran_collection')
    except FileNotFoundError:
        logger.warning("BM25 Quran index not found at %s", index_dir / 'quran_collection.pkl')

    _bm25_hadith = BM25Index(index_dir)
    try:
        _bm25_hadith.load('hadith_collection')
    except FileNotFoundError:
        logger.warning("BM25 Hadith index not found at %s", index_dir / 'hadith_collection.pkl')

    # Load canonical corpus from ChromaDB for citation verification
    if not _canonical_loaded:
        _load_canonical_corpus()
        _canonical_loaded = True


def _load_canonical_corpus() -> None:
    """Load canonical texts from ChromaDB collections for citation verification."""
    try:
        from chroma.chroma_utils import get_or_create_collection
        from retrieval.dense_rag import OllamaEmbeddingFunction, _normalize_metadata

        emb_fn = OllamaEmbeddingFunction()

        for coll_name in [QURAN_COLLECTION, HADITH_COLLECTION]:
            try:
                collection = get_or_create_collection(name=coll_name, embedding_function=emb_fn)
                docs = collection.get()
                if docs.get('ids'):
                    records = []
                    for i, doc_id in enumerate(docs['ids']):
                        meta = docs['metadatas'][i] if docs.get('metadatas') else {}
                        normalized = _normalize_metadata(meta, coll_name)
                        records.append({
                            'source_tag': normalized.get('source_tag', ''),
                            'text_ar': normalized.get('text_ar', ''),
                        })
                    load_canonical_corpus(records)
                    logger.info("Loaded canonical corpus from '%s' (%d docs)", coll_name, len(records))
            except Exception as exc:
                logger.warning("Could not load canonical corpus from '%s': %s", coll_name, exc)
    except Exception as exc:
        logger.warning("Could not load canonical corpus: %s", exc)


# ---------------------------------------------------------------------------
# Grounded generation prompt
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are an Islamic knowledge assistant. Your sole purpose is to answer
questions using ONLY the Quran and Hadith passages provided to you.

RULES:
1. Answer ONLY from the provided context. Do not add any external knowledge.
2. Cite every Quranic reference as [Q surah:ayah], e.g. [Q 2:255]
3. Cite every Hadith as [C collection/number], e.g. [C Bukhari/52]
4. If the answer is not found in the provided context, respond with:
   "I do not have a grounded source for this in the provided passages."
5. Do not issue fatwas or definitive rulings. Present what the sources say.
6. If the question involves sensitive jurisprudence, add:
   "For a definitive ruling, please consult a qualified scholar."
7. Respond in the same language as the user's question.

Context:
{context}"""


# ---------------------------------------------------------------------------
# Pipeline Service
# ---------------------------------------------------------------------------

class PipelineService:
    """Stateless orchestrator — each call runs the full pipeline."""

    def __init__(self, phase: int = 1):
        self.phase = phase

    def run(self, query: str, language: str = 'en', max_sources: int = 5) -> dict:
        """Run the RAG pipeline and return a response dict."""
        _load_indexes()
        start = time.time()
        pipeline_meta = {
            'phase': self.phase,
            'llm_calls': 0,
            'retrieval_iterations': 1,
        }

        # --- Phase 2: Intent Router + Scope Guard ---
        intent = 'general'
        if self.phase >= 2:
            from .intent_router import classify_intent
            from .scope_guard import check_scope

            intent_result = classify_intent(query)
            pipeline_meta['llm_calls'] += 1
            intent = intent_result['type']
            confidence = intent_result['confidence']

            scope_check = check_scope(intent, confidence)
            if not scope_check['allowed']:
                return {
                    'query': query,
                    'intent': intent,
                    'answer': scope_check['message'],
                    'sources': [],
                    'citations': [],
                    'safety': {
                        'hallucination_detected': False,
                        'flagged_spans': [],
                        'fatwa_boundary_triggered': False,
                        'disclaimer': None,
                    },
                    'pipeline_meta': {
                        **pipeline_meta,
                        'elapsed': round(time.time() - start, 3),
                    },
                }

        # --- Phase 2: Query Rewriting ---
        query_variants = [query]
        if self.phase >= 2 and intent != 'quran_verse':
            from .query_rewriter import rewrite_queries
            additional = rewrite_queries(query, intent)
            pipeline_meta['llm_calls'] += 1
            query_variants = [query] + additional.get('hyde', []) + additional.get('sub_queries', [])

        # --- Hybrid Retrieval ---
        fused = retrieve_from_all_corpora(
            query_variants=query_variants,
            bm25_quran=_bm25_quran,
            bm25_hadith=_bm25_hadith,
            bm25_k=10,
            dense_k=10,
            top_n=10,
        )

        # --- Citation Verification ---
        verified = verify_chunks(fused)

        # --- Phase 2: Evidence Sufficiency Check ---
        evidence_sufficient = True
        if self.phase >= 2 and intent == 'fiqh':
            from .evidence_checker import check_evidence_sufficiency
            evidence_sufficient = check_evidence_sufficiency(query, verified)
            pipeline_meta['llm_calls'] += 1
            # Simplified: single check; loop logic could be added in future

        # --- Grounded Generation ---
        answer = self._generate(query, verified, language)
        pipeline_meta['llm_calls'] += 1

        # --- Phase 2: Safety Layer ---
        safety = {
            'hallucination_detected': False,
            'flagged_spans': [],
            'fatwa_boundary_triggered': False,
            'disclaimer': None,
        }
        if self.phase >= 2:
            from .hallucination_detector import detect_hallucinations
            h_result = detect_hallucinations(answer, verified)
            pipeline_meta['llm_calls'] += 1
            safety['hallucination_detected'] = h_result.get('hallucinated', False)
            safety['flagged_spans'] = h_result.get('flagged_spans', [])

            from .fatwa_boundary import check_fatwa_boundary
            fb_result = check_fatwa_boundary(answer)
            safety['fatwa_boundary_triggered'] = fb_result['triggered']
            safety['disclaimer'] = fb_result.get('disclaimer')

        # --- Assemble response ---
        sources = verified[:max_sources]
        citations = list(set(
            chunk.get('metadata', {}).get('source_tag', '')
            for chunk in sources
            if chunk.get('metadata', {}).get('source_tag')
        ))

        source_serialized = []
        for chunk in sources:
            meta = chunk.get('metadata', {})
            source_serialized.append({
                'source_tag': meta.get('source_tag', chunk.get('id', '')),
                'corpus': meta.get('corpus', 'quran'),
                'text_ar': meta.get('text_ar', ''),
                'text_en': meta.get('text_en', ''),
                'verification_status': chunk.get('verification_status', 'unknown'),
                'retrieval_score': chunk.get('rrf_score', 0),
            })

        return {
            'query': query,
            'intent': intent,
            'answer': answer,
            'sources': source_serialized,
            'citations': citations,
            'safety': safety,
            'pipeline_meta': {
                **pipeline_meta,
                'elapsed': round(time.time() - start, 3),
            },
        }

    def _generate(self, query: str, context_chunks: list[dict], language: str) -> str:
        """Build the grounded generation prompt from context and generate."""
        if not context_chunks:
            return "I do not have a grounded source for this in the provided passages."

        context_lines = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get('metadata', {})
            source_tag = meta.get('source_tag', chunk.get('id', ''))
            text_ar = meta.get('text_ar', '')
            text_en = meta.get('text_en', '')
            context_lines.append(
                f"[Source {i}] ({source_tag})\n"
                f"Arabic: {text_ar}\n"
                f"English: {text_en}"
            )

        context_str = "\n---\n".join(context_lines)
        system_prompt = GENERATION_SYSTEM_PROMPT.format(context=context_str)

        generation_model = getattr(
            settings,
            'OPENROUTER_GENERATION_MODEL',
            'google/gemini-2.5-flash',
        )

        return generate(
            prompt=query,
            system=system_prompt,
            model=generation_model,
            temperature=0.3,
        )