"""
Pipeline orchestrator — wires together all RAG stages.

Phase 1: direct retrieval → citation verify → grounded generation
Phase 2: intent → scope → rewrite → retrieve → verify → check → generate → safety
"""

import logging
import time

from django.conf import settings

from retrieval.citation_verifier import verify_chunks, load_canonical_corpus
from retrieval.dense_rag import (
    QURAN_COLLECTION,
    HADITH_COLLECTION,
    retrieve_dense_all_corpora,
    _normalize_metadata,
)
from generation.llm_service import generate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded resources (loaded once per process)
# ---------------------------------------------------------------------------

_canonical_loaded: bool = False


def _load_canonical() -> None:
    """Load canonical corpus from ChromaDB for citation verification."""
    global _canonical_loaded

    if _canonical_loaded:
        return

    # Only latch on success — a failed load is retried on the next request
    # instead of leaving citation verification blind for the process lifetime.
    if _load_canonical_corpus():
        _canonical_loaded = True


# Rows per ChromaDB fetch when loading the canonical corpus.
CANONICAL_PAGE_SIZE = 5000


def _load_collection_canonical(collection, coll_name: str) -> int:
    """Load one collection's canonical texts, a page at a time.

    Fetches metadatas only — ``source_tag`` and ``text_ar`` both come from
    metadata, so pulling every document body as well would double the cost
    of a load that already blocks the first request of each worker.

    Returns the number of rows read.
    """
    total = 0
    offset = 0
    while True:
        page = collection.get(
            include=["metadatas"],
            limit=CANONICAL_PAGE_SIZE,
            offset=offset,
        )
        ids = page.get('ids') or []
        if not ids:
            break

        metadatas = page.get('metadatas') or []
        records = []
        for i in range(len(ids)):
            meta = (metadatas[i] if i < len(metadatas) else None) or {}
            normalized = _normalize_metadata(meta, coll_name)
            records.append({
                'source_tag': normalized.get('source_tag', ''),
                'text_ar': normalized.get('text_ar', ''),
            })
        load_canonical_corpus(records)

        total += len(ids)
        offset += len(ids)
        if len(ids) < CANONICAL_PAGE_SIZE:
            break

    return total


def _load_canonical_corpus() -> bool:
    """Load canonical texts from ChromaDB collections for citation verification.

    Returns True if at least one collection loaded successfully.
    """
    loaded_any = False
    try:
        from chroma.chroma_utils import get_chroma_client

        client = get_chroma_client()

        for coll_name in [QURAN_COLLECTION, HADITH_COLLECTION]:
            try:
                collection = client.get_collection(coll_name)
                count = _load_collection_canonical(collection, coll_name)
                if count:
                    loaded_any = True
                    logger.info("Loaded canonical corpus from '%s' (%d docs)", coll_name, count)
            except Exception as exc:
                logger.warning("Could not load canonical corpus from '%s': %s", coll_name, exc)
    except Exception as exc:
        logger.warning("Could not load canonical corpus: %s", exc)
    return loaded_any


# ---------------------------------------------------------------------------
# Grounded generation prompt
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are an Islamic knowledge assistant. Answer the user's question from the
Quran and Hadith passages provided below.

RULES:
1. Every claim must come from the provided passages — never add facts from
   outside them. Within that limit, answer as fully as the passages allow:
   summarise them, explain them, and draw them together.
2. The source tags are part of the material. A tag such as [Q 2:255] states
   the surah number and ayah number, and [C Bukhari/52] states the collection
   — use that when the question is about a surah, a collection, or a
   reference itself.
3. Cite every Quranic reference as [Q surah:ayah], e.g. [Q 2:255]
4. Cite every Hadith as [C collection/number], e.g. [C Bukhari/52]
5. Prefer a partial answer over no answer. If the passages cover only part of
   the question, give that part and say which part they do not cover. Reply
   with "I do not have a grounded source for this in the provided passages."
   only when no passage relates to the question at all.
6. Do not issue fatwas or definitive rulings. Present what the sources say.
7. If the question involves sensitive jurisprudence, add:
   "For a definitive ruling, please consult a qualified scholar."
8. TONE & SAFETY (STRICT):
   - Keep answers tame, calm, balanced, peaceful, and constructive.
   - NEVER encourage, incite, justify, or instruct on violence, terrorism, extremism, harm to self or others, vigilantism, hatred, or dangerous or illegal acts.
   - Never provide actionable guidance or encouragement for dangerous, harmful, or unlawful behavior.
   - Never give instructions that could cause physical, legal, or financial harm. Point the user to a qualified scholar or professional instead.
   - Present classical or historical texts strictly in an educational, descriptive, and peaceful context, emphasizing the Islamic principles of preserving life, peace, justice, and lawful order.
   - If a topic touches upon conflict or harm, strictly de-escalate and emphasize peaceful and ethical conduct under qualified legal and scholarly authority.
9. Respond in the same language as the user's question.

LENGTH AND STRUCTURE:
Give a thorough, explanatory answer — not a one-line reply. Aim for several
paragraphs (roughly 250–500 words when the passages allow it), using this
structure:
- Direct answer: open with a clear 1–2 sentence answer to the question.
- Evidence: walk through each relevant passage. Quote or closely paraphrase
  the key wording, cite it, and explain in your own words what it says and
  how it bears on the question. Cover every relevant passage, not just one.
- Explanation: draw the passages together — what themes or principles they
  share, how they relate to each other, and any context needed to understand
  them (based only on what the passages themselves state).
- Closing: end with a short summary of the key points and, where rule 7
  applies, the scholar disclaimer.
Only stay brief when the passages genuinely contain very little on the
question; in that case, still explain what they do say and what is missing.

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
        _canonical_load_start = time.time()
        _load_canonical()
        print(f"[pipeline] canonical corpus loaded in {time.time() - _canonical_load_start:.2f}s", flush=True)
        logger.info("canonical corpus loaded in %.2fs", time.time() - _canonical_load_start)

        start = time.time()
        pipeline_meta = {
            'phase': self.phase,
            'llm_calls': 0,
            'retrieval_iterations': 1,
        }

        print(f"[pipeline] phase={self.phase} | query='{query[:80]}' | language={language}", flush=True)
        logger.info("phase=%s query='%s' language=%s", self.phase, query[:80], language)

        # --- Phase 2: Intent Router + Scope Guard ---
        intent = 'general'
        if self.phase >= 2:
            from .intent_router import classify_intent
            from .scope_guard import check_scope

            t0 = time.time()
            intent_result = classify_intent(query)
            print(f"[pipeline] intent classified as '{intent_result['type']}' (confidence={intent_result['confidence']}) in {time.time() - t0:.2f}s", flush=True)
            logger.info("intent=%s confidence=%s elapsed=%.2fs", intent_result['type'], intent_result['confidence'], time.time() - t0)

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
            t0 = time.time()
            additional = rewrite_queries(query, intent)
            print(f"[pipeline] query rewriting produced {len(additional.get('hyde', []))} hyde + {len(additional.get('sub_queries', []))} sub-queries in {time.time() - t0:.2f}s", flush=True)
            logger.info("query rewriting: %d hyde + %d sub-queries, elapsed=%.2fs", len(additional.get('hyde', [])), len(additional.get('sub_queries', [])), time.time() - t0)
            pipeline_meta['llm_calls'] += 1
            query_variants = [query] + additional.get('hyde', []) + additional.get('sub_queries', [])

        # --- Dense-Only Retrieval ---
        t0 = time.time()
        print(f"[pipeline] starting dense retrieval with {len(query_variants)} query variants...", flush=True)
        logger.info("starting dense retrieval: %d query variants", len(query_variants))
        fused = retrieve_dense_all_corpora(
            query_variants=query_variants,
            dense_k=10,
            top_n=10,
            max_distance=getattr(settings, 'RAG_MAX_DISTANCE', None),
        )
        print(f"[pipeline] dense retrieval returned {len(fused)} chunks in {time.time() - t0:.2f}s", flush=True)
        logger.info("dense retrieval: %d chunks, elapsed=%.2fs", len(fused), time.time() - t0)

        # --- Citation Verification ---
        t0 = time.time()
        verified = verify_chunks(fused)
        # PRD §5.6 — chunks that fail verification are removed before
        # generation so fabricated references never reach the LLM context.
        hallucinated_count = sum(1 for c in verified if c.get('verification_status') == 'hallucinated')
        if hallucinated_count:
            verified = [c for c in verified if c.get('verification_status') != 'hallucinated']
            print(f"[pipeline] dropped {hallucinated_count} hallucinated chunks before generation", flush=True)
            logger.warning("dropped %d hallucinated chunks before generation", hallucinated_count)
        print(f"[pipeline] citation verification completed in {time.time() - t0:.2f}s", flush=True)
        logger.info("citation verification: elapsed=%.2fs", time.time() - t0)

        # --- Phase 2: Evidence Sufficiency Check ---
        evidence_sufficient = True
        if self.phase >= 2 and intent == 'fiqh':
            from .evidence_checker import check_evidence_sufficiency
            t0 = time.time()
            evidence_sufficient = check_evidence_sufficiency(query, verified)
            print(f"[pipeline] evidence check: sufficient={evidence_sufficient} in {time.time() - t0:.2f}s", flush=True)
            logger.info("evidence check: sufficient=%s, elapsed=%.2fs", evidence_sufficient, time.time() - t0)
            pipeline_meta['llm_calls'] += 1
            # Simplified: single check; loop logic could be added in future

        # --- Grounded Generation ---
        t0 = time.time()
        print(f"[pipeline] generating answer with {len(verified)} context chunks...", flush=True)
        logger.info("generating answer: %d context chunks", len(verified))
        answer = self._generate(query, verified, language)
        print(f"[pipeline] generation completed in {time.time() - t0:.2f}s", flush=True)
        logger.info("generation completed: elapsed=%.2fs", time.time() - t0)
        pipeline_meta['llm_calls'] += 1

        if not answer:
            logger.warning("LLM returned an empty answer for query: %s", query[:100])
            answer = (
                "Sorry — I could not generate an answer right now. "
                "Please try again in a moment."
            )

        # --- Phase 2: Safety Layer ---
        safety = {
            'hallucination_detected': False,
            'flagged_spans': [],
            'fatwa_boundary_triggered': False,
            'disclaimer': None,
        }
        if self.phase >= 2:
            from .hallucination_detector import detect_hallucinations
            t0 = time.time()
            h_result = detect_hallucinations(answer, verified)
            print(f"[pipeline] hallucination check: detected={h_result.get('hallucinated', False)} in {time.time() - t0:.2f}s", flush=True)
            logger.info("hallucination check: detected=%s, elapsed=%.2fs", h_result.get('hallucinated', False), time.time() - t0)
            pipeline_meta['llm_calls'] += 1
            safety['hallucination_detected'] = h_result.get('hallucinated', False)
            # Detector returns spans as dicts ({"text", "reason"}); the API
            # contract (SafetyResultSerializer) expects a list of strings.
            safety['flagged_spans'] = [
                f"{s.get('text', '')} — {s.get('reason', '')}" if isinstance(s, dict) else str(s)
                for s in h_result.get('flagged_spans', [])
            ]
            if safety['hallucination_detected']:
                answer += (
                    "\n\n⚠️ Note: parts of this answer could not be verified "
                    "against the retrieved sources. Please double-check the "
                    "citations before relying on it."
                )

            from .fatwa_boundary import check_fatwa_boundary
            t0 = time.time()
            fb_result = check_fatwa_boundary(answer)
            safety['fatwa_boundary_triggered'] = fb_result['triggered']
            safety['disclaimer'] = fb_result.get('disclaimer')
            print(f"[pipeline] fatwa boundary check: triggered={fb_result['triggered']} in {time.time() - t0:.2f}s", flush=True)

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
                'retrieval_score': chunk.get('distance', 0),
            })

        elapsed = round(time.time() - start, 3)
        print(f"[pipeline] DONE — total elapsed={elapsed}s | llm_calls={pipeline_meta['llm_calls']} | sources={len(source_serialized)}", flush=True)
        logger.info("pipeline complete: elapsed=%.3fs llm_calls=%d sources=%d", elapsed, pipeline_meta['llm_calls'], len(source_serialized))

        return {
            'query': query,
            'intent': intent,
            'answer': answer,
            'sources': source_serialized,
            'citations': citations,
            'safety': safety,
            'pipeline_meta': {
                **pipeline_meta,
                'elapsed': elapsed,
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
            max_tokens=2048,
        )