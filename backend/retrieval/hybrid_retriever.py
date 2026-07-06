"""
Hybrid retrieval — BM25 + Dense (ChromaDB/BGE-M3) with RRF fusion.

Reciprocal Rank Fusion formula:
    RRF(doc) = Σ 1 / (k + rank_i)
where k = 60 (standard from the RRF paper).

All results across all corpora and query variants are merged into a
single top-10 pool.
"""

import logging
from typing import List

from corpus.bm25_index import BM25Index
from .dense_rag import QURAN_COLLECTION, HADITH_COLLECTION, query_dense

logger = logging.getLogger(__name__)

RRF_K = 60


def rrf_merge(
    bm25_results: list[tuple[str, float]],
    dense_results: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """Merge BM25 and dense results via Reciprocal Rank Fusion.

    Parameters
    ----------
    bm25_results:
        List of ``(doc_id, bm25_score)`` tuples from BM25, sorted descending.
    dense_results:
        List of ChromaDB result dicts with ``id``, ``text``, ``metadata``, ``distance`` keys.
    top_n:
        Number of top results to return.

    Returns
    -------
    List of result dicts with ``id``, ``text``, ``metadata``, ``rrf_score`` keys,
    sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # BM25 results: rank 1 = top result
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        score = 1.0 / (RRF_K + rank)
        scores[doc_id] = scores.get(doc_id, 0) + score

    # Dense results: rank 1 = closest (lowest distance)
    for rank, result in enumerate(dense_results, 1):
        doc_id = result['id']
        score = 1.0 / (RRF_K + rank)
        scores[doc_id] = scores.get(doc_id, 0) + score
        if doc_id not in doc_map:
            doc_map[doc_id] = result

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    output = []
    for doc_id, rrf_score in ranked[:top_n]:
        entry = doc_map.get(doc_id, {'id': doc_id})
        entry['rrf_score'] = rrf_score
        output.append(entry)

    return output


def retrieve_hybrid(
    query: str,
    bm25_index: BM25Index,
    dense_collection: str,
    bm25_k: int = 10,
    dense_k: int = 10,
    top_n: int = 10,
) -> list[dict]:
    """Run BM25 + Dense retrieval and fuse via RRF.

    Returns ``top_n`` results with RRF scores and metadata.
    """
    # BM25 retrieval (normalization happens inside BM25Index.retrieve)
    bm25_results = bm25_index.retrieve(query, k=bm25_k)

    # Dense retrieval (ChromaDB handles its own tokenization)
    dense_results = query_dense(query, dense_collection, k=dense_k)

    if not bm25_results and not dense_results:
        return []

    return rrf_merge(bm25_results, dense_results, top_n=top_n)


def retrieve_from_all_corpora(
    query_variants: list[str],
    bm25_quran: BM25Index,
    bm25_hadith: BM25Index | None,
    bm25_k: int = 10,
    dense_k: int = 10,
    top_n: int = 10,
) -> list[dict]:
    """Run hybrid retrieval across all query variants and both corpora.

    This is the main retrieval entry point for the pipeline.  It
    accumulates BM25 and dense results from all sources and fuses
    them into a single ranked list.

    Parameters
    ----------
    query_variants:
        List of query strings to retrieve for (original + HyDE + sub-queries).
    bm25_quran:
        Loaded BM25 index for the Quran corpus.
    bm25_hadith:
        Loaded BM25 index for the Hadith corpus (may be None).
    """
    all_bm25: list[tuple[str, float]] = []
    all_dense: list[dict] = []

    for qv in query_variants:
        all_bm25.extend(bm25_quran.retrieve(qv, k=bm25_k))
        all_dense.extend(query_dense(qv, QURAN_COLLECTION, k=dense_k))

        if bm25_hadith is not None:
            all_bm25.extend(bm25_hadith.retrieve(qv, k=bm25_k))
            all_dense.extend(query_dense(qv, HADITH_COLLECTION, k=dense_k))

    if not all_bm25 and not all_dense:
        return []

    return rrf_merge(all_bm25, all_dense, top_n=top_n)