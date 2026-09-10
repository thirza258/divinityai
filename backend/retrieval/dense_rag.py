"""
Dense retrieval using Ollama embeddinggemma embeddings.

Ollama provides a local embedding API — no GPU or external API token
required.  The embedding model runs on the host machine or inside
a Docker container.

Collection names per ``docs/retrieval_changes.md``:

- Quran:   ``quran_collection``
- Hadith:  ``hadith_collection``
"""

import logging
import os
import time
from typing import List

import requests
from chromadb.errors import NotFoundError
from django.conf import settings

from chroma.chroma_utils import get_chroma_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")

# Collection names
QURAN_COLLECTION = os.getenv("QURAN_COLLECTION", "quran_collection")
HADITH_COLLECTION = os.getenv("HADITH_COLLECTION", "hadith_collection")


# ---------------------------------------------------------------------------
# Ollama Embedding Function (ChromaDB-compatible)
# ---------------------------------------------------------------------------

class OllamaEmbeddingFunction:
    """Custom ChromaDB embedding function using Ollama's embeddinggemma.

    Implements ChromaDB's embedding function protocol:
    - ``__call__(self, input)`` — legacy interface
    - ``embed_query(input)`` / ``embed_documents(input)`` — v1.5+ interface
    - ``name()`` — model identifier
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        return embed_texts(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return embed_texts(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return embed_texts(input)

    def name(self) -> str:
        return f"ollama/{OLLAMA_EMBED_MODEL}"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Call Ollama's batch embedding API for a list of texts.

    Uses ``/api/embed`` with ``truncate=True`` — the exact request the
    ingest script makes — so query vectors land in the same space as the
    stored document embeddings.  Vectors are returned as-is (no
    post-processing) to match the ingest side.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": OLLAMA_EMBED_MODEL, "input": texts, "truncate": True}

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            embeddings = resp.json()["embeddings"]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
                )
            return embeddings
        except (requests.RequestException, KeyError, ValueError) as exc:
            if attempt == 2:
                logger.error("Ollama embedding failed after 3 attempts: %s", exc)
                raise
            wait = 2 ** attempt
            logger.warning("Ollama error (attempt %d/3), retrying in %ds: %s", attempt + 1, wait, exc)
            time.sleep(wait)

    return []  # unreachable


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using Ollama embeddinggemma.

    Sends all texts in one ``/api/embed`` batch request and returns the
    vectors exactly as Ollama returns them, matching how the ingest
    script embedded the stored documents.
    """
    return _embed_batch(texts)


# ---------------------------------------------------------------------------
# Metadata normalization
# ---------------------------------------------------------------------------

def _normalize_metadata(meta: dict, collection_name: str) -> dict:
    """Normalize ChromaDB metadata to consistent field names.

    The actual stored metadata uses different field names depending on how
    the data was ingested.  This function maps the actual fields to the
    expected names (``source_tag``, ``corpus``, ``text_ar``, ``text_en``)
    so downstream code can rely on a uniform schema.

    Detection is based on the presence of Quran-specific or Hadith-specific
    fields in the metadata, with *collection_name* as a fallback hint.
    """
    normalized = dict(meta)  # preserve all original fields

    # --- Quran detection ---
    if 'ayah_no_quran' in meta or 'surah_no' in meta or 'ayah_ar' in meta \
       or 'quran' in collection_name.lower():
        normalized['corpus'] = 'quran'
        normalized['text_ar'] = meta.get('text_ar') or meta.get('ayah_ar', '')
        normalized['text_en'] = meta.get('text_en') or meta.get('ayah_en', '')
        if 'source_tag' not in normalized:
            surah = meta.get('surah_no', '')
            ayah = meta.get('ayah_no_surah', '')
            normalized['source_tag'] = f"Q {surah}:{ayah}" if surah and ayah else ''

    # --- Hadith detection ---
    elif 'source' in meta or 'hadith_no' in meta or 'hadith_id' in meta \
         or 'hadith' in collection_name.lower():
        normalized['corpus'] = 'hadith'
        source = (meta.get('source', '') or '').strip()
        hadith_no = meta.get('hadith_no', '')
        if 'source_tag' not in normalized:
            normalized['source_tag'] = f"C {source}/{hadith_no}" if source and hadith_no else ''
        normalized['text_ar'] = meta.get('text_ar', '')
        normalized['text_en'] = meta.get('text_en', '')

    return normalized


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def query_dense(
    query_text: str,
    collection_name: str,
    k: int = 10,
    where_filter: dict | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Query a ChromaDB collection using Ollama dense embeddings.

    Returns a list of result dicts with keys: ``id``, ``text``,
    ``metadata``, ``distance``.  Metadata is normalized to use
    consistent field names (``source_tag``, ``corpus``, ``text_ar``,
    ``text_en``).

    Pre-embeds the query with Ollama and passes ``query_embeddings``
    to ChromaDB so results are in the same vector space as the
    ingested documents, regardless of the collection's persisted
    embedding function.  Pass *query_embedding* to reuse a vector that
    has already been computed for *query_text*.
    """
    t0 = time.time()

    # Pre-embed with Ollama (same function used at ingestion time)
    if query_embedding is None:
        query_embedding = embed_texts([query_text])[0]

    # Get the existing collection WITHOUT any embedding function — the
    # ingest created it that way, and we pass pre-embedded query vectors
    # ourselves.  Never auto-create: a missing collection means the ingest
    # has not run against this Chroma, which should fail loudly.
    try:
        collection = get_chroma_client().get_collection(name=collection_name)
    except (NotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"Chroma collection '{collection_name}' not found — run "
            "ingest_divinityai_to_chroma.py against this Chroma server first."
        ) from exc

    print(f"[dense_rag] querying '{collection_name}' (k={k}): '{query_text[:60]}...'", flush=True)
    logger.info("querying '%s' k=%d query='%s'", collection_name, k, query_text[:60])

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    records = []
    for i, doc_id in enumerate(results['ids'][0]):
        raw_meta = results['metadatas'][0][i] if results.get('metadatas') else {}
        records.append({
            'id': doc_id,
            'text': results['documents'][0][i] if results.get('documents') else '',
            'metadata': _normalize_metadata(raw_meta, collection_name),
            'distance': results['distances'][0][i] if results.get('distances') else None,
        })

    print(f"[dense_rag] '{collection_name}' returned {len(records)} results in {time.time() - t0:.2f}s", flush=True)
    logger.info("'%s' returned %d results in %.2fs", collection_name, len(records), time.time() - t0)
    return records


# ---------------------------------------------------------------------------
# Dense-only retrieval across all corpora (replaces hybrid BM25+dense)
# ---------------------------------------------------------------------------

def retrieve_dense_all_corpora(
    query_variants: list[str],
    dense_k: int = 10,
    top_n: int = 10,
    max_distance: float | None = None,
) -> list[dict]:
    """Dense-only retrieval across all query variants and both corpora.

    Queries the Quran and Hadith ChromaDB collections using Ollama
    dense embeddings, merges results by distance, and returns the
    top-N most relevant chunks.

    Parameters
    ----------
    query_variants:
        List of query strings to retrieve for (original + HyDE + sub-queries).
    dense_k:
        Number of results to retrieve per collection per query variant.
    top_n:
        Number of top results to return after merging.
    max_distance:
        Relevance cutoff — chunks with a cosine distance above this value
        are dropped so unrelated passages never reach generation.
        ``None`` disables the cutoff.  Chunks without a distance are kept.
    """
    t0 = time.time()
    all_results: list[dict] = []

    # One Ollama round-trip for every variant, instead of one per variant
    # per collection — embedding dominates retrieval latency.
    t_embed = time.time()
    variant_embeddings = embed_texts(list(query_variants))
    print(f"[dense_rag] embedded {len(query_variants)} variants in one batch in {time.time() - t_embed:.2f}s", flush=True)
    logger.info("embedded %d variants in one batch, elapsed=%.2fs", len(query_variants), time.time() - t_embed)

    for idx, (qv, qvec) in enumerate(zip(query_variants, variant_embeddings)):
        t_qv = time.time()
        all_results.extend(query_dense(qv, QURAN_COLLECTION, k=dense_k, query_embedding=qvec))
        all_results.extend(query_dense(qv, HADITH_COLLECTION, k=dense_k, query_embedding=qvec))
        print(f"[dense_rag] variant {idx+1}/{len(query_variants)} done in {time.time() - t_qv:.2f}s", flush=True)

    if not all_results:
        print(f"[dense_rag] NO results across all corpora in {time.time() - t0:.2f}s", flush=True)
        logger.warning("dense retrieval returned 0 results across all corpora")
        return []

    # Deduplicate by id, keeping the best (lowest distance) entry
    seen: dict[str, dict] = {}
    for r in all_results:
        doc_id = r['id']
        if doc_id not in seen or (r.get('distance') or float('inf')) < (seen[doc_id].get('distance') or float('inf')):
            seen[doc_id] = r

    # Sort by distance ascending (closest = most relevant)
    ranked = sorted(seen.values(), key=lambda r: r.get('distance') or float('inf'))

    if max_distance is not None:
        before = len(ranked)
        ranked = [r for r in ranked if r.get('distance') is None or r['distance'] <= max_distance]
        if before != len(ranked):
            print(f"[dense_rag] relevance cutoff (max_distance={max_distance}) dropped {before - len(ranked)}/{before} chunks", flush=True)
            logger.info("relevance cutoff max_distance=%s dropped %d/%d chunks", max_distance, before - len(ranked), before)

    result = ranked[:top_n]
    print(f"[dense_rag] total={len(all_results)} raw, {len(seen)} unique, returning top {len(result)} in {time.time() - t0:.2f}s", flush=True)
    logger.info("dense retrieval: %d raw, %d unique, top %d, elapsed=%.2fs", len(all_results), len(seen), len(result), time.time() - t0)
    return result