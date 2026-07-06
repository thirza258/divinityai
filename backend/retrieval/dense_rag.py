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
from django.conf import settings

from chroma.chroma_utils import get_or_create_collection

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

def _embed_single(text: str) -> List[float]:
    """Call Ollama embedding API for a single text."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["embedding"]
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

    Ollama does not support batching natively, so we embed texts
    sequentially.  For large ingestion jobs, this is acceptable
    since Ollama runs locally and each call is fast.
    """
    embeddings = []
    for text in texts:
        emb = _embed_single(text)
        # Normalize to unit vector for cosine similarity
        norm = sum(x * x for x in emb) ** 0.5
        if norm > 0:
            embeddings.append([x / norm for x in emb])
        else:
            embeddings.append(emb)
    return embeddings


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
) -> list[dict]:
    """Query a ChromaDB collection using Ollama dense embeddings.

    Returns a list of result dicts with keys: ``id``, ``text``,
    ``metadata``, ``distance``.  Metadata is normalized to use
    consistent field names (``source_tag``, ``corpus``, ``text_ar``,
    ``text_en``).
    """
    emb_fn = OllamaEmbeddingFunction()
    collection = get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
    )
    results = collection.query(
        query_texts=[query_text],
        n_results=k,
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
    return records