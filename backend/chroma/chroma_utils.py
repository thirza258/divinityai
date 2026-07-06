"""
ChromaDB utility functions — client factory, collection helpers, and CRUD
operations over documents stored in ChromaDB.
"""

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from .chroma_settings import (
    CHROMA_CLIENT_SERVER_MODE,
    CHROMA_DEFAULT_COLLECTION,
    CHROMA_DISTANCE_METRIC,
    CHROMA_EMBEDDING_MODEL,
    CHROMA_HOST,
    CHROMA_PERSIST_DIR,
    CHROMA_PORT,
    CHROMA_SSL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazily initialised)
# ---------------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None
_embedding_fn: embedding_functions.EmbeddingFunction | None = None


# ======================================================================
# Client factory
# ======================================================================

def get_chroma_client() -> chromadb.ClientAPI:
    """Return a singleton ChromaDB client.

    In *client-server mode* returns an :class:`HttpClient` pointing at the
    configured host/port.  In *local mode* returns a
    :class:`PersistentClient` that stores data under ``CHROMA_PERSIST_DIR``.
    """
    global _client
    if _client is not None:
        return _client

    if CHROMA_CLIENT_SERVER_MODE:
        logger.info(
            "Creating ChromaDB HttpClient → %s:%s (ssl=%s)",
            CHROMA_HOST,
            CHROMA_PORT,
            CHROMA_SSL,
        )
        _client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=CHROMA_SSL,
        )
    else:
        logger.info(
            "Creating ChromaDB PersistentClient → %s",
            CHROMA_PERSIST_DIR,
        )
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
        )

    return _client


def reset_chroma_client() -> None:
    """Discard the cached client so the next call to :func:`get_chroma_client`
    creates a fresh instance.  Useful in tests."""
    global _client
    _client = None


# ======================================================================
# Embedding function
# ======================================================================

def get_embedding_function() -> embedding_functions.EmbeddingFunction:
    """Return the default Sentence Transformers embedding function."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn

    _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=CHROMA_EMBEDDING_MODEL,
    )
    return _embedding_fn


# ======================================================================
# Collection helpers
# ======================================================================

def get_or_create_collection(
    name: str | None = None,
    *,
    embedding_function: embedding_functions.EmbeddingFunction | None = None,
    metadata: dict[str, Any] | None = None,
) -> Collection:
    """Get an existing collection or create it if it doesn't exist.

    Parameters
    ----------
    name:
        Collection name.  Defaults to :data:`CHROMA_DEFAULT_COLLECTION`.
    embedding_function:
        Embedding function to use.  Defaults to the one returned by
        :func:`get_embedding_function`.
    metadata:
        Optional metadata dict (e.g. ``{"hnsw:space": "cosine"}``).
    """
    client = get_chroma_client()
    collection_name = name or CHROMA_DEFAULT_COLLECTION
    emb_fn = embedding_function or get_embedding_function()

    if metadata is None:
        metadata = {"hnsw:space": CHROMA_DISTANCE_METRIC}

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata=metadata,
    )
    return collection


def list_collections() -> list[str]:
    """Return the names of every collection in the database."""
    client = get_chroma_client()
    return client.list_collections()


def delete_collection(name: str) -> None:
    """Delete a collection by name.  No-op if it doesn't exist."""
    client = get_chroma_client()
    try:
        client.delete_collection(name)
        logger.info("Deleted collection '%s'", name)
    except ValueError:
        logger.warning("Collection '%s' does not exist — nothing to delete", name)
    except Exception:
        logger.exception("Failed to delete collection '%s'", name)
        raise


def get_collection_count(collection: Collection | None = None, name: str | None = None) -> int:
    """Return the number of documents in a collection.

    Either pass a *collection* object **or** a *name* to look up the
    collection on the fly.
    """
    if collection is None:
        collection = get_or_create_collection(name)
    return collection.count()


# ======================================================================
# Document CRUD
# ======================================================================

def add_documents(
    documents: list[str],
    *,
    metadatas: list[dict[str, Any]] | None = None,
    ids: list[str] | None = None,
    collection_name: str | None = None,
    collection: Collection | None = None,
) -> None:
    """Add documents to a ChromaDB collection.

    Parameters
    ----------
    documents:
        List of document strings to embed and store.
    metadatas:
        Optional list of metadata dicts (one per document).
    ids:
        Optional list of unique ids (auto-generated when omitted).
    collection_name:
        Look up / create the collection by this name.
    collection:
        An existing :class:`Collection` object (takes precedence over
        *collection_name*).
    """
    col = collection or get_or_create_collection(collection_name)

    if ids is None:
        import uuid

        ids = [str(uuid.uuid4()) for _ in documents]

    col.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info("Added %d document(s) to collection '%s'", len(documents), col.name)


def query_documents(
    query_texts: list[str],
    *,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    collection_name: str | None = None,
    collection: Collection | None = None,
) -> dict[str, Any]:
    """Query a ChromaDB collection by text.

    Returns the standard ChromaDB query result dict with keys
    ``ids``, ``distances``, ``metadatas``, ``documents``,
    ``embeddings`` (if requested).
    """
    col = collection or get_or_create_collection(collection_name)
    return col.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        where_document=where_document,
    )


def get_documents(
    *,
    ids: list[str] | None = None,
    where: dict[str, Any] | None = None,
    limit: int | None = None,
    offset: int | None = None,
    include: list[str] | None = None,
    collection_name: str | None = None,
    collection: Collection | None = None,
) -> dict[str, Any]:
    """Retrieve documents from a ChromaDB collection by id or metadata filter."""
    col = collection or get_or_create_collection(collection_name)

    kwargs: dict[str, Any] = {}
    if ids is not None:
        kwargs["ids"] = ids
    if where is not None:
        kwargs["where"] = where
    if limit is not None:
        kwargs["limit"] = limit
    if offset is not None:
        kwargs["offset"] = offset
    if include is not None:
        kwargs["include"] = include

    return col.get(**kwargs)


def update_documents(
    ids: list[str],
    *,
    documents: list[str] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
    collection_name: str | None = None,
    collection: Collection | None = None,
) -> None:
    """Update documents by id — supply new *documents* and/or *metadatas*."""
    col = collection or get_or_create_collection(collection_name)
    col.update(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Updated %d document(s) in collection '%s'", len(ids), col.name)


def delete_documents(
    ids: list[str],
    *,
    collection_name: str | None = None,
    collection: Collection | None = None,
) -> None:
    """Delete documents from a collection by id."""
    col = collection or get_or_create_collection(collection_name)
    col.delete(ids=ids)
    logger.info("Deleted %d document(s) from collection '%s'", len(ids), col.name)
