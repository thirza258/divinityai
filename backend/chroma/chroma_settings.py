"""
ChromaDB settings and configuration for the DivinityAI project.

Supports both local persistent mode and client-server mode.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# ChromaDB connection mode
# ---------------------------------------------------------------------------
# Set to True for client-server mode (remote ChromaDB server).
# Set to False for local persistent mode (embedded, file-based).
CHROMA_CLIENT_SERVER_MODE = os.getenv("CHROMA_CLIENT_SERVER_MODE", "False").lower() in (
    "true", "1", "yes",
)

# ---------------------------------------------------------------------------
# Client-server mode settings (used when CHROMA_CLIENT_SERVER_MODE=True)
# ---------------------------------------------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8040"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "False").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Local persistent mode settings (used when CHROMA_CLIENT_SERVER_MODE=False)
# ---------------------------------------------------------------------------
# Default persist directory — placed alongside the chroma package in dev,
# override via CHROMA_PERSIST_DIR env var.
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent / "chroma_data"),
)

# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------
# ChromaDB's built-in Sentence Transformers embedding function name.
# Change to "all-MiniLM-L6-v2" for a smaller model or any sentence-transformers
# model name on HuggingFace.
CHROMA_EMBEDDING_MODEL = os.getenv(
    "CHROMA_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

# ---------------------------------------------------------------------------
# Default collection name
# ---------------------------------------------------------------------------
CHROMA_DEFAULT_COLLECTION = os.getenv(
    "CHROMA_DEFAULT_COLLECTION",
    "divinityai_documents",
)

# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------
# One of "l2", "ip", "cosine".  ChromaDB defaults to "l2" when not specified.
CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
