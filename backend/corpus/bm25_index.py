"""
BM25 index — build, persist, and query using rank_bm25.

The index operates on pre-normalized Arabic text.  The caller is
responsible for calling :func:`corpus.arabic_utils.normalize_arabic`
before passing text to :meth:`BM25Index.build` or :meth:`BM25Index.retrieve`.

Tokenization is whitespace-split on the normalized text, which works
well for Arabic because Arabic script is already space-delimited
for words and BM25 is a bag-of-words model.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .arabic_utils import normalize_arabic

logger = logging.getLogger(__name__)


class BM25Index:
    """Wraps ``rank_bm25.BM25Okapi`` with persistence and Arabic normalization."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._corpus_map: dict[str, int] = {}  # doc_id → position

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, documents: list[dict]) -> None:
        """Build the BM25 index from a list of ``{id, text_normalized}`` dicts.

        The caller is responsible for providing pre-normalized text
        in the ``text_normalized`` key.
        """
        self._doc_ids = [doc['id'] for doc in documents]
        self._corpus_map = {doc['id']: i for i, doc in enumerate(documents)}
        tokenized_corpus = [doc['text_normalized'].split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index with %d documents", len(self._doc_ids))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str) -> Path:
        """Save the index to ``{index_dir}/{name}.pkl``."""
        if self._bm25 is None:
            raise RuntimeError("Cannot save — index not built or loaded")
        path = self.index_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self._bm25,
                'doc_ids': self._doc_ids,
                'corpus_map': self._corpus_map,
            }, f)
        logger.info("Saved BM25 index to %s (%d docs)", path, len(self._doc_ids))
        return path

    def load(self, name: str) -> None:
        """Load the index from ``{index_dir}/{name}.pkl``."""
        path = self.index_dir / f"{name}.pkl"
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._bm25 = data['bm25']
        self._doc_ids = data['doc_ids']
        self._corpus_map = data['corpus_map']
        logger.info("Loaded BM25 index from %s (%d docs)", path, len(self._doc_ids))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Query the BM25 index.

        Returns a list of ``(doc_id, score)`` tuples sorted descending
        by score.  Only documents with a non-zero score are returned.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built or loaded — call build() or load() first")

        normalized = normalize_arabic(query)
        tokens = normalized.split()
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        """Number of documents in the index."""
        return len(self._doc_ids)

    @property
    def is_loaded(self) -> bool:
        """Whether the index has been built or loaded."""
        return self._bm25 is not None