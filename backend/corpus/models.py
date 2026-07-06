"""
Django models for tracking corpus ingestion state.

These models track *what* has been ingested — the actual ayah and
hadith text lives in ChromaDB and BM25 indexes.
"""

from django.db import models


class IngestedCollection(models.Model):
    """Tracks which corpus collections have been ingested into ChromaDB."""

    name = models.CharField(max_length=64, unique=True)
    corpus_type = models.CharField(max_length=16)  # "quran" or "hadith"
    document_count = models.IntegerField(default=0)
    bm25_index_path = models.CharField(max_length=256, blank=True)
    ingested_at = models.DateTimeField(auto_now_add=True)
    checksum = models.CharField(max_length=64, blank=True)

    def __str__(self):
        return f"{self.name} ({self.corpus_type}, {self.document_count} docs)"