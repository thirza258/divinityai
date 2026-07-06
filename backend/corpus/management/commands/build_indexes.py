"""
Django management command: rebuild BM25 indexes from ChromaDB data.

Use this when you want to re-index without re-embedding (faster than
re-running the full ingestion commands).

Usage::

    python manage.py build_indexes [--collection quran_collection]
"""

import logging
from pathlib import Path

from django.core.management.base import BaseCommand

from corpus.arabic_utils import normalize_arabic
from corpus.bm25_index import BM25Index
from chroma.chroma_utils import get_or_create_collection
from retrieval.dense_rag import QURAN_COLLECTION, HADITH_COLLECTION

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Rebuild BM25 indexes from ChromaDB data (no re-embedding)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--collection', type=str, default='',
            help='Rebuild a specific collection (default: all)',
        )

    def handle(self, *args, **options):
        target = options['collection']

        if not target or target == QURAN_COLLECTION:
            self._rebuild(QURAN_COLLECTION, 'quran')

        if not target or target == HADITH_COLLECTION:
            self._rebuild(HADITH_COLLECTION, 'hadith')

    def _rebuild(self, name: str, corpus_type: str):
        self.stdout.write(f"Rebuilding BM25 index for '{name}' ...")

        collection = get_or_create_collection(name=name)
        data = collection.get()

        if not data['ids']:
            self.stderr.write(self.style.WARNING(f"No documents in '{name}' — skipping"))
            return

        bm25_docs = []
        for i, doc_id in enumerate(data['ids']):
            meta = data['metadatas'][i] if data.get('metadatas') else {}
            text_ar = meta.get('text_ar', '')
            bm25_docs.append({
                'id': doc_id,
                'text_normalized': normalize_arabic(text_ar),
            })

        bm25 = BM25Index(Path.home() / '.divinityai' / 'bm25_indexes')
        bm25.build(bm25_docs)
        bm25_path = bm25.save(name)

        from corpus.models import IngestedCollection
        IngestedCollection.objects.update_or_create(
            name=name,
            defaults={
                'corpus_type': corpus_type,
                'document_count': len(bm25_docs),
                'bm25_index_path': str(bm25_path),
            },
        )

        self.stdout.write(self.style.SUCCESS(
            f"Rebuilt BM25 index for '{name}': {len(bm25_docs)} docs → {bm25_path}"
        ))