"""
Django management command: ingest the Quran corpus into ChromaDB + BM25.

Usage::

    python manage.py ingest_quran [--data-dir corpus/data] [--reindex]
"""

import logging
import time
from pathlib import Path

from django.core.management.base import BaseCommand

from corpus.arabic_utils import normalize_arabic
from corpus.bm25_index import BM25Index
from corpus.ingestion import load_quran_corpus
from corpus.models import IngestedCollection
from chroma.chroma_utils import add_documents, get_or_create_collection
from retrieval.dense_rag import OllamaEmbeddingFunction, QURAN_COLLECTION

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_RETRIES = 3


class Command(BaseCommand):
    help = 'Ingest Quran corpus into ChromaDB + BM25'

    def add_arguments(self, parser):
        parser.add_argument(
            '--data-dir', type=str, default='corpus/data',
            help='Directory containing corpus JSON files',
        )
        parser.add_argument(
            '--reindex', action='store_true',
            help='Recreate index from scratch (delete existing)',
        )

    def handle(self, *args, **options):
        data_dir = Path(options['data_dir'])
        if not data_dir.is_absolute():
            # Resolve relative to the Django project root
            from django.conf import settings
            data_dir = settings.BASE_DIR / data_dir

        self.stdout.write(f"Loading Quran corpus from {data_dir} ...")

        # Check if already ingested
        existing = IngestedCollection.objects.filter(name=QURAN_COLLECTION).first()
        if existing and not options['reindex']:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Quran already ingested ({existing.document_count} ayahs, "
                    f"ingested {existing.ingested_at}). "
                    f"Use --reindex to re-ingest."
                )
            )
            return

        # Load corpus
        ayahs = load_quran_corpus(data_dir)
        if not ayahs:
            self.stderr.write(self.style.ERROR("No Quran ayahs found — aborting"))
            return

        self.stdout.write(f"Loaded {len(ayahs)} ayahs. Ingesting into ChromaDB ...")

        # Create ChromaDB collection with Ollama embeddings
        emb_fn = OllamaEmbeddingFunction()
        collection = get_or_create_collection(
            name=QURAN_COLLECTION,
            embedding_function=emb_fn,
        )

        if options['reindex']:
            self._clear_collection(collection)

        # Ingest in batches
        total = len(ayahs)
        ingested = 0
        bm25_docs = []

        for i in range(0, total, BATCH_SIZE):
            batch = ayahs[i:i + BATCH_SIZE]
            documents = []
            metadatas = []
            ids = []

            for ayah in batch:
                doc_id = ayah.get('id', f"q_{ayah['surah_number']}_{ayah['ayah_number']}")
                text_ar = ayah.get('text_ar', '')
                text_en = ayah.get('text_en', '')
                # The document text for embedding is the Arabic text
                document = text_ar
                documents.append(document)
                metadatas.append({
                    'source_tag': ayah.get('source_tag', f"Q {ayah['surah_number']}:{ayah['ayah_number']}"),
                    'corpus': 'quran',
                    'surah_number': ayah.get('surah_number'),
                    'ayah_number': ayah.get('ayah_number'),
                    'surah_name_ar': ayah.get('surah_name_ar', ''),
                    'surah_name_en': ayah.get('surah_name_en', ''),
                    'juz': ayah.get('juz'),
                    'text_ar': text_ar,
                    'text_en': text_en,
                })
                ids.append(doc_id)

                # Prepare BM25 document (normalized Arabic for indexing)
                bm25_docs.append({
                    'id': doc_id,
                    'text_normalized': normalize_arabic(text_ar),
                })

            # Add to ChromaDB with retries
            for attempt in range(MAX_RETRIES):
                try:
                    add_documents(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        collection=collection,
                    )
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1:
                        self.stderr.write(f"Failed to ingest batch {i}: {exc}")
                        raise
                    wait = 2 ** attempt
                    self.stdout.write(f"  Retrying in {wait}s (attempt {attempt + 1}) ...")
                    time.sleep(wait)

            ingested += len(batch)
            if ingested % 500 == 0 or ingested == total:
                self.stdout.write(f"  {ingested}/{total} ayahs ingested ...")

        self.stdout.write(self.style.SUCCESS(f"ChromaDB: {ingested} ayahs stored in '{QURAN_COLLECTION}'"))

        # Build BM25 index
        self.stdout.write("Building BM25 index ...")
        bm25 = BM25Index(Path.home() / '.divinityai' / 'bm25_indexes')
        bm25.build(bm25_docs)
        bm25_path = bm25.save('quran_collection')

        # Record ingestion
        IngestedCollection.objects.update_or_create(
            name=QURAN_COLLECTION,
            defaults={
                'corpus_type': 'quran',
                'document_count': ingested,
                'bm25_index_path': str(bm25_path),
            },
        )

        self.stdout.write(self.style.SUCCESS(
            f"Done. {ingested} ayahs ingested. BM25 index: {bm25_path}"
        ))

    def _clear_collection(self, collection):
        """Remove all documents from a collection."""
        count = collection.count()
        if count > 0:
            ids = collection.get()['ids']
            collection.delete(ids=ids)
            self.stdout.write(f"Cleared {count} existing documents from '{collection.name}'")