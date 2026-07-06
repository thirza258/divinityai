"""
Django management command: ingest Hadith collections into ChromaDB + BM25.

Usage::

    python manage.py ingest_hadith [--data-dir corpus/data] [--collections bukhari,muslim] [--reindex]
"""

import logging
import time
from pathlib import Path

from django.core.management.base import BaseCommand

from corpus.arabic_utils import normalize_arabic
from corpus.bm25_index import BM25Index
from corpus.ingestion import load_hadith_corpus
from corpus.models import IngestedCollection
from chroma.chroma_utils import add_documents, get_or_create_collection
from retrieval.dense_rag import OllamaEmbeddingFunction, HADITH_COLLECTION

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_RETRIES = 3

# Phase 1 default collections; Phase 2 adds abu_dawud, tirmidhi, nasai, ibn_majah
DEFAULT_COLLECTIONS = ['bukhari', 'muslim']


class Command(BaseCommand):
    help = 'Ingest Hadith corpus into ChromaDB + BM25'

    def add_arguments(self, parser):
        parser.add_argument(
            '--data-dir', type=str, default='corpus/data',
            help='Directory containing corpus JSON files',
        )
        parser.add_argument(
            '--collections', type=str, default=','.join(DEFAULT_COLLECTIONS),
            help='Comma-separated list of collection slugs (e.g. bukhari,muslim)',
        )
        parser.add_argument(
            '--reindex', action='store_true',
            help='Recreate index from scratch (delete existing)',
        )

    def handle(self, *args, **options):
        data_dir = Path(options['data_dir'])
        if not data_dir.is_absolute():
            from django.conf import settings
            data_dir = settings.BASE_DIR / data_dir

        collections = [c.strip() for c in options['collections'].split(',') if c.strip()]
        self.stdout.write(f"Loading Hadith corpus from {data_dir}")
        self.stdout.write(f"Collections: {', '.join(collections)}")

        # Check if already ingested
        existing = IngestedCollection.objects.filter(name=HADITH_COLLECTION).first()
        if existing and not options['reindex']:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Hadith already ingested ({existing.document_count} hadiths, "
                    f"ingested {existing.ingested_at}). "
                    f"Use --reindex to re-ingest."
                )
            )
            return

        # Load corpus
        hadiths = load_hadith_corpus(data_dir, collections)
        if not hadiths:
            self.stderr.write(self.style.ERROR("No Hadith records found — aborting"))
            return

        self.stdout.write(f"Loaded {len(hadiths)} hadiths. Ingesting into ChromaDB ...")

        # Create ChromaDB collection with Ollama embeddings
        emb_fn = OllamaEmbeddingFunction()
        collection = get_or_create_collection(
            name=HADITH_COLLECTION,
            embedding_function=emb_fn,
        )

        if options['reindex']:
            self._clear_collection(collection)

        # Ingest in batches
        total = len(hadiths)
        ingested = 0
        bm25_docs = []

        for i in range(0, total, BATCH_SIZE):
            batch = hadiths[i:i + BATCH_SIZE]
            documents = []
            metadatas = []
            ids = []

            for hadith in batch:
                collection_name = hadith.get('collection', '')
                hadith_number = hadith.get('hadith_number', '')
                doc_id = hadith.get('id', f"h_{collection_name}_{hadith_number}")
                text_ar = hadith.get('text_ar', '')
                text_en = hadith.get('text_en', '')
                document = text_ar
                documents.append(document)
                metadatas.append({
                    'source_tag': hadith.get('source_tag', f"C {collection_name}/{hadith_number}"),
                    'corpus': 'hadith',
                    'collection': collection_name,
                    'hadith_number': hadith_number,
                    'book': hadith.get('book', ''),
                    'chapter': hadith.get('chapter', ''),
                    'narrator': hadith.get('narrator', ''),
                    'grade': hadith.get('grade', ''),
                    'text_ar': text_ar,
                    'text_en': text_en,
                })
                ids.append(doc_id)

                bm25_docs.append({
                    'id': doc_id,
                    'text_normalized': normalize_arabic(text_ar),
                })

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
                self.stdout.write(f"  {ingested}/{total} hadiths ingested ...")

        self.stdout.write(self.style.SUCCESS(f"ChromaDB: {ingested} hadiths stored in '{HADITH_COLLECTION}'"))

        # Build BM25 index
        self.stdout.write("Building BM25 index ...")
        bm25 = BM25Index(Path.home() / '.divinityai' / 'bm25_indexes')
        bm25.build(bm25_docs)
        bm25_path = bm25.save('hadith_collection')

        IngestedCollection.objects.update_or_create(
            name=HADITH_COLLECTION,
            defaults={
                'corpus_type': 'hadith',
                'document_count': ingested,
                'bm25_index_path': str(bm25_path),
            },
        )

        self.stdout.write(self.style.SUCCESS(
            f"Done. {ingested} hadiths ingested. BM25 index: {bm25_path}"
        ))

    def _clear_collection(self, collection):
        count = collection.count()
        if count > 0:
            ids = collection.get()['ids']
            collection.delete(ids=ids)
            self.stdout.write(f"Cleared {count} existing documents from '{collection.name}'")