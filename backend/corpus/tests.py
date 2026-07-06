"""Unit tests for corpus app — Arabic normalization + BM25 index."""

import tempfile
from pathlib import Path

from django.test import TestCase

from corpus.arabic_utils import normalize_arabic
from corpus.bm25_index import BM25Index


class ArabicNormalizationTests(TestCase):
    """Verify normalize_arabic handles all normalization steps."""

    def test_strips_tashkeel(self):
        self.assertEqual(
            normalize_arabic('بِسْمِ اللَّهِ'),
            'بسم الله',
        )

    def test_normalizes_alef_variants(self):
        self.assertEqual(normalize_arabic('أإآ'), 'ااا')

    def test_strips_tatweel(self):
        self.assertEqual(normalize_arabic('اللــــه'), 'الله')

    def test_handles_empty_string(self):
        self.assertEqual(normalize_arabic(''), '')

    def test_preserves_non_arabic(self):
        self.assertEqual(normalize_arabic('Hello 123'), 'Hello 123')

    def test_ayatul_kursi(self):
        """Full ayah with diacritics should normalize cleanly."""
        text = 'اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ'
        expected = 'الله لا اله الا هو الحي القيوم'
        self.assertEqual(normalize_arabic(text), expected)

    def test_preserves_ya_alef(self):
        """ى should not be changed to ي — it's a word-final distinction."""
        text = 'عَلَىٰ'
        # Tashkeel stripped, but the base alef-maqsura stays
        result = normalize_arabic(text)
        self.assertIn('على', result)

    def test_consistent_for_query_and_corpus(self):
        """Same text with different diacritic forms should normalize identically."""
        a = 'الرَّحْمَٰنِ'
        b = 'الرحمن'
        self.assertEqual(normalize_arabic(a), normalize_arabic(b))


class BM25IndexTests(TestCase):
    """Verify BM25Index build, persistence, and retrieval."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp(prefix="bm25_test_")

    def setUp(self):
        self.index = BM25Index(self.tmpdir)
        docs = [
            {"id": "q_1_1", "text_normalized": normalize_arabic("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")},
            {"id": "q_2_255", "text_normalized": normalize_arabic("اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ")},
            {"id": "q_112_1", "text_normalized": normalize_arabic("قُلْ هُوَ اللَّهُ أَحَدٌ")},
            {"id": "q_2_153", "text_normalized": normalize_arabic("يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ")},
        ]
        self.index.build(docs)

    def test_build_and_count(self):
        self.assertEqual(self.index.doc_count, 4)
        self.assertTrue(self.index.is_loaded)

    def test_retrieve_returns_relevant(self):
        results = self.index.retrieve("الله", k=3)
        self.assertGreaterEqual(len(results), 1)
        doc_ids = [r[0] for r in results]
        # All docs containing "الله" should appear
        self.assertIn("q_1_1", doc_ids)
        self.assertIn("q_112_1", doc_ids)
        self.assertIn("q_2_255", doc_ids)

    def test_retrieve_respects_k(self):
        results = self.index.retrieve("الرحمن", k=1)
        self.assertEqual(len(results), 1)

    def test_retrieve_empty_query(self):
        results = self.index.retrieve("", k=5)
        self.assertEqual(results, [])

    def test_retrieve_no_match(self):
        results = self.index.retrieve("zzz_nonexistent_zzz", k=5)
        self.assertEqual(results, [])

    def test_save_and_load_roundtrip(self):
        self.index.save("roundtrip_test")

        loaded = BM25Index(self.tmpdir)
        loaded.load("roundtrip_test")
        self.assertEqual(loaded.doc_count, 4)

        # Verify same results from loaded index
        orig = self.index.retrieve("الله", k=3)
        reloaded = loaded.retrieve("الله", k=3)
        self.assertEqual(orig, reloaded)

    def test_normalized_query_matches(self):
        """Query with diacritics should match normalized corpus."""
        results = self.index.retrieve("اللَّهُ", k=3)
        self.assertGreaterEqual(len(results), 1)

    def test_retrieve_not_loaded(self):
        empty = BM25Index(self.tmpdir)
        with self.assertRaises(RuntimeError):
            empty.retrieve("test")

    def test_save_not_loaded(self):
        empty = BM25Index(self.tmpdir)
        with self.assertRaises(RuntimeError):
            empty.save("should_fail")