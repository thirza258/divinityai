"""Unit tests for retrieval — citation verifier + RRF fusion."""

from django.test import TestCase

from corpus.arabic_utils import normalize_arabic
from retrieval.citation_verifier import (
    VerificationResult,
    load_canonical_corpus,
    verify_citation,
    verify_chunks,
    get_hallucinated,
)
from retrieval.hybrid_retriever import rrf_merge


class CitationVerifierTests(TestCase):
    """Verify citation verification cascade."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        load_canonical_corpus([
            {
                "source_tag": "Q 2:255",
                "text_ar": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
            },
            {
                "source_tag": "Q 112:1",
                "text_ar": "قُلْ هُوَ اللَّهُ أَحَدٌ",
            },
            {
                "source_tag": "C Bukhari/1",
                "text_ar": "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ",
            },
        ])

    def test_exact_match(self):
        result = verify_citation(
            "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
            "Q 2:255",
        )
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.score, 1.0)

    def test_normalized_match(self):
        """Diacritic-free version of ayah should pass normalized match."""
        result = verify_citation(
            "الله لا اله الا هو الحي القيوم",
            "Q 2:255",
        )
        self.assertEqual(result.status, "normalized")
        self.assertEqual(result.score, 0.95)

    def test_fuzzy_match(self):
        """Slightly altered text should pass fuzzy match (85%+ on normalized)."""
        result = verify_citation(
            "إنما الأعمال بالنياتي",  # extra ي at end
            "C Bukhari/1",
        )
        self.assertEqual(result.status, "fuzzy")
        self.assertGreater(result.score, 0.85)

    def test_hallucinated_missing_tag(self):
        """Unknown source tag should be hallucinated."""
        result = verify_citation("some text", "Q 999:999")
        self.assertEqual(result.status, "hallucinated")
        self.assertEqual(result.score, 0.0)

    def test_hallucinated_wrong_text(self):
        """Completely wrong text for a known tag."""
        result = verify_citation(
            "this is completely wrong text",
            "Q 2:255",
        )
        self.assertEqual(result.status, "hallucinated")
        self.assertEqual(result.score, 0.0)

    def test_verify_chunks_adds_status(self):
        chunks = [
            {
                "id": "q_2_255",
                "metadata": {
                    "source_tag": "Q 2:255",
                    "text_ar": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
                },
            },
            {
                "id": "fake_1",
                "metadata": {
                    "source_tag": "Q 999:1",
                    "text_ar": "made up text",
                },
            },
        ]
        result = verify_chunks(chunks)
        self.assertEqual(result[0]["verification_status"], "exact")
        self.assertEqual(result[1]["verification_status"], "hallucinated")

    def test_verify_chunks_unknown_tag(self):
        """Chunk with no source_tag gets 'unknown' status."""
        chunks = [{"id": "no_tag", "metadata": {"text_ar": "hello"}}]
        result = verify_chunks(chunks)
        self.assertEqual(result[0]["verification_status"], "unknown")

    def test_get_hallucinated(self):
        chunks = [
            {"id": "a", "verification_status": "exact"},
            {"id": "b", "verification_status": "hallucinated"},
            {"id": "c", "verification_status": "normalized"},
            {"id": "d", "verification_status": "hallucinated"},
        ]
        bad = get_hallucinated(chunks)
        self.assertEqual(bad, ["b", "d"])


class RRFTests(TestCase):
    """Verify Reciprocal Rank Fusion correctness."""

    def test_merge_basic(self):
        bm25 = [("doc_a", 5.0), ("doc_b", 3.0)]
        dense = [
            {"id": "doc_a", "text": "aaa", "metadata": {}, "distance": 0.1},
            {"id": "doc_c", "text": "ccc", "metadata": {}, "distance": 0.3},
        ]
        result = rrf_merge(bm25, dense, top_n=3)
        self.assertEqual(len(result), 3)
        # doc_a gets BM25 rank 1 + dense rank 1 → highest RRF
        self.assertEqual(result[0]["id"], "doc_a")

    def test_merge_empty_inputs(self):
        result = rrf_merge([], [], top_n=5)
        self.assertEqual(result, [])

    def test_merge_only_bm25(self):
        bm25 = [("x", 1.0)]
        result = rrf_merge(bm25, [], top_n=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "x")

    def test_merge_only_dense(self):
        dense = [{"id": "y", "text": "", "metadata": {}, "distance": 0.5}]
        result = rrf_merge([], dense, top_n=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "y")

    def test_merge_respects_top_n(self):
        bm25 = [("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 1.0)]
        dense = []
        result = rrf_merge(bm25, dense, top_n=3)
        self.assertEqual(len(result), 3)

    def test_rrf_scores_descending(self):
        bm25 = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        dense = []
        result = rrf_merge(bm25, dense, top_n=3)
        scores = [r["rrf_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_overlapping_docs_get_boost(self):
        """Documents appearing in both BM25 and dense should get higher RRF."""
        bm25 = [("shared", 1.0), ("bm25_only", 1.0)]
        dense = [
            {"id": "shared", "metadata": {}, "distance": 0.1},
            {"id": "dense_only", "metadata": {}, "distance": 0.2},
        ]
        result = rrf_merge(bm25, dense, top_n=3)
        # "shared" should be ranked first
        self.assertEqual(result[0]["id"], "shared")