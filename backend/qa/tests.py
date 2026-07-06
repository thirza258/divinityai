"""Unit + integration tests for QA pipeline components."""

import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from django.test import TestCase, override_settings

from qa.scope_guard import check_scope
from qa.fatwa_boundary import check_fatwa_boundary
from qa.serializers import (
    QueryRequestSerializer,
    QueryResponseSerializer,
    SafetyResultSerializer,
)
from qa.intent_router import classify_intent
from qa.pipeline import PipelineService, _load_indexes
from corpus.arabic_utils import normalize_arabic
from corpus.bm25_index import BM25Index
from retrieval.citation_verifier import load_canonical_corpus

# ---------------------------------------------------------------------------
# Sample Quran/Hadith data for integration tests
# ---------------------------------------------------------------------------

SAMPLE_QURAN = [
    {
        "id": "q_2_255",
        "source_tag": "Q 2:255",
        "text_ar": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
        "text_en": "Allah! There is no deity except Him, the Ever-Living, the Sustainer of existence.",
        "surah_number": 2,
        "ayah_number": 255,
        "surah_name_ar": "البقرة",
        "surah_name_en": "Al-Baqarah",
        "juz": 3,
    },
    {
        "id": "q_2_153",
        "source_tag": "Q 2:153",
        "text_ar": "يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ",
        "text_en": "O you who have believed, seek help through patience and prayer.",
        "surah_number": 2,
        "ayah_number": 153,
        "surah_name_ar": "البقرة",
        "surah_name_en": "Al-Baqarah",
        "juz": 2,
    },
    {
        "id": "q_112_1",
        "source_tag": "Q 112:1",
        "text_ar": "قُلْ هُوَ اللَّهُ أَحَدٌ",
        "text_en": "Say, He is Allah, [who is] One.",
        "surah_number": 112,
        "ayah_number": 1,
        "surah_name_ar": "الإخلاص",
        "surah_name_en": "Al-Ikhlas",
        "juz": 30,
    },
]

SAMPLE_HADITH = [
    {
        "id": "h_Bukhari_1",
        "source_tag": "C Bukhari/1",
        "collection": "Bukhari",
        "hadith_number": 1,
        "text_ar": "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ",
        "text_en": "Actions are but by intentions.",
        "book": "Revelation",
        "chapter": "How revelation began",
        "narrator": "Umar ibn al-Khattab",
        "grade": "sahih",
    },
]

MOCK_GENERATED_ANSWER = (
    "The Quran emphasizes patience extensively. Allah says 'O you who have believed, "
    "seek help through patience and prayer. Indeed, Allah is with the patient.' [Q 2:153]. "
    "This is a core Islamic virtue mentioned throughout the Quran."
)


# =========================================================================
# Unit tests: Scope Guard
# =========================================================================

class ScopeGuardTests(TestCase):
    def test_rejects_off_domain(self):
        result = check_scope('off_domain', 0.95)
        self.assertFalse(result['allowed'])
        self.assertIn('outside this scope', result['message'])

    def test_rejects_low_confidence(self):
        result = check_scope('hadith', 0.5)
        self.assertFalse(result['allowed'])
        self.assertIn('not confident', result['message'])

    def test_allows_valid_intent(self):
        result = check_scope('quran_verse', 0.85)
        self.assertTrue(result['allowed'])

    def test_allows_fiqh_with_high_confidence(self):
        result = check_scope('fiqh', 0.92)
        self.assertTrue(result['allowed'])


# =========================================================================
# Unit tests: Fatwa Boundary
# =========================================================================

class FatwaBoundaryTests(TestCase):
    def test_triggers_on_divorce(self):
        result = check_fatwa_boundary(
            'According to the Quran, talaq should be done properly.'
        )
        self.assertTrue(result['triggered'])
        self.assertIsNotNone(result['disclaimer'])

    def test_triggers_on_arabic_keyword(self):
        result = check_fatwa_boundary('حكم الطلاق في الإسلام')
        self.assertTrue(result['triggered'])

    def test_no_trigger_on_safe_answer(self):
        result = check_fatwa_boundary(
            'The Quran emphasizes patience and prayer.'
        )
        self.assertFalse(result['triggered'])
        self.assertIsNone(result['disclaimer'])

    def test_handles_empty_answer(self):
        result = check_fatwa_boundary('')
        self.assertFalse(result['triggered'])


# =========================================================================
# Unit tests: Intent Router (mocked LLM)
# =========================================================================

class IntentRouterTests(TestCase):
    @patch('qa.intent_router.generate_dashscope')
    def test_classifies_quran_verse(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "type": "quran_verse",
            "confidence": 0.95,
        })
        result = classify_intent("What does the Quran say about patience?")
        self.assertEqual(result['type'], 'quran_verse')
        self.assertEqual(result['confidence'], 0.95)

    @patch('qa.intent_router.generate_dashscope')
    def test_classifies_hadith(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "type": "hadith",
            "confidence": 0.88,
        })
        result = classify_intent("What did the Prophet say about intentions?")
        self.assertEqual(result['type'], 'hadith')

    @patch('qa.intent_router.generate_dashscope')
    def test_classifies_off_domain(self, mock_generate):
        mock_generate.return_value = json.dumps({
            "type": "off_domain",
            "confidence": 0.92,
        })
        result = classify_intent("What is the weather today?")
        self.assertEqual(result['type'], 'off_domain')

    @patch('qa.intent_router.generate_dashscope')
    def test_fallback_on_parse_error(self, mock_generate):
        mock_generate.return_value = "not valid json!!!"
        result = classify_intent("some query")
        self.assertEqual(result['type'], 'hadith')
        self.assertEqual(result['confidence'], 0.5)


# =========================================================================
# Unit tests: Serializers
# =========================================================================

class SerializerTests(TestCase):
    def test_valid_request(self):
        data = {"query": "What does the Quran say about patience?", "language": "en"}
        serializer = QueryRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid())
        self.assertEqual(serializer.validated_data['language'], 'en')

    def test_valid_request_arabic(self):
        data = {"query": "ماذا يقول القرآن عن الصبر؟", "language": "ar"}
        serializer = QueryRequestSerializer(data=data)
        self.assertTrue(serializer.is_valid())

    def test_missing_query(self):
        serializer = QueryRequestSerializer(data={"language": "en"})
        self.assertFalse(serializer.is_valid())
        self.assertIn('query', serializer.errors)

    def test_empty_query(self):
        serializer = QueryRequestSerializer(data={"query": "", "language": "en"})
        self.assertFalse(serializer.is_valid())

    def test_query_too_long(self):
        serializer = QueryRequestSerializer(data={"query": "x" * 2001, "language": "en"})
        self.assertFalse(serializer.is_valid())

    def test_invalid_language(self):
        serializer = QueryRequestSerializer(data={"query": "test", "language": "fr"})
        self.assertFalse(serializer.is_valid())

    def test_default_language(self):
        serializer = QueryRequestSerializer(data={"query": "test"})
        self.assertTrue(serializer.is_valid())
        self.assertEqual(serializer.validated_data['language'], 'en')

    def test_max_sources_bounds(self):
        s1 = QueryRequestSerializer(data={"query": "test", "max_sources": 0})
        self.assertFalse(s1.is_valid())
        s2 = QueryRequestSerializer(data={"query": "test", "max_sources": 21})
        self.assertFalse(s2.is_valid())
        s3 = QueryRequestSerializer(data={"query": "test", "max_sources": 10})
        self.assertTrue(s3.is_valid())

    def test_response_serializer_valid(self):
        """Full response should serialize correctly."""
        data = {
            "query": "test",
            "intent": "quran_verse",
            "answer": "The Quran says...",
            "sources": [
                {
                    "source_tag": "Q 2:153",
                    "corpus": "quran",
                    "text_ar": "...",
                    "text_en": "...",
                    "verification_status": "exact",
                    "retrieval_score": 0.94,
                }
            ],
            "citations": ["Q 2:153"],
            "safety": {
                "hallucination_detected": False,
                "flagged_spans": [],
                "fatwa_boundary_triggered": False,
                "disclaimer": None,
            },
            "pipeline_meta": {"phase": "1", "llm_calls": "1", "elapsed": "0.5"},
        }
        serializer = QueryResponseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)

    def test_safety_serializer_defaults(self):
        serializer = SafetyResultSerializer(data={})
        self.assertTrue(serializer.is_valid())
        self.assertFalse(serializer.validated_data['hallucination_detected'])
        self.assertEqual(serializer.validated_data['flagged_spans'], [])


# =========================================================================
# Integration tests: Full Pipeline
# =========================================================================

class PipelineIntegrationTests(TestCase):
    """End-to-end pipeline tests with real ChromaDB + BM25, mocked LLM."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.mkdtemp(prefix="pipeline_test_")

        # Build BM25 Quran index
        bm25_dir = Path(cls.tmpdir) / "bm25"
        bm25_quran = BM25Index(bm25_dir)
        bm25_docs = []
        for ayah in SAMPLE_QURAN:
            bm25_docs.append({
                "id": ayah["id"],
                "text_normalized": normalize_arabic(ayah["text_ar"]),
            })
        bm25_quran.build(bm25_docs)
        bm25_quran.save("quran_collection")

        # Build BM25 Hadith index
        bm25_hadith = BM25Index(bm25_dir)
        hadith_docs = []
        for h in SAMPLE_HADITH:
            hadith_docs.append({
                "id": h["id"],
                "text_normalized": normalize_arabic(h["text_ar"]),
            })
        bm25_hadith.build(hadith_docs)
        bm25_hadith.save("hadith_collection")

        # Load canonical corpus
        load_canonical_corpus(SAMPLE_QURAN + SAMPLE_HADITH)

        cls.bm25_dir = bm25_dir

    def _patch_pipeline_loaders(self):
        """Replace the module-level _load_indexes to use test data."""
        import qa.pipeline as pipeline_mod

        bm25_dir = self.bm25_dir
        bm25_quran = BM25Index(bm25_dir)
        bm25_quran.load("quran_collection")
        bm25_hadith = BM25Index(bm25_dir)
        bm25_hadith.load("hadith_collection")

        pipeline_mod._bm25_quran = bm25_quran
        pipeline_mod._bm25_hadith = bm25_hadith
        pipeline_mod._canonical_loaded = True

    def tearDown(self):
        """Reset pipeline globals after each test."""
        import qa.pipeline as pipeline_mod
        pipeline_mod._bm25_quran = None
        pipeline_mod._bm25_hadith = None
        pipeline_mod._canonical_loaded = False

    @patch('qa.pipeline.generate')
    @patch('retrieval.hybrid_retriever.query_dense')
    def test_phase1_pipeline_returns_correct_structure(self, mock_dense, mock_generate):
        """Phase 1 pipeline should return a properly structured response."""
        self._patch_pipeline_loaders()

        # Mock dense retrieval to return sample data
        mock_dense.return_value = [
            {
                "id": "q_2_153",
                "text": SAMPLE_QURAN[1]["text_ar"],
                "metadata": {
                    "source_tag": "Q 2:153",
                    "corpus": "quran",
                    "text_ar": SAMPLE_QURAN[1]["text_ar"],
                    "text_en": SAMPLE_QURAN[1]["text_en"],
                },
                "distance": 0.15,
            },
            {
                "id": "q_2_255",
                "text": SAMPLE_QURAN[0]["text_ar"],
                "metadata": {
                    "source_tag": "Q 2:255",
                    "corpus": "quran",
                    "text_ar": SAMPLE_QURAN[0]["text_ar"],
                    "text_en": SAMPLE_QURAN[0]["text_en"],
                },
                "distance": 0.25,
            },
        ]

        # Mock LLM generation
        mock_generate.return_value = MOCK_GENERATED_ANSWER

        pipeline = PipelineService(phase=1)
        result = pipeline.run(
            query="What does the Quran say about patience?",
            language="en",
            max_sources=3,
        )

        # Verify response structure
        self.assertIn("query", result)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("citations", result)
        self.assertIn("safety", result)
        self.assertIn("pipeline_meta", result)

        # Phase 1 has 'general' intent
        self.assertEqual(result["intent"], "general")

        # Answer should be the mocked response
        self.assertEqual(result["answer"], MOCK_GENERATED_ANSWER)

        # Sources should be present
        self.assertGreater(len(result["sources"]), 0)
        for source in result["sources"]:
            self.assertIn("source_tag", source)
            self.assertIn("verification_status", source)

        # Pipeline meta should have timing
        self.assertIn("elapsed", result["pipeline_meta"])
        self.assertEqual(result["pipeline_meta"]["phase"], 1)

    @patch('qa.pipeline.generate')
    @patch('retrieval.hybrid_retriever.query_dense')
    def test_phase1_no_results_graceful(self, mock_dense, mock_generate):
        """When no retrieval results, pipeline should return graceful message."""
        self._patch_pipeline_loaders()

        mock_dense.return_value = []
        mock_generate.return_value = None  # won't be called

        pipeline = PipelineService(phase=1)
        result = pipeline.run(query="xyzzy_nonexistent", language="en")

        self.assertIn("do not have a grounded source", result["answer"].lower())
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["citations"], [])

    @patch('qa.pipeline.generate')
    @patch('retrieval.hybrid_retriever.query_dense')
    @patch('qa.intent_router.generate_dashscope')
    def test_phase2_off_domain_rejected(self, mock_intent, mock_dense, mock_gen):
        """Phase 2 should reject off-domain queries via scope guard."""
        self._patch_pipeline_loaders()

        mock_intent.return_value = json.dumps({
            "type": "off_domain",
            "confidence": 0.95,
        })

        pipeline = PipelineService(phase=2)
        result = pipeline.run(query="What is the weather today?", language="en")

        self.assertEqual(result["intent"], "off_domain")
        self.assertIn("outside this scope", result["answer"])
        self.assertEqual(result["sources"], [])

        # Dense retrieval and generation should NOT have been called
        mock_dense.assert_not_called()
        mock_gen.assert_not_called()

    @patch('qa.pipeline.generate')
    @patch('retrieval.hybrid_retriever.query_dense')
    @patch('qa.intent_router.generate_dashscope')
    @patch('qa.hallucination_detector.generate_dashscope')
    def test_phase2_full_pipeline(self, mock_halluc, mock_intent, mock_dense, mock_gen):
        """Phase 2 full pipeline with intent routing, safety, and generation."""
        self._patch_pipeline_loaders()

        # Mock intent
        mock_intent.return_value = json.dumps({
            "type": "quran_verse",
            "confidence": 0.92,
        })

        # Mock dense retrieval
        mock_dense.return_value = [
            {
                "id": "q_2_153",
                "text": SAMPLE_QURAN[1]["text_ar"],
                "metadata": {
                    "source_tag": "Q 2:153",
                    "corpus": "quran",
                    "text_ar": SAMPLE_QURAN[1]["text_ar"],
                    "text_en": SAMPLE_QURAN[1]["text_en"],
                },
                "distance": 0.15,
            },
        ]

        # Mock generation
        mock_gen.return_value = MOCK_GENERATED_ANSWER

        # Mock hallucination detector
        mock_halluc.return_value = json.dumps({
            "hallucinated": False,
            "flagged_spans": [],
        })

        pipeline = PipelineService(phase=2)
        result = pipeline.run(
            query="What does the Quran say about patience?",
            language="en",
        )

        # Verify Phase 2 features
        self.assertEqual(result["intent"], "quran_verse")
        self.assertFalse(result["safety"]["hallucination_detected"])
        self.assertFalse(result["safety"]["fatwa_boundary_triggered"])

        # LLM calls should be tracked
        self.assertGreaterEqual(result["pipeline_meta"]["llm_calls"], 2)

    @patch('qa.pipeline.generate')
    @patch('retrieval.hybrid_retriever.query_dense')
    @patch('qa.evidence_checker.generate_dashscope')
    @patch('qa.query_rewriter.generate_dashscope')
    @patch('qa.intent_router.generate_dashscope')
    def test_phase2_fatwa_triggers_disclaimer(self, mock_intent, mock_rewrite, mock_evidence, mock_dense, mock_gen):
        """Phase 2 should add disclaimer when answer triggers fatwa boundary."""
        self._patch_pipeline_loaders()

        mock_intent.return_value = json.dumps({
            "type": "fiqh",
            "confidence": 0.88,
        })

        # Mock query rewriting (HyDE + sub-query decomposition)
        mock_rewrite.return_value = "A hypothetical passage about divorce in Islamic law."

        # Mock evidence check — sufficient
        mock_evidence.return_value = json.dumps({"sufficient": True})

        mock_dense.return_value = [
            {
                "id": "h_Bukhari_1",
                "text": SAMPLE_HADITH[0]["text_ar"],
                "metadata": {
                    "source_tag": "C Bukhari/1",
                    "corpus": "hadith",
                    "text_ar": SAMPLE_HADITH[0]["text_ar"],
                    "text_en": SAMPLE_HADITH[0]["text_en"],
                },
                "distance": 0.2,
            },
        ]

        # Answer mentions talaq (divorce) — should trigger fatwa boundary
        mock_gen.return_value = (
            "Regarding talaq, the Prophet (ﷺ) said that divorce is permitted "
            "but disliked by Allah. [C Bukhari/1]"
        )

        with patch('qa.hallucination_detector.generate_dashscope') as mock_halluc:
            mock_halluc.return_value = json.dumps({
                "hallucinated": False,
                "flagged_spans": [],
            })

            pipeline = PipelineService(phase=2)
            result = pipeline.run(query="What is the ruling on divorce?", language="en")

            # Fatwa boundary should be triggered
            self.assertTrue(result["safety"]["fatwa_boundary_triggered"])
            self.assertIsNotNone(result["safety"]["disclaimer"])
            self.assertIn("consult a qualified scholar", result["safety"]["disclaimer"])

    @patch('qa.pipeline.generate')
    @patch('retrieval.hybrid_retriever.query_dense')
    def test_citation_verification_in_pipeline(self, mock_dense, mock_gen):
        """Citations in the generated answer should be verified against canonical corpus."""
        self._patch_pipeline_loaders()

        mock_dense.return_value = [
            {
                "id": "q_2_153",
                "text": SAMPLE_QURAN[1]["text_ar"],
                "metadata": {
                    "source_tag": "Q 2:153",
                    "corpus": "quran",
                    "text_ar": SAMPLE_QURAN[1]["text_ar"],
                    "text_en": SAMPLE_QURAN[1]["text_en"],
                },
                "distance": 0.15,
            },
        ]

        mock_gen.return_value = "The Quran says seek help through patience and prayer. [Q 2:153]"

        pipeline = PipelineService(phase=1)
        result = pipeline.run(query="What does the Quran say about patience?", language="en")

        # Source should be verified as exact match
        self.assertEqual(result["sources"][0]["verification_status"], "exact")
        # Citations should include the source tag
        self.assertIn("Q 2:153", result["citations"])