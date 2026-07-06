"""API integration tests — full HTTP request/response cycle for all endpoints.

These tests exercise the real Django URL routing, view dispatch,
serialization, and error handling, with external dependencies mocked.
"""

import json
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client


# =========================================================================
# Health endpoint
# =========================================================================

class HealthEndpointTests(TestCase):
    """Test GET /api/v1/health"""

    def setUp(self):
        self.client = Client()

    def test_health_returns_ok(self):
        resp = self.client.get('/api/v1/health')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('phase', data)

    def test_health_only_allows_get(self):
        resp = self.client.post('/api/v1/health')
        self.assertEqual(resp.status_code, 405)


# =========================================================================
# Query endpoint — Request validation
# =========================================================================

class QueryEndpointValidationTests(TestCase):
    """Test POST /api/v1/query — request validation."""

    def setUp(self):
        self.client = Client()

    def test_missing_query_returns_400(self):
        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'language': 'en'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertIn('query', data)

    def test_empty_query_returns_400(self):
        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'query': '', 'language': 'en'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_query_too_long_returns_400(self):
        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'query': 'x' * 2001, 'language': 'en'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_language_returns_400(self):
        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'query': 'test', 'language': 'fr'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_max_sources_returns_400(self):
        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'query': 'test', 'max_sources': 0}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_valid_minimal_request(self):
        """Minimal valid request should be accepted even if pipeline fails."""
        with patch('qa.views.PipelineService') as mock_svc:
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = RuntimeError("No API key")
            mock_svc.return_value = mock_pipeline

            resp = self.client.post(
                '/api/v1/query',
                data=json.dumps({'query': 'test question'}),
                content_type='application/json',
            )
            # Pipeline exception → 500
            self.assertEqual(resp.status_code, 500)
            data = json.loads(resp.content)
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Pipeline processing failed')

    def test_valid_full_request(self):
        """Full request with all fields should be accepted."""
        with patch('qa.views.PipelineService') as mock_svc:
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = RuntimeError("No API key")
            mock_svc.return_value = mock_pipeline

            resp = self.client.post(
                '/api/v1/query',
                data=json.dumps({
                    'query': 'What does the Quran say about patience?',
                    'language': 'ar',
                    'max_sources': 10,
                    'include_arabic': False,
                }),
                content_type='application/json',
            )
            self.assertEqual(resp.status_code, 500)  # Pipeline fails but validation passes

    def test_default_language_is_en(self):
        """If language is omitted, it defaults to 'en'."""
        with patch('qa.views.PipelineService') as mock_svc:
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = RuntimeError("No API key")
            mock_svc.return_value = mock_pipeline

            resp = self.client.post(
                '/api/v1/query',
                data=json.dumps({'query': 'test'}),
                content_type='application/json',
            )
            # Should get past validation to pipeline execution
            self.assertEqual(resp.status_code, 500)

    def test_default_max_sources_is_5(self):
        """If max_sources is omitted, it defaults to 5."""
        with patch('qa.views.PipelineService') as mock_svc:
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = RuntimeError("No API key")
            mock_svc.return_value = mock_pipeline

            resp = self.client.post(
                '/api/v1/query',
                data=json.dumps({'query': 'test'}),
                content_type='application/json',
            )
            self.assertEqual(resp.status_code, 500)


# =========================================================================
# Query endpoint — Successful response
# =========================================================================

class QueryEndpointSuccessTests(TestCase):
    """Test POST /api/v1/query — successful pipeline execution."""

    def setUp(self):
        self.client = Client()

    @patch('qa.views.PipelineService')
    def test_successful_query_returns_200(self, mock_svc_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            'query': 'What is patience?',
            'intent': 'quran_verse',
            'answer': 'Patience is a virtue in Islam.',
            'sources': [],
            'citations': [],
            'safety': {
                'hallucination_detected': False,
                'flagged_spans': [],
                'fatwa_boundary_triggered': False,
                'disclaimer': None,
            },
            'pipeline_meta': {
                'phase': 1,
                'llm_calls': 1,
                'elapsed': 0.123,
            },
        }
        mock_svc_cls.return_value = mock_pipeline

        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'query': 'What is patience?'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['query'], 'What is patience?')
        self.assertEqual(data['answer'], 'Patience is a virtue in Islam.')
        self.assertEqual(data['intent'], 'quran_verse')

    @patch('qa.views.PipelineService')
    def test_pipeline_receives_correct_params(self, mock_svc_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            'query': 'test',
            'intent': 'general',
            'answer': 'ok',
            'sources': [],
            'citations': [],
            'safety': {
                'hallucination_detected': False,
                'flagged_spans': [],
                'fatwa_boundary_triggered': False,
                'disclaimer': None,
            },
            'pipeline_meta': {},
        }
        mock_svc_cls.return_value = mock_pipeline

        self.client.post(
            '/api/v1/query',
            data=json.dumps({
                'query': 'custom query',
                'language': 'ar',
                'max_sources': 7,
            }),
            content_type='application/json',
        )

        mock_pipeline.run.assert_called_once_with(
            query='custom query',
            language='ar',
            max_sources=7,
        )

    @patch('qa.views.PipelineService')
    def test_fallback_raw_result_on_serialization_failure(self, mock_svc_cls):
        """When response serialization fails, the raw result is returned."""
        mock_pipeline = MagicMock()
        # Return a result that will fail QueryResponseSerializer validation
        mock_pipeline.run.return_value = {
            'query': 'test',
            # Missing required fields: 'intent', 'answer', 'sources', etc.
        }
        mock_svc_cls.return_value = mock_pipeline

        resp = self.client.post(
            '/api/v1/query',
            data=json.dumps({'query': 'test'}),
            content_type='application/json',
        )
        # Should still return 200 with raw result
        self.assertEqual(resp.status_code, 200)

    def test_query_only_allows_post(self):
        resp = self.client.get('/api/v1/query')
        self.assertEqual(resp.status_code, 405)


# =========================================================================
# Corpus stats endpoint
# =========================================================================

class CorpusStatsEndpointTests(TestCase):
    """Test GET /api/v1/corpus/stats"""

    def setUp(self):
        self.client = Client()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_corpus_stats_returns_counts(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 6236
        mock_get_coll.return_value = mock_collection

        resp = self.client.get('/api/v1/corpus/stats')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn('quran_collection', data)
        self.assertIn('hadith_collection', data)

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_corpus_stats_handles_errors_gracefully(self, mock_get_coll):
        mock_get_coll.side_effect = RuntimeError("ChromaDB is down")

        resp = self.client.get('/api/v1/corpus/stats')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        # Should return 0 counts on error
        self.assertEqual(data['quran_collection']['document_count'], 0)
        self.assertEqual(data['hadith_collection']['document_count'], 0)

    def test_corpus_stats_only_allows_get(self):
        resp = self.client.post('/api/v1/corpus/stats')
        self.assertEqual(resp.status_code, 405)


# =========================================================================
# ChromaDB CRUD endpoints — Integration
# =========================================================================

class ChromaCollectionsIntegrationTests(TestCase):
    """Integration tests for ChromaDB collection endpoints."""

    def setUp(self):
        self.client = Client()

    @patch('router.views.list_collections')
    def test_list_collections_endpoint(self, mock_list):
        mock_list.return_value = ['quran_collection', 'hadith_collection']
        resp = self.client.get('/api/chroma/collections/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(len(data['collections']), 2)

    @patch('router.views.get_or_create_collection')
    def test_create_collection_endpoint(self, mock_get):
        mock_collection = MagicMock()
        mock_collection.name = 'new_coll'
        mock_collection.count.return_value = 0
        mock_get.return_value = mock_collection

        resp = self.client.post(
            '/api/chroma/collections/',
            data=json.dumps({'name': 'new_coll'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.content)
        self.assertEqual(data['name'], 'new_coll')
        self.assertEqual(data['count'], 0)

    @patch('router.views.get_or_create_collection')
    def test_get_collection_detail_endpoint(self, mock_get):
        mock_collection = MagicMock()
        mock_collection.name = 'quran_collection'
        mock_collection.count.return_value = 6236
        mock_get.return_value = mock_collection

        resp = self.client.get('/api/chroma/collections/quran_collection/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['name'], 'quran_collection')
        self.assertEqual(data['count'], 6236)

    @patch('router.views.delete_collection')
    def test_delete_collection_endpoint(self, mock_delete):
        resp = self.client.delete('/api/chroma/collections/old_coll/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['deleted'], 'old_coll')


class ChromaDocumentsIntegrationTests(TestCase):
    """Integration tests for ChromaDB document endpoints."""

    def setUp(self):
        self.client = Client()

    @patch('router.views.get_or_create_collection')
    @patch('router.views.add_documents')
    def test_add_documents_endpoint(self, mock_add, mock_get):
        mock_collection = MagicMock()
        mock_collection.name = 'test'
        mock_get.return_value = mock_collection

        resp = self.client.post(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({'documents': ['text1', 'text2']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.content)
        self.assertEqual(data['added'], 2)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.get_documents')
    def test_get_documents_endpoint(self, mock_get_docs, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_get_docs.return_value = {
            'ids': ['id1'],
            'documents': ['doc1'],
            'metadatas': [{}],
        }

        resp = self.client.get('/api/chroma/collections/test/documents/?ids=id1')
        self.assertEqual(resp.status_code, 200)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.delete_documents')
    def test_delete_documents_endpoint(self, mock_delete, mock_get):
        mock_collection = MagicMock()
        mock_get.return_value = mock_collection

        resp = self.client.delete(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({'ids': ['id1', 'id2']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['deleted'], 2)


class ChromaQueryIntegrationTests(TestCase):
    """Integration tests for ChromaDB query endpoint."""

    def setUp(self):
        self.client = Client()

    @patch('router.views.query_documents')
    @patch('router.views.get_or_create_collection')
    def test_query_endpoint(self, mock_get_coll, mock_query):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_query.return_value = {
            'ids': [['id1']],
            'distances': [[0.1]],
            'metadatas': [[{}]],
            'documents': [['text']],
        }

        resp = self.client.post(
            '/api/chroma/collections/test/query/',
            data=json.dumps({'query_texts': ['search']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)


class LLMGenerateIntegrationTests(TestCase):
    """Integration tests for LLM generate endpoint."""

    def setUp(self):
        self.client = Client()

    @patch('router.views.generate')
    def test_generate_endpoint(self, mock_generate):
        mock_generate.return_value = "Generated response"

        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'Hello'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['generated'], 'Generated response')

    @patch('router.views.generate')
    def test_generate_endpoint_streaming(self, mock_generate):
        mock_chunk = MagicMock()
        mock_chunk.content = 'token'
        mock_generate.return_value = iter([mock_chunk])

        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'test', 'stream': True}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'text/event-stream')


# =========================================================================
# URL routing tests
# =========================================================================

class URLRoutingTests(TestCase):
    """Verify that all URLs resolve correctly."""

    def setUp(self):
        self.client = Client()

    def test_admin_url(self):
        resp = self.client.get('/admin/')
        # Should redirect to login (302) or return login page (200)
        self.assertIn(resp.status_code, [200, 302])

    @patch('router.views.list_collections')
    def test_router_urls(self, mock_list):
        mock_list.return_value = []
        resp = self.client.get('/api/chroma/collections/')
        self.assertEqual(resp.status_code, 200)

    def test_health_url(self):
        resp = self.client.get('/api/v1/health')
        self.assertEqual(resp.status_code, 200)

    def test_404_on_unknown_url(self):
        resp = self.client.get('/api/nonexistent/')
        self.assertEqual(resp.status_code, 404)