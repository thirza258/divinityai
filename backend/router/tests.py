"""Unit tests for router views — ChromaDB CRUD + LLM generate."""

import json
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client


# =========================================================================
# Unit tests: ChromaDB Collections API
# =========================================================================

class ChromaCollectionsTests(TestCase):
    """Test GET/POST /api/chroma/collections/"""

    def setUp(self):
        self.client = Client()

    @patch('router.views.list_collections')
    def test_list_collections(self, mock_list):
        mock_list.return_value = ['quran_collection', 'hadith_collection']
        resp = self.client.get('/api/chroma/collections/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn('collections', data)
        self.assertEqual(len(data['collections']), 2)

    @patch('router.views.list_collections')
    def test_list_collections_empty(self, mock_list):
        mock_list.return_value = []
        resp = self.client.get('/api/chroma/collections/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['collections'], [])

    @patch('router.views.list_collections')
    def test_list_collections_error(self, mock_list):
        mock_list.side_effect = RuntimeError("ChromaDB is down")
        resp = self.client.get('/api/chroma/collections/')
        self.assertEqual(resp.status_code, 500)
        data = json.loads(resp.content)
        self.assertIn('error', data)

    @patch('router.views.get_or_create_collection')
    def test_create_collection(self, mock_get):
        mock_collection = MagicMock()
        mock_collection.name = 'my_collection'
        mock_collection.count.return_value = 0
        mock_get.return_value = mock_collection

        resp = self.client.post(
            '/api/chroma/collections/',
            data=json.dumps({'name': 'my_collection'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.content)
        self.assertEqual(data['name'], 'my_collection')
        self.assertEqual(data['count'], 0)

    @patch('router.views.get_or_create_collection')
    def test_create_collection_with_metadata(self, mock_get):
        mock_collection = MagicMock()
        mock_collection.name = 'test'
        mock_collection.count.return_value = 0
        mock_get.return_value = mock_collection

        resp = self.client.post(
            '/api/chroma/collections/',
            data=json.dumps({'name': 'test', 'metadata': {'hnsw:space': 'cosine'}}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)

    @patch('router.views.get_or_create_collection')
    def test_create_collection_error(self, mock_get):
        mock_get.side_effect = RuntimeError("Cannot create")
        resp = self.client.post(
            '/api/chroma/collections/',
            data=json.dumps({'name': 'bad'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 500)

    def test_collections_method_not_allowed(self):
        resp = self.client.put('/api/chroma/collections/')
        self.assertEqual(resp.status_code, 405)


# =========================================================================
# Unit tests: ChromaDB Collection Detail API
# =========================================================================

class ChromaCollectionDetailTests(TestCase):
    """Test GET/DELETE /api/chroma/collections/<name>/"""

    def setUp(self):
        self.client = Client()

    @patch('router.views.get_or_create_collection')
    def test_get_collection_detail(self, mock_get):
        mock_collection = MagicMock()
        mock_collection.name = 'quran_collection'
        mock_collection.count.return_value = 6236
        mock_get.return_value = mock_collection

        resp = self.client.get('/api/chroma/collections/quran_collection/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['name'], 'quran_collection')
        self.assertEqual(data['count'], 6236)

    @patch('router.views.get_or_create_collection')
    def test_get_collection_detail_error(self, mock_get):
        mock_get.side_effect = RuntimeError("Not found")
        resp = self.client.get('/api/chroma/collections/missing/')
        self.assertEqual(resp.status_code, 500)

    @patch('router.views.delete_collection')
    def test_delete_collection(self, mock_delete):
        mock_delete.return_value = None
        resp = self.client.delete('/api/chroma/collections/old_collection/')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['deleted'], 'old_collection')

    @patch('router.views.delete_collection')
    def test_delete_collection_error(self, mock_delete):
        mock_delete.side_effect = RuntimeError("Cannot delete")
        resp = self.client.delete('/api/chroma/collections/locked/')
        self.assertEqual(resp.status_code, 500)

    def test_detail_method_not_allowed(self):
        resp = self.client.post('/api/chroma/collections/test/')
        self.assertEqual(resp.status_code, 405)


# =========================================================================
# Unit tests: ChromaDB Documents API
# =========================================================================

class ChromaDocumentsTests(TestCase):
    """Test GET/POST/PUT/DELETE /api/chroma/collections/<name>/documents/"""

    def setUp(self):
        self.client = Client()
        # Mock the collection lookup that happens before request validation
        self._coll_patcher = patch('router.views.get_or_create_collection')
        self.mock_get_coll = self._coll_patcher.start()
        self.mock_collection = MagicMock()
        self.mock_get_coll.return_value = self.mock_collection

    def tearDown(self):
        self._coll_patcher.stop()

    @patch('router.views.get_or_create_collection')
    @patch('router.views.get_documents')
    def test_get_documents(self, mock_get_docs, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_get_docs.return_value = {
            'ids': ['id1', 'id2'],
            'documents': ['doc1', 'doc2'],
            'metadatas': [{}, {}],
        }

        resp = self.client.get('/api/chroma/collections/test/documents/')
        self.assertEqual(resp.status_code, 200)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.get_documents')
    def test_get_documents_with_ids(self, mock_get_docs, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_get_docs.return_value = {'ids': ['id1'], 'documents': ['doc1'], 'metadatas': [{}]}

        resp = self.client.get('/api/chroma/collections/test/documents/?ids=id1,id2')
        self.assertEqual(resp.status_code, 200)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.get_documents')
    def test_get_documents_with_limit_offset(self, mock_get_docs, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_get_docs.return_value = {'ids': [], 'documents': [], 'metadatas': []}

        resp = self.client.get('/api/chroma/collections/test/documents/?limit=10&offset=0')
        self.assertEqual(resp.status_code, 200)

    def test_get_documents_invalid_limit(self):
        resp = self.client.get('/api/chroma/collections/test/documents/?limit=abc')
        self.assertEqual(resp.status_code, 400)

    def test_get_documents_invalid_offset(self):
        resp = self.client.get('/api/chroma/collections/test/documents/?offset=abc')
        self.assertEqual(resp.status_code, 400)

    def test_get_documents_invalid_where(self):
        resp = self.client.get('/api/chroma/collections/test/documents/?where=not-json')
        self.assertEqual(resp.status_code, 400)

    @patch('router.views.get_or_create_collection')
    def test_get_documents_collection_error(self, mock_get_coll):
        mock_get_coll.side_effect = RuntimeError("Down")
        resp = self.client.get('/api/chroma/collections/test/documents/')
        self.assertEqual(resp.status_code, 500)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.add_documents')
    def test_add_documents(self, mock_add, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.name = 'test'
        mock_get_coll.return_value = mock_collection

        resp = self.client.post(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({'documents': ['text1', 'text2']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.content)
        self.assertEqual(data['added'], 2)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.add_documents')
    def test_add_documents_with_ids_and_metadata(self, mock_add, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.name = 'test'
        mock_get_coll.return_value = mock_collection

        resp = self.client.post(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({
                'documents': ['text1'],
                'ids': ['custom_id_1'],
                'metadatas': [{'source': 'quran'}],
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.content)
        self.assertEqual(data['added'], 1)

    def test_add_documents_missing_documents_field(self):
        resp = self.client.post(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({'not_documents': 'oops'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_add_documents_empty_body(self):
        resp = self.client.post(
            '/api/chroma/collections/test/documents/',
            data='{}',
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.update_documents')
    def test_update_documents(self, mock_update, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        resp = self.client.put(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({
                'ids': ['id1', 'id2'],
                'documents': ['new1', 'new2'],
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['updated'], 2)

    def test_update_documents_missing_ids(self):
        resp = self.client.put(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({'documents': ['new']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    @patch('router.views.get_or_create_collection')
    @patch('router.views.delete_documents')
    def test_delete_documents(self, mock_delete, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        resp = self.client.delete(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({'ids': ['id1', 'id2']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['deleted'], 2)

    def test_delete_documents_missing_ids(self):
        resp = self.client.delete(
            '/api/chroma/collections/test/documents/',
            data=json.dumps({}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_documents_method_not_allowed(self):
        resp = self.client.patch('/api/chroma/collections/test/documents/')
        self.assertEqual(resp.status_code, 405)


# =========================================================================
# Unit tests: ChromaDB Query API
# =========================================================================

class ChromaQueryTests(TestCase):
    """Test POST /api/chroma/collections/<name>/query/"""

    def setUp(self):
        self.client = Client()

    def test_query_method_not_allowed(self):
        resp = self.client.get('/api/chroma/collections/test/query/')
        self.assertEqual(resp.status_code, 405)

    def test_query_missing_query_texts(self):
        resp = self.client.post(
            '/api/chroma/collections/test/query/',
            data=json.dumps({'n_results': 5}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    @patch('router.views.query_documents')
    @patch('router.views.get_or_create_collection')
    def test_query_success(self, mock_get_coll, mock_query):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_query.return_value = {
            'ids': [['id1']],
            'distances': [[0.1]],
            'metadatas': [[{'source': 'quran'}]],
            'documents': [['text']],
        }

        resp = self.client.post(
            '/api/chroma/collections/test/query/',
            data=json.dumps({'query_texts': ['search term']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)

    @patch('router.views.query_documents')
    @patch('router.views.get_or_create_collection')
    def test_query_with_filters(self, mock_get_coll, mock_query):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_query.return_value = {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}

        resp = self.client.post(
            '/api/chroma/collections/test/query/',
            data=json.dumps({
                'query_texts': ['test'],
                'n_results': 3,
                'where': {'source': 'quran'},
                'where_document': {'$contains': 'allah'},
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)

    @patch('router.views.get_or_create_collection')
    def test_query_error(self, mock_get_coll):
        mock_get_coll.side_effect = RuntimeError("Down")
        resp = self.client.post(
            '/api/chroma/collections/test/query/',
            data=json.dumps({'query_texts': ['test']}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 500)


# =========================================================================
# Unit tests: LLM Generate API
# =========================================================================

class LLMGenerateTests(TestCase):
    """Test POST /api/generate/"""

    def setUp(self):
        self.client = Client()

    def test_generate_method_not_allowed(self):
        resp = self.client.get('/api/generate/')
        self.assertEqual(resp.status_code, 405)

    def test_generate_missing_prompt(self):
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    def test_generate_empty_prompt(self):
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': '   '}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    @patch('router.views.generate')
    def test_generate_success(self, mock_generate):
        mock_generate.return_value = "This is the generated response."
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'What is Islam?'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn('generated', data)
        self.assertEqual(data['generated'], 'This is the generated response.')

    @patch('router.views.generate')
    def test_generate_with_all_params(self, mock_generate):
        mock_generate.return_value = "Response with params."
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({
                'prompt': 'Test',
                'system': 'You are helpful.',
                'model': 'openai/gpt-4o',
                'temperature': 0.5,
                'max_tokens': 100,
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['generated'], 'Response with params.')

    def test_generate_invalid_temperature(self):
        """Non-numeric temperature raises ValueError → 500."""
        with self.assertRaises(ValueError):
            self.client.post(
                '/api/generate/',
                data=json.dumps({'prompt': 'test', 'temperature': 'cold'}),
                content_type='application/json',
            )

    def test_generate_invalid_max_tokens(self):
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'test', 'max_tokens': 'many'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 400)

    @patch('router.views.generate')
    def test_generate_error(self, mock_generate):
        mock_generate.side_effect = RuntimeError("API key invalid")
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'test'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 500)

    @patch('router.views.generate')
    def test_generate_returns_model_in_response(self, mock_generate):
        mock_generate.return_value = "Hello"
        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'Hi', 'model': 'google/gemini-2.5-flash'}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn('model', data)

    @patch('router.views.generate')
    def test_generate_streaming(self, mock_generate):
        """Streaming should return SSE response."""
        mock_chunk = MagicMock()
        mock_chunk.content = "token"
        mock_generate.return_value = iter([mock_chunk])

        resp = self.client.post(
            '/api/generate/',
            data=json.dumps({'prompt': 'test', 'stream': True}),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'text/event-stream')
        self.assertEqual(resp['Cache-Control'], 'no-cache')
        self.assertEqual(resp['X-Accel-Buffering'], 'no')