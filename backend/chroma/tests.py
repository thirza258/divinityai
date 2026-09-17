"""Unit tests for ChromaDB utility functions."""

from unittest.mock import patch, MagicMock

from chromadb.errors import NotFoundError
from django.test import TestCase

from chroma.chroma_settings import CHROMA_DEFAULT_COLLECTION, CHROMA_DISTANCE_METRIC
from chroma.chroma_utils import (
    add_documents,
    delete_collection,
    delete_documents,
    get_chroma_client,
    get_collection_count,
    get_documents,
    get_or_create_collection,
    list_collections,
    query_documents,
    reset_chroma_client,
    update_documents,
)


# =========================================================================
# Unit tests: Client factory
# =========================================================================

class ChromaClientTests(TestCase):
    """Test ChromaDB client creation and reset."""

    def tearDown(self):
        reset_chroma_client()

    def test_reset_clears_cached_client(self):
        """reset_chroma_client clears the cached client."""
        # First get a client to cache it
        with patch('chroma.chroma_utils.chromadb.PersistentClient') as mock_pc:
            mock_pc.return_value = MagicMock()
            client1 = get_chroma_client()
            self.assertIsNotNone(client1)

        # Reset should clear the cache
        reset_chroma_client()

        # Next call should create a new client
        with patch('chroma.chroma_utils.chromadb.PersistentClient') as mock_pc:
            mock_pc.return_value = MagicMock()
            client2 = get_chroma_client()
            self.assertIsNotNone(client2)
            self.assertIsNot(client1, client2)


# =========================================================================
# Unit tests: Collection management
# =========================================================================

class CollectionManagementTests(TestCase):
    """Test get_or_create_collection, list_collections, delete_collection."""

    def tearDown(self):
        reset_chroma_client()

    @patch('chroma.chroma_utils.get_embedding_function')
    @patch('chroma.chroma_utils.get_chroma_client')
    def test_get_or_create_collection_reuses_existing(self, mock_get_client, mock_get_emb):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        mock_get_emb.return_value = MagicMock()

        coll = get_or_create_collection(name='test_collection')
        self.assertEqual(coll, mock_collection)
        mock_client.get_collection.assert_called_once_with(name='test_collection')
        mock_client.get_or_create_collection.assert_not_called()
        mock_get_emb.assert_not_called()

    @patch('chroma.chroma_utils.get_embedding_function')
    @patch('chroma.chroma_utils.get_chroma_client')
    def test_get_or_create_collection_default_name(self, mock_get_client, mock_get_emb):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        mock_get_emb.return_value = MagicMock()

        coll = get_or_create_collection()
        self.assertEqual(coll, mock_collection)
        mock_client.get_collection.assert_called_once_with(name=CHROMA_DEFAULT_COLLECTION)
        mock_client.get_or_create_collection.assert_not_called()
        mock_get_emb.assert_not_called()

    @patch('chroma.chroma_utils.get_embedding_function')
    @patch('chroma.chroma_utils.get_chroma_client')
    def test_get_or_create_collection_with_embedding_fn(self, mock_get_client, mock_get_emb):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.side_effect = NotFoundError('collection missing')
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        mock_get_emb.return_value = MagicMock()

        emb_fn = MagicMock()
        coll = get_or_create_collection(name='test', embedding_function=emb_fn)
        self.assertEqual(coll, mock_collection)
        mock_client.get_or_create_collection.assert_called_once_with(
            name='test', embedding_function=emb_fn,
            metadata={'hnsw:space': CHROMA_DISTANCE_METRIC},
        )
        mock_get_emb.assert_not_called()

    @patch('chroma.chroma_utils.get_embedding_function')
    @patch('chroma.chroma_utils.get_chroma_client')
    def test_creates_missing_collection_with_default_embeddings(self, mock_get_client, mock_get_emb):
        mock_client = mock_get_client.return_value
        mock_client.get_collection.side_effect = NotFoundError('collection missing')
        metadata = {'hnsw:space': 'l2', 'corpus': 'quran'}

        coll = get_or_create_collection(name='new_collection', metadata=metadata)

        self.assertEqual(coll, mock_client.get_or_create_collection.return_value)
        mock_get_emb.assert_called_once_with()
        mock_client.get_or_create_collection.assert_called_once_with(
            name='new_collection', embedding_function=mock_get_emb.return_value,
            metadata=metadata,
        )

    @patch('chroma.chroma_utils.get_embedding_function')
    @patch('chroma.chroma_utils.get_chroma_client')
    def test_existing_collection_keeps_its_persisted_configuration(self, mock_get_client, mock_get_emb):
        mock_client = mock_get_client.return_value

        coll = get_or_create_collection(
            name='existing_collection', embedding_function=MagicMock(),
            metadata={'hnsw:space': 'cosine'},
        )

        self.assertEqual(coll, mock_client.get_collection.return_value)
        mock_client.get_collection.assert_called_once_with(name='existing_collection')
        mock_client.get_or_create_collection.assert_not_called()
        mock_get_emb.assert_not_called()

    @patch('chroma.chroma_utils.get_embedding_function')
    @patch('chroma.chroma_utils.get_chroma_client')
    def test_lookup_errors_do_not_trigger_collection_creation(self, mock_get_client, mock_get_emb):
        mock_client = mock_get_client.return_value
        for error in [ValueError('embedding configuration conflict'), RuntimeError('connection failed')]:
            with self.subTest(error=error):
                mock_client.get_collection.side_effect = error
                with self.assertRaises(type(error)):
                    get_or_create_collection(name='test_collection')
        mock_client.get_or_create_collection.assert_not_called()
        mock_get_emb.assert_not_called()

    @patch('chroma.chroma_utils.get_chroma_client')
    def test_list_collections(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_collections.return_value = ['coll_a', 'coll_b']
        mock_get_client.return_value = mock_client

        names = list_collections()
        self.assertEqual(names, ['coll_a', 'coll_b'])

    @patch('chroma.chroma_utils.get_chroma_client')
    def test_list_collections_empty(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_get_client.return_value = mock_client

        names = list_collections()
        self.assertEqual(names, [])

    @patch('chroma.chroma_utils.get_chroma_client')
    def test_delete_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        delete_collection('old_collection')
        mock_client.delete_collection.assert_called_once_with('old_collection')

    @patch('chroma.chroma_utils.get_chroma_client')
    def test_delete_collection_not_found(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = NotFoundError("not found")
        mock_get_client.return_value = mock_client

        # Deleting an already absent collection is a no-op.
        delete_collection('nonexistent')

    @patch('chroma.chroma_utils.get_chroma_client')
    def test_delete_collection_other_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        for error in [ValueError('invalid collection name'), RuntimeError('fatal')]:
            with self.subTest(error=error):
                mock_client.delete_collection.side_effect = error
                with self.assertRaises(type(error)):
                    delete_collection('doomed')

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_get_collection_count(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_get_coll.return_value = mock_collection

        count = get_collection_count(name='test')
        self.assertEqual(count, 42)

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_get_collection_count_with_collection_obj(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10

        count = get_collection_count(collection=mock_collection)
        self.assertEqual(count, 10)
        mock_get_coll.assert_not_called()


# =========================================================================
# Unit tests: Document CRUD
# =========================================================================

class DocumentCRUDTests(TestCase):
    """Test add_documents, get_documents, update_documents, delete_documents."""

    def tearDown(self):
        reset_chroma_client()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_add_documents_with_ids(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        add_documents(
            documents=['text1', 'text2'],
            ids=['id1', 'id2'],
            collection_name='test',
        )
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args.kwargs
        self.assertEqual(call_kwargs['ids'], ['id1', 'id2'])

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_add_documents_auto_generates_ids(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        add_documents(
            documents=['text1'],
            collection_name='test',
        )
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args.kwargs
        # Auto-generated UUIDs should be strings
        self.assertIsNotNone(call_kwargs['ids'])
        self.assertEqual(len(call_kwargs['ids']), 1)

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_add_documents_with_metadatas(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        add_documents(
            documents=['text1'],
            metadatas=[{'source': 'quran'}],
            ids=['id1'],
            collection_name='test',
        )
        mock_collection.add.assert_called_once()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_add_documents_with_existing_collection_obj(self, mock_get_coll):
        mock_collection = MagicMock()

        add_documents(
            documents=['text1'],
            ids=['id1'],
            collection=mock_collection,
        )
        mock_collection.add.assert_called_once()
        mock_get_coll.assert_not_called()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_query_documents(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['id1']],
            'distances': [[0.1]],
            'metadatas': [[{}]],
            'documents': [['text']],
        }
        mock_get_coll.return_value = mock_collection

        result = query_documents(
            query_texts=['search'],
            n_results=5,
            collection_name='test',
        )
        self.assertIn('ids', result)
        mock_collection.query.assert_called_once()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_query_documents_with_filters(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {'ids': [[]]}
        mock_get_coll.return_value = mock_collection

        result = query_documents(
            query_texts=['test'],
            n_results=3,
            where={'source': 'quran'},
            where_document={'$contains': 'allah'},
            collection_name='test',
        )
        self.assertIn('ids', result)

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_get_documents_by_ids(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'ids': ['id1', 'id2'],
            'documents': ['doc1', 'doc2'],
        }
        mock_get_coll.return_value = mock_collection

        result = get_documents(ids=['id1', 'id2'], collection_name='test')
        self.assertEqual(result['ids'], ['id1', 'id2'])

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_get_documents_by_metadata_filter(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': ['id1']}
        mock_get_coll.return_value = mock_collection

        result = get_documents(
            where={'source': 'quran'},
            limit=10,
            offset=0,
            collection_name='test',
        )
        self.assertEqual(result['ids'], ['id1'])

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_update_documents(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        update_documents(
            ids=['id1', 'id2'],
            documents=['new1', 'new2'],
            collection_name='test',
        )
        mock_collection.update.assert_called_once()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_update_documents_metadatas_only(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        update_documents(
            ids=['id1'],
            metadatas=[{'updated': True}],
            collection_name='test',
        )
        mock_collection.update.assert_called_once()

    @patch('chroma.chroma_utils.get_or_create_collection')
    def test_delete_documents(self, mock_get_coll):
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection

        delete_documents(ids=['id1', 'id2'], collection_name='test')
        mock_collection.delete.assert_called_once_with(ids=['id1', 'id2'])
