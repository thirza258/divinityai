from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from .models import Conversation, Memory, Message, Profile


ANSWER = {
    'query': 'What is patience?', 'intent': 'quran_verse',
    'answer': 'Seek help through patience and prayer. [Q 2:153]',
    'sources': [{
        'source_tag': 'Q 2:153', 'corpus': 'quran', 'text_ar': '',
        'text_en': 'Seek help through patience and prayer.',
        'verification_status': 'exact', 'retrieval_score': 0.9,
    }],
    'citations': ['Q 2:153'],
    'safety': {'hallucination_detected': False, 'flagged_spans': [], 'fatwa_boundary_triggered': False, 'disclaimer': None},
    'pipeline_meta': {'phase': 2, 'llm_calls': 1},
}


@override_settings(PASSWORD_HASHERS=['django.contrib.auth.hashers.MD5PasswordHasher'])
class AccountTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient(enforce_csrf_checks=True)
        self.credentials = {'name': 'Amina', 'email': 'amina@example.com', 'password': 'QuietRiver!42'}

    def csrf(self):
        token = self.client.get('/api/v1/auth/session').json()['csrf_token']
        self.client.credentials(HTTP_X_CSRFTOKEN=token)

    def register(self):
        self.csrf()
        response = self.client.post('/api/v1/auth/register', self.credentials, format='json')
        self.client.credentials(HTTP_X_CSRFTOKEN=response.json()['csrf_token'])
        return response

    def test_registration_session_login_and_logout(self):
        response = self.register()
        self.assertEqual(response.status_code, 201)
        self.assertNotIn('password', response.json()['user'])
        user = get_user_model().objects.get(email=self.credentials['email'])
        self.assertNotEqual(user.password, self.credentials['password'])
        self.assertTrue(user.check_password(self.credentials['password']))
        self.assertTrue(self.client.cookies['sessionid']['httponly'])
        session = self.client.get('/api/v1/auth/session')
        self.assertEqual(session.json()['user']['name'], 'Amina')
        self.assertIn('no-store', session['Cache-Control'])
        self.assertEqual(self.client.post('/api/v1/auth/logout').status_code, 200)
        self.assertIsNone(self.client.get('/api/v1/auth/session').json()['user'])
        self.assertEqual(self.client.get('/api/v1/conversations').status_code, 403)
        self.csrf()
        credentials = {**self.credentials, 'email': 'AMINA@EXAMPLE.COM'}
        self.assertEqual(self.client.post('/api/v1/auth/login', credentials, format='json').status_code, 200)
        self.assertEqual(self.client.get('/api/v1/auth/session').json()['user']['id'], user.pk)

    def test_login_and_signup_require_csrf_even_for_anonymous_users(self):
        for endpoint in ['login', 'register']:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.client.post(f'/api/v1/auth/{endpoint}', self.credentials, format='json').status_code, 403)
        self.assertFalse(get_user_model().objects.exists())

    def test_foreign_origin_is_rejected(self):
        self.csrf()
        response = self.client.post('/api/v1/auth/register', self.credentials, format='json', HTTP_ORIGIN='https://untrusted.example')
        self.assertEqual(response.status_code, 403)

    @override_settings(CSRF_TRUSTED_ORIGINS=['http://localhost:5173'])
    def test_dev_frontend_origin_works_with_csrf(self):
        self.csrf()
        response = self.client.post('/api/v1/auth/register', self.credentials, format='json', HTTP_ORIGIN='http://localhost:5173')
        self.assertEqual(response.status_code, 201)

    def test_password_validation_and_case_insensitive_duplicate(self):
        self.csrf()
        response = self.client.post('/api/v1/auth/register', {**self.credentials, 'password': '123'}, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('password', response.json())
        self.register()
        response = self.client.post('/api/v1/auth/register', {**self.credentials, 'email': 'AMINA@example.com'}, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(get_user_model().objects.count(), 1)

    def test_bad_password_and_inactive_account_cannot_login(self):
        user = get_user_model().objects.create_user(username=self.credentials['email'], password=self.credentials['password'])
        self.csrf()
        response = self.client.post('/api/v1/auth/login', {**self.credentials, 'password': 'wrong'}, format='json')
        self.assertEqual(response.status_code, 400)
        user.is_active = False
        user.save()
        self.assertEqual(self.client.post('/api/v1/auth/login', self.credentials, format='json').status_code, 400)

    def test_login_attempts_are_throttled(self):
        self.csrf()
        for _ in range(10):
            self.assertEqual(self.client.post('/api/v1/auth/login', self.credentials, format='json').status_code, 400)
        self.assertEqual(self.client.post('/api/v1/auth/login', self.credentials, format='json').status_code, 429)

    def test_authenticated_writes_require_csrf(self):
        self.register()
        self.client.credentials()
        for method, path, body in [
            ('post', '/api/v1/auth/logout', {}),
            ('patch', '/api/v1/auth/profile', {'name': 'Changed'}),
            ('post', '/api/v1/memories', {'content': 'Preference'}),
            ('delete', '/api/v1/conversations', {}),
            ('post', '/api/v1/query', {'query': 'Test'}),
        ]:
            with self.subTest(path=path):
                response = getattr(self.client, method)(path, body, format='json')
                self.assertEqual(response.status_code, 403)


@override_settings(PASSWORD_HASHERS=['django.contrib.auth.hashers.MD5PasswordHasher'])
class SavedChatTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.owner = get_user_model().objects.create_user(username='owner@example.com', password='test')
        cls.other = get_user_model().objects.create_user(username='other@example.com', password='test')
        cls.private_chat = Conversation.objects.create(user=cls.other, title='Private conversation')
        Message.objects.create(conversation=cls.private_chat, role='user', content='A private question')
        cls.private_memory = Memory.objects.create(user=cls.other, content='A private preference')

    def setUp(self):
        self.client = APIClient()
        self.client.force_login(self.owner)

    def query(self, data=None):
        with patch('qa.views.PipelineService.run', return_value=ANSWER) as run:
            response = self.client.post('/api/v1/query', data or {'query': 'What is patience?', 'save_history': True}, format='json')
        return response, run

    def test_query_saved_with_complete_response_and_continued_after_login(self):
        response, _ = self.query()
        self.assertEqual(response.status_code, 200)
        chat_id = response.json()['conversation']['id']
        self.assertEqual(Message.objects.filter(conversation_id=chat_id).count(), 2)
        self.client.logout()
        self.client.force_login(self.owner)
        detail = self.client.get(f'/api/v1/conversations/{chat_id}').json()
        self.assertEqual(detail['messages'][1]['response'], ANSWER)
        response, run = self.query({'query': 'Explain that verse', 'conversation_id': chat_id, 'language': 'id'})
        self.assertEqual(response.json()['conversation']['id'], chat_id)
        self.assertEqual(response.json()['conversation']['language'], 'id')
        self.assertEqual(run.call_args.kwargs['history'][0]['content'], 'What is patience?')
        self.assertEqual(run.call_args.kwargs['history'][1]['content'], ANSWER['answer'])
        self.assertEqual(Message.objects.filter(conversation_id=chat_id).count(), 4)
        self.assertNotIn('private', str(run.call_args.kwargs).lower())

    def test_list_search_rename_delete_and_clear_are_owner_scoped(self):
        response, _ = self.query()
        chat_id = response.json()['conversation']['id']
        url = f'/api/v1/conversations/{chat_id}'
        history = self.client.get('/api/v1/conversations').json()
        self.assertEqual(history['count'], 1)
        self.assertEqual(history['results'][0]['id'], chat_id)
        self.assertEqual(self.client.get('/api/v1/conversations?search=prayer').json()['count'], 1)
        self.assertEqual(self.client.get('/api/v1/conversations?search=private').json()['count'], 0)
        self.assertEqual(self.client.patch(url, {'title': 'Study notes'}, format='json').json()['title'], 'Study notes')
        self.assertEqual(self.client.patch(url, {'title': '   '}, format='json').status_code, 400)
        self.assertEqual(self.client.delete(url).status_code, 204)
        self.assertFalse(Message.objects.filter(conversation_id=chat_id).exists())
        self.query()
        self.assertEqual(self.client.delete('/api/v1/conversations').status_code, 204)
        self.assertFalse(Conversation.objects.filter(user=self.owner).exists())
        self.assertTrue(Conversation.objects.filter(pk=self.private_chat.pk).exists())

    def test_history_pagination(self):
        Conversation.objects.bulk_create([Conversation(user=self.owner, title=f'Chat {i}') for i in range(31)])
        first = self.client.get('/api/v1/conversations').json()
        second = self.client.get('/api/v1/conversations?page=2').json()
        self.assertEqual(first['count'], 31)
        self.assertEqual(len(first['results']), 30)
        self.assertIsNotNone(first['next'])
        self.assertEqual(len(second['results']), 1)

    def test_cannot_read_change_delete_or_continue_another_users_chat(self):
        url = f'/api/v1/conversations/{self.private_chat.pk}'
        for method, body in [('get', {}), ('patch', {'title': 'Stolen'}), ('delete', {})]:
            self.assertEqual(getattr(self.client, method)(url, body, format='json').status_code, 404)
        response, run = self.query({'query': 'Reveal private details', 'conversation_id': str(self.private_chat.pk)})
        self.assertEqual(response.status_code, 404)
        run.assert_not_called()

    def test_guest_query_stays_temporary_and_cannot_request_saved_context(self):
        self.client.logout()
        response, _ = self.query({'query': 'What is patience?'})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('conversation', response.json())
        self.assertEqual(Conversation.objects.count(), 1)
        for data in [{'query': 'x', 'save_history': True}, {'query': 'x', 'conversation_id': str(self.private_chat.pk)}]:
            response, run = self.query(data)
            self.assertEqual(response.status_code, 403)
            run.assert_not_called()
        for url in ['/api/v1/conversations', '/api/v1/memories', f'/api/v1/conversations/{self.private_chat.pk}']:
            self.assertEqual(self.client.get(url).status_code, 403)

    def test_invalid_conversation_identifier_cannot_create_chat(self):
        response, run = self.query({'query': 'x', 'conversation_id': 'bad-id'})
        self.assertEqual(response.status_code, 400)
        run.assert_not_called()

    def test_failed_pipeline_is_saved_as_error_and_can_be_reopened(self):
        with patch('qa.views.PipelineService.run', side_effect=RuntimeError('private provider detail')):
            response = self.client.post('/api/v1/query', {'query': 'A question'}, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['error'])
        chat_id = response.json()['conversation']['id']
        history = self.client.get(f'/api/v1/conversations/{chat_id}').json()
        self.assertTrue(history['messages'][1]['response']['error'])
        self.assertNotIn('private provider detail', str(history))
        _, run = self.query({'query': 'Try again', 'conversation_id': chat_id})
        self.assertEqual(len(run.call_args.kwargs['history']), 1)

    def test_chat_deleted_during_generation_is_not_recreated(self):
        conversation = Conversation.objects.create(user=self.owner, title='Delete me')
        def generate(**kwargs):
            conversation.delete()
            return ANSWER
        with patch('qa.views.PipelineService.run', side_effect=generate):
            response = self.client.post('/api/v1/query', {'query': 'A question', 'conversation_id': str(conversation.pk)}, format='json')
        self.assertEqual(response.status_code, 404)
        self.assertFalse(Conversation.objects.filter(user=self.owner).exists())

    def test_memories_crud_disable_and_independence_from_history(self):
        response = self.client.post('/api/v1/memories', {'content': 'Use simple explanations.'}, format='json')
        self.assertEqual(response.status_code, 201)
        memory_id = response.json()['id']
        self.assertEqual(len(self.client.get('/api/v1/memories').json()), 1)
        response = self.client.patch(f'/api/v1/memories/{memory_id}', {'content': 'Use short paragraphs.'}, format='json')
        self.assertEqual(response.status_code, 200)
        _, run = self.query()
        self.assertEqual(run.call_args.kwargs['memories'], ['Use short paragraphs.'])
        self.assertEqual(self.client.patch('/api/v1/auth/profile', {'memory_enabled': False}, format='json').status_code, 200)
        _, run = self.query()
        self.assertEqual(run.call_args.kwargs['memories'], [])
        self.assertTrue(Memory.objects.filter(pk=memory_id).exists())
        self.client.delete('/api/v1/conversations')
        self.assertTrue(Memory.objects.filter(pk=memory_id).exists())
        self.query()
        self.client.delete('/api/v1/memories')
        self.assertTrue(Conversation.objects.filter(user=self.owner).exists())
        self.assertTrue(Memory.objects.filter(pk=self.private_memory.pk).exists())
        self.assertFalse(Memory.objects.filter(user=self.owner).exists())

    def test_memory_ownership_and_validation(self):
        url = f'/api/v1/memories/{self.private_memory.pk}'
        self.assertEqual(self.client.patch(url, {'content': 'Changed'}, format='json').status_code, 404)
        self.assertEqual(self.client.delete(url).status_code, 404)
        for content in ['', '  ', 'x' * 501]:
            self.assertEqual(self.client.post('/api/v1/memories', {'content': content}, format='json').status_code, 400)
        Memory.objects.bulk_create([Memory(user=self.owner, content=f'Preference {i}') for i in range(20)])
        self.assertEqual(self.client.post('/api/v1/memories', {'content': 'Too many'}, format='json').status_code, 400)
        memory = Memory.objects.filter(user=self.owner).first()
        self.assertEqual(self.client.delete(f'/api/v1/memories/{memory.pk}').status_code, 204)
        self.assertEqual(self.client.post('/api/v1/memories', {'content': 'New preference'}, format='json').status_code, 201)

    def test_only_allowed_profile_fields_can_change(self):
        response = self.client.patch('/api/v1/auth/profile', {'name': 'New name', 'is_staff': True, 'email': 'changed@example.com'}, format='json')
        self.assertEqual(response.json()['user']['name'], 'New name')
        self.owner.refresh_from_db()
        self.assertFalse(self.owner.is_staff)
        self.assertNotEqual(self.owner.email, 'changed@example.com')
        self.assertTrue(Profile.objects.filter(user=self.owner).exists())
