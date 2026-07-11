"""Unit tests for generation/llm_service.py."""

from unittest.mock import patch, MagicMock

from django.test import TestCase

from generation.llm_service import (
    generate,
    generate_with_history,
    get_cached_llm,
    get_llm,
)


# =========================================================================
# Unit tests: get_llm
# =========================================================================

class GetLLMTests(TestCase):
    """Test LLM client factory."""

    @patch('generation.llm_service.ChatOpenAI')
    def test_get_llm_creates_client(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = get_llm(model='openai/gpt-4o', temperature=0.5)
        self.assertEqual(llm, mock_llm)
        mock_chat_openai.assert_called_once()

    @patch('generation.llm_service.ChatOpenAI')
    def test_get_llm_uses_defaults(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = get_llm()
        self.assertEqual(llm, mock_llm)

    @patch('generation.llm_service.ChatOpenAI')
    def test_get_llm_warns_on_missing_key(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        with patch('generation.llm_service.OPENROUTER_API_KEY', ''):
            llm = get_llm()
            self.assertEqual(llm, mock_llm)

    @patch('generation.llm_service.ChatOpenAI')
    def test_get_llm_with_custom_api_key(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = get_llm(api_key='sk-custom-key')
        self.assertEqual(llm, mock_llm)
        call_kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(call_kwargs['api_key'], 'sk-custom-key')

    @patch('generation.llm_service.ChatOpenAI')
    def test_get_llm_with_all_params(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = get_llm(
            model='google/gemini-2.5-flash',
            temperature=0.3,
            max_tokens=512,
            api_key='sk-key',
            base_url='https://custom.api/v1',
        )
        self.assertEqual(llm, mock_llm)
        call_kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(call_kwargs['model'], 'google/gemini-2.5-flash')
        self.assertEqual(call_kwargs['temperature'], 0.3)
        self.assertEqual(call_kwargs['max_tokens'], 512)
        self.assertEqual(call_kwargs['api_key'], 'sk-key')
        self.assertEqual(call_kwargs['base_url'], 'https://custom.api/v1')


# =========================================================================
# Unit tests: get_cached_llm
# =========================================================================

class GetCachedLLMTests(TestCase):
    """Test cached LLM singleton."""

    def tearDown(self):
        # Reset the cached LLM singleton
        import generation.llm_service as mod
        mod._llm = None

    @patch('generation.llm_service.ChatOpenAI')
    def test_cached_llm_returns_same_instance(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm1 = get_cached_llm()
        llm2 = get_cached_llm()
        self.assertIs(llm1, llm2)
        # ChatOpenAI should only be called once
        self.assertEqual(mock_chat_openai.call_count, 1)

    @patch('generation.llm_service.ChatOpenAI')
    def test_cached_llm_passes_params_on_first_call(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        get_cached_llm(model='openai/gpt-4o', temperature=0.0, max_tokens=256)
        call_kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(call_kwargs['model'], 'openai/gpt-4o')
        self.assertEqual(call_kwargs['temperature'], 0.0)
        self.assertEqual(call_kwargs['max_tokens'], 256)


# =========================================================================
# Unit tests: generate
# =========================================================================

class GenerateTests(TestCase):
    """Test generate() function."""

    @patch('generation.llm_service.get_llm')
    def test_generate_returns_content(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated response text"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = generate(prompt="Hello")
        self.assertEqual(result, "Generated response text")

    @patch('generation.llm_service.get_llm')
    def test_generate_with_system_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = generate(
            prompt="What is Islam?",
            system="You are an Islamic scholar.",
        )
        self.assertEqual(result, "Response")
        # Verify invoke was called with a SystemMessage + HumanMessage
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 2)

    @patch('generation.llm_service.get_llm')
    def test_generate_with_model_override(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        generate(prompt="test", model="google/gemini-2.5-flash")
        mock_get_llm.assert_called_once()
        call_kwargs = mock_get_llm.call_args.kwargs
        self.assertEqual(call_kwargs['model'], 'google/gemini-2.5-flash')

    @patch('generation.llm_service.get_llm')
    def test_generate_streaming(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_stream = MagicMock()
        mock_llm.stream.return_value = mock_stream
        mock_get_llm.return_value = mock_llm

        result = generate(prompt="test", stream=True)
        self.assertEqual(result, mock_stream)
        mock_llm.stream.assert_called_once()
        mock_llm.invoke.assert_not_called()

    @patch('generation.llm_service.get_llm')
    def test_generate_with_temperature(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Creative"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        generate(prompt="test", temperature=0.9)
        call_kwargs = mock_get_llm.call_args.kwargs
        self.assertEqual(call_kwargs['temperature'], 0.9)


# =========================================================================
# Unit tests: generate_with_history
# =========================================================================

class GenerateWithHistoryTests(TestCase):
    """Test generate_with_history() function."""

    @patch('generation.llm_service.get_llm')
    def test_generate_with_history(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response to history"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = generate_with_history(messages=[
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hello'},
        ])
        self.assertEqual(result, "Response to history")

    @patch('generation.llm_service.get_llm')
    def test_generate_with_history_maps_roles(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Ok"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        generate_with_history(messages=[
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'User message'},
            {'role': 'user', 'content': 'Another user message'},
        ])
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 3)

    @patch('generation.llm_service.get_llm')
    def test_generate_with_history_empty_messages(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = generate_with_history(messages=[])
        self.assertEqual(result, "")

    @patch('generation.llm_service.get_llm')
    def test_generate_with_history_streaming(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_stream = MagicMock()
        mock_llm.stream.return_value = mock_stream
        mock_get_llm.return_value = mock_llm

        result = generate_with_history(
            messages=[{'role': 'user', 'content': 'Hi'}],
            stream=True,
        )
        self.assertEqual(result, mock_stream)
        mock_llm.stream.assert_called_once()


