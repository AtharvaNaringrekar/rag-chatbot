import unittest
from unittest.mock import patch, MagicMock
import requests
from services.llm.ollama import OllamaLLMService
from core.exceptions import LLMException


class TestOllamaLLMService(unittest.TestCase):
    """
    Unit tests for the OllamaLLMService.
    Mocks requests.post to avoid hitting a running Ollama server.
    """

    def setUp(self):
        self.service = OllamaLLMService(
            base_url="http://mock-ollama:11434",
            model_name="phi3:mini",
            temperature=0.2,
            max_tokens=100,
            timeout_seconds=30.0
        )

    @patch("services.llm.ollama.requests.post")
    def test_generate_response_success(self, mock_post):
        """
        generate_response should make a POST request and return response text on HTTP 200.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is the answer."}
        mock_post.return_value = mock_response

        answer = self.service.generate_response(prompt="Hello", system_prompt="Be polite")

        self.assertEqual(answer, "This is the answer.")
        mock_post.assert_called_once_with(
            "http://mock-ollama:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": "Hello",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 100},
                "system": "Be polite"
            },
            timeout=30.0
        )

    @patch("services.llm.ollama.requests.post")
    def test_generate_response_timeout(self, mock_post):
        """
        If requests.post raises a Timeout, the service should wrap it in an LLMException.
        """
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        with self.assertRaises(LLMException) as ctx:
            self.service.generate_response(prompt="Hello")
            
        self.assertIn("Request timed out", ctx.exception.message)

    @patch("services.llm.ollama.requests.post")
    def test_generate_response_connection_error(self, mock_post):
        """
        If requests.post raises a connection error, the service should wrap it in an LLMException.
        """
        mock_post.side_effect = requests.exceptions.ConnectionError("Could not resolve host")

        with self.assertRaises(LLMException) as ctx:
            self.service.generate_response(prompt="Hello")

        self.assertIn("Connection failed", ctx.exception.message)

    @patch("services.llm.ollama.requests.post")
    def test_generate_response_stream(self, mock_post):
        """
        generate_response_stream should parse streamed chunks line by line and yield content.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Mocking streaming lines returned by Ollama
        mock_response.iter_lines.return_value = [
            b'{"response": "token1", "done": false}',
            b'{"response": " token2", "done": false}',
            b'{"response": "", "done": true}'
        ]
        mock_post.return_value = mock_response

        stream_gen = self.service.generate_response_stream(prompt="Stream query")
        chunks = list(stream_gen)

        self.assertEqual(chunks, ["token1", " token2"])
        mock_post.assert_called_once_with(
            "http://mock-ollama:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": "Stream query",
                "stream": True,
                "options": {"temperature": 0.2, "num_predict": 100}
            },
            stream=True,
            timeout=30.0
        )


if __name__ == "__main__":
    unittest.main()
