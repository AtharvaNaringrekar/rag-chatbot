import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from api.router import get_rag_pipeline
from core.exceptions import TechnicalSupportBotException


class TestAPIRoutes(unittest.TestCase):
    """
    Unit tests for FastAPI endpoint routing and schemas.
    """

    def setUp(self):
        self.client = TestClient(app)
        app.dependency_overrides = {}

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("api.router.requests.get")
    @patch("api.router.get_db")
    def test_health_check_endpoint(self, mock_get_db, mock_req_get):
        """
        GET /api/v1/health should return a valid health check payload.
        """
        # Mock database session execution
        mock_db = MagicMock()
        mock_db.execute.return_value = True
        mock_get_db.return_value = mock_db

        # Mock Ollama ping request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_req_get.return_value = mock_response

        # Execute call
        response = self.client.get("/api/v1/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["database_connected"])
        self.assertTrue(data["ollama_reachable"])
        self.assertIn("timestamp", data)

    def test_chat_endpoint_validation(self):
        """
        POST /api/v1/chat should validate input and delegate to pipeline.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "query": "How do I log in?",
            "answer": "Use OAuth.",
            "sources": [
                {
                    "document_id": "12345678-1234-1234-1234-123456789012",
                    "filename": "auth.txt", 
                    "similarity_score": 0.85
                }
            ],
            "metrics": {}
        }
        
        app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

        # Valid request
        payload = {"prompt": "How do I log in?", "top_k": 3}
        response = self.client.post("/api/v1/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Use OAuth.")
        self.assertEqual(len(data["sources"]), 1)
        mock_pipeline.query.assert_called_once_with("How do I log in?", top_k=3)

    def test_chat_vision_endpoint_validation(self):
        """
        POST /api/v1/chat/vision should validate files and route prompts.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query_image.return_value = {
            "query": "Fix this",
            "extracted_text": "Error 401",
            "vision_description": "Terminal",
            "answer": "Check token.",
            "sources": [],
            "metrics": {}
        }
        
        app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

        # Mock multipart form-data request
        file_data = {"image": ("test.png", b"fake_image_bytes", "image/png")}
        data = {"prompt": "Fix this", "top_k": 5}
        
        response = self.client.post("/api/v1/chat/vision", files=file_data, data=data)
        
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["answer"], "Check token.")
        self.assertEqual(json_data["extracted_text"], "Error 401")
        mock_pipeline.query_image.assert_called_once_with(
            b"fake_image_bytes", 
            user_prompt="Fix this", 
            top_k=5
        )

    def test_chat_endpoint_invalid_payload(self):
        """
        POST /api/v1/chat should return 400 Bad Request when pipeline throws exception.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = TechnicalSupportBotException("Prompt cannot be empty")
        app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

        payload = {"prompt": "", "top_k": 3}
        response = self.client.post("/api/v1/chat", json=payload)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Prompt cannot be empty")
