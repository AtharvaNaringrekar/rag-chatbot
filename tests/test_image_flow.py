import unittest
from unittest.mock import MagicMock
from uuid import uuid4
from pipeline.rag_pipeline import RAGPipeline
from core.interfaces.embedding import IEmbeddingService
from core.interfaces.vector_store import IVectorStore
from core.interfaces.llm import ILLMService
from core.interfaces.ocr import IOCRService
from core.interfaces.vision import IVisionService
from core.models import DocumentChunk, QueryResult, OCRResult, VisionAnalysisResult


class TestRAGImagePipelineFlow(unittest.TestCase):
    """
    Unit tests for the extended query_image method inside RAGPipeline.
    Mocks all five service boundaries to run isolated test checks.
    """

    def setUp(self):
        self.mock_embed = MagicMock(spec=IEmbeddingService)
        self.mock_vs = MagicMock(spec=IVectorStore)
        self.mock_llm = MagicMock(spec=ILLMService)
        self.mock_ocr = MagicMock(spec=IOCRService)
        self.mock_vision = MagicMock(spec=IVisionService)

        self.pipeline = RAGPipeline(
            embedding_service=self.mock_embed,
            vector_store=self.mock_vs,
            llm_service=self.mock_llm,
            ocr_service=self.mock_ocr,
            vision_service=self.mock_vision
        )

        self.doc_id = uuid4()
        self.chunk = DocumentChunk(
            document_id=self.doc_id,
            content="Error 500 represents server exception.",
            chunk_index=0,
            metadata={"filename": "troubleshooting.pdf"}
        )
        self.query_result = QueryResult(chunk=self.chunk, similarity_score=0.95)

    def test_query_image_pipeline_success(self):
        """
        query_image should run the full sequence: OCR -> Vision -> Embed -> Search -> LLM.
        """
        # 1. Setup Mock responses
        self.mock_ocr.extract_text.return_value = OCRResult(
            extracted_text="FATAL: Connection timed out at port 5432",
            confidence=0.98,
            raw_detections=[]
        )
        self.mock_vision.analyze_image.return_value = VisionAnalysisResult(
            description="Dark CLI terminal window showing database connect logs.",
            detected_errors=["Connection timed out"],
            source_component="Terminal"
        )
        self.mock_embed.embed_query.return_value = [0.1] * 384
        self.mock_vs.similarity_search.return_value = [self.query_result]
        self.mock_llm.generate_response.return_value = "Ensure port 5432 is open and the service is active."

        # 2. Run test
        response = self.pipeline.query_image(
            image_bytes=b"dummy_screenshot_payload",
            user_prompt="Help fix this database connection issue"
        )

        # 3. Assertions on service calls
        self.mock_ocr.extract_text.assert_called_once_with(b"dummy_screenshot_payload")
        self.mock_vision.analyze_image.assert_called_once_with(b"dummy_screenshot_payload", prompt="Help fix this database connection issue")
        self.mock_embed.embed_query.assert_called_once()
        self.mock_vs.similarity_search.assert_called_once()
        self.mock_llm.generate_response.assert_called_once()

        # 4. Assert response payload values
        self.assertEqual(response["query"], "Help fix this database connection issue")
        self.assertEqual(response["extracted_text"], "FATAL: Connection timed out at port 5432")
        self.assertEqual(response["vision_description"], "Dark CLI terminal window showing database connect logs.")
        self.assertIn("Ensure port 5432 is open", response["answer"])
        
        # Sources
        self.assertEqual(len(response["sources"]), 1)
        self.assertEqual(response["sources"][0]["filename"], "troubleshooting.pdf")

        # Check metrics timings
        metrics = response["metrics"]
        self.assertIn("ocr_time_seconds", metrics)
        self.assertIn("vision_time_seconds", metrics)
        self.assertIn("embedding_time_seconds", metrics)
        self.assertIn("retrieval_time_seconds", metrics)
        self.assertIn("llm_time_seconds", metrics)
        self.assertEqual(metrics["chunks_retrieved"], 1)


if __name__ == "__main__":
    unittest.main()
