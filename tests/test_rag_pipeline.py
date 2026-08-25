import unittest
from unittest.mock import MagicMock
from uuid import uuid4
from pipeline.rag_pipeline import RAGPipeline
from core.interfaces.embedding import IEmbeddingService
from core.interfaces.vector_store import IVectorStore
from core.interfaces.llm import ILLMService
from core.models import DocumentChunk, QueryResult


class TestRAGPipeline(unittest.TestCase):
    """
    Unit tests for the RAGPipeline orchestrator.
    Mocks all dependency services to isolate pipeline logic.
    """

    def setUp(self):
        self.mock_embed = MagicMock(spec=IEmbeddingService)
        self.mock_vs = MagicMock(spec=IVectorStore)
        self.mock_llm = MagicMock(spec=ILLMService)

        self.pipeline = RAGPipeline(
            embedding_service=self.mock_embed,
            vector_store=self.mock_vs,
            llm_service=self.mock_llm
        )

        self.doc_id = uuid4()
        # Mock matching query result
        self.chunk = DocumentChunk(
            document_id=self.doc_id,
            content="Standard API port is 8000.",
            chunk_index=0,
            metadata={"filename": "api_manual.pdf", "page_number": 5}
        )
        self.query_result = QueryResult(chunk=self.chunk, similarity_score=0.92)

    def test_query_pipeline_success(self):
        """
        RAGPipeline query should orchestrate embed, search, prompt, and LLM services successfully.
        """
        # Set mock behaviors
        self.mock_embed.embed_query.return_value = [0.2] * 384
        self.mock_vs.similarity_search.return_value = [self.query_result]
        self.mock_llm.generate_response.return_value = "The port is 8000."

        response = self.pipeline.query(user_query="What is the port?", top_k=1)

        # Assert correct workflow orchestration calls
        self.mock_embed.embed_query.assert_called_once_with("What is the port?")
        self.mock_vs.similarity_search.assert_called_once_with([0.2] * 384, limit=30)
        self.mock_llm.generate_response.assert_called_once()

        # Assert response schema structures
        self.assertEqual(response["query"], "What is the port?")
        self.assertEqual(response["answer"], "The port is 8000.")
        
        # Verify deduplicated sources list
        sources = response["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["filename"], "api_manual.pdf")
        self.assertEqual(sources[0]["page_number"], 5)
        self.assertEqual(sources[0]["similarity_score"], 0.92)

        # Verify time metrics
        metrics = response["metrics"]
        self.assertIn("embedding_time_seconds", metrics)
        self.assertIn("retrieval_time_seconds", metrics)
        self.assertIn("llm_time_seconds", metrics)
        self.assertEqual(metrics["chunks_retrieved"], 1)


if __name__ == "__main__":
    unittest.main()
