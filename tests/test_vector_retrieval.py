import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from services.vector_store.postgres import PostgresVectorStore
from database.models import DBDocumentChunk
from core.models import QueryResult


class TestPostgresVectorRetrieval(unittest.TestCase):
    """
    Unit tests for similarity search logic within PostgresVectorStore.
    Mocks SQLAlchemy queries to isolate database tests.
    """

    def setUp(self):
        self.doc_id = uuid4()
        self.mock_session = MagicMock()
        self.vector_store = PostgresVectorStore(db_session=self.mock_session)

        # Mock database rows
        self.db_chunk1 = DBDocumentChunk(
            id=uuid4(),
            document_id=self.doc_id,
            content="Retrieve checkout details.",
            chunk_index=0,
            meta_data={"filename": "doc.pdf", "page_number": 2}
        )
        self.db_chunk2 = DBDocumentChunk(
            id=uuid4(),
            document_id=self.doc_id,
            content="Create a checkout cart.",
            chunk_index=1,
            meta_data={"filename": "doc.pdf", "page_number": 3}
        )

    def test_similarity_search_returns_query_results(self):
        """
        similarity_search should execute queries, calculate scores, and map models.
        """
        # Mocking query chain
        mock_query = self.mock_session.query.return_value
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        # Mock database returning (DBDocumentChunk, cosine_distance)
        # Cosine distance = 0.15 (similarity = 0.85)
        # Cosine distance = 0.40 (similarity = 0.60)
        mock_query.all.return_value = [
            (self.db_chunk1, 0.15),
            (self.db_chunk2, 0.40)
        ]

        query_emb = [0.1] * 384
        results = self.vector_store.similarity_search(query_emb, limit=2)

        # Assert query is correct
        self.assertEqual(len(results), 2)
        
        # Verify first match
        r1 = results[0]
        self.assertEqual(r1.chunk.content, "Retrieve checkout details.")
        self.assertEqual(r1.chunk.document_id, self.doc_id)
        # Similarity score = 1.0 - cosine_distance
        self.assertAlmostEqual(r1.similarity_score, 0.85)
        self.assertEqual(r1.chunk.metadata["page_number"], 2)

        # Verify second match
        r2 = results[1]
        self.assertEqual(r2.chunk.content, "Create a checkout cart.")
        self.assertAlmostEqual(r2.similarity_score, 0.60)

    def test_similarity_search_with_metadata_filters(self):
        """
        If filter_metadata is provided, it should inject filter clauses into query.
        """
        mock_query = self.mock_session.query.return_value
        mock_query.filter.return_value = mock_query  # Mock filter calls
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        query_emb = [0.1] * 384
        filters = {"filename": "doc.pdf"}
        
        self.vector_store.similarity_search(query_emb, limit=5, filter_metadata=filters)

        # Verify filter was called
        mock_query.filter.assert_called()


if __name__ == "__main__":
    unittest.main()
