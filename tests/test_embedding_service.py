import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from services.embedding.sentence_transformer import SentenceTransformersEmbedding
from core.exceptions import EmbeddingException


class TestSentenceTransformersEmbedding(unittest.TestCase):
    """
    Unit tests for the SentenceTransformersEmbedding wrapper class.
    Mocks the sentence-transformers library to run tests quickly and offline.
    """

    @patch("services.embedding.sentence_transformer.SentenceTransformer")
    def test_initialization_success(self, mock_transformer):
        """
        Initialization should correctly instantiate the SentenceTransformer model.
        """
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance

        service = SentenceTransformersEmbedding(model_name="dummy-model", device="cpu")

        # Verify constructor parameters routed to model loader
        mock_transformer.assert_called_once_with("dummy-model", device="cpu")
        self.assertEqual(service.model_name, "dummy-model")
        self.assertEqual(service.device, "cpu")

    @patch("services.embedding.sentence_transformer.SentenceTransformer")
    def test_initialization_raises_exception(self, mock_transformer):
        """
        If SentenceTransformer raises an error, constructor should wrap it in an EmbeddingException.
        """
        mock_transformer.side_effect = RuntimeError("Failed to load weight layers")

        with self.assertRaises(EmbeddingException):
            SentenceTransformersEmbedding(model_name="broken-model")

    @patch("services.embedding.sentence_transformer.SentenceTransformer")
    def test_embed_query(self, mock_transformer):
        """
        embed_query should encode a single string and return a list of float values.
        """
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance
        # Mock encoding output as numpy array
        mock_instance.encode.return_value = np.array([0.1, -0.2, 0.3])

        service = SentenceTransformersEmbedding()
        vector = service.embed_query("search text")

        mock_instance.encode.assert_called_once_with("search text", convert_to_numpy=True)
        self.assertEqual(vector, [0.1, -0.2, 0.3])

    @patch("services.embedding.sentence_transformer.SentenceTransformer")
    def test_embed_documents(self, mock_transformer):
        """
        embed_documents should encode multiple strings and return a list of list of float values.
        """
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance
        mock_instance.encode.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ])

        service = SentenceTransformersEmbedding()
        vectors = service.embed_documents(["doc one", "doc two"])

        mock_instance.encode.assert_called_once_with(
            ["doc one", "doc two"], 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])

    @patch("services.embedding.sentence_transformer.SentenceTransformer")
    def test_get_embedding_dimension(self, mock_transformer):
        """
        get_embedding_dimension should return the correct vector size.
        """
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        service = SentenceTransformersEmbedding()
        dim = service.get_embedding_dimension()

        mock_instance.get_sentence_embedding_dimension.assert_called_once()
        self.assertEqual(dim, 384)


if __name__ == "__main__":
    unittest.main()
