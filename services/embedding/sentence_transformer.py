from typing import List
from sentence_transformers import SentenceTransformer
from core.interfaces.embedding import IEmbeddingService
from config.settings import settings
from core.exceptions import EmbeddingException


class SentenceTransformersEmbedding(IEmbeddingService):
    """
    Embedding service implementing IEmbeddingService using the local
    HuggingFace sentence-transformers library.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, device: str = settings.EMBEDDING_DEVICE):
        """
        Initialize the sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier.
            device: Computing device ('cpu', 'cuda', etc.).
        """
        self.model_name = model_name
        self.device = device
        try:
            # Load the sentence transformer model into memory
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingException(model_name, f"Failed to initialize sentence transformer model: {e}")

    def embed_query(self, text: str) -> List[float]:
        try:
            if not text:
                raise ValueError("Text query to embed cannot be empty")
            
            # Generate embedding vector
            vector = self._model.encode(text, convert_to_numpy=True)
            return vector.tolist()
        except Exception as e:
            raise EmbeddingException(self.model_name, f"Error embedding query: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                return []
            
            # Batch encode documents
            vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            return vectors.tolist()
        except Exception as e:
            raise EmbeddingException(self.model_name, f"Error embedding document list: {e}")

    def get_embedding_dimension(self) -> int:
        try:
            dimension = self._model.get_sentence_embedding_dimension()
            return int(dimension)
        except Exception as e:
            # Fallback for all-MiniLM-L6-v2 if call fails
            return 384
