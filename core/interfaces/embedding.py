from abc import ABC, abstractmethod
from typing import List


class IEmbeddingService(ABC):
    """
    Interface for text embedding generation.
    Enables swapping local models (sentence-transformers) with API models (OpenAI, HuggingFace).
    """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text query.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of document strings.

        Args:
            texts: List of text items to embed.

        Returns:
            A list of list of floats representing the vectors.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the vector dimension output size of the underlying embedding model.

        Returns:
            Integer dimension (e.g., 384 for all-MiniLM-L6-v2).
        """
        pass
