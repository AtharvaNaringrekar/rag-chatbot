from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from core.models import Document, DocumentChunk, QueryResult


class IVectorStore(ABC):
    """
    Interface for vector database operations.
    Enables storing embeddings and performing similarity searches.
    Allows swapping pgvector with Pinecone, Qdrant, Chroma, etc.
    """

    @abstractmethod
    def save_document(self, document: Document) -> None:
        """
        Save the master document metadata to the database.

        Args:
            document: The Document domain model containing metadata.
        """
        pass

    @abstractmethod
    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Save a batch of document chunks, including their embeddings, to the vector store.

        Args:
            chunks: A list of DocumentChunk domain models containing valid embeddings.
        """
        pass

    @abstractmethod
    def similarity_search(
        self, 
        query_embedding: List[float], 
        limit: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """
        Perform a vector similarity search (e.g. Cosine Similarity) against stored document chunks.

        Args:
            query_embedding: The vector embedding of the user's prompt.
            limit: Maximum number of matching chunks to return.
            filter_metadata: Optional metadata filters (e.g. matching a specific document_id).

        Returns:
            A list of QueryResult models containing matching chunks and their similarity scores.
        """
        pass

    @abstractmethod
    def delete_document_chunks(self, document_id: UUID) -> None:
        """
        Remove all document chunks and embeddings associated with a specific document ID.

        Args:
            document_id: The UUID of the document to clean up.
        """
        pass
