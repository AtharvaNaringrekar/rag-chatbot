import logging
import time
from typing import Dict, Any
from core.interfaces.embedding import IEmbeddingService
from core.interfaces.vector_store import IVectorStore
from services.parser.factory import DocumentParserFactory
from core.exceptions import TechnicalSupportBotException

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestration pipeline for document ingestion.
    Coordinates file parsing, text normalization, semantic chunking,
    vector embedding generation, and storage operations.
    """

    def __init__(self, embedding_service: IEmbeddingService, vector_store: IVectorStore):
        """
        Initialize the Ingestion Pipeline.

        Args:
            embedding_service: Service to generate vector embeddings.
            vector_store: Target vector database store.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Ingest an uploaded file, run it through the processing pipeline,
        and save its embeddings and contents to pgvector.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: Original name of the file.

        Returns:
            A dictionary containing ingestion execution statistics.
            
        Raises:
            TechnicalSupportBotException: On pipeline failures.
        """
        start_time = time.time()
        logger.info(f"Starting ingestion process for file: '{filename}'")

        try:
            # 1. Resolve parser using the factory
            parser = DocumentParserFactory.get_parser(filename, file_content)
            logger.info(f"Resolved parser '{parser.__class__.__name__}' for file '{filename}'")

            # 2. Parse the document (this yields a Document and its constituent List[DocumentChunk])
            # The concrete parsers also perform text normalization and chunking internally.
            doc, chunks = parser.parse(file_content, filename)
            logger.info(f"Parsed file '{filename}' into Document ID '{doc.id}' with {len(chunks)} raw chunks.")

            if not chunks:
                logger.warning(f"No text content could be extracted from '{filename}'. Ingestion skipped.")
                return {
                    "status": "skipped",
                    "filename": filename,
                    "reason": "No text content found",
                    "total_chunks": 0
                }

            # 3. Generate embeddings in a single batch
            logger.info(f"Generating vector embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.embed_documents(chunk_texts)
            
            # Map embeddings back to chunks
            for idx, embedding in enumerate(embeddings):
                chunks[idx].embedding = embedding

            # 4. Save master document metadata in the database
            logger.info(f"Persisting master Document metadata for ID: '{doc.id}'")
            self.vector_store.save_document(doc)

            # 5. Save the chunks + embeddings to pgvector
            logger.info(f"Persisting {len(chunks)} chunks and embeddings into PostgreSQL...")
            self.vector_store.save_chunks(chunks)

            duration = time.time() - start_time
            logger.info(f"Ingestion of '{filename}' completed successfully in {duration:.2f} seconds.")

            return {
                "status": "success",
                "document_id": str(doc.id),
                "filename": doc.filename,
                "file_type": doc.file_type,
                "total_chunks": len(chunks),
                "duration_seconds": round(duration, 3),
                "metadata": doc.metadata
            }

        except TechnicalSupportBotException as tbe:
            # Re-raise application-specific exceptions
            logger.error(f"Application error during ingestion of '{filename}': {tbe.message}")
            raise tbe
        except Exception as e:
            # Wrap any unhandled infrastructure exception in a standard container
            err_msg = f"Unexpected failure in ingestion pipeline: {e}"
            logger.error(err_msg, exc_info=True)
            raise TechnicalSupportBotException(err_msg)
