from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from core.interfaces.vector_store import IVectorStore
from core.models import Document, DocumentChunk, QueryResult
from database.models import DBDocument, DBDocumentChunk
from database.connection import SessionLocal
from core.exceptions import VectorStoreException


class PostgresVectorStore(IVectorStore):
    """
    PostgreSQL vector store adapter using pgvector and SQLAlchemy.
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the vector store.

        Args:
            db_session: Optional SQLAlchemy Session. If omitted,
                        a session will be created locally for each transaction.
        """
        self._session = db_session

    def _get_session(self) -> Session:
        """
        Helper to return the injected session or a new one.
        """
        return self._session if self._session is not None else SessionLocal()

    def save_document(self, document: Document) -> None:
        session = self._get_session()
        is_local_session = self._session is None

        try:
            # Check if document already exists, if so merge/update
            db_doc = session.query(DBDocument).filter(DBDocument.id == document.id).first()
            if db_doc:
                db_doc.filename = document.filename
                db_doc.file_type = document.file_type
                db_doc.uploaded_at = document.uploaded_at
                db_doc.meta_data = document.metadata
            else:
                db_doc = DBDocument(
                    id=document.id,
                    filename=document.filename,
                    file_type=document.file_type,
                    uploaded_at=document.uploaded_at,
                    meta_data=document.metadata
                )
                session.add(db_doc)

            if is_local_session:
                session.commit()
            else:
                session.flush()
        except Exception as e:
            if is_local_session:
                session.rollback()
            raise VectorStoreException("save_document", str(e))
        finally:
            if is_local_session:
                session.close()

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return

        session = self._get_session()
        is_local_session = self._session is None

        try:
            # Check if parent document needs to be created or if it's already there
            # Since all chunks share the same parent document_id, check the first chunk
            doc_id = chunks[0].document_id
            
            # We assume the parent Document was already stored or we check and insert it.
            # To be safe and robust, we verify document existence.
            db_doc = session.query(DBDocument).filter(DBDocument.id == doc_id).first()
            if not db_doc:
                # If document metadata isn't present, create a stub document
                # This prevents foreign key constraint failures
                db_doc = DBDocument(
                    id=doc_id,
                    filename=chunks[0].metadata.get("filename", "unknown"),
                    file_type="unknown",
                    meta_data={"created_automatically": True}
                )
                session.add(db_doc)
                session.flush()  # Push document to DB so foreign key checks succeed

            # Insert chunks
            for chunk in chunks:
                db_chunk = DBDocumentChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    embedding=chunk.embedding,
                    chunk_index=chunk.chunk_index,
                    meta_data=chunk.metadata
                )
                session.add(db_chunk)

            # Commit the session if managed locally
            if is_local_session:
                session.commit()
            else:
                session.flush()  # Hand off transaction control tocaller

        except Exception as e:
            if is_local_session:
                session.rollback()
            raise VectorStoreException("save_chunks", str(e))
        finally:
            if is_local_session:
                session.close()

    def similarity_search(
        self, 
        query_embedding: List[float], 
        limit: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        session = self._get_session()
        is_local_session = self._session is None

        try:
            # Compute cosine distance
            c_distance = DBDocumentChunk.embedding.cosine_distance(query_embedding)
            
            # Setup base query
            query = session.query(DBDocumentChunk, c_distance)

            # Apply metadata filters if provided
            if filter_metadata:
                for key, val in filter_metadata.items():
                    # Check path variables or metadata keys
                    query = query.filter(DBDocumentChunk.meta_data[key].astext == str(val))

            # Order by smallest distance and execute
            results = query.order_by(c_distance).limit(limit).all()

            query_results = []
            for db_chunk, dist in results:
                # Map back to domain model
                chunk_domain = DocumentChunk(
                    id=db_chunk.id,
                    document_id=db_chunk.document_id,
                    content=db_chunk.content,
                    embedding=db_chunk.embedding,
                    chunk_index=db_chunk.chunk_index,
                    metadata=db_chunk.meta_data
                )
                
                # Cosine Similarity = 1.0 - Cosine Distance
                score = 1.0 - dist if dist is not None else 0.0
                
                query_results.append(
                    QueryResult(chunk=chunk_domain, similarity_score=score)
                )

            return query_results

        except Exception as e:
            raise VectorStoreException("similarity_search", str(e))
        finally:
            if is_local_session:
                session.close()

    def delete_document_chunks(self, document_id: UUID) -> None:
        session = self._get_session()
        is_local_session = self._session is None

        try:
            # Delete parent document. Cascades handle deleting child chunks.
            doc = session.query(DBDocument).filter(DBDocument.id == document_id).first()
            if doc:
                session.delete(doc)
                if is_local_session:
                    session.commit()
                else:
                    session.flush()

        except Exception as e:
            if is_local_session:
                session.rollback()
            raise VectorStoreException("delete_document_chunks", str(e))
        finally:
            if is_local_session:
                session.close()
