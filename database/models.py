import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from database.connection import Base


class DBDocument(Base):
    """
    SQLAlchemy model representing the 'documents' table.
    Stores metadata about files uploaded to the technical support assistant.
    """
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    meta_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Relationship to delete chunks when parent document is deleted
    chunks = relationship(
        "DBDocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class DBDocumentChunk(Base):
    """
    SQLAlchemy model representing the 'document_chunks' table.
    Stores partitioned text content and their high-dimensional vector embeddings.
    """
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # 384 dimensions matching the output of sentence-transformers/all-MiniLM-L6-v2
    embedding: Mapped[list] = mapped_column(Vector(384), nullable=True)
    
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    meta_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    # Reference back to parent document
    document = relationship("DBDocument", back_populates="chunks")


class DBChatMessage(Base):
    """
    SQLAlchemy model representing the 'chat_history' table.
    Maintains stateless session conversations for user support requests.
    """
    __tablename__ = "chat_history"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    meta_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
