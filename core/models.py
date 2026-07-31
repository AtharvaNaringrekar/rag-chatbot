from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Domain model representing a file uploaded for technical documentation ingestion.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier of the document")
    filename: str = Field(..., description="Original name of the uploaded file")
    file_type: str = Field(..., description="Extension or format category (e.g., 'pdf', 'openapi', 'docx')")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of when the document was uploaded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (e.g., file size, author, description)")


class DocumentChunk(BaseModel):
    """
    Domain model representing a parsed, split, and optional embedded segment of a Document.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the chunk")
    document_id: UUID = Field(..., description="ID of the parent document this chunk belongs to")
    content: str = Field(..., description="Text content of the document chunk")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation (typically 384 dimensions for all-MiniLM-L6-v2)")
    chunk_index: int = Field(..., description="Zero-based sequence order index of this chunk within the parent document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Segment-specific metadata (e.g., api_endpoint, page number, header key)")


class QueryResult(BaseModel):
    """
    Domain model representing a retrieval match result from the vector database.
    """
    chunk: DocumentChunk = Field(..., description="The matching document chunk retrieved")
    similarity_score: float = Field(..., description="The cosine similarity score (typically 0.0 to 1.0)")


class OCRResult(BaseModel):
    """
    Domain model representing text extracted from a screenshot or image via OCR.
    """
    extracted_text: str = Field(..., description="Raw text combined from OCR output")
    confidence: float = Field(default=1.0, description="Average OCR extraction confidence score")
    raw_detections: List[Any] = Field(default_factory=list, description="Raw coordinates, text segments, and scores from the OCR backend")


class VisionAnalysisResult(BaseModel):
    """
    Domain model representing semantic layout and context analysis from a Vision LLM.
    """
    description: str = Field(..., description="Semantic description of the screenshot (e.g., 'A CLI printout depicting a connection timeout error')")
    detected_errors: List[str] = Field(default_factory=list, description="List of recognized error phrases or exceptions in the visual frame")
    source_component: str = Field(default="unknown", description="Determined source app (e.g., 'VS Code', 'Swagger UI', 'Terminal', 'Postman')")


class ChatMessage(BaseModel):
    """
    Domain model representing a message in the chat history.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier of the message")
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(..., description="Text message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when message was sent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata (e.g. links to source chunks, error analysis results)")
