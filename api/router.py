import os
import logging
import time
from datetime import datetime
from typing import Optional
import requests
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from config.settings import settings
from database.connection import get_db
from api.schemas import (
    ChatRequest, ChatResponse, VisionChatResponse, 
    IngestionResponse, HealthResponse
)
from services.embedding.sentence_transformer import SentenceTransformersEmbedding
from services.vector_store.postgres import PostgresVectorStore
from services.llm.ollama import OllamaLLMService
from services.ocr.easy_ocr import EasyOCRService
from services.vision.vision_stub import VisionLLMStubService
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.rag_pipeline import RAGPipeline
from core.exceptions import TechnicalSupportBotException

logger = logging.getLogger(__name__)

# Initialize APIRouter
router = APIRouter(prefix="/api/v1", tags=["Technical Support Assistant"])

# PERFORMANCE OPTIMIZATION: Instantiate heavy model loaders globally
# This prevents loading PyTorch weights and models on every request cycle.
logger.info("APIRouter: Preloading SentenceTransformers embedding model...")
embedding_service = SentenceTransformersEmbedding()

logger.info("APIRouter: Preloading EasyOCR engine...")
ocr_service = EasyOCRService()

logger.info("APIRouter: Initializing Vision stub...")
vision_service = VisionLLMStubService()

logger.info("APIRouter: Preloading Ollama LLM Service connection session...")
llm_service = OllamaLLMService(max_tokens=300)


# Dependency injection providers
def get_ingestion_pipeline(db: Session = Depends(get_db)) -> IngestionPipeline:
    vector_store = PostgresVectorStore(db_session=db)
    return IngestionPipeline(embedding_service=embedding_service, vector_store=vector_store)


def get_rag_pipeline(db: Session = Depends(get_db)) -> RAGPipeline:
    vector_store = PostgresVectorStore(db_session=db)
    return RAGPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        ocr_service=ocr_service,
        vision_service=vision_service
    )


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    Perform connectivity checks against PostgreSQL database and local Ollama API.
    """
    db_ok = False
    ollama_ok = False

    # 1. Test Database
    try:
        db.execute(text("SELECT 1;"))
        db_ok = True
    except Exception as e:
        logger.error(f"Health check database connection failed: {e}")

    # 2. Test Ollama
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        if response.status_code == 200:
            ollama_ok = True
    except Exception as e:
        logger.error(f"Health check Ollama connection failed: {e}")

    return HealthResponse(
        status="ok" if (db_ok and ollama_ok) else "degraded",
        database_connected=db_ok,
        ollama_reachable=ollama_ok,
        timestamp=datetime.utcnow().isoformat()
    )


@router.post("/ingest", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
):
    """
    Upload and parse documentation file (PDF, DOCX, TXT, OpenAPI JSON/YAML, Postman Collection).
    Extracts text, generates sentence vectors, and stores chunks in PostgreSQL pgvector.
    """
    # Validate file size or availability
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file payload is empty."
        )

    # Restrict file size (DoS prevention)
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    if len(file_bytes) > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file size exceeds maximum permitted limit (50MB)."
        )

    try:
        # Trigger ingestion pipeline
        logger.info(f"Route: Ingesting file '{file.filename}'...")
        stats = pipeline.ingest_file(file_bytes, file.filename)
        
        return IngestionResponse(
            status=stats["status"],
            document_id=stats.get("document_id"),
            filename=stats["filename"],
            file_type=stats["file_type"],
            total_chunks=stats["total_chunks"],
            duration_seconds=stats["duration_seconds"],
            metadata=stats.get("metadata", {})
        )

    except TechnicalSupportBotException as tbe:
        logger.error(f"Route Ingestion failed: {tbe.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=tbe.message
        )


@router.post("/chat", response_model=ChatResponse)
def chat_query(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Submit a standard text query to search matching technical documentation
    and synthesize a response using local Ollama.
    """
    try:
        logger.info(f"Route: Incoming Chat request from client.")
        logger.info(f"Route: Prompt: '{request.prompt}'")
        logger.info(f"Route: Top_k: {request.top_k}")
        start_route = time.time()
        
        result = pipeline.query(request.prompt, top_k=request.top_k)
        
        duration = time.time() - start_route
        logger.info(f"Route chat completed successfully in {duration:.3f} seconds.")
        logger.info(f"Route chat metrics details: {result['metrics']}")
        
        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            metrics=result["metrics"]
        )

    except TechnicalSupportBotException as tbe:
        logger.error(f"Route Chat query failed: {tbe.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=tbe.message
        )


@router.post("/chat/vision", response_model=VisionChatResponse)
def chat_vision_query(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    top_k: Optional[int] = Form(5),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Submit a screenshot image (PNG/JPEG) of errors, logs, or UI layouts.
    Extracts text via OCR, analyzes environment via Vision, matches documents,
    and returns diagnostic troubleshooting steps.
    """
    # 1. Validate image format extension
    _, ext = os.path.splitext(image.filename.lower())
    if ext not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image file format '{ext}'. Must be .png, .jpg, or .jpeg"
        )

    # 2. Read raw binary
    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image binary payload is empty."
        )

    # Restrict screenshot size (DoS prevention)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image size exceeds maximum permitted limit (10MB)."
        )

    try:
        logger.info(f"Route: Incoming Image chat request from client.")
        logger.info(f"Route: Prompt: '{prompt}'")
        logger.info(f"Route: Top_k: {top_k}")
        logger.info(f"Route: Image Size: {len(image_bytes)} bytes")
        start_route = time.time()
        
        result = pipeline.query_image(image_bytes, user_prompt=prompt, top_k=top_k)
        
        duration = time.time() - start_route
        logger.info(f"Route vision chat completed successfully in {duration:.3f} seconds.")
        logger.info(f"Route vision chat metrics details: {result['metrics']}")
        
        return VisionChatResponse(
            query=result["query"] or "Image Troubleshooting",
            extracted_text=result["extracted_text"],
            vision_description=result["vision_description"],
            answer=result["answer"],
            sources=result["sources"],
            metrics=result["metrics"]
        )

    except TechnicalSupportBotException as tbe:
        logger.error(f"Route Vision chat failed: {tbe.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=tbe.message
        )
