import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings configuration class.
    
    Loads configuration values from environment variables or a .env file,
    providing strong typing and validation for application settings.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # FastAPI settings
    APP_NAME: str = Field(default="AI Technical Support Assistant", description="The name of the application")
    DEBUG: bool = Field(default=False, description="Enable or disable debug mode")
    API_HOST: str = Field(default="127.0.0.1", description="Host address for FastAPI server")
    API_PORT: int = Field(default=8000, description="Port for FastAPI server")

    # Database settings
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/tech_support_db",
        description="PostgreSQL database connection string (must have pgvector extension installed)"
    )
    DB_ECHO: bool = Field(default=False, description="SQLAlchemy query logging flag")

    # Ollama LLM settings
    OLLAMA_BASE_URL: str = Field(default="http://127.0.0.1:11434", description="Ollama API base URL")
    LLM_MODEL: str = Field(default="qwen2.5:1.5b", description="Name of the LLM model configured in Ollama")
    LLM_TEMPERATURE: float = Field(default=0.0, description="Temperature for response generation (0.0 for deterministic answers)")

    # Embedding settings
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2", description="HuggingFace model name for embeddings")
    EMBEDDING_DEVICE: str = Field(default="cpu", description="Device to run embedding models ('cpu', 'cuda', etc.)")

    # Ingestion & Chunking settings
    DEFAULT_CHUNK_SIZE: int = Field(default=1000, description="Default character size for text chunks")
    DEFAULT_CHUNK_OVERLAP: int = Field(default=100, description="Character overlap between consecutive chunks")

    # OCR settings
    OCR_LANGUAGES: List[str] = Field(default=["en"], description="List of languages for EasyOCR/Tesseract parsing")
    OCR_USE_GPU: bool = Field(default=False, description="Flag to leverage GPU for OCR acceleration if available")

    # Vision settings
    VISION_MODEL_NAME: str = Field(default="llava:latest", description="Vision-capable model name or API stub identifier")
    VISION_API_KEY: str = Field(default="", description="API key if using external vision providers (e.g. Gemini, OpenAI)")


# Create a single global settings instance
settings = Settings()
