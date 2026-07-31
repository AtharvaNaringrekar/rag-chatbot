class TechnicalSupportBotException(Exception):
    """Base exception class for the AI Technical Support Assistant application."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# Parser Exceptions
class ParserException(TechnicalSupportBotException):
    """Base exception for document parsing failures."""
    pass


class UnsupportedFormatException(ParserException):
    """Raised when an uploaded file format is not supported by the system."""
    def __init__(self, file_extension: str):
        super().__init__(f"The file extension '{file_extension}' is not supported. Supported: .json, .yaml, .yml, .pdf, .docx, .txt")


class ParsingException(ParserException):
    """Raised when there is a syntax or semantic error during document processing."""
    def __init__(self, filename: str, reason: str):
        super().__init__(f"Failed to parse document '{filename}': {reason}")


# Database / Vector Store Exceptions
class DatabaseException(TechnicalSupportBotException):
    """Base exception for general database failures."""
    pass


class DatabaseConnectionException(DatabaseException):
    """Raised when connection to the PostgreSQL database fails."""
    def __init__(self, connection_details: str, reason: str):
        super().__init__(f"Failed to connect to database using details '{connection_details}'. Reason: {reason}")


class VectorStoreException(TechnicalSupportBotException):
    """Raised when saving or querying operations fail within the vector store."""
    def __init__(self, operation: str, reason: str):
        super().__init__(f"Vector store operation '{operation}' failed: {reason}")


# LLM & Embedding Exceptions
class EmbeddingException(TechnicalSupportBotException):
    """Raised when generating embeddings from text chunks fails."""
    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Failed to generate embeddings using model '{model_name}': {reason}")


class LLMException(TechnicalSupportBotException):
    """Raised when communicating with the local Ollama instance or external LLM API fails."""
    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Failed to communicate with LLM '{model_name}': {reason}")


# OCR & Vision Exceptions
class OCRException(TechnicalSupportBotException):
    """Raised when OCR text extraction from screenshots fails."""
    def __init__(self, reason: str):
        super().__init__(f"OCR extraction failed: {reason}")


class VisionException(TechnicalSupportBotException):
    """Raised when visual analysis of screenshots fails."""
    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Vision model '{model_name}' analysis failed: {reason}")
