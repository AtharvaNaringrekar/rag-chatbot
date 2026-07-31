from abc import ABC, abstractmethod
from typing import Tuple, List
from core.models import Document, DocumentChunk


class IDocumentParser(ABC):
    """
    Interface for parsing various technical and manual document formats.
    Implementations will exist for OpenAPI/Swagger (JSON/YAML), Postman Collections, PDF, DOCX, and TXT.
    """

    @abstractmethod
    def parse(self, file_content: bytes, filename: str) -> Tuple[Document, List[DocumentChunk]]:
        """
        Parse raw file bytes into a Document model and a list of its constituent DocumentChunks.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename: The original name of the file (useful for determining metadata).

        Returns:
            A tuple containing:
                - Document: The metadata model for the entire file.
                - List[DocumentChunk]: Individual text segments mapped with structural metadata.
        """
        pass
