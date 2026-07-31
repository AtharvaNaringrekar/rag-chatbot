import os
from abc import ABC, abstractmethod
from typing import Tuple, List
from core.interfaces.parser import IDocumentParser
from core.models import Document, DocumentChunk
from ingestion.normalizers import normalize_text
from core.exceptions import ParsingException


class BaseParser(IDocumentParser, ABC):
    """
    Abstract Base Parser implementing the IDocumentParser interface.
    Provides shared helper utilities for text normalization, extension parsing,
    and standard exception routing.
    """

    @abstractmethod
    def parse(self, file_content: bytes, filename: str) -> Tuple[Document, List[DocumentChunk]]:
        """
        Parse raw file bytes into a Document and list of chunks.
        Must be implemented by concrete subclasses.
        """
        pass

    def _get_extension(self, filename: str) -> str:
        """
        Extract the lowercase file extension from the filename.
        """
        _, ext = os.path.splitext(filename)
        return ext.lower()

    def _normalize(self, text: str) -> str:
        """
        Helper method to normalize text segments.
        Delegates to the ingestion package normalizer.
        """
        return normalize_text(text)

    def _handle_error(self, filename: str, error: Exception) -> None:
        """
        Helper to standardise exception wrapping.
        """
        raise ParsingException(filename, str(error))
