from abc import ABC, abstractmethod
from core.models import OCRResult


class IOCRService(ABC):
    """
    Interface for Optical Character Recognition (OCR).
    Allows extracting raw code tokens, console outputs, and traceback errors from screenshots.
    Enables swapping EasyOCR, Tesseract, or cloud OCR engines (like Google Cloud Vision API).
    """

    @abstractmethod
    def extract_text(self, image_bytes: bytes) -> OCRResult:
        """
        Extract text and structure metadata from image bytes.

        Args:
            image_bytes: The raw binary content of the screenshot.

        Returns:
            An OCRResult domain model with the compiled text and raw detection metadata.
        """
        pass

    @abstractmethod
    def extract_text_from_file(self, file_path: str) -> OCRResult:
        """
        Extract text from a locally saved image file.

        Args:
            file_path: The absolute filesystem path to the image file.

        Returns:
            An OCRResult domain model with the compiled text and raw detection metadata.
        """
        pass
