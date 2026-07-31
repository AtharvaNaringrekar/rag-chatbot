import io
import logging
from typing import List, Optional
import numpy as np
from PIL import Image
import easyocr
from core.interfaces.ocr import IOCRService
from core.models import OCRResult
from config.settings import settings
from core.exceptions import OCRException

logger = logging.getLogger(__name__)


class EasyOCRService(IOCRService):
    """
    OCR service implementing IOCRService using the EasyOCR engine.
    Extracts text tokens and bounding box details from raw image bytes.
    """

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Initialize the EasyOCR adapter.
        Uses lazy loading for the reader object to avoid slow startup.

        Args:
            languages: Language codes to parse (e.g. ['en']).
            use_gpu: Enable CUDA acceleration if available.
        """
        self.languages = languages or settings.OCR_LANGUAGES
        self.use_gpu = use_gpu if use_gpu is not None else settings.OCR_USE_GPU
        self._reader: Optional[easyocr.Reader] = None

    @property
    def reader(self) -> easyocr.Reader:
        """
        Property getter implementing thread-safe lazy load pattern for easyocr.Reader.
        """
        if self._reader is None:
            try:
                logger.info(f"Initializing EasyOCR Reader with languages={self.languages} (gpu={self.use_gpu})...")
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR reader: {e}")
                raise OCRException(f"Model initialization failed: {e}")
        return self._reader

    def extract_text(self, image_bytes: bytes) -> OCRResult:
        if not image_bytes:
            raise OCRException("Image bytes payload is empty")

        try:
            # 1. Load image from bytes using Pillow
            image = Image.open(io.BytesIO(image_bytes))
            
            # 2. Convert to RGB numpy array which EasyOCR reader expects
            img_np = np.array(image.convert("RGB"))

            # 3. Perform text detection
            logger.info("Executing EasyOCR text detection on image array...")
            detections = self.reader.readtext(img_np)
            
            if not detections:
                logger.warning("EasyOCR detection finished: No text detected in the image.")
                return OCRResult(extracted_text="", confidence=1.0, raw_detections=[])

            # 4. Compile result segments and compute confidence averages
            extracted_segments = []
            confidence_scores = []
            raw_details = []

            for bbox, text, conf in detections:
                extracted_segments.append(text)
                confidence_scores.append(float(conf))
                # Map bounding box coordinates to serializable structure
                # bbox is typically list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                raw_details.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox]
                })

            full_text = "\n".join(extracted_segments)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0

            logger.info(f"OCR transcription finished. Extracted {len(extracted_segments)} lines with confidence={avg_confidence:.3f}")
            return OCRResult(
                extracted_text=full_text,
                confidence=avg_confidence,
                raw_detections=raw_details
            )

        except Exception as e:
            logger.error(f"Error during EasyOCR text extraction: {e}", exc_info=True)
            raise OCRException(f"Text extraction failed: {e}")

    def extract_text_from_file(self, file_path: str) -> OCRResult:
        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            return self.extract_text(image_bytes)
        except OCRException as oe:
            raise oe
        except Exception as e:
            logger.error(f"Failed to read image file at '{file_path}': {e}")
            raise OCRException(f"Failed to parse image from file: {e}")
