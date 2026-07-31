import unittest
from unittest.mock import patch, MagicMock
from services.ocr.easy_ocr import EasyOCRService
from core.exceptions import OCRException


class TestEasyOCRService(unittest.TestCase):
    """
    Unit tests for the EasyOCRService adapter.
    Mocks easyocr.Reader and Pillow to run tests offline and fast.
    """

    @patch("services.ocr.easy_ocr.easyocr.Reader")
    @patch("services.ocr.easy_ocr.Image.open")
    @patch("services.ocr.easy_ocr.np.array")
    def test_extract_text_success(self, mock_np_array, mock_image_open, mock_reader_class):
        """
        extract_text should open the image, transcribe it via EasyOCR,
        and compile the result DTO correctly.
        """
        # 1. Setup mock Pillow Image
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        # 2. Setup mock Reader instance
        mock_reader_inst = MagicMock()
        mock_reader_class.return_value = mock_reader_inst
        
        # Mock easyocr detections return format:
        # [ (bbox, text, confidence) ]
        mock_bbox1 = [[10, 10], [50, 10], [50, 20], [10, 20]]
        mock_bbox2 = [[10, 30], [80, 30], [80, 40], [10, 40]]
        mock_reader_inst.readtext.return_value = [
            (mock_bbox1, "Connection refused", 0.90),
            (mock_bbox2, "at port 8000", 0.80)
        ]

        # Initialize and test
        service = EasyOCRService()
        result = service.extract_text(b"mock_jpeg_bytes")

        # Assertions
        self.assertEqual(result.extracted_text, "Connection refused\nat port 8000")
        self.assertAlmostEqual(result.confidence, 0.85)  # Average of 0.90 and 0.80
        self.assertEqual(len(result.raw_detections), 2)
        self.assertEqual(result.raw_detections[0]["text"], "Connection refused")
        self.assertEqual(result.raw_detections[0]["confidence"], 0.90)
        self.assertEqual(result.raw_detections[0]["bbox"], mock_bbox1)

    @patch("services.ocr.easy_ocr.Image.open")
    def test_extract_text_empty_bytes(self, mock_image_open):
        """
        Passing empty bytes should raise an OCRException immediately.
        """
        service = EasyOCRService()
        with self.assertRaises(OCRException):
            service.extract_text(b"")

    @patch("services.ocr.easy_ocr.easyocr.Reader")
    @patch("services.ocr.easy_ocr.Image.open")
    def test_extract_text_handles_crashes(self, mock_image_open, mock_reader_class):
        """
        If easyocr or pillow throws an error, it should raise an OCRException.
        """
        mock_image_open.side_effect = RuntimeError("Corrupted PNG headers")
        
        service = EasyOCRService()
        with self.assertRaises(OCRException):
            service.extract_text(b"broken_bytes")


if __name__ == "__main__":
    unittest.main()
