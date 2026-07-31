import unittest
from unittest.mock import patch, MagicMock
from services.vision.vision_stub import VisionLLMStubService
from core.exceptions import VisionException


class TestVisionLLMStubService(unittest.TestCase):
    """
    Unit tests for the pluggable VisionLLMStubService.
    Mocks Pillow Image loading to run tests fast and locally.
    """

    @patch("services.vision.vision_stub.Image.open")
    def test_analyze_image_success(self, mock_image_open):
        """
        analyze_image should decode image properties and return a structured layout classification.
        """
        # Mock Pillow Image
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        mock_image.format = "PNG"
        mock_image_open.return_value = mock_image

        service = VisionLLMStubService()
        
        # Test terminal classifier
        res_terminal = service.analyze_image(b"fake_png_bytes", prompt="Analyze this terminal error.")
        self.assertEqual(res_terminal.source_component, "Terminal")
        self.assertIn("exit code 1", res_terminal.detected_errors)
        self.assertIn("800x600", res_terminal.description)

        # Test postman classifier
        res_postman = service.analyze_image(b"fake_png_bytes", prompt="Look at the postman response.")
        self.assertEqual(res_postman.source_component, "Postman")
        self.assertIn("401 Unauthorized", res_postman.detected_errors)

    def test_analyze_image_empty_bytes(self):
        """
        Passing empty bytes should raise a VisionException.
        """
        service = VisionLLMStubService()
        with self.assertRaises(VisionException):
            service.analyze_image(b"")

    @patch("services.vision.vision_stub.Image.open")
    def test_analyze_image_invalid_image(self, mock_image_open):
        """
        If Pillow raises a decoding error, the service should wrap it in a VisionException.
        """
        mock_image_open.side_effect = IOError("Unknown image header format")
        
        service = VisionLLMStubService()
        with self.assertRaises(VisionException):
            service.analyze_image(b"corrupted_headers_bytes")


if __name__ == "__main__":
    unittest.main()
