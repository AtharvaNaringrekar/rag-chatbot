import io
import logging
from typing import Optional
from PIL import Image
from core.interfaces.vision import IVisionService
from core.models import VisionAnalysisResult
from core.exceptions import VisionException

logger = logging.getLogger(__name__)


class VisionLLMStubService(IVisionService):
    """
    Pluggable Vision service stub implementing IVisionService.
    Decodes layout descriptors and prepares structures for future Vision API bindings.
    """

    def __init__(self, model_name: str = "stub-vision-v1"):
        """
        Initialize the vision stub service.

        Args:
            model_name: Pluggable model version name identifier.
        """
        self.model_name = model_name

    def analyze_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> VisionAnalysisResult:
        if not image_bytes:
            raise VisionException(self.model_name, "Empty image bytes payload provided")

        try:
            # 1. Load image to verify formatting validity
            logger.info("Vision Stub: Verifying image validity via Pillow...")
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            img_format = image.format or "Unknown Format"
            logger.info(f"Vision Stub: Loaded valid {img_format} image ({width}x{height})")

            # 2. Rule-based heuristic classifier to simulate visual analysis
            # In a live production setting, this would call LLaVA or Gemini
            prompt_lower = (prompt or "").lower()
            
            description = (
                f"A screenshot in {img_format} format ({width}x{height} pixels). "
                "The image contains user interface text representing a software error trace."
            )
            detected_errors = []
            source_component = "unknown"

            # Check keywords in the prompt to simulate human visual recognition
            if "terminal" in prompt_lower or "cli" in prompt_lower or "console" in prompt_lower:
                description = f"A dark-themed command line interface (CLI) terminal panel. Size: {width}x{height}."
                detected_errors = ["connection refused", "exit code 1"]
                source_component = "Terminal"
            elif "vscode" in prompt_lower or "code" in prompt_lower or "editor" in prompt_lower:
                description = f"An IDE workspace (VS Code style editor) showing active debugger alerts. Size: {width}x{height}."
                detected_errors = ["SyntaxError", "unexpected token"]
                source_component = "VS Code"
            elif "postman" in prompt_lower or "response" in prompt_lower:
                description = f"A Postman HTTP client view depicting a failed request session. Size: {width}x{height}."
                detected_errors = ["401 Unauthorized", "500 Internal Server Error"]
                source_component = "Postman"
            elif "swagger" in prompt_lower or "browser" in prompt_lower or "console" in prompt_lower:
                description = f"A web browser viewport displaying a swagger-ui specification or developer console traceback. Size: {width}x{height}."
                detected_errors = ["400 Bad Request", "CORS policy blocked"]
                source_component = "Swagger UI / Browser"
            else:
                # Generic fallback technical profile
                detected_errors = ["technical exception"]
                source_component = "Desktop Environment"

            logger.info(f"Vision Stub analysis completed. Source classified as '{source_component}'.")
            return VisionAnalysisResult(
                description=description,
                detected_errors=detected_errors,
                source_component=source_component
            )

        except Exception as e:
            logger.error(f"Failed to analyze image visually: {e}", exc_info=True)
            raise VisionException(self.model_name, f"Visual analysis failed: {e}")

    def analyze_image_file(self, file_path: str, prompt: Optional[str] = None) -> VisionAnalysisResult:
        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            return self.analyze_image(image_bytes, prompt)
        except VisionException as ve:
            raise ve
        except Exception as e:
            logger.error(f"Failed to read image file at '{file_path}': {e}")
            raise VisionException(self.model_name, f"Failed to parse image from file: {e}")
