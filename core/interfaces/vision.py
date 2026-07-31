from abc import ABC, abstractmethod
from typing import Optional
from core.models import VisionAnalysisResult


class IVisionService(ABC):
    """
    Interface for Vision AI interaction.
    Analyzes layout context, identifies which software is shown, and interprets errors visually.
    Allows plugging in multimodal LLMs (Ollama-based Llava, Gemini Pro Vision, GPT-4o).
    """

    @abstractmethod
    def analyze_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> VisionAnalysisResult:
        """
        Analyze the image visually to extract high-level context, layout, and issues.

        Args:
            image_bytes: The raw binary content of the screenshot.
            prompt: Optional context guiding prompt (e.g. "Focus on identifying what application this is").

        Returns:
            A VisionAnalysisResult domain model indicating description, errors, and source component.
        """
        pass

    @abstractmethod
    def analyze_image_file(self, file_path: str, prompt: Optional[str] = None) -> VisionAnalysisResult:
        """
        Analyze a saved image file visually.

        Args:
            file_path: The absolute path of the image.
            prompt: Optional context guiding prompt.

        Returns:
            A VisionAnalysisResult domain model indicating description, errors, and source component.
        """
        pass
