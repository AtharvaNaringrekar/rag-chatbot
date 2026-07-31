from abc import ABC, abstractmethod
from typing import Generator, Optional


class ILLMService(ABC):
    """
    Interface for LLM (Large Language Model) interaction.
    Allows swapping local LLMs (like Ollama/phi3:mini) with cloud-based APIs (OpenAI, Gemini).
    """

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Generate a complete response from the LLM based on the user prompt and system prompt.

        Args:
            prompt: The main user prompt containing instructions and context.
            system_prompt: Optional instructions indicating the model's persona/constraints.
            max_tokens: Optional maximum tokens override for generation.

        Returns:
            The generated response as a string.
        """
        pass

    @abstractmethod
    def generate_response_stream(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """
        Generate a streaming response from the LLM token by token.

        Args:
            prompt: The main user prompt containing instructions and context.
            system_prompt: Optional instructions indicating the model's persona/constraints.
            max_tokens: Optional maximum tokens override for generation.

        Yields:
            Chunks of the generated response as they become available.
        """
        pass
