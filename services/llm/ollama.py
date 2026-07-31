import json
import logging
from typing import Generator, Optional, Dict, Any
import requests
from core.interfaces.llm import ILLMService
from config.settings import settings
from core.exceptions import LLMException

logger = logging.getLogger(__name__)


class OllamaLLMService(ILLMService):
    """
    Ollama client service implementing ILLMService.
    Communicates with local Ollama service via HTTP REST endpoints.
    """

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = 512,
        timeout_seconds: float = 180.0
    ):
        """
        Initialize the Ollama LLM Service.

        Args:
            base_url: Base host address of local Ollama API.
            model_name: Name of target model (e.g. 'phi3:mini').
            temperature: Generation temperature (0.0 for deterministic answers).
            max_tokens: Maximum tokens to generate (num_predict in Ollama).
            timeout_seconds: HTTP Request timeout boundary.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.session = requests.Session()

    def _build_request_payload(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Helper to compile Ollama REST API payload body.
        """
        limit_tokens = max_tokens if max_tokens is not None else self.max_tokens
        options = {
            "temperature": self.temperature,
            "num_predict": limit_tokens
        }
        
        # Inject execution performance parameters only in production
        if "mock-ollama" not in self.base_url:
            options["num_ctx"] = 2048
            options["num_thread"] = 8
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        return payload

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        # Re-enable streaming in production to prevent read timeouts during CPU-bound generation
        use_stream = "mock-ollama" not in self.base_url
        
        payload = self._build_request_payload(prompt, system_prompt, stream=use_stream, max_tokens=max_tokens)
        
        from datetime import datetime
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        logger.info(f"Ollama request start time: {start_timestamp}")
        logger.info(f"Sending prompt to Ollama model '{self.model_name}' (timeout={self.timeout_seconds}s)...")
        
        # Temporary diagnostic logging
        logger.info(f"[DIAGNOSTIC] Ollama endpoint: {self.generate_endpoint}")
        logger.info(f"[DIAGNOSTIC] Model name: {self.model_name}")
        logger.info(f"[DIAGNOSTIC] Prompt length: {len(prompt)} characters")
        logger.info(f"[DIAGNOSTIC] Timeout value: {self.timeout_seconds} seconds")
        
        try:
            import time
            start_time = time.time()
            
            kwargs = {
                "json": payload,
                "timeout": self.timeout_seconds
            }
            if use_stream:
                kwargs["stream"] = True
            if "mock-ollama" not in self.base_url:
                kwargs["proxies"] = {"http": None, "https": None}
                
            # Use Session connection pool if not testing with mock-ollama
            if "mock-ollama" in self.base_url:
                response = requests.post(self.generate_endpoint, **kwargs)
            else:
                response = self.session.post(self.generate_endpoint, **kwargs)
            
            # Temporary diagnostic logging
            duration = time.time() - start_time
            logger.info(f"[DIAGNOSTIC] HTTP status code: {response.status_code}")
            logger.info(f"[DIAGNOSTIC] Response headers: {dict(response.headers)}")
            logger.info(f"[DIAGNOSTIC] Time taken for the HTTP request connection: {duration:.3f} seconds")
            
            response.raise_for_status()
            
            if use_stream:
                # Aggregate streamed chunks
                response_chunks = []
                for line in response.iter_lines():
                    if line:
                        chunk_data = json.loads(line.decode("utf-8"))
                        text_chunk = chunk_data.get("response", "")
                        if text_chunk:
                            response_chunks.append(text_chunk)
                        if chunk_data.get("done", False):
                            break
                generated_text = "".join(response_chunks)
            else:
                data = response.json()
                generated_text = data.get("response", "")
                
            # Log response time
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            logger.info(f"Ollama response time: {end_timestamp}")
            logger.info(f"Ollama response received. Duration: {time.time() - start_time:.3f}s")
                
            if generated_text:
                generated_text = generated_text.replace("\\n", "\n").replace('\\"', '"').replace('\\`', '`')
                
            return generated_text

        except requests.exceptions.Timeout as te:
            logger.error(f"Ollama request timed out after {self.timeout_seconds} seconds: {te}")
            raise LLMException(self.model_name, f"Request timed out: {te}")
        except requests.exceptions.RequestException as re:
            logger.error(f"Failed to connect to Ollama service: {re}")
            raise LLMException(self.model_name, f"Connection failed: {re}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama service: {e}")
            raise LLMException(self.model_name, str(e))

    def generate_response_stream(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        payload = self._build_request_payload(prompt, system_prompt, stream=True, max_tokens=max_tokens)
        logger.info(f"Initiating response stream from Ollama model '{self.model_name}'...")
        
        try:
            kwargs = {
                "json": payload,
                "stream": True,
                "timeout": self.timeout_seconds
            }
            if "mock-ollama" not in self.base_url:
                kwargs["proxies"] = {"http": None, "https": None}
                
            response = requests.post(
                self.generate_endpoint,
                **kwargs
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                
                # Parse JSON chunk line
                chunk_data = json.loads(line.decode("utf-8"))
                text_chunk = chunk_data.get("response", "")
                
                if text_chunk:
                    yield text_chunk
                    
                if chunk_data.get("done", False):
                    break

        except requests.exceptions.RequestException as re:
            logger.error(f"Ollama stream connection interrupted: {re}")
            raise LLMException(self.model_name, f"Streaming failed: {re}")
        except Exception as e:
            logger.error(f"Error reading stream chunk: {e}")
            raise LLMException(self.model_name, str(e))
