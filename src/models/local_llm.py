"""
Local LLM client for offline inference (no API costs).
Supports llama.cpp, ollama, or any OpenAI-compatible local endpoint.
"""

from typing import Generator, Optional
import requests
import json
import logging
from src.core.llm_base import LLMBase

logger = logging.getLogger("legal_llm")


class LocalLLMClient(LLMBase):
    """Local LLM provider (llama.cpp, ollama, etc.) via REST API."""

    def __init__(self, model: str = "local-model",
                 temperature: float = 0.1, max_tokens: int = 4096,
                 endpoint: str = "http://localhost:11434/api/generate"):
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._endpoint = endpoint

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = requests.post(
                self._endpoint,
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", data.get("text", ""))
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            return f"⚠️ Local LLM error: {str(e)}"

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        try:
            response = requests.post(
                self._endpoint,
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            full_response = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            full_response.append(data["response"])
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            return "".join(full_response)
        except Exception as e:
            logger.error(f"Local LLM stream error: {e}")
            yield f"⚠️ Local LLM error: {str(e)}"
            return ""

    def count_tokens(self, text: str) -> int:
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = len(text) - arabic_chars
        return (english_chars // 4) + (arabic_chars // 2) + 1

    def get_provider_name(self) -> str:
        return f"local/{self._model_name}"
