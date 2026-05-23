"""
Google Gemini LLM client with streaming and token estimation.
"""

import google.generativeai as genai
from typing import Generator, Optional
from src.core.llm_base import LLMBase


class GoogleGenAIClient(LLMBase):
    """Gemini LLM provider with streaming support."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 temperature: float = 0.1, max_tokens: int = 4096):
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        stream = self._model.generate_content(prompt, stream=True)
        full_response = []
        for chunk in stream:
            if chunk.text:
                full_response.append(chunk.text)
                yield chunk.text
        return "".join(full_response)

    def count_tokens(self, text: str) -> int:
        # ~4 chars per token for English, ~2 for Arabic
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = len(text) - arabic_chars
        return (english_chars // 4) + (arabic_chars // 2) + 1

    def get_provider_name(self) -> str:
        return f"google/{self._model_name}"
