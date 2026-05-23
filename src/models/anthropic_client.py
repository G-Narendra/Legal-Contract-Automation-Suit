"""
Anthropic Claude LLM client with streaming and token estimation.
"""

from anthropic import Anthropic
from typing import Generator
from src.core.llm_base import LLMBase


class AnthropicClient(LLMBase):
    """Anthropic Claude LLM provider with streaming support."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307",
                 temperature: float = 0.1, max_tokens: int = 4096):
        self._client = Anthropic(api_key=api_key)
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        with self._client.messages.stream(
            model=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            full_response = []
            for chunk in stream.text_stream:
                full_response.append(chunk)
                yield chunk
            return "".join(full_response)

    def count_tokens(self, text: str) -> int:
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = len(text) - arabic_chars
        return (english_chars // 4) + (arabic_chars // 2) + 1

    def get_provider_name(self) -> str:
        return f"anthropic/{self._model_name}"
