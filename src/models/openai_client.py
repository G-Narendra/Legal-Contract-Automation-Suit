"""
OpenAI LLM client with streaming and token estimation.
"""

from openai import OpenAI
from typing import Generator
from src.core.llm_base import LLMBase


class OpenAIClient(LLMBase):
    """OpenAI LLM provider with streaming support."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 temperature: float = 0.1, max_tokens: int = 4096):
        self._client = OpenAI(api_key=api_key)
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        stream = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        full_response = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response.append(token)
                yield token
        return "".join(full_response)

    def count_tokens(self, text: str) -> int:
        # Approximate: 1 token ~= 4 chars for English
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = len(text) - arabic_chars
        return (english_chars // 4) + (arabic_chars // 2) + 1

    def get_provider_name(self) -> str:
        return f"openai/{self._model_name}"
