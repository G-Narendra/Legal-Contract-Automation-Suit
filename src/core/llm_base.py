"""
Abstract LLM base class with dynamic provider switching.

Design:
- Single interface for Google, OpenAI, Anthropic, and local LLMs
- Switch providers by changing one config line
- Built-in streaming, caching, and token estimation
- Cost tracking per provider for optimization
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
import time
import hashlib
import logging

logger = logging.getLogger("legal_llm")


class LLMBase(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response synchronously."""
        ...

    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, str]:
        """Stream tokens as they arrive. Yields each token, returns full text."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for cost tracking."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for logging/metrics."""
        ...


class LLMFactory:
    """Factory for creating LLM instances based on provider selection."""

    @staticmethod
    def create(provider: str, api_key: str, model: str,
               temperature: float = 0.1, max_tokens: int = 4096) -> LLMBase:
        """Create an LLM instance for the given provider."""
        if provider == "google":
            from src.models.google_client import GoogleGenAIClient
            return GoogleGenAIClient(api_key=api_key, model=model,
                                     temperature=temperature, max_tokens=max_tokens)
        elif provider == "openai":
            from src.models.openai_client import OpenAIClient
            return OpenAIClient(api_key=api_key, model=model,
                                temperature=temperature, max_tokens=max_tokens)
        elif provider == "anthropic":
            from src.models.anthropic_client import AnthropicClient
            return AnthropicClient(api_key=api_key, model=model,
                                   temperature=temperature, max_tokens=max_tokens)
        elif provider == "local":
            from src.models.local_llm import LocalLLMClient
            return LocalLLMClient(model=model, temperature=temperature,
                                  max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def from_config(config: Dict) -> LLMBase:
        """Create LLM from config dict (for dynamic switching)."""
        provider = config.get("provider", "google")
        api_key = config.get("api_key", "")
        model = config.get("model", "gemini-2.5-flash")
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 4096)
        return LLMFactory.create(provider, api_key, model, temperature, max_tokens)


class LLMWithCache:
    """LLM wrapper with TTL caching for cost savings.

    Cache hit rate target: >40% — saves ~$0.001-0.003 per hit.
    """

    def __init__(self, llm: LLMBase, ttl_seconds: int = 300):
        self._llm = llm
        self._cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds

    @property
    def llm(self) -> LLMBase:
        return self._llm

    def _make_key(self, prompt: str, **kwargs) -> str:
        content = f"{prompt}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate(self, prompt: str, use_cache: bool = True, **kwargs) -> str:
        cache_key = self._make_key(prompt, **kwargs) if use_cache else ""

        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["ts"] < self.ttl:
                logger.debug(f"Cache HIT | saved ~$0.002 | {self._llm.get_provider_name()}")
                return entry["value"]
            else:
                del self._cache[cache_key]

        start = time.time()
        result = self._llm.generate(prompt, **kwargs)
        duration = (time.time() - start) * 1000

        if use_cache:
            self._cache[cache_key] = {"value": result, "ts": time.time()}

        tokens = self._llm.count_tokens(prompt) + self._llm.count_tokens(result)
        logger.info(f"LLM [{self._llm.get_provider_name()}] {duration:.0f}ms ~{tokens} tokens")

        return result

    def stream_generate(self, prompt: str, use_cache: bool = True, **kwargs) -> Generator[str, None, str]:
        cache_key = self._make_key(prompt, **kwargs) if use_cache else ""

        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["ts"] < self.ttl:
                logger.debug(f"Cache HIT (stream) | {self._llm.get_provider_name()}")
                yield entry["value"]
                return entry["value"]
            else:
                del self._cache[cache_key]

        start = time.time()
        full_response = []
        for token in self._llm.stream_generate(prompt, **kwargs):
            full_response.append(token)
            yield token

        result = "".join(full_response)
        duration = (time.time() - start) * 1000

        if use_cache:
            self._cache[cache_key] = {"value": result, "ts": time.time()}

        tokens = self._llm.count_tokens(prompt) + self._llm.count_tokens(result)
        logger.info(f"LLM stream [{self._llm.get_provider_name()}] {duration:.0f}ms ~{tokens} tokens")

        return result

    def count_tokens(self, text: str) -> int:
        return self._llm.count_tokens(text)

    def get_provider_name(self) -> str:
        return self._llm.get_provider_name()

    def clear_cache(self):
        self._cache.clear()
        logger.info("LLM cache cleared")

    @property
    def cache_size(self) -> int:
        return len(self._cache)
