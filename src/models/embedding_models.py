"""
Embedding models for multiple providers.
"""

import hashlib
import time
from typing import List, Optional, Dict
from src.core.embedder_base import EmbedderBase
import logging

logger = logging.getLogger("legal_embedder")


class GoogleEmbedder(EmbedderBase):
    """Google Gemini embedding provider."""

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-2"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model_name = model
        self._dimension = 768

    def embed_text(self, text: str) -> List[float]:
        import google.generativeai as genai
        result = genai.embed_content(model=self._model_name, content=text)
        return result["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import google.generativeai as genai
        result = genai.embed_content(model=self._model_name, content=texts)
        return result["embedding"]

    def get_dimension(self) -> int:
        return self._dimension

    def get_provider_name(self) -> str:
        return f"google/{self._model_name}"


class OpenAIEmbedder(EmbedderBase):
    """OpenAI embedding provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model_name = model
        self._dimension = 1536 if "3-large" in model else 512

    def embed_text(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        return self._dimension

    def get_provider_name(self) -> str:
        return f"openai/{self._model_name}"


class LocalEmbedder(EmbedderBase):
    """Local embedding model using sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model
        self._model = None  # Lazy load
        self._dimension = 384

    def _lazy_load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                logger.warning("sentence-transformers not installed; using random embeddings")
                return False
        return True

    def embed_text(self, text: str) -> List[float]:
        if not self._lazy_load():
            return [0.0] * self._dimension
        return self._model.encode(text).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self._lazy_load():
            return [[0.0] * self._dimension for _ in texts]
        return self._model.encode(texts).tolist()

    def get_dimension(self) -> int:
        return self._dimension

    def get_provider_name(self) -> str:
        return f"local/{self._model_name}"
