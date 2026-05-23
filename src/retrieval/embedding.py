"""
Embedding service using configured provider.

Wraps the embedder with batch processing and retry logic.
"""

from typing import List
from src.models.embedding_models import GoogleEmbedder, OpenAIEmbedder, LocalEmbedder
import logging

logger = logging.getLogger("embedding_service")


def get_embedder(provider: str = "google", api_key: str = "",
                 model: str = "models/gemini-embedding-2"):
    """Get embedder instance for the given provider."""
    if provider == "google":
        return GoogleEmbedder(api_key=api_key, model=model)
    elif provider == "openai":
        return OpenAIEmbedder(api_key=api_key, model=model)
    elif provider == "local":
        return LocalEmbedder(model=model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def embed_batch(texts: List[str], embedder, batch_size: int = 10) -> List[List[float]]:
    """Embed a batch of texts with error handling."""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_results = embedder.embed_texts(batch)
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Batch embedding error at index {i}: {e}")
            # Retry one by one
            for text in batch:
                try:
                    results.append(embedder.embed_text(text))
                except Exception:
                    # Return zero vector on failure
                    results.append([0.0] * embedder.get_dimension())

    return results
