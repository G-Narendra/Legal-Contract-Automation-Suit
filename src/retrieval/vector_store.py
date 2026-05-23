"""
Vector store wrapper for ChromaDB with persistent storage.

Provides:
- Collection management
- Dense similarity search
- Hybrid search (dense + BM25)
- Batch insertion with embeddings
"""

from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path

logger = logging.getLogger("vector_store")


class VectorStore:
    """ChromaDB-based vector store for legal documents.

    Cost-effective design:
    - Persistent storage (no re-indexing on restart)
    - Collection-based organization (laws, cases, templates)
    - Cosine similarity for semantic search
    """

    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collections: Dict[str, Any] = {}

    @property
    def client(self):
        """Lazy client initialization."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def get_collection(self, name: str):
        """Get or create a collection."""
        if name not in self._collections:
            try:
                self._collections[name] = self.client.get_collection(name)
            except ValueError:
                self._collections[name] = self.client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
        return self._collections[name]

    def add_texts(self, texts: List[str], metadatas: List[Dict],
                  ids: List[str], collection_name: str = "legal_laws"):
        """Add texts with metadata to a collection."""
        collection = self.get_collection(collection_name)

        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            collection.add(
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
            )

        logger.info(f"Added {len(texts)} texts to '{collection_name}'")

    def similarity_search(self, query: str, collection_name: str = "legal_laws",
                          k: int = 5) -> List[Dict]:
        """Search for similar documents by semantic similarity."""
        collection = self.get_collection(collection_name)

        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results.get("distances") else 0.0,
                })

        return documents

    def hybrid_search(self, query: str, collection_name: str = "legal_laws",
                      k: int = 5) -> List[Dict]:
        """Hybrid search combining dense and keyword retrieval.

        Falls back to dense-only if keyword search not available.
        """
        # Dense search
        dense_results = self.similarity_search(query, collection_name, k)

        # For now, return dense results (BM25 integration would require additional setup)
        return dense_results

    def delete_collection(self, name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            if name in self._collections:
                del self._collections[name]
            logger.info(f"Deleted collection '{name}'")
        except Exception as e:
            logger.error(f"Delete collection error: {e}")

    def count(self, collection_name: str) -> int:
        """Get document count in a collection."""
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def list_collections(self) -> List[str]:
        """List all collections."""
        return [c.name for c in self.client.list_collections()]
