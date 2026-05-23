"""
Build Vector Database — Creates and populates ChromaDB collections.

Usage:
    python scripts/build_vector_db.py
    python scripts/build_vector_db.py --provider google --reset

Populates:
    - legal_laws: UAE Federal Laws and regulations
    - case_law: UAE court precedents
    - templates: Contract templates
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.models.embedding_models import GoogleEmbedder, OpenAIEmbedder, LocalEmbedder
from src.retrieval.chunking import chunk_document


def parse_args():
    parser = argparse.ArgumentParser(description="Build vector database for legal documents")
    parser.add_argument("--provider", default="google", choices=["google", "openai", "local"],
                       help="Embedding provider")
    parser.add_argument("--reset", action="store_true", help="Reset collections before building")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"🚀 Building vector database with provider: {args.provider}")

    # Initialize embedder
    if args.provider == "google":
        import os
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            sys.exit(1)
        embedder = GoogleEmbedder(api_key=api_key)
    elif args.provider == "openai":
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        embedder = OpenAIEmbedder(api_key=api_key)
    else:
        embedder = LocalEmbedder()

    # Initialize vector store
    store = VectorStore(persist_directory=f"{args.data_dir}/chroma_db")

    # Reset if requested
    if args.reset:
        for collection in ["legal_laws", "case_law", "templates", "legal_terminology"]:
            try:
                store.delete_collection(collection)
                print(f"🗑️  Deleted collection: {collection}")
            except Exception:
                pass

    # Load and index sample legal documents
    data_dir = Path(args.data_dir)

    # 1. Legal Laws collection
    print("\n📚 Building Legal Laws collection...")
    laws_path = data_dir / "laws.json"
    if laws_path.exists():
        with open(laws_path, "r") as f:
            laws = json.load(f)

        texts, metadatas, ids = [], [], []
        for i, law in enumerate(laws):
            chunks = chunk_document(
                law.get("text", ""),
                chunk_size=512,
                strategy="semantic",
            )
            for j, chunk in enumerate(chunks):
                texts.append(chunk["text"])
                metadatas.append({
                    **chunk["metadata"],
                    "title": law.get("title", ""),
                    "law_number": law.get("law_number", ""),
                    "year": law.get("year", ""),
                    "category": law.get("category", "general"),
                })
                ids.append(f"law_{i}_{j}")

        if texts:
            store.add_texts(texts, metadatas, ids, "legal_laws")
            print(f"✅ Indexed {len(texts)} law chunks")
    else:
        print(f"ℹ️  {laws_path} not found. Create it with UAE legal documents to populate.")
        print("   Creating empty collection for later population...")
        store.get_collection("legal_laws")

    # 2. Templates collection
    print("\n📄 Building Templates collection...")
    templates_dir = data_dir / "templates"
    if templates_dir.exists():
        texts, metadatas, ids = [], [], []
        for f in templates_dir.glob("*.yaml"):
            import yaml
            with open(f, "r") as fh:
                templates = yaml.safe_load(fh) or {}
            for name, template in templates.items():
                texts.append(str(template))
                metadatas.append({"name": name, "source": f.name})
                ids.append(f"template_{name}")

        if texts:
            store.add_texts(texts, metadatas, ids, "templates")
            print(f"✅ Indexed {len(texts)} templates")
    else:
        print("ℹ️  No templates directory found. Creating empty collection.")
        store.get_collection("templates")

    # Print stats
    print("\n📊 Database Summary:")
    for collection in store.list_collections():
        count = store.count(collection)
        print(f"   - {collection}: {count} documents")

    print("\n✅ Vector database build complete!")
    print(f"   Location: {args.data_dir}/chroma_db")


if __name__ == "__main__":
    main()
