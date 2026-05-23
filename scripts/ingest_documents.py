"""
Ingest Legal Documents — Parses and indexes legal documents.

Usage:
    python scripts/ingest_documents.py data/laws/ --collection legal_laws
    python scripts/ingest_documents.py data/contracts/ --collection templates
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.retrieval.chunking import chunk_document


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest legal documents into vector store")
    parser.add_argument("input_path", help="Path to document(s) or directory")
    parser.add_argument("--collection", default="legal_laws", help="Target collection name")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in chars")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in chars")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        sys.exit(1)

    # Initialize vector store
    store = VectorStore()

    # Collect all files
    files = []
    if input_path.is_file():
        files.append(input_path)
    else:
        files.extend(input_path.glob("*.txt"))
        files.extend(input_path.glob("*.json"))
        files.extend(input_path.glob("*.md"))
        files.extend(input_path.glob("*.yaml"))
        files.extend(input_path.glob("*.yml"))

    if not files:
        print(f"❌ No supported files found in {input_path}")
        sys.exit(1)

    print(f"📂 Found {len(files)} files to ingest")

    texts, metadatas, ids = [], [], []
    doc_index = 0

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as e:
                print(f"⚠️  Skipping {file_path.name}: {e}")
                continue

        # Chunk the document
        chunks = chunk_document(
            content,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            strategy="semantic" if len(content) > args.chunk_size else "simple",
        )

        for chunk in chunks:
            texts.append(chunk["text"])
            metadatas.append({
                **chunk["metadata"],
                "source_file": file_path.name,
                "source_path": str(file_path),
            })
            ids.append(f"{file_path.stem}_{doc_index}_{chunk['metadata']['chunk_index']}")

        doc_index += 1
        print(f"  ✓ {file_path.name}: {len(chunks)} chunks")

    # Add to vector store
    store.add_texts(texts, metadatas, ids, args.collection)
    total = store.count(args.collection)
    print(f"\n✅ Ingested {len(texts)} chunks into '{args.collection}'")
    print(f"   Total documents in collection: {total}")


if __name__ == "__main__":
    main()
