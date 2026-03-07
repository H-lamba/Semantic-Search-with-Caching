"""
Build Embeddings & Vector Database (Tasks 29-35, 37-39)
========================================================
This script:
  1. Loads the cleaned corpus from Phase 3
  2. Generates vector embeddings using sentence-transformers
  3. Saves raw embeddings to disk (Task 30)
  4. Builds and populates a FAISS vector store (Tasks 33-34)
  5. Runs a test semantic search query (Task 37)
  6. Persists the vector store to disk (Task 39)

Run from project root:
    python scripts/build_vector_db.py
"""

import os
import sys
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "cleaned_corpus.parquet")


def build():
    # ============================================================
    # Step 1: Load cleaned corpus
    # ============================================================
    print("=" * 60)
    print("Step 1: Loading cleaned corpus...")
    print("=" * 60)
    df = pd.read_parquet(CORPUS_PATH)
    print(f"Loaded {len(df)} documents from {CORPUS_PATH}\n")

    texts = df["cleaned_text"].tolist()
    categories = df["category"].tolist()
    filenames = df["filename"].tolist()

    # ============================================================
    # Step 2: Generate embeddings (Task 29)
    # ============================================================
    print("=" * 60)
    print("Step 2: Generating embeddings (batch processing)...")
    print("=" * 60)
    embed_service = EmbeddingService()

    # Task 29: Process in batches to manage memory
    embeddings = embed_service.encode(texts, batch_size=64, show_progress=True)
    print(f"Generated embeddings: {embeddings.shape}\n")

    # Task 30: Save raw embeddings to prevent re-computation
    print("=" * 60)
    print("Step 3: Saving raw embeddings to disk...")
    print("=" * 60)
    embed_service.save_embeddings(embeddings)
    print()

    # ============================================================
    # Step 4: Build FAISS vector store (Tasks 33-35)
    # ============================================================
    print("=" * 60)
    print("Step 4: Building FAISS vector store...")
    print("=" * 60)
    vector_store = VectorStore(dimension=embeddings.shape[1])

    # Task 34: Insert documents with metadata
    metadata_list = []
    for i in range(len(df)):
        metadata_list.append({
            "category": categories[i],
            "filename": filenames[i],
            "text": texts[i][:500],  # Store first 500 chars for display
        })

    vector_store.insert(embeddings, metadata_list)
    print()

    # ============================================================
    # Step 5: Test semantic search (Task 37)
    # ============================================================
    print("=" * 60)
    print("Step 5: Testing semantic search...")
    print("=" * 60)
    test_queries = [
        "What is the best graphics card for gaming?",
        "Is there life on other planets?",
        "How does encryption work?",
        "What are the rules of baseball?",
        "I want to sell my old car",
    ]

    for query in test_queries:
        query_embedding = embed_service.encode_single(query)
        results = vector_store.search(query_embedding, top_k=3)

        print(f"\nQuery: '{query}'")
        print("-" * 40)
        for j, r in enumerate(results, 1):
            print(f"  {j}. [{r['category']}] score={r['score']:.4f}")
            print(f"     {r['text'][:100]}...")
    print()

    # ============================================================
    # Step 6: Persist vector store (Task 39)
    # ============================================================
    print("=" * 60)
    print("Step 6: Persisting vector store to disk...")
    print("=" * 60)
    vector_store.save()

    # Final summary
    print(f"\n{'=' * 60}")
    print("BUILD COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  Documents indexed:     {vector_store.total_documents}")
    print(f"  Embedding dimension:   {embeddings.shape[1]}")
    print(f"  Embedding model:       all-MiniLM-L6-v2")
    print(f"  Vector DB:             FAISS (IndexFlatIP)")
    print(f"  Index size:            ~{embeddings.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Persisted to:          models/")


if __name__ == "__main__":
    build()
