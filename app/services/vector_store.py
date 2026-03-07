"""
Vector Store (FAISS)
====================
Wraps FAISS for indexing, searching, and persisting document embeddings.

Why FAISS over alternatives (ChromaDB, Pinecone, Weaviate):
    - In-process C++ library — no external server or Docker dependency
    - Native disk persistence with simple save/load
    - At ~19K documents, exact search (IndexFlatIP) completes in <10ms,
      so approximate indexes (IVF, HNSW) aren't needed yet
    - Well-documented, battle-tested at scale by Meta

Why IndexFlatIP (Inner Product):
    - Our embeddings are L2-normalized, so inner product = cosine similarity
    - Exact search at this scale is fast enough — approximate indexes only
      help beyond ~100K documents
    - Simpler to debug and reason about than IVF or HNSW
"""

import os
import json
import faiss
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "faiss_index.bin")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "metadata.json")


class VectorStore:
    """Manages a FAISS index with associated document metadata.

    Supports building, searching, saving, and loading the index.
    Metadata (category, text snippet, filename) is stored alongside
    the vectors for filtered retrieval.
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.metadata = []

    def build_index(self, dimension=None):
        """Create a new FAISS index.

        Uses IndexFlatIP (exact inner product) — appropriate for our
        corpus size (~19K docs). At 100K+ you'd want IVF or HNSW.
        """
        dim = dimension or self.dimension
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add_documents(self, embeddings, metadata_list):
        """Insert documents and their embeddings into the index.

        Args:
            embeddings: numpy array of shape (n_docs, dimension).
            metadata_list: List of dicts with keys like 'category', 'text', 'filename'.
        """
        if self.index is None:
            self.build_index(embeddings.shape[1])

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, top_k=5, category_filter=None):
        """Perform semantic search with optional category filtering.

        Args:
            query_embedding: 1D array of shape (dimension,).
            top_k: Number of results to return.
            category_filter: If set, only return docs from this category.

        Returns:
            List of dicts with 'category', 'score', 'text', 'filename'.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Fetch extra results if filtering, since some will be dropped
        fetch_k = top_k * 5 if category_filter else top_k
        scores, indices = self.index.search(query, min(fetch_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            meta = self.metadata[idx]

            if category_filter and meta.get("category") != category_filter:
                continue

            results.append({
                "category": meta.get("category", "unknown"),
                "score": float(score),
                "text": meta.get("text", "")[:500],
                "filename": meta.get("filename", ""),
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, index_path=None, metadata_path=None):
        """Persist the index and metadata to disk."""
        index_path = index_path or INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

        print(f"Saved FAISS index ({self.index.ntotal} vectors) to {index_path}")
        print(f"Saved metadata ({len(self.metadata)} entries) to {metadata_path}")

    def load(self, index_path=None, metadata_path=None):
        """Load a previously saved index and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        index_path = index_path or INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        print(f"Loaded FAISS index ({self.index.ntotal} vectors) from {index_path}")
        print(f"Loaded metadata ({len(self.metadata)} entries) from {metadata_path}")
        return True
