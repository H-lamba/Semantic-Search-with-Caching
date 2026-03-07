"""
Vector Store Service (Tasks 31-36, 38-39)
==========================================
Handles all vector database operations using FAISS.

VECTOR DB SELECTION (Task 31):
    We chose FAISS (Facebook AI Similarity Search) as our vector database.

JUSTIFICATION (Task 32):
    1. LIGHTWEIGHT & LOCAL: FAISS runs entirely in-process with no external
       server or daemon. This makes it ideal for a self-contained FastAPI
       application that boots with a single `uvicorn` command.

    2. PERFORMANCE: FAISS is written in C++ with Python bindings, making it
       one of the fastest similarity search libraries available. It supports
       both exact (IndexFlatIP) and approximate (IndexIVFFlat) search.

    3. PERSISTENCE: FAISS indexes can be saved to and loaded from disk
       natively, so the vector store survives application restarts without
       rebuilding from scratch (Task 39).

    4. MEMORY EFFICIENT: For our ~19K documents with 384-dim embeddings,
       the index requires only ~28MB of RAM (19K * 384 * 4 bytes).

    5. FILTERING SUPPORT: By combining FAISS with a metadata store, we can
       filter results by category, cluster, or other attributes (Task 35).

    Why not ChromaDB?
        ChromaDB adds an abstraction layer and SQLite dependency. For our
        use case (single-user API with ~20K documents), the added complexity
        is unnecessary. FAISS gives us direct control over the index type
        and search parameters, which is critical for Task 38 (optimization).
"""

import os
import json
import numpy as np
import faiss

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "faiss_index.bin")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "metadata.json")


class VectorStore:
    """Python wrapper class for all vector DB operations (Task 36).

    Supports insert, search, save, and load operations with FAISS
    as the underlying similarity search engine.
    """

    def __init__(self, dimension=384):
        """Initialize the vector store (Task 33).

        Args:
            dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = None
        self.metadata = []  # List of dicts with document info
        self._build_index()

    def _build_index(self):
        """Build the FAISS index (Task 33 & 38).

        INDEX SELECTION & OPTIMIZATION (Task 38):
            We use IndexFlatIP (Inner Product) for exact search.

            For ~19K documents, exact search is fast enough (<10ms per query)
            and guarantees the best possible results. Approximate indexes
            (like IndexIVFFlat) only provide speedups for 100K+ documents
            at the cost of recall accuracy.

            Since embeddings are L2-normalized in the EmbeddingService,
            inner product equals cosine similarity:
                cos(a, b) = dot(a, b) / (||a|| * ||b||) = dot(a, b)  when ||a|| = ||b|| = 1
        """
        self.index = faiss.IndexFlatIP(self.dimension)

    def insert(self, embeddings, metadata_list):
        """Insert documents and their embeddings into the database (Task 34).

        Args:
            embeddings: numpy array of shape (n, dimension).
            metadata_list: List of dicts with document info
                          (category, filename, text snippet, etc.)
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        print(f"Inserted {len(metadata_list)} documents. Total: {self.index.ntotal}")

    def search(self, query_embedding, top_k=5, category_filter=None):
        """Perform semantic search with optional filtering (Task 35).

        Args:
            query_embedding: numpy array of shape (dimension,)
            top_k: Number of results to return.
            category_filter: Optional category name to filter results.

        Returns:
            List of dicts with keys: score, category, filename, text, index
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Reshape for FAISS (expects 2D array)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Task 35: Configure filtered retrieval
        # If filtering, search more results than needed, then filter
        search_k = top_k * 5 if category_filter else top_k

        scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc = self.metadata[idx].copy()
            doc["score"] = float(score)
            doc["index"] = int(idx)

            # Apply category filter if specified
            if category_filter and doc.get("category") != category_filter:
                continue

            results.append(doc)
            if len(results) >= top_k:
                break

        return results

    def save(self, index_path=None, metadata_path=None):
        """Persist the vector store to disk (Task 39).

        Saves both the FAISS index and the metadata separately.
        This ensures the vector store does not rebuild entirely
        on every application startup.
        """
        idx_path = index_path or INDEX_PATH
        meta_path = metadata_path or METADATA_PATH

        os.makedirs(os.path.dirname(idx_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, idx_path)
        print(f"Saved FAISS index ({self.index.ntotal} vectors) to {idx_path}")

        # Save metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata ({len(self.metadata)} entries) to {meta_path}")

    def load(self, index_path=None, metadata_path=None):
        """Load a persisted vector store from disk (Task 39).

        Returns True if loaded successfully, False if files not found.
        """
        idx_path = index_path or INDEX_PATH
        meta_path = metadata_path or METADATA_PATH

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            print("No persisted vector store found. Starting fresh.")
            return False

        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loaded FAISS index ({self.index.ntotal} vectors) from {idx_path}")
        print(f"Loaded metadata ({len(self.metadata)} entries) from {meta_path}")
        return True

    @property
    def total_documents(self):
        """Total number of documents in the store."""
        return self.index.ntotal if self.index else 0

    def get_all_embeddings(self):
        """Extract all embeddings from the FAISS index.

        Useful for Phase 5 (clustering) which needs the raw vectors.
        """
        if self.index.ntotal == 0:
            return np.array([], dtype=np.float32)
        return faiss.rev_swig_ptr(
            self.index.get_xb(), self.index.ntotal * self.dimension
        ).reshape(self.index.ntotal, self.dimension).copy()
