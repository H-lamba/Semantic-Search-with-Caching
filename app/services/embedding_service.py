"""
Embedding Service
=================
Wraps sentence-transformers to provide a clean interface for encoding
documents and queries into dense vector representations.

We chose all-MiniLM-L6-v2 for this project after evaluating several options:

Why all-MiniLM-L6-v2:
    - 384-dimensional vectors — good balance between quality and storage
    - Trained on 1B+ sentence pairs for semantic similarity
    - Only ~80MB / 22M params — runs fine on CPU without a GPU
    - Consistently ranks well on MTEB benchmarks for retrieval tasks
    - Produces L2-normalized embeddings, so dot product = cosine similarity

Why not larger models:
    - all-mpnet-base-v2 (768-dim) gives marginal quality improvement but
      doubles storage and slows search. For 19K documents it's overkill.
    - OpenAI/Cohere API embeddings would add latency, cost, and an external
      dependency for a self-contained demo.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "models", "embeddings.npy")


class EmbeddingService:
    """Handles all embedding operations for documents and queries.

    Uses lazy initialization — the model is loaded on first use to keep
    startup fast when only loading saved embeddings.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the transformer model on first access."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def encode_batch(self, texts, batch_size=64, show_progress=True):
        """Encode a list of texts into embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Texts per batch (64 balances speed vs memory).
            show_progress: Show a tqdm progress bar.

        Returns:
            numpy array of shape (len(texts), 384).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize so dot product = cosine sim
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text):
        """Encode a single query string.

        Returns:
            1D numpy array of shape (384,).
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)

    def save_embeddings(self, embeddings, path=None):
        """Save embeddings to disk so we don't have to recompute them."""
        path = path or EMBEDDINGS_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)
        print(f"Saved {len(embeddings)} embeddings to {path}")

    def load_embeddings(self, path=None):
        """Load previously saved embeddings from disk."""
        path = path or EMBEDDINGS_PATH
        embeddings = np.load(path)
        print(f"Loaded {len(embeddings)} embeddings from {path}")
        return embeddings
