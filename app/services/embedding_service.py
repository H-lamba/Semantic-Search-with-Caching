"""
Embedding Service (Tasks 26-28, 30)
====================================
Handles loading the embedding model and generating vector embeddings.

MODEL SELECTION (Task 26):
    We chose 'all-MiniLM-L6-v2' from the sentence-transformers library.

JUSTIFICATION (Task 27):
    1. SEMANTIC QUALITY: This model is specifically trained for semantic
       similarity tasks using a contrastive learning objective on 1B+ sentence
       pairs. It maps sentences to a 384-dimensional dense vector space where
       semantically similar texts have high cosine similarity.

    2. EFFICIENCY: At only 22M parameters (80MB), it is lightweight enough to
       run on CPU without a GPU, making it ideal for development and Docker
       deployment. It encodes ~2800 sentences/sec on GPU and ~50/sec on CPU.

    3. DIMENSIONALITY: 384 dimensions provides a good balance — high enough for
       rich semantic representation, low enough for efficient FAISS indexing and
       storage. Compare: all-mpnet-base-v2 produces 768-dim vectors (2x storage
       and slower search) with only marginally better quality.

    4. PROVEN TRACK RECORD: Consistently ranks among the top models on the
       MTEB (Massive Text Embedding Benchmark) for its size class.

    5. COMPATIBILITY: Designed for natural English text (which aligns with our
       Phase 3 decision to preserve natural sentence structure).
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Model config
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "models", "embeddings.npy")


class EmbeddingService:
    """Wrapper for the sentence-transformers embedding model.

    Task 28: Loads the model efficiently with lazy initialization—the model
    is only loaded into memory when first needed, not at import time.
    """

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model on first access."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def encode(self, texts, batch_size=64, show_progress=True):
        """Encode a list of texts into vector embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts to process at once (Task 29).
            show_progress: Show progress bar during encoding.

        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text):
        """Encode a single text string into a vector embedding.

        Used for encoding individual search queries at runtime.
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)[0]

    def save_embeddings(self, embeddings, path=None):
        """Save embeddings to disk to prevent re-computation (Task 30)."""
        save_path = path or EMBEDDINGS_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embeddings)
        print(f"Saved embeddings ({embeddings.shape}) to {save_path}")

    def load_embeddings(self, path=None):
        """Load pre-computed embeddings from disk."""
        load_path = path or EMBEDDINGS_PATH
        embeddings = np.load(load_path)
        print(f"Loaded embeddings ({embeddings.shape}) from {load_path}")
        return embeddings
