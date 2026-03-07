"""
Unit tests for the semantic cache.
Tests cover cache hits, misses, statistics, and flushing.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.semantic_cache import SemanticCache


class TestSemanticCacheHit(unittest.TestCase):
    """Test that similar queries produce cache hits."""

    def setUp(self):
        self.cache = SemanticCache(similarity_threshold=0.85)

    def test_exact_same_query_hits(self):
        """The exact same embedding should always hit."""
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.set("What is machine learning?", {"result": "test"}, query_embedding=emb)
        result = self.cache.get("What is machine learning?", query_embedding=emb)

        self.assertIsNotNone(result)
        self.assertEqual(result["matched_query"], "What is machine learning?")

    def test_similar_query_hits(self):
        """A very similar embedding (sim > 0.85) should hit."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        # Create a slightly perturbed version (high similarity)
        noise = np.random.randn(384).astype(np.float32) * 0.05
        emb2 = emb1 + noise
        emb2 = emb2 / np.linalg.norm(emb2)

        self.cache.set("What is deep learning?", {"result": "test"}, query_embedding=emb1)
        result = self.cache.get("What is neural networks?", query_embedding=emb2)

        similarity = float(np.dot(emb1, emb2))
        if similarity >= 0.85:
            self.assertIsNotNone(result)
        else:
            self.assertIsNone(result)

    def test_hit_increments_counter(self):
        """Cache hit should increase the hit count."""
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.set("test query", {"result": "data"}, query_embedding=emb)
        self.cache.get("test query", query_embedding=emb)

        self.assertEqual(self.cache.hit_count, 1)


class TestSemanticCacheMiss(unittest.TestCase):
    """Test that dissimilar queries produce cache misses."""

    def setUp(self):
        self.cache = SemanticCache(similarity_threshold=0.85)

    def test_different_query_misses(self):
        """Completely different embeddings should miss."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0  # Orthogonal — similarity = 0

        self.cache.set("machine learning basics", {"result": "ml"}, query_embedding=emb1)
        result = self.cache.get("medieval history of France", query_embedding=emb2)

        self.assertIsNone(result)

    def test_empty_cache_misses(self):
        """Empty cache should always miss."""
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        result = self.cache.get("anything", query_embedding=emb)
        self.assertIsNone(result)

    def test_miss_increments_counter(self):
        """Cache miss should increase the miss count."""
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.cache.get("new query", query_embedding=emb)
        self.assertEqual(self.cache.miss_count, 1)


class TestCacheStatistics(unittest.TestCase):
    """Test hit rate calculation and stats."""

    def test_hit_rate_calculation(self):
        """Hit rate should be hits / (hits + misses)."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        cache.set("q1", {"r": 1}, query_embedding=emb)
        cache.get("q1", query_embedding=emb)  # hit

        ortho = np.zeros(384, dtype=np.float32)
        ortho[100] = 1.0
        cache.get("q2", query_embedding=ortho)  # miss

        self.assertAlmostEqual(cache.hit_rate, 0.5, places=2)

    def test_initial_stats_are_zero(self):
        """Fresh cache should have all zeros."""
        cache = SemanticCache()
        stats = cache.get_stats()
        self.assertEqual(stats["total_entries"], 0)
        self.assertEqual(stats["hit_count"], 0)
        self.assertEqual(stats["miss_count"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)


class TestCacheFlush(unittest.TestCase):
    """Test that flushing clears everything."""

    def test_flush_clears_everything(self):
        cache = SemanticCache(similarity_threshold=0.85)
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        cache.set("q1", {"r": 1}, query_embedding=emb)
        cache.get("q1", query_embedding=emb)

        cache.flush()

        self.assertEqual(cache.total_entries, 0)
        self.assertEqual(cache.hit_count, 0)
        self.assertEqual(cache.miss_count, 0)
        self.assertIsNone(cache.get("q1", query_embedding=emb))


if __name__ == "__main__":
    unittest.main()
