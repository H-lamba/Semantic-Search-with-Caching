"""
Unit Tests for Semantic Cache (Tasks 74-75)
============================================
Tests that:
  - Similarly phrased queries trigger a cache HIT (Task 74)
  - Distinct queries trigger a cache MISS (Task 75)
"""

import sys
import os
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.semantic_cache import SemanticCache


def make_embedding(seed, dim=384):
    """Create a deterministic normalized embedding for testing."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def make_similar_embedding(base, noise_level=0.05):
    """Create an embedding similar to base (high cosine similarity)."""
    noise = np.random.RandomState(99).randn(*base.shape).astype(np.float32) * noise_level
    vec = base + noise
    return vec / np.linalg.norm(vec)


def make_different_embedding(seed=999, dim=384):
    """Create an embedding very different from others."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestSemanticCacheHit:
    """Task 74: Validate that similarly phrased queries trigger a cache hit."""

    def test_exact_same_query_hits(self):
        """Exact same embedding should always hit."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = make_embedding(42)

        cache.set("What is the best GPU?", {"answer": "RTX 4090"}, query_embedding=emb)
        result = cache.get("What is the best GPU?", query_embedding=emb)

        assert result is not None
        assert result["similarity_score"] == pytest.approx(1.0, abs=1e-5)
        assert result["matched_query"] == "What is the best GPU?"
        assert result["result"]["answer"] == "RTX 4090"

    def test_similar_query_hits(self):
        """Slightly different embedding (paraphrase) should hit."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb_original = make_embedding(42)
        emb_paraphrase = make_similar_embedding(emb_original, noise_level=0.03)

        # Verify they are similar enough
        sim = float(np.dot(emb_original, emb_paraphrase))
        assert sim > 0.85, f"Test setup error: similarity {sim} < 0.85"

        cache.set("best graphics card", {"answer": "RTX 4090"}, query_embedding=emb_original)
        result = cache.get("top GPU for gaming", query_embedding=emb_paraphrase)

        assert result is not None
        assert result["similarity_score"] >= 0.85

    def test_hit_increments_counter(self):
        """Cache hit should increment hit_count."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = make_embedding(42)

        cache.set("query", {"data": 1}, query_embedding=emb)
        cache.get("query", query_embedding=emb)

        assert cache.hit_count == 1
        assert cache.miss_count == 0


class TestSemanticCacheMiss:
    """Task 75: Validate that distinct queries trigger a cache miss."""

    def test_different_query_misses(self):
        """Very different embedding should miss."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb_gpu = make_embedding(42)
        emb_baseball = make_different_embedding(seed=999)

        # Verify they are different enough
        sim = float(np.dot(emb_gpu, emb_baseball))
        assert sim < 0.85, f"Test setup error: similarity {sim} >= 0.85"

        cache.set("best graphics card", {"answer": "RTX 4090"}, query_embedding=emb_gpu)
        result = cache.get("rules of baseball", query_embedding=emb_baseball)

        assert result is None

    def test_empty_cache_misses(self):
        """Empty cache should always miss."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = make_embedding(42)

        result = cache.get("any query", query_embedding=emb)
        assert result is None

    def test_miss_increments_counter(self):
        """Cache miss should increment miss_count."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = make_embedding(42)

        cache.get("query", query_embedding=emb)

        assert cache.miss_count == 1
        assert cache.hit_count == 0


class TestCacheStatistics:
    """Test statistics tracking (Tasks 71-72)."""

    def test_hit_rate_calculation(self):
        """Hit rate should be dynamically calculated (Task 72)."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = make_embedding(42)
        diff_emb = make_different_embedding()

        cache.set("query", {"data": 1}, query_embedding=emb)

        # 1 hit + 1 miss = 50% hit rate
        cache.get("query", query_embedding=emb)       # hit
        cache.get("different", query_embedding=diff_emb)  # miss

        assert cache.hit_rate == pytest.approx(0.5)
        assert cache.total_entries == 1

    def test_initial_stats_are_zero(self):
        """Fresh cache should have all zero stats."""
        cache = SemanticCache()
        assert cache.total_entries == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert cache.hit_rate == 0.0


class TestCacheFlush:
    """Test flush method (Task 73)."""

    def test_flush_clears_everything(self):
        """Flush should clear all data and reset stats."""
        cache = SemanticCache(similarity_threshold=0.85)
        emb = make_embedding(42)

        cache.set("query", {"data": 1}, query_embedding=emb)
        cache.get("query", query_embedding=emb)  # hit

        assert cache.total_entries == 1
        assert cache.hit_count == 1

        cache.flush()

        assert cache.total_entries == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert cache.hit_rate == 0.0

        # Verify cache is actually empty
        result = cache.get("query", query_embedding=emb)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
