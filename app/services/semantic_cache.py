"""
Custom Semantic Cache (Tasks 59-75)
====================================
A from-scratch semantic cache that recognizes semantically similar queries
even when phrased differently, using embedding similarity and cluster-aware
routing for efficiency.

CRITICAL CONSTRAINT (Task 60):
    Absolutely NO caching libraries (Redis, Memcached, etc.) are used.
    This is a pure Python implementation using dictionaries and numpy.

CORE MECHANISM (Task 61):
    The cache uses cosine similarity between query embeddings to determine
    if an incoming query is "close enough" to a previously cached query.
    Since our embeddings are L2-normalized, cosine similarity = dot product.

    similarity(q1, q2) = dot(embed(q1), embed(q2))

    If similarity > SIMILARITY_THRESHOLD, we consider it a cache HIT.

TUNABLE PARAMETER (Tasks 65-67):
    The SIMILARITY_THRESHOLD is the critical tunable decision. See
    config.yaml to change it and scripts/experiment_cache_threshold.py
    for analysis of different threshold values.

CLUSTER-AWARE ROUTING (Tasks 63-64):
    Entries are organized by dominant cluster ID. Only the top N clusters
    (configurable) are searched, reducing O(n) to O(n/k).

EVICTION POLICY:
    LRU (Least Recently Used) eviction is applied when the cache exceeds
    max_entries. The oldest unused entry across all clusters is evicted.
    This prevents unbounded memory growth in production.
"""

import time
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger("semantic_search.cache")


class SemanticCache:
    """Custom semantic cache with cluster-aware routing and LRU eviction.

    No external caching libraries are used — pure Python + numpy.

    Attributes:
        total_entries: Number of cached query-result pairs
        hit_count: Number of cache hits
        miss_count: Number of cache misses
        hit_rate: Dynamic property computing hit_count / (hit_count + miss_count)
    """

    def __init__(self, similarity_threshold=0.85, max_entries=10000,
                 cluster_search_depth=3, embedding_service=None, cluster_service=None):
        """Initialize the semantic cache.

        Args:
            similarity_threshold: Cosine similarity threshold for cache hits.
            max_entries: Maximum cache entries before LRU eviction kicks in.
            cluster_search_depth: Number of top clusters to search.
            embedding_service: EmbeddingService instance for encoding queries.
            cluster_service: FuzzyClusterService instance for cluster routing.
        """
        # Core data structure:
        # Dictionary mapping cluster_id -> list of cache entries
        self._cache = defaultdict(list)

        # Configurable parameters (from config.yaml)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.cluster_search_depth = cluster_search_depth

        # Services
        self.embedding_service = embedding_service
        self.cluster_service = cluster_service

        # Statistics counters
        self._total_entries = 0
        self._hit_count = 0
        self._miss_count = 0

        logger.info(
            f"Cache initialized: threshold={similarity_threshold}, "
            f"max_entries={max_entries}, cluster_depth={cluster_search_depth}"
        )

    @property
    def total_entries(self):
        return self._total_entries

    @property
    def hit_count(self):
        return self._hit_count

    @property
    def miss_count(self):
        return self._miss_count

    @property
    def hit_rate(self):
        """Dynamic hit rate calculation."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def _evict_lru(self):
        """LRU eviction: remove the least recently used cache entry.

        Scans all clusters to find the entry with the oldest last_accessed
        timestamp and removes it. This prevents unbounded memory growth.
        """
        oldest_time = float("inf")
        oldest_cluster = None
        oldest_idx = None

        for cluster_id, entries in self._cache.items():
            for i, entry in enumerate(entries):
                if entry["last_accessed"] < oldest_time:
                    oldest_time = entry["last_accessed"]
                    oldest_cluster = cluster_id
                    oldest_idx = i

        if oldest_cluster is not None and oldest_idx is not None:
            evicted = self._cache[oldest_cluster].pop(oldest_idx)
            self._total_entries -= 1
            logger.debug(f"Evicted LRU entry: '{evicted['query'][:50]}...'")

            # Clean up empty cluster buckets
            if not self._cache[oldest_cluster]:
                del self._cache[oldest_cluster]

    def get(self, query, query_embedding=None):
        """Search the cache for a semantically similar query.

        Uses embedding similarity to find matches, even if the query
        is phrased differently from the cached version.

        Cluster-aware routing: Only searches entries in the same or
        nearby clusters for efficiency.

        Args:
            query: The natural language query string.
            query_embedding: Pre-computed embedding (optional).

        Returns:
            dict or None: If hit, returns matched info. If miss, None.
        """
        if query_embedding is None and self.embedding_service is not None:
            query_embedding = self.embedding_service.encode_single(query)

        if query_embedding is None:
            self._miss_count += 1
            return None

        # Get cluster assignment for routing
        if self.cluster_service is not None and self.cluster_service.gmm is not None:
            cluster_probs = self.cluster_service.predict(query_embedding)
            top_clusters = np.argsort(cluster_probs)[-self.cluster_search_depth:][::-1]
        else:
            top_clusters = list(self._cache.keys())

        # Search for similar queries in relevant clusters
        best_match = None
        best_score = -1.0
        best_cluster = None
        best_idx = None

        for cluster_id in top_clusters:
            cluster_id = int(cluster_id)
            for i, entry in enumerate(self._cache.get(cluster_id, [])):
                score = float(np.dot(query_embedding, entry["embedding"]))
                if score > best_score:
                    best_score = score
                    best_match = entry
                    best_cluster = cluster_id
                    best_idx = i

        # Check if best match exceeds threshold
        if best_match is not None and best_score >= self.similarity_threshold:
            self._hit_count += 1
            # Update LRU timestamp
            self._cache[best_cluster][best_idx]["last_accessed"] = time.time()
            logger.info(f"Cache HIT: '{query[:50]}' matched '{best_match['query'][:50]}' (score={best_score:.4f})")
            return {
                "matched_query": best_match["query"],
                "similarity_score": best_score,
                "result": best_match["result"],
                "dominant_cluster": best_match.get("dominant_cluster", 0),
            }

        self._miss_count += 1
        logger.info(f"Cache MISS: '{query[:50]}' (best_score={best_score:.4f})")
        return None

    def set(self, query, result, query_embedding=None, cluster_probs=None):
        """Store a query-result pair in the cache with LRU tracking.

        Args:
            query: The natural language query string.
            result: The search result to cache.
            query_embedding: Pre-computed embedding (optional).
            cluster_probs: Pre-computed cluster probabilities (optional).
        """
        if query_embedding is None and self.embedding_service is not None:
            query_embedding = self.embedding_service.encode_single(query)

        if cluster_probs is None and self.cluster_service is not None and self.cluster_service.gmm is not None:
            cluster_probs = self.cluster_service.predict(query_embedding)

        if cluster_probs is not None:
            dominant_cluster = int(np.argmax(cluster_probs))
        else:
            dominant_cluster = 0

        # Evict LRU entry if at capacity
        if self._total_entries >= self.max_entries:
            self._evict_lru()

        entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "cluster_probs": cluster_probs,
            "dominant_cluster": dominant_cluster,
            "last_accessed": time.time(),  # LRU timestamp
        }

        self._cache[dominant_cluster].append(entry)
        self._total_entries += 1
        logger.debug(f"Cached: '{query[:50]}' in cluster {dominant_cluster}")

    def flush(self):
        """Clear all cached data and reset statistics."""
        self._cache.clear()
        self._total_entries = 0
        self._hit_count = 0
        self._miss_count = 0
        logger.info("Cache flushed: all entries and stats reset")

    def get_stats(self):
        """Return current cache statistics."""
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_rate, 4),
        }
