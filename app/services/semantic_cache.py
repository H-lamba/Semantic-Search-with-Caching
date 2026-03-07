"""
Custom Semantic Cache
=====================
A from-scratch semantic cache that recognizes semantically similar queries
even when phrased differently.

How it works:
    The cache stores query embeddings alongside their results. When a new
    query comes in, we compute its embedding and check cosine similarity
    against cached entries. If similarity > threshold, it's a cache hit.

    Since our embeddings are L2-normalized, cosine similarity = dot product.

Cluster-aware routing:
    Entries are organized by their dominant cluster ID. On lookup, we only
    search the top N clusters (default 3), reducing search from O(n) to
    O(n/k). This matters when the cache grows large.

Eviction policy:
    LRU (Least Recently Used) — when the cache exceeds max_entries, the
    oldest unused entry is evicted. Prevents unbounded memory growth.

The similarity threshold (default 0.85) is the key tunable parameter.
See scripts/experiment_cache_threshold.py for analysis of different values.

No Redis, Memcached, or any caching library. Pure Python + numpy.
"""

import time
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger("semantic_search.cache")


class SemanticCache:
    """Semantic cache with cluster-aware routing and LRU eviction."""

    def __init__(self, similarity_threshold=0.85, max_entries=10000,
                 cluster_search_depth=3, embedding_service=None, cluster_service=None):
        self._cache = defaultdict(list)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.cluster_search_depth = cluster_search_depth
        self.embedding_service = embedding_service
        self.cluster_service = cluster_service

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
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def _evict_lru(self):
        """Remove the least recently used cache entry."""
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
            self._cache[oldest_cluster].pop(oldest_idx)
            self._total_entries -= 1
            if not self._cache[oldest_cluster]:
                del self._cache[oldest_cluster]

    def get(self, query, query_embedding=None):
        """Look up a semantically similar cached query.

        Returns cached result if similarity > threshold, else None.
        """
        if query_embedding is None and self.embedding_service is not None:
            query_embedding = self.embedding_service.encode_single(query)

        if query_embedding is None:
            self._miss_count += 1
            return None

        # Route to relevant clusters only
        if self.cluster_service is not None and self.cluster_service.gmm is not None:
            cluster_probs = self.cluster_service.predict(query_embedding)
            top_clusters = np.argsort(cluster_probs)[-self.cluster_search_depth:][::-1]
        else:
            top_clusters = list(self._cache.keys())

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

        if best_match is not None and best_score >= self.similarity_threshold:
            self._hit_count += 1
            self._cache[best_cluster][best_idx]["last_accessed"] = time.time()
            logger.info(f"Cache HIT: '{query[:50]}' -> '{best_match['query'][:50]}' (sim={best_score:.4f})")
            return {
                "matched_query": best_match["query"],
                "similarity_score": best_score,
                "result": best_match["result"],
                "dominant_cluster": best_match.get("dominant_cluster", 0),
            }

        self._miss_count += 1
        logger.info(f"Cache MISS: '{query[:50]}' (best={best_score:.4f})")
        return None

    def set(self, query, result, query_embedding=None, cluster_probs=None):
        """Store a query-result pair in the cache."""
        if query_embedding is None and self.embedding_service is not None:
            query_embedding = self.embedding_service.encode_single(query)

        if cluster_probs is None and self.cluster_service is not None and self.cluster_service.gmm is not None:
            cluster_probs = self.cluster_service.predict(query_embedding)

        dominant_cluster = int(np.argmax(cluster_probs)) if cluster_probs is not None else 0

        if self._total_entries >= self.max_entries:
            self._evict_lru()

        self._cache[dominant_cluster].append({
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "cluster_probs": cluster_probs,
            "dominant_cluster": dominant_cluster,
            "last_accessed": time.time(),
        })
        self._total_entries += 1

    def flush(self):
        """Clear all cached data and reset stats."""
        self._cache.clear()
        self._total_entries = 0
        self._hit_count = 0
        self._miss_count = 0
        logger.info("Cache flushed")

    def get_stats(self):
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_rate, 4),
        }
