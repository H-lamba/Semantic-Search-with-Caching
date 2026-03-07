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
    The SIMILARITY_THRESHOLD is the critical tunable decision at the heart
    of this component.

    - HIGH threshold (e.g., 0.95): Very strict matching. Only nearly identical
      queries trigger a hit. Low hit rate, but very high precision — you almost
      never return a wrong cached result.

    - LOW threshold (e.g., 0.70): Loose matching. Paraphrased queries and related
      topics trigger hits. High hit rate, but risk of returning semantically
      different results (false positives).

    - SWEET SPOT (0.85): Balances hit rate with precision. Catches paraphrases
      ("best GPU" ↔ "top graphics card") while rejecting unrelated queries
      ("best GPU" ↔ "best CPU").

    The optimal value depends on the application's tolerance for stale/incorrect
    cached results vs. the cost of a cache miss (full vector DB retrieval).

CLUSTER-AWARE ROUTING (Tasks 63-64):
    To maintain lookup efficiency as the cache grows, entries are organized
    by their dominant cluster ID. When searching, we only compare the incoming
    query against cache entries in the same or nearby clusters.

    This reduces O(n) comparisons to O(n/k) where k = number of clusters,
    providing a natural partitioning of the cache space.
"""

import numpy as np
from collections import defaultdict


class SemanticCache:
    """Custom semantic cache with cluster-aware routing (Task 68).

    No external caching libraries are used — pure Python + numpy.

    Attributes:
        total_entries: Number of cached query-result pairs
        hit_count: Number of cache hits
        miss_count: Number of cache misses
        hit_rate: Dynamic property computing hit_count / (hit_count + miss_count)
    """

    def __init__(self, similarity_threshold=0.85, embedding_service=None, cluster_service=None):
        """Initialize the semantic cache.

        Args:
            similarity_threshold: Cosine similarity threshold for cache hits (Task 65).
                The tunable parameter at the heart of this component.
            embedding_service: EmbeddingService instance for encoding queries.
            cluster_service: FuzzyClusterService instance for cluster routing.
        """
        # Core data structure (Task 59):
        # Dictionary mapping cluster_id -> list of cache entries
        # Each entry: {"query": str, "embedding": np.array, "result": dict, "cluster_probs": np.array}
        self._cache = defaultdict(list)

        # The tunable parameter (Task 65)
        self.similarity_threshold = similarity_threshold

        # Services
        self.embedding_service = embedding_service
        self.cluster_service = cluster_service

        # Statistics counters (Task 71)
        self._total_entries = 0
        self._hit_count = 0
        self._miss_count = 0

    @property
    def total_entries(self):
        """Total number of cached entries (Task 71)."""
        return self._total_entries

    @property
    def hit_count(self):
        """Total cache hits (Task 71)."""
        return self._hit_count

    @property
    def miss_count(self):
        """Total cache misses (Task 71)."""
        return self._miss_count

    @property
    def hit_rate(self):
        """Dynamic hit rate calculation (Task 72).

        Returns the cache hit rate as a float between 0.0 and 1.0.
        Returns 0.0 if no queries have been made yet.
        """
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total

    def get(self, query, query_embedding=None):
        """Search the cache for a semantically similar query (Task 69).

        Uses embedding similarity to find matches, even if the query
        is phrased differently from the cached version (Task 62).

        Cluster-aware routing (Task 64): Only searches entries in the
        same or nearby clusters for efficiency.

        Args:
            query: The natural language query string.
            query_embedding: Pre-computed embedding (optional, computed if None).

        Returns:
            dict or None: If hit, returns {
                "matched_query": str,  # The original cached query
                "similarity_score": float,
                "result": dict,  # The cached result
                "dominant_cluster": int
            }
            If miss, returns None.
        """
        # Compute embedding if not provided
        if query_embedding is None and self.embedding_service is not None:
            query_embedding = self.embedding_service.encode_single(query)

        if query_embedding is None:
            self._miss_count += 1
            return None

        # Get cluster assignment for routing (Task 64)
        if self.cluster_service is not None and self.cluster_service.gmm is not None:
            cluster_probs = self.cluster_service.predict(query_embedding)
            dominant_cluster = int(np.argmax(cluster_probs))
            # Search the dominant cluster + nearby clusters (top 3 by probability)
            top_clusters = np.argsort(cluster_probs)[-3:][::-1]
        else:
            # No clustering available — search all entries
            dominant_cluster = 0
            top_clusters = list(self._cache.keys())

        # Search for similar queries in relevant clusters
        best_match = None
        best_score = -1.0

        for cluster_id in top_clusters:
            cluster_id = int(cluster_id)
            for entry in self._cache.get(cluster_id, []):
                # Cosine similarity via dot product (embeddings are L2-normalized)
                score = float(np.dot(query_embedding, entry["embedding"]))

                if score > best_score:
                    best_score = score
                    best_match = entry

        # Check if best match exceeds threshold
        if best_match is not None and best_score >= self.similarity_threshold:
            self._hit_count += 1
            return {
                "matched_query": best_match["query"],
                "similarity_score": best_score,
                "result": best_match["result"],
                "dominant_cluster": best_match.get("dominant_cluster", 0),
            }

        self._miss_count += 1
        return None

    def set(self, query, result, query_embedding=None, cluster_probs=None):
        """Store a query-result pair in the cache (Task 70).

        Args:
            query: The natural language query string.
            result: The search result to cache.
            query_embedding: Pre-computed embedding (optional).
            cluster_probs: Pre-computed cluster probabilities (optional).
        """
        # Compute embedding if not provided
        if query_embedding is None and self.embedding_service is not None:
            query_embedding = self.embedding_service.encode_single(query)

        # Compute cluster assignment
        if cluster_probs is None and self.cluster_service is not None and self.cluster_service.gmm is not None:
            cluster_probs = self.cluster_service.predict(query_embedding)

        if cluster_probs is not None:
            dominant_cluster = int(np.argmax(cluster_probs))
        else:
            dominant_cluster = 0

        entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "cluster_probs": cluster_probs,
            "dominant_cluster": dominant_cluster,
        }

        self._cache[dominant_cluster].append(entry)
        self._total_entries += 1

    def flush(self):
        """Clear all cached data and reset statistics (Task 73).

        Completely empties the cache and resets hit/miss counters to zero.
        """
        self._cache.clear()
        self._total_entries = 0
        self._hit_count = 0
        self._miss_count = 0

    def get_stats(self):
        """Return current cache statistics.

        Returns:
            dict with total_entries, hit_count, miss_count, hit_rate
        """
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_rate, 4),
        }
