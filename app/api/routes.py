"""
API Route Handlers (Tasks 79-87)
==================================
Implements the three core API endpoints:
  - POST /query     : Semantic search with cache (Tasks 79-83)
  - GET /cache/stats : Cache statistics (Tasks 84-85)
  - DELETE /cache    : Flush cache (Tasks 86-87)
"""

from fastapi import APIRouter, Depends
from .schemas import QueryRequest, QueryResponse, SearchResult, CacheStatsResponse, FlushResponse

router = APIRouter()

# These will be injected via dependency injection from main.py (Task 88)
_embedding_service = None
_vector_store = None
_semantic_cache = None
_cluster_service = None


def set_services(embedding_service, vector_store, semantic_cache, cluster_service):
    """Set service instances (called from main.py during startup)."""
    global _embedding_service, _vector_store, _semantic_cache, _cluster_service
    _embedding_service = embedding_service
    _vector_store = vector_store
    _semantic_cache = semantic_cache
    _cluster_service = cluster_service


# ============================================================
# POST /query (Tasks 79-83)
# ============================================================
@router.post("/query", response_model=QueryResponse)
async def search_query(request: QueryRequest):
    """Semantic search endpoint with semantic cache integration.

    Task 80: Generate embedding for the incoming query.
    Task 81: Check semantic cache first.
    Task 82: On cache miss, perform vector DB retrieval and cache the result.
    Task 83: Return exact keys as specified.
    """
    query = request.query

    # Task 80: Generate embedding for query
    query_embedding = _embedding_service.encode_single(query)

    # Get cluster info for the query (fuzzy probability distribution)
    dominant_cluster = 0
    cluster_probs = None
    cluster_probs_dict = None
    if _cluster_service and _cluster_service.gmm is not None:
        cluster_probs = _cluster_service.predict(query_embedding)
        dominant_cluster = int(cluster_probs.argmax())
        # Show top 5 cluster probabilities as a fuzzy distribution
        top_indices = cluster_probs.argsort()[-5:][::-1]
        cluster_probs_dict = {
            f"cluster_{int(i)}": round(float(cluster_probs[i]), 4)
            for i in top_indices
        }

    # Task 81: Check semantic cache
    cache_result = _semantic_cache.get(query, query_embedding=query_embedding)

    if cache_result is not None:
        # Cache HIT
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=cache_result["matched_query"],
            similarity_score=round(cache_result["similarity_score"], 4),
            result=cache_result["result"],
            dominant_cluster=cache_result["dominant_cluster"],
            cluster_probabilities=cluster_probs_dict,
        )

    # Task 82: Cache MISS — perform vector DB retrieval
    search_results = _vector_store.search(query_embedding, top_k=5)

    formatted_results = [
        SearchResult(
            category=r["category"],
            score=round(r["score"], 4),
            text=r["text"],
        )
        for r in search_results
    ]

    # Store in cache for future queries
    _semantic_cache.set(
        query=query,
        result=formatted_results,
        query_embedding=query_embedding,
        cluster_probs=cluster_probs,
    )

    return QueryResponse(
        query=query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=formatted_results,
        dominant_cluster=dominant_cluster,
        cluster_probabilities=cluster_probs_dict,
    )


# ============================================================
# GET /cache/stats (Tasks 84-85)
# ============================================================
@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Return current cache statistics (Task 85).

    Returns: total_entries, hit_count, miss_count, hit_rate.
    """
    stats = _semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


# ============================================================
# DELETE /cache (Tasks 86-87)
# ============================================================
@router.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    """Flush the semantic cache completely (Task 87).

    Calls the flush method to empty the cache and reset all stats.
    """
    _semantic_cache.flush()
    return FlushResponse(
        message="Cache flushed successfully. All entries and stats reset.",
        status="ok",
    )
