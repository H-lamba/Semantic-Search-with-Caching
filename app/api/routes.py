"""
API Route Handlers
==================
Three endpoints: search with cache, cache stats, and cache flush.
"""

from fastapi import APIRouter
from .schemas import QueryRequest, QueryResponse, SearchResult, CacheStatsResponse, FlushResponse

router = APIRouter()

# Service instances (injected at startup)
_embedding_service = None
_vector_store = None
_semantic_cache = None
_cluster_service = None


def set_services(embedding_service, vector_store, semantic_cache, cluster_service):
    """Called from main.py during startup to inject service instances."""
    global _embedding_service, _vector_store, _semantic_cache, _cluster_service
    _embedding_service = embedding_service
    _vector_store = vector_store
    _semantic_cache = semantic_cache
    _cluster_service = cluster_service


@router.post("/query", response_model=QueryResponse)
async def search_query(request: QueryRequest):
    """Semantic search with cache integration.

    Flow: embed query -> check cache -> on miss, search FAISS -> cache result.
    """
    query = request.query
    query_embedding = _embedding_service.encode_single(query)

    # Get fuzzy cluster assignment for the query
    dominant_cluster = 0
    cluster_probs = None
    cluster_probs_dict = None
    if _cluster_service and _cluster_service.gmm is not None:
        cluster_probs = _cluster_service.predict(query_embedding)
        dominant_cluster = int(cluster_probs.argmax())
        # Return top 5 cluster probabilities as a distribution
        top_indices = cluster_probs.argsort()[-5:][::-1]
        cluster_probs_dict = {
            f"cluster_{int(i)}": round(float(cluster_probs[i]), 4)
            for i in top_indices
        }

    # Check cache first
    cache_result = _semantic_cache.get(query, query_embedding=query_embedding)

    if cache_result is not None:
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=cache_result["matched_query"],
            similarity_score=round(cache_result["similarity_score"], 4),
            result=cache_result["result"],
            dominant_cluster=cache_result["dominant_cluster"],
            cluster_probabilities=cluster_probs_dict,
        )

    # Cache miss — search the vector store
    search_results = _vector_store.search(query_embedding, top_k=5)

    formatted_results = [
        SearchResult(
            category=r["category"],
            score=round(r["score"], 4),
            text=r["text"],
        )
        for r in search_results
    ]

    # Store in cache for future similar queries
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


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Return current cache statistics."""
    stats = _semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    """Flush the cache and reset all stats."""
    _semantic_cache.flush()
    return FlushResponse(
        message="Cache flushed successfully. All entries and stats reset.",
        status="ok",
    )
