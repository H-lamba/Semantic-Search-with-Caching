"""
FastAPI API Routes (Tasks 76-88)
=================================
Defines the API endpoints for the semantic search service.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


# ============================================================
# Request/Response Models (Tasks 77-78)
# ============================================================

class QueryRequest(BaseModel):
    """POST /query request body (Task 77)."""
    query: str = Field(..., description="Natural language search query", min_length=1)


class SearchResult(BaseModel):
    """Individual search result item."""
    category: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """POST /query response body (Task 78).

    Returns exact keys as specified in Task 83.
    """
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: List[SearchResult]
    dominant_cluster: int
    cluster_probabilities: Optional[Dict[str, float]] = None


class CacheStatsResponse(BaseModel):
    """GET /cache/stats response body (Task 85)."""
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class FlushResponse(BaseModel):
    """DELETE /cache response body (Task 87)."""
    message: str
    status: str
