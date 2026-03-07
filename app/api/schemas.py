"""
Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class QueryRequest(BaseModel):
    """POST /query request body."""
    query: str = Field(..., description="Natural language search query", min_length=1)


class SearchResult(BaseModel):
    """Individual search result item."""
    category: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """POST /query response body."""
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: List[SearchResult]
    dominant_cluster: int
    cluster_probabilities: Optional[Dict[str, float]] = None


class CacheStatsResponse(BaseModel):
    """GET /cache/stats response body."""
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class FlushResponse(BaseModel):
    """DELETE /cache response body."""
    message: str
    status: str
