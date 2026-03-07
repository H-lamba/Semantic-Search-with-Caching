"""
Trademarkia AI&ML Engineer Task
================================
Semantic Search & Fuzzy Clustering Pipeline over the 20 Newsgroups Dataset.

This is the main entry point for the FastAPI application.

Task 88: Services are mounted using FastAPI's lifespan events for proper
state management. The embedding model, vector store, clustering model, and
semantic cache are all initialized once at startup and shared across requests.

Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app.api.routes import router, set_services
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.clustering_service import FuzzyClusterService
from app.services.semantic_cache import SemanticCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Task 88: Mount services using FastAPI lifespan events.

    Task 90: Optimized startup — loads pre-built indexes from disk
    instead of rebuilding. Boots in seconds after initial build.
    """
    print("=" * 50)
    print("Starting Semantic Search API...")
    print("=" * 50)

    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    cluster_service = FuzzyClusterService()
    semantic_cache = SemanticCache(
        similarity_threshold=0.85,
        embedding_service=embedding_service,
        cluster_service=cluster_service,
    )

    # Load pre-built models from disk (Task 39, 55)
    vector_loaded = vector_store.load()
    if not vector_loaded:
        print("WARNING: No vector store found. Run 'python scripts/build_vector_db.py' first.")

    try:
        cluster_service.load()
    except FileNotFoundError:
        print("WARNING: No clustering model found. Run 'python scripts/build_clusters.py' first.")

    # Pre-load embedding model (so first query isn't slow)
    _ = embedding_service.model

    # Inject services into routes
    set_services(embedding_service, vector_store, semantic_cache, cluster_service)

    print("\nAPI ready! All services loaded.")
    print("=" * 50)

    yield  # App is running

    # Cleanup on shutdown
    print("Shutting down...")


# ============================================================
# Task 76: Initialize FastAPI application
# ============================================================
app = FastAPI(
    title="Semantic Search & Clustering API",
    description=(
        "A full-stack NLP pipeline performing semantic search over the "
        "20 Newsgroups dataset with fuzzy clustering and a custom semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task 91: Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": str(exc), "status": "error"}

# Mount routes
app.include_router(router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Semantic Search & Clustering API is running.",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
