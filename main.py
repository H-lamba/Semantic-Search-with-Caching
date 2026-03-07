"""
Semantic Search & Fuzzy Clustering API
=======================================
Main entry point for the FastAPI application.

Run with: uvicorn main:app --host 127.0.0.1 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app.core.config import config
from app.core.logger import setup_logger
from app.api.routes import router, set_services
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.clustering_service import FuzzyClusterService
from app.services.semantic_cache import SemanticCache

logger = setup_logger("semantic_search.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all services at startup, tear down on shutdown."""
    logger.info("=" * 50)
    logger.info("Starting Semantic Search API...")
    logger.info("=" * 50)

    embedding_service = EmbeddingService(
        model_name=config["embedding"]["model_name"]
    )
    vector_store = VectorStore(
        dimension=config["embedding"]["dimension"]
    )
    cluster_service = FuzzyClusterService()
    semantic_cache = SemanticCache(
        similarity_threshold=config["cache"]["similarity_threshold"],
        max_entries=config["cache"]["max_entries"],
        cluster_search_depth=config["cache"]["cluster_search_depth"],
        embedding_service=embedding_service,
        cluster_service=cluster_service,
    )

    vector_loaded = vector_store.load()
    if not vector_loaded:
        logger.warning("No vector store found. Run 'python scripts/build_vector_db.py' first.")

    try:
        cluster_service.load()
    except FileNotFoundError:
        logger.warning("No clustering model found. Run 'python scripts/build_clusters.py' first.")

    _ = embedding_service.model  # warm up

    set_services(embedding_service, vector_store, semantic_cache, cluster_service)

    logger.info("API ready! All services loaded.")
    logger.info("=" * 50)

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Semantic Search & Clustering API",
    description=(
        "A semantic search pipeline over the 20 Newsgroups dataset "
        "with fuzzy clustering and a custom semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": str(exc), "status": "error"}


app.include_router(router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Semantic Search & Clustering API is running.",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=True,
    )
