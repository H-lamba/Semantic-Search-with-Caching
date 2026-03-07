# Semantic Search & Fuzzy Clustering API

A full-stack NLP pipeline that performs **semantic search** over the 20 Newsgroups dataset (~20K documents), using **fuzzy clustering**, a **custom semantic cache**, and a **FastAPI** web service.

## Architecture

```
User Query
  → FastAPI Endpoint (/query)
    → Embedding (all-MiniLM-L6-v2, 384-dim)
    → Semantic Cache Check (cosine similarity threshold)
      → HIT: Return cached result
      → MISS: FAISS Vector Search → GMM Cluster Assignment → Cache & Return
```

## Key Design Decisions

### Embedding Model: `all-MiniLM-L6-v2`
- **Why**: 384-dim vectors, trained on 1B+ sentence pairs for semantic similarity. Lightweight (22M params, 80MB) but ranks highly on MTEB benchmarks. Perfect for CPU deployment.
- **Why not larger models**: `all-mpnet-base-v2` (768-dim) offers marginal quality gains with 2x storage and slower search.

### Vector Database: FAISS (`IndexFlatIP`)
- **Why**: In-process C++ library — no external server needed. Native disk persistence. ~28MB for 19K docs.
- **Why not ChromaDB**: Adds SQLite dependency and abstraction overhead unnecessary for our scale.
- **Why `IndexFlatIP`**: With ~19K documents, exact search is <10ms. Approximate indexes (IVF) only help at 100K+ scale.

### Fuzzy Clustering: Gaussian Mixture Model (GMM)
- **Why GMM over FCM**: Probabilistic output (probability distribution per document), flexible elliptical cluster shapes, and built-in BIC/AIC for optimal cluster count selection.
- **Why not K-Means**: Hard clustering is strictly forbidden — documents genuinely belong to multiple topics.

### Semantic Cache: Custom (No Redis/Memcached)
- **Mechanism**: Cosine similarity between query embeddings. If `similarity > 0.85`, it's a cache hit.
- **Cluster-Aware Routing**: Cache entries are organized by cluster ID, reducing search from O(n) to O(n/k).
- **Tunable Parameter**: The similarity threshold (0.85 default) balances hit rate vs. precision. See `scripts/experiment_cache_threshold.py` for analysis.

### Data Cleaning: Preserve Natural Language
- **What we strip**: Email headers, quoted replies, signatures, HTML, URLs, email addresses, duplicates.
- **What we keep**: Natural sentence structure, punctuation, grammar, stopwords.
- **Why**: Transformer models need syntactic structure for accurate embeddings. Aggressive tokenization/stopword removal (appropriate for TF-IDF) degrades dense embedding quality.

## Project Structure

```
semantic-search-pipeline/
├── app/
│   ├── api/
│   │   ├── routes.py          # API endpoint handlers
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/                  # Configuration (future use)
│   └── services/
│       ├── embedding_service.py   # Sentence-transformer wrapper
│       ├── vector_store.py        # FAISS wrapper class
│       ├── clustering_service.py  # GMM fuzzy clustering
│       └── semantic_cache.py      # Custom cache (no external libs)
├── scripts/
│   ├── fetch_data.py          # Auto-download dataset
│   ├── load_data.py           # Load raw data
│   ├── eda_inspect.py         # Exploratory data analysis
│   ├── clean_data.py          # Data preprocessing
│   ├── data_pipeline.py       # Orchestrate fetch→load→save
│   ├── build_vector_db.py     # Generate embeddings + FAISS index
│   ├── build_clusters.py      # Train GMM clustering
│   └── experiment_cache_threshold.py  # Cache tuning experiment
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Cleaned corpus (parquet + JSON)
├── models/                    # Saved embeddings, FAISS index, GMM model
├── tests/
│   └── test_cache.py          # Cache unit tests
├── docs/
│   └── eda_findings.md        # EDA results documentation
├── main.py                    # FastAPI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Build pipeline (run once)
python scripts/data_pipeline.py       # Fetch + load + save corpus
python scripts/clean_data.py          # Clean corpus
python scripts/build_vector_db.py     # Generate embeddings + FAISS index
python scripts/build_clusters.py      # Train fuzzy clustering

# 3. Start API
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `POST /query`
```json
// Request
{ "query": "What is the best graphics card?" }

// Response
{
  "query": "What is the best graphics card?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    { "category": "comp.graphics", "score": 0.82, "text": "..." }
  ],
  "dominant_cluster": 3
}
```

### `GET /cache/stats`
```json
{ "total_entries": 42, "hit_count": 15, "miss_count": 27, "hit_rate": 0.3571 }
```

### `DELETE /cache`
```json
{ "message": "Cache flushed successfully.", "status": "ok" }
```

## Docker

```bash
docker-compose up --build
# API available at http://localhost:8000
```

## Tests

```bash
pytest tests/ -v
```
