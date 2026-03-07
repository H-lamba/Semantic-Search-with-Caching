"""
Build Fuzzy Clustering Model
==============================
Trains a Gaussian Mixture Model on document embeddings and analyzes
cluster quality. Uses BIC/AIC to find the optimal number of clusters.

Steps:
  1. Load embeddings and corpus
  2. Find optimal cluster count via BIC/AIC
  3. Train GMM at the optimal k
  4. Analyze cluster quality and composition
  5. Save model to disk
  6. Tag vector store metadata with cluster assignments

Run from project root:
    python scripts/build_clusters.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.clustering_service import FuzzyClusterService
from app.services.vector_store import VectorStore

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_clusters():
    # Step 1: Load embeddings and corpus
    print("=" * 60)
    print("Step 1: Loading embeddings and corpus...")
    print("=" * 60)
    embeddings = np.load(os.path.join(MODELS_DIR, "embeddings.npy"))
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "cleaned_corpus.parquet"))
    categories = df["category"].tolist()
    print(f"Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    print(f"Loaded {len(df)} documents\n")

    # Step 2: Find optimal number of clusters
    print("=" * 60)
    print("Step 2: Finding optimal cluster count (BIC/AIC)...")
    print("=" * 60)
    cluster_service = FuzzyClusterService()
    results, optimal_k = cluster_service.find_optimal_clusters(
        embeddings, min_k=10, max_k=30, step=2
    )

    print(f"\nCluster selection results:")
    print(f"{'k':>4} | {'BIC':>12} | {'AIC':>12} | {'Silhouette':>10}")
    print("-" * 45)
    for k, v in sorted(results.items()):
        marker = " <-- optimal" if k == optimal_k else ""
        print(f"{k:>4} | {v['bic']:>12.0f} | {v['aic']:>12.0f} | {v['silhouette']:>10.4f}{marker}")

    # Save selection results
    with open(os.path.join(MODELS_DIR, "cluster_selection.json"), "w") as f:
        serializable_results = {
            str(k): {key: float(val) for key, val in v.items()}
            for k, v in results.items()
        }
        json.dump({"results": serializable_results, "optimal_k": int(optimal_k)}, f, indent=2)
    print(f"\nSaved cluster selection results to models/cluster_selection.json\n")

    # Step 3: Train GMM
    print("=" * 60)
    print(f"Step 3: Training GMM with k={optimal_k}...")
    print("=" * 60)
    cluster_probs = cluster_service.train(embeddings, n_clusters=optimal_k)

    # Verify output is a distribution (not a hard label)
    print("\nVerification:")
    sample_probs = cluster_probs[0]
    print(f"  Sample doc cluster probs: {sample_probs[:5]}... (first 5)")
    print(f"  Sum = {sample_probs.sum():.6f} (should be 1.0)")
    print(f"  Is distribution: {len(sample_probs) > 1 and abs(sample_probs.sum() - 1.0) < 1e-5}")

    # Step 4: Analyze clusters
    print(f"\n{'=' * 60}")
    print("Step 4: Analyzing cluster quality...")
    print("=" * 60)
    analysis = cluster_service.analyze_clusters(cluster_probs, categories)

    print(f"\n  High-confidence docs (>0.7):  {analysis['high_confidence_count']} ({analysis['high_confidence_pct']})")
    print(f"  Boundary cases:               {analysis['boundary_count']} ({analysis['boundary_pct']})")
    print(f"  Uncertain docs (<0.3):         {analysis['uncertain_count']} ({analysis['uncertain_pct']})")

    print(f"\n  Cluster-to-Category Mapping:")
    print(f"  {'Cluster':>8} | {'Docs':>5} | Top Categories")
    print(f"  {'-'*60}")
    for c_id, info in sorted(analysis["cluster_category_mapping"].items()):
        top_cats = ", ".join([f"{cat}({cnt})" for cat, cnt in info["top_categories"][:3]])
        print(f"  {c_id:>8} | {info['total_docs']:>5} | {top_cats}")

    # Save analysis
    analysis_json = analysis.copy()
    for c_id in analysis_json["cluster_category_mapping"]:
        info = analysis_json["cluster_category_mapping"][c_id]
        info["top_categories"] = [[cat, cnt] for cat, cnt in info["top_categories"]]

    with open(os.path.join(MODELS_DIR, "cluster_analysis.json"), "w") as f:
        json.dump(analysis_json, f, indent=2)
    print(f"\n  Saved analysis to models/cluster_analysis.json")

    # Step 5: Save model
    print(f"\n{'=' * 60}")
    print("Step 5: Saving trained model...")
    print("=" * 60)
    cluster_service.save()

    # Step 6: Add cluster info to vector store metadata
    print(f"\n{'=' * 60}")
    print("Step 6: Adding cluster info to vector store metadata...")
    print("=" * 60)
    vector_store = VectorStore()
    if vector_store.load():
        dominant_clusters = np.argmax(cluster_probs, axis=1)
        for i in range(len(vector_store.metadata)):
            if i < len(dominant_clusters):
                vector_store.metadata[i]["dominant_cluster"] = int(dominant_clusters[i])
        vector_store.save()
        print("Updated vector store metadata with cluster info.")

    # Summary
    print(f"\n{'=' * 60}")
    print("CLUSTERING COMPLETE")
    print("=" * 60)
    print(f"  Algorithm:             Gaussian Mixture Model (GMM)")
    print(f"  Optimal clusters:      {optimal_k}")
    print(f"  Documents clustered:   {len(cluster_probs)}")
    print(f"  High confidence:       {analysis['high_confidence_pct']}")
    print(f"  Boundary cases:        {analysis['boundary_pct']}")
    print(f"  Uncertain:             {analysis['uncertain_pct']}")


if __name__ == "__main__":
    build_clusters()
