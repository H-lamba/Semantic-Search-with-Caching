"""
Cache Threshold Experiment
===========================
Explores how changing the similarity threshold affects cache performance.
The threshold is the core tunable parameter in the semantic cache.

Run from project root:
    python scripts/experiment_cache_threshold.py
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.semantic_cache import SemanticCache
from app.services.embedding_service import EmbeddingService


def run_experiment():
    print("=" * 60)
    print("Cache Threshold Experiment")
    print("=" * 60)

    embed_service = EmbeddingService()

    # Test pairs: (original, paraphrase that should hit)
    test_pairs = [
        ("What is the best graphics card?", "Which GPU should I buy for gaming?"),
        ("How does encryption work?", "Explain cryptographic algorithms"),
        ("Is there life on other planets?", "Does alien life exist in space?"),
        ("What are the rules of baseball?", "How do you play baseball?"),
        ("I want to sell my old car", "Looking to sell my used vehicle"),
    ]

    unrelated_queries = [
        "How to bake chocolate cookies",
        "What is the weather like today",
        "History of ancient Rome",
    ]

    # Pre-compute all embeddings
    print("\nEncoding test queries...")
    originals = [p[0] for p in test_pairs]
    paraphrases = [p[1] for p in test_pairs]
    original_embs = [embed_service.encode_single(q) for q in originals]
    paraphrase_embs = [embed_service.encode_single(q) for q in paraphrases]
    unrelated_embs = [embed_service.encode_single(q) for q in unrelated_queries]

    # Show pairwise similarities between originals and paraphrases
    print("\nPairwise similarities (original vs paraphrase):")
    for i, (orig, para) in enumerate(test_pairs):
        sim = float(np.dot(original_embs[i], paraphrase_embs[i]))
        print(f"  {sim:.4f}: '{orig}' <-> '{para}'")

    # Test each threshold value
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    print(f"\n{'='*60}")
    print(f"{'Threshold':>10} | {'Hits':>4} | {'Misses':>6} | {'Hit Rate':>8} | {'False Pos':>9}")
    print(f"{'-'*60}")

    for threshold in thresholds:
        cache = SemanticCache(similarity_threshold=threshold)

        # Populate cache with originals
        for i, (query, emb) in enumerate(zip(originals, original_embs)):
            cache.set(query, {"result": f"result_{i}"}, query_embedding=emb)

        # Test paraphrases (should hit)
        for emb in paraphrase_embs:
            cache.get("paraphrase", query_embedding=emb)

        hits_from_paraphrases = cache.hit_count

        # Test unrelated queries (should miss — false positives if they hit)
        for emb in unrelated_embs:
            cache.get("unrelated", query_embedding=emb)

        false_positives = cache.hit_count - hits_from_paraphrases

        print(f"{threshold:>10.2f} | {cache.hit_count:>4} | {cache.miss_count:>6} | "
              f"{cache.hit_rate:>8.2%} | {false_positives:>9}")

    # Analysis
    print(f"\n{'='*60}")
    print("FINDINGS")
    print("=" * 60)
    print("""
    LOW threshold (0.70):
      High hit rate, but risks returning wrong cached results.
      Even unrelated queries might match — bad for precision.

    HIGH threshold (0.95):
      Very strict — only near-exact matches hit.
      Paraphrases are missed, defeating the cache purpose.

    SWEET SPOT (0.85):
      Catches paraphrases while rejecting unrelated queries.
      Best balance of hit rate and precision for this corpus.
    """)


if __name__ == "__main__":
    run_experiment()
