"""
Fuzzy Clustering Service (Tasks 41-58)
=======================================
Implements fuzzy (soft) clustering on document embeddings using
Gaussian Mixture Models (GMM).

ALGORITHM SELECTION (Task 41):
    We chose Gaussian Mixture Models (GMM) over Fuzzy C-Means (FCM).

    JUSTIFICATION:
    1. PROBABILISTIC OUTPUT: GMM naturally outputs a probability distribution
       over clusters for each document (P(cluster_k | document)). This is
       exactly the "fuzzy cluster membership" the task requires — each document
       gets a vector of probabilities summing to 1, NOT a hard label.

    2. CLUSTER SHAPE FLEXIBILITY: GMM models each cluster as a multivariate
       Gaussian with its own covariance, allowing elliptical cluster shapes.
       FCM assumes spherical clusters (like K-Means), which is unrealistic
       for high-dimensional embedding spaces where topics overlap unevenly.

    3. MODEL SELECTION: GMM provides BIC (Bayesian Information Criterion) and
       AIC (Akaike Information Criterion) scores for principled selection of
       the optimal number of clusters (Task 43-44).

    4. SCIKIT-LEARN INTEGRATION: sklearn's GaussianMixture is battle-tested,
       well-documented, and integrates seamlessly with our numpy embeddings.

    Why not hard clustering (K-Means)?
        Hard assignments are STRICTLY FORBIDDEN by the task specification.
        A document about "space electronics" genuinely belongs to both
        sci.space and sci.electronics — hard labels would lose this signal.
"""

import os
import json
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gmm_model.pkl")
CLUSTER_PROBS_PATH = os.path.join(PROJECT_ROOT, "models", "cluster_probs.npy")


class FuzzyClusterService:
    """Manages fuzzy clustering of document embeddings.

    Each document receives a probability distribution over clusters,
    NOT a single hard label. This enables identification of:
      - Dominant cluster members (high single-cluster probability)
      - Boundary cases (split between 2-3 clusters)
      - Uncertain documents (flat distribution across many clusters)
    """

    def __init__(self):
        self.gmm = None
        self.n_clusters = None
        self.cluster_probs = None  # shape: (n_docs, n_clusters)

    def find_optimal_clusters(self, embeddings, min_k=10, max_k=30, step=2):
        """Determine optimal number of clusters using BIC/AIC (Tasks 43-44).

        We test a range of cluster counts and select the one that minimizes
        BIC (Bayesian Information Criterion). BIC penalizes model complexity,
        preventing overfitting to noise in the embedding space.

        Args:
            embeddings: numpy array of shape (n_docs, dim)
            min_k: Minimum clusters to test
            max_k: Maximum clusters to test
            step: Step size for cluster range

        Returns:
            dict with BIC, AIC, and silhouette scores for each k
        """
        # Reduce dimensionality for faster clustering (384 -> 50)
        # This also helps with the curse of dimensionality in GMM
        print("Reducing dimensionality with PCA (384 -> 50)...")
        pca = PCA(n_components=50, random_state=42)
        reduced = pca.fit_transform(embeddings)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}\n")

        results = {}
        k_range = range(min_k, max_k + 1, step)

        for k in k_range:
            print(f"Testing k={k}...", end=" ")
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",  # Diagonal covariance for efficiency
                random_state=42,
                n_init=3,
                max_iter=200,
            )
            gmm.fit(reduced)

            bic = gmm.bic(reduced)
            aic = gmm.aic(reduced)

            # Silhouette score (sample for speed)
            labels = gmm.predict(reduced)
            sample_idx = np.random.RandomState(42).choice(
                len(reduced), min(5000, len(reduced)), replace=False
            )
            sil = silhouette_score(reduced[sample_idx], labels[sample_idx])

            results[k] = {"bic": bic, "aic": aic, "silhouette": sil}
            print(f"BIC={bic:.0f}, AIC={aic:.0f}, Silhouette={sil:.4f}")

        # Select k with minimum BIC
        optimal_k = min(results, key=lambda k: results[k]["bic"])
        print(f"\nOptimal k by BIC: {optimal_k}")

        return results, optimal_k

    def train(self, embeddings, n_clusters, use_pca=True):
        """Train the fuzzy clustering model (Task 46).

        Args:
            embeddings: numpy array of shape (n_docs, dim)
            n_clusters: Number of clusters to use
            use_pca: Whether to reduce dimensionality first
        """
        self.n_clusters = n_clusters

        if use_pca:
            print("Applying PCA dimensionality reduction...")
            self.pca = PCA(n_components=50, random_state=42)
            reduced = self.pca.fit_transform(embeddings)
        else:
            self.pca = None
            reduced = embeddings

        print(f"Training GMM with {n_clusters} clusters...")
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=42,
            n_init=5,
            max_iter=300,
        )
        self.gmm.fit(reduced)
        print("GMM training complete.\n")

        # Task 47: Generate cluster probability distribution for every document
        self.cluster_probs = self.gmm.predict_proba(reduced)
        print(f"Generated cluster probabilities: {self.cluster_probs.shape}")
        print(f"Sum of probs per doc (should be 1.0): {self.cluster_probs[0].sum():.6f}")

        return self.cluster_probs

    def predict(self, embedding):
        """Calculate cluster distribution for a new query (Task 56).

        Args:
            embedding: numpy array of shape (dim,)

        Returns:
            numpy array of cluster probabilities, shape (n_clusters,)
        """
        if self.gmm is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        vec = embedding.reshape(1, -1)
        if self.pca is not None:
            vec = self.pca.transform(vec)
        probs = self.gmm.predict_proba(vec)
        return probs[0]

    def get_dominant_cluster(self, probs):
        """Get the dominant cluster ID from a probability distribution."""
        return int(np.argmax(probs))

    def analyze_clusters(self, cluster_probs, categories):
        """Analyze cluster quality and produce report (Tasks 49-54, 58).

        Args:
            cluster_probs: shape (n_docs, n_clusters)
            categories: list of true category labels

        Returns:
            dict with analysis results
        """
        n_docs = len(cluster_probs)
        dominant_clusters = np.argmax(cluster_probs, axis=1)
        max_probs = np.max(cluster_probs, axis=1)

        analysis = {
            "n_documents": n_docs,
            "n_clusters": cluster_probs.shape[1],
        }

        # Task 50: High confidence documents (dominant cluster > 0.7)
        high_conf_mask = max_probs > 0.7
        analysis["high_confidence_count"] = int(high_conf_mask.sum())
        analysis["high_confidence_pct"] = f"{high_conf_mask.sum() / n_docs * 100:.1f}%"

        # Task 51: Boundary cases (top prob < 0.5, but top 2 > 0.7 combined)
        sorted_probs = np.sort(cluster_probs, axis=1)[:, ::-1]
        boundary_mask = (sorted_probs[:, 0] < 0.5) & (sorted_probs[:, 0] + sorted_probs[:, 1] > 0.6)
        analysis["boundary_count"] = int(boundary_mask.sum())
        analysis["boundary_pct"] = f"{boundary_mask.sum() / n_docs * 100:.1f}%"

        # Task 52: Uncertain documents (max prob < 0.3)
        uncertain_mask = max_probs < 0.3
        analysis["uncertain_count"] = int(uncertain_mask.sum())
        analysis["uncertain_pct"] = f"{uncertain_mask.sum() / n_docs * 100:.1f}%"

        # Task 49: Map original labels to fuzzy clusters
        cluster_category_mapping = {}
        for c in range(cluster_probs.shape[1]):
            mask = dominant_clusters == c
            if mask.sum() == 0:
                continue
            cat_counts = {}
            for cat in np.array(categories)[mask]:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            # Sort by count descending
            sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
            cluster_category_mapping[int(c)] = {
                "total_docs": int(mask.sum()),
                "top_categories": sorted_cats[:5],
            }
        analysis["cluster_category_mapping"] = cluster_category_mapping

        return analysis

    def save(self, model_path=None, probs_path=None):
        """Serialize the trained model to disk (Task 55)."""
        m_path = model_path or MODEL_PATH
        p_path = probs_path or CLUSTER_PROBS_PATH

        os.makedirs(os.path.dirname(m_path), exist_ok=True)

        # Save GMM model + PCA
        with open(m_path, "wb") as f:
            pickle.dump({"gmm": self.gmm, "pca": self.pca, "n_clusters": self.n_clusters}, f)
        print(f"Saved GMM model to {m_path}")

        # Save cluster probabilities
        if self.cluster_probs is not None:
            np.save(p_path, self.cluster_probs)
            print(f"Saved cluster probs ({self.cluster_probs.shape}) to {p_path}")

    def load(self, model_path=None, probs_path=None):
        """Load a trained model from disk."""
        m_path = model_path or MODEL_PATH
        p_path = probs_path or CLUSTER_PROBS_PATH

        with open(m_path, "rb") as f:
            data = pickle.load(f)
        self.gmm = data["gmm"]
        self.pca = data["pca"]
        self.n_clusters = data["n_clusters"]
        print(f"Loaded GMM model ({self.n_clusters} clusters) from {m_path}")

        if os.path.exists(p_path):
            self.cluster_probs = np.load(p_path)
            print(f"Loaded cluster probs ({self.cluster_probs.shape}) from {p_path}")

        return True
