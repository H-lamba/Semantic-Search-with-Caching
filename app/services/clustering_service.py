"""
Fuzzy Clustering Service (GMM)
===============================
Uses Gaussian Mixture Models to assign soft cluster memberships to documents.

Why GMM over other approaches:
    - Produces probability distributions, not hard labels — a document about
      gun legislation can be 42% politics, 35% firearms, 23% misc
    - Flexible elliptical cluster shapes (unlike K-Means' spherical assumption)
    - Built-in model selection via BIC/AIC for choosing cluster count
    - sklearn's implementation is fast and well-tested

Why not Fuzzy C-Means:
    - GMM handles high-dimensional data better with PCA preprocessing
    - BIC/AIC give a principled way to pick k (FCM lacks this)
    - sklearn has no FCM — would need skfuzzy which is less maintained
"""

import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gmm_model.pkl")
CLUSTER_PROBS_PATH = os.path.join(PROJECT_ROOT, "models", "cluster_probs.npy")


class FuzzyClusterService:
    """GMM-based fuzzy clustering for document embeddings."""

    def __init__(self):
        self.gmm = None
        self.pca = None
        self.cluster_probs = None

    def find_optimal_clusters(self, embeddings, min_k=10, max_k=30, step=2):
        """Search for the best number of clusters using BIC, AIC, and silhouette.

        We reduce dimensionality with PCA first (384 -> 50) to avoid GMM's
        curse of dimensionality and speed up fitting.

        Returns:
            results: dict mapping k -> {bic, aic, silhouette}
            optimal_k: the k with lowest BIC
        """
        print("Reducing dimensionality with PCA (384 -> 50)...")
        self.pca = PCA(n_components=50, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}\n")

        results = {}
        best_bic = float("inf")
        optimal_k = min_k

        for k in range(min_k, max_k + 1, step):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=42,
                n_init=5,
                max_iter=300,
            )
            gmm.fit(reduced)

            bic = gmm.bic(reduced)
            aic = gmm.aic(reduced)
            labels = gmm.predict(reduced)

            # Silhouette on a sample (full dataset is slow)
            sample_size = min(5000, len(reduced))
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(reduced), sample_size, replace=False)
            sil = silhouette_score(reduced[sample_idx], labels[sample_idx])

            results[k] = {"bic": bic, "aic": aic, "silhouette": sil}

            if bic < best_bic:
                best_bic = bic
                optimal_k = k

            print(f"Testing k={k}... BIC={bic:.0f}, AIC={aic:.0f}, Silhouette={sil:.4f}")

        print(f"\nOptimal k by BIC: {optimal_k}")
        return results, optimal_k

    def train(self, embeddings, n_clusters=20):
        """Train the GMM and return soft cluster assignments.

        Returns:
            numpy array of shape (n_docs, n_clusters) — each row sums to 1.0
        """
        if self.pca is None:
            self.pca = PCA(n_components=50, random_state=42)
            self.pca.fit(embeddings)

        reduced = self.pca.transform(embeddings)

        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=42,
            n_init=5,
            max_iter=300,
        )
        self.gmm.fit(reduced)

        self.cluster_probs = self.gmm.predict_proba(reduced)
        return self.cluster_probs

    def predict(self, embedding):
        """Get cluster probabilities for a single embedding.

        Returns:
            1D array of shape (n_clusters,) — probability distribution.
        """
        if self.gmm is None:
            raise RuntimeError("Model not trained or loaded yet")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        reduced = self.pca.transform(embedding)
        return self.gmm.predict_proba(reduced)[0]

    def analyze_clusters(self, cluster_probs, categories):
        """Analyze cluster quality and composition.

        Computes confidence distribution and maps clusters to real categories.
        """
        dominant = np.max(cluster_probs, axis=1)
        assignments = np.argmax(cluster_probs, axis=1)

        high = np.sum(dominant > 0.7)
        boundary = np.sum((dominant >= 0.3) & (dominant <= 0.5))
        uncertain = np.sum(dominant < 0.3)
        total = len(cluster_probs)

        # Map each cluster to its top real categories
        cluster_mapping = {}
        for c_id in range(cluster_probs.shape[1]):
            mask = assignments == c_id
            cluster_cats = [categories[i] for i in range(len(categories)) if mask[i]]

            from collections import Counter
            cat_counts = Counter(cluster_cats).most_common()

            cluster_mapping[c_id] = {
                "total_docs": int(mask.sum()),
                "top_categories": cat_counts[:5],
            }

        return {
            "high_confidence_count": int(high),
            "high_confidence_pct": f"{high / total * 100:.1f}%",
            "boundary_count": int(boundary),
            "boundary_pct": f"{boundary / total * 100:.1f}%",
            "uncertain_count": int(uncertain),
            "uncertain_pct": f"{uncertain / total * 100:.1f}%",
            "cluster_category_mapping": cluster_mapping,
        }

    def save(self, model_path=None, probs_path=None):
        """Save the trained model and cluster probabilities to disk."""
        model_path = model_path or MODEL_PATH
        probs_path = probs_path or CLUSTER_PROBS_PATH
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump({"gmm": self.gmm, "pca": self.pca}, f)

        np.save(probs_path, self.cluster_probs)
        print(f"Saved GMM model to {model_path}")
        print(f"Saved cluster probs ({self.cluster_probs.shape}) to {probs_path}")

    def load(self, model_path=None, probs_path=None):
        """Load a previously saved model from disk."""
        model_path = model_path or MODEL_PATH
        probs_path = probs_path or CLUSTER_PROBS_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.gmm = data["gmm"]
            self.pca = data["pca"]

        if os.path.exists(probs_path):
            self.cluster_probs = np.load(probs_path)

        print(f"Loaded GMM model ({self.gmm.n_components} clusters) from {model_path}")
        if self.cluster_probs is not None:
            print(f"Loaded cluster probs ({self.cluster_probs.shape}) from {probs_path}")
