"""
clustering.py
─────────────
Unsupervised Learning: K-Means clustering + PCA visualisation.

Tasks:
  • Elbow method to choose optimal k
  • K-Means on TF-IDF feature matrix
  • PCA (2D) for cluster visualisation
  • matplotlib / seaborn charts

Run:  python clustering.py
      (requires processed_resumes.csv from data_preprocessing.py)
"""

import os
import sys
import warnings

# Force UTF-8 output on Windows to avoid UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster                  import KMeans
from sklearn.decomposition            import PCA
from sklearn.metrics                  import silhouette_score

warnings.filterwarnings('ignore')

PROCESSED_CSV = 'processed_resumes.csv'

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(
            f"{PROCESSED_CSV} not found. Run data_preprocessing.py first."
        )
    df = pd.read_csv(PROCESSED_CSV).fillna('')
    print(f"✅ Loaded {len(df)} resumes")
    return df


# ─────────────────────────────────────────────────────────────
# 2. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df['cleaned_resume'])
    print(f"📐 TF-IDF matrix: {X.shape}")
    return X


# ─────────────────────────────────────────────────────────────
# 3. ELBOW METHOD
# ─────────────────────────────────────────────────────────────
def plot_elbow(X, k_range=range(2, 11)):
    inertias    = []
    silhouettes = []

    print("📈 Running elbow method …")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, sample_size=1000)
        silhouettes.append(sil)
        print(f"   k={k}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, 'bo-')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, 'rs-')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('elbow_plot.png', dpi=120)
    plt.close()
    print("📊 Saved elbow_plot.png")

    # Return optimal k (highest silhouette)
    optimal_k = list(k_range)[np.argmax(silhouettes)]
    return optimal_k


# ─────────────────────────────────────────────────────────────
# 4. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────
def run_kmeans(X, k: int):
    print(f"\n🔵 Fitting K-Means with k={k} …", end=" ", flush=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    print("done")
    return km, labels


# ─────────────────────────────────────────────────────────────
# 5. PCA REDUCTION & VISUALISATION
# ─────────────────────────────────────────────────────────────
def plot_clusters(X, labels, df: pd.DataFrame, k: int):
    print("🔵 Reducing to 2D with PCA …", end=" ", flush=True)
    pca      = PCA(n_components=2, random_state=42)
    X_2d     = pca.fit_transform(X.toarray())
    var_exp  = pca.explained_variance_ratio_.sum()
    print(f"done  (variance explained: {var_exp:.1%})")

    palette = sns.color_palette("husl", k)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1: Cluster assignments ──────────────────────────
    for cluster_id in range(k):
        mask = labels == cluster_id
        axes[0].scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=f"Cluster {cluster_id}",
            color=palette[cluster_id], alpha=0.7, s=20
        )
    axes[0].set_title(f'PCA Cluster View (k={k}, var={var_exp:.1%})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(markerscale=2, fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── Plot 2: Colour by Category if available ───────────────
    cat_col = next((c for c in ['Category','category'] if c in df.columns), None)
    if cat_col:
        categories = df[cat_col].values
        unique_cats = list(set(categories))
        cat_palette = sns.color_palette("tab20", len(unique_cats))
        for i, cat in enumerate(unique_cats):
            mask = categories == cat
            axes[1].scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                label=cat, color=cat_palette[i], alpha=0.6, s=20
            )
        axes[1].set_title('PCA – Ground Truth Categories')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        if len(unique_cats) <= 15:
            axes[1].legend(markerscale=2, fontsize=7, bbox_to_anchor=(1, 1))
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('cluster_plot.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("📊 Saved cluster_plot.png")

    return X_2d, pca


# ─────────────────────────────────────────────────────────────
# 6. CLUSTER SUMMARY
# ─────────────────────────────────────────────────────────────
def summarise_clusters(df: pd.DataFrame, labels: np.ndarray):
    df = df.copy()
    df['cluster'] = labels

    print("\n── Cluster Summary ────────────────────────────")
    num_cols = [c for c in ['experience', 'skills_count', 'score'] if c in df.columns]
    summary  = df.groupby('cluster')[num_cols].agg(['mean', 'count']).round(2)
    print(summary.to_string())

    cat_col = next((c for c in ['Category','category'] if c in df.columns), None)
    if cat_col:
        print("\n── Top Category Per Cluster ────────────────────")
        top = df.groupby('cluster')[cat_col].agg(lambda x: x.value_counts().index[0])
        print(top.to_string())

    return df


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────
def run_clustering(k: int = None):
    print("=" * 55)
    print("  TalentAI – Clustering & PCA Pipeline")
    print("=" * 55)

    df = load_data()
    X  = build_features(df)

    if k is None:
        k = plot_elbow(X, k_range=range(2, 8))
        print(f"\n🎯 Optimal k selected: {k}")

    km, labels = run_kmeans(X, k)
    _, pca     = plot_clusters(X, labels, df, k)
    df_out     = summarise_clusters(df, labels)

    # Save artefacts
    joblib.dump(km,  'kmeans_model.pkl')
    joblib.dump(pca, 'pca_model.pkl')
    print("\n💾 Saved kmeans_model.pkl  +  pca_model.pkl")

    df_out.to_csv('clustered_resumes.csv', index=False)
    print("💾 Saved clustered_resumes.csv")

    return km, labels, pca


if __name__ == "__main__":
    run_clustering()
