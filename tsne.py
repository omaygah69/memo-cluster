import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import hdbscan
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# Custom Stopwords
memo_stopwords = [
    "student", "students", "school", "memo", "memorandum",
    "office", "university", "college", "campus",
    "psu", "lingayen", "pangasinan",
    "aaral", "academic"
]

# Load Preprocessed Memos
with open("memos.json", "r", encoding="utf-8") as f:
    memos = json.load(f)

documents = [m["clean_text"] for m in memos]
filenames = [m["filename"] for m in memos]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words=memo_stopwords,
    max_df=0.8,
    min_df=2,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(documents)

# Dimensionality Reduction for clustering (UMAP 10D is still recommended here)
# If you want to replace this with t-SNE too, it will be extremely slow.
import umap.umap_ as umap
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    random_state=42,
    metric="cosine"
)
X_reduced = umap_model.fit_transform(X)

# HDBSCAN Clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    metric='euclidean',
    cluster_selection_method='eom'
).fit(X_reduced)

labels = clusterer.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} clusters (+ noise)")

# Assign cluster labels to memos
for i, memo in enumerate(memos):
    memo["cluster"] = int(labels[i])

# --- Visualization with t-SNE 3D ---
print("\nRunning t-SNE 3D (this may take a while)...")
tsne_3d = TSNE(
    n_components=3,
    random_state=42,
    perplexity=30,
    learning_rate=200,
    metric="cosine",
    init="random"
)
X_3d = tsne_3d.fit_transform(X.toarray())  # t-SNE requires dense input

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
unique_labels = set(labels)

for lbl in unique_labels:
    mask = labels == lbl
    ax.scatter(
        X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
        label=f"Cluster {lbl}" if lbl != -1 else "Noise",
        alpha=0.7, s=40
    )

ax.set_title("Memo Clusters (t-SNE 3D + HDBSCAN)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.legend()
plt.show()
