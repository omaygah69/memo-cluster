import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
import hdbscan
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# from transformers import pipeline
from keybert import KeyBERT

warnings.filterwarnings("ignore")

# Load a small and fast LLM
# llm = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-small",
#     max_new_tokens=50,   # limit output length (no warnings)
#     do_sample=False      # deterministic output
# )
kw_model = KeyBERT()
# llm = pipeline("text2text-generation", model="google/flan-t5-base")  # more fluent and larger

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

# UMAP Dimensionality Reduction
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

# Representative Documents
def get_representative_docs(X_reduced, labels, filenames, top_n=3):
    reps = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_points = np.where(labels == cluster_id)[0]
        cluster_center = X_reduced[cluster_points].mean(axis=0)
        distances = np.linalg.norm(X_reduced[cluster_points] - cluster_center, axis=1)
        closest_idxs = cluster_points[np.argsort(distances)[:top_n]]
        reps[cluster_id] = [filenames[idx] for idx in closest_idxs]
    return reps

representatives = get_representative_docs(X_reduced, labels, filenames, top_n=3)
for cluster, docs in representatives.items():
    print(f"\nCluster {cluster} representative memos:")
    for doc in docs:
        print("-", doc)

# Top Keywords per Cluster
def get_top_keywords_per_cluster(X, labels, vectorizer, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    clusters = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_docs = np.where(labels == cluster_id)[0]
        if len(cluster_docs) == 0:
            continue
        cluster_mean = X[cluster_docs].mean(axis=0).A1
        top_indices = cluster_mean.argsort()[::-1][:n_terms]
        clusters[cluster_id] = [terms[i] for i in top_indices]
    return clusters

top_keywords = get_top_keywords_per_cluster(X, labels, vectorizer)

# Cluster Explanations
def summarize_cluster(cluster_id, docs, top_n=3):
    if not docs:
        return f"Cluster {cluster_id}: No clear theme."
    # Join all docs in the cluster
    text = " ".join(docs)
    # Extract top_n keywords
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    words = [w for w, _ in keywords]
    return f"Cluster {cluster_id}: Related to {', '.join(words)}"

# Print Summaries
print()
for cluster_id, keywords in top_keywords.items():
    cluster_docs = [documents[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
    # print(f"\n=== Cluster {cluster_id} ===")
    # print("Top keywords:", ", ".join(keywords))
    print(summarize_cluster(cluster_id, cluster_docs))

# Silhouette Scores
mask = labels != -1
if len(set(labels[mask])) > 1:
    global_score = silhouette_score(X_reduced[mask], labels[mask])
    sample_scores = silhouette_samples(X_reduced[mask], labels[mask])
    print(f"\nGlobal Silhouette Score (excluding noise): {global_score:.4f}")

    for cluster_id in set(labels[mask]):
        cluster_scores = sample_scores[labels[mask] == cluster_id]
        avg_score = cluster_scores.mean()
        print(f"Cluster {cluster_id}: silhouette = {avg_score:.4f}")
else:
    print("\nNot enough clusters for silhouette scores")

# Visualization (UMAP 3D projection)
umap_3d = umap.UMAP(n_neighbors=15, n_components=3, random_state=42, metric="cosine")
X_3d = umap_3d.fit_transform(X)

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

ax.set_title("Memo Clusters (UMAP 3D + HDBSCAN)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.legend()
plt.show()
