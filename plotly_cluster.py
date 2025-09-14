import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Load preprocessed memos
with open("memos.json", "r", encoding="utf-8") as f:
    memos = json.load(f)

documents = [m["clean_text"] for m in memos]
filenames = [m["filename"] for m in memos]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, max_df=0.6, min_df=2)
X = vectorizer.fit_transform(documents)

# KMeans Clustering
num_clusters = min(3, len(memos))  # safe if fewer memos
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)

for i, memo in enumerate(memos):
    memo["cluster"] = int(kmeans.labels_[i])

# Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

# Add coordinates back to memos
for i, memo in enumerate(memos):
    memo["pca1"] = reduced_X[i, 0]
    memo["pca2"] = reduced_X[i, 1]

# Plotly Scatter
fig = px.scatter(
    memos,
    x="pca1",
    y="pca2",
    color="cluster",
    hover_data={
        "filename": True,
        "cluster": True,
        "pca1": False,
        "pca2": False
    },
    title="Interactive Memo Clusters (PCA 2D Projection)"
)

fig.show()
