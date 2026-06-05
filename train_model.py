from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

# -----------------------------
# LOAD DATASET
# -----------------------------

iris = load_iris()

X = iris.data

# -----------------------------
# SCALING
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------
# TSNE
# -----------------------------

tsne = TSNE(n_components=2, random_state=42, perplexity=30)

X_tsne = tsne.fit_transform(X_scaled)

# -----------------------------
# OUTPUT
# -----------------------------

print("Original Shape:", X.shape)

print("Reduced Shape:", X_tsne.shape)
