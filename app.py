import streamlit as st

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

# -----------------------------
# LOAD DATA
# -----------------------------

iris = load_iris()

X = iris.data

y = iris.target

# -----------------------------
# SCALE
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------
# PAGE
# -----------------------------

st.title("t-SNE Visualization")

st.write("Dimensionality Reduction using t-SNE")

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

perplexity = st.slider("Perplexity", 5, 50, 30)

# -----------------------------
# TSNE
# -----------------------------

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

X_tsne = tsne.fit_transform(X_scaled)

# -----------------------------
# VISUALIZATION
# -----------------------------

fig, ax = plt.subplots()

scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)

ax.set_title("t-SNE Projection")

st.pyplot(fig)
