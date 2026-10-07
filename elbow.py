"""Choose K for k-means (elbow plot) and look at the cluster sizes.
Run:  python elbow.py
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from cleandata import load_movies
from recommender import MovieRecommender

K_VALUES = [5, 10, 15, 20, 25, 30]
CHOSEN_K = 15

df = load_movies()
rec = MovieRecommender(df, mode="lemma", text_column="full_text", sublinear_tf=True)

# ---- elbow plot: inertia = how spread out the clusters are (lower = tighter) ----
inertias = []
for k in K_VALUES:
    model = KMeans(n_clusters=k, random_state=42, n_init=3).fit(rec.matrix)
    inertias.append(model.inertia_)
    print(f"K={k:<3} inertia={model.inertia_:.1f}")

plt.figure(figsize=(6, 4))
plt.plot(K_VALUES, inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow plot")
plt.tight_layout()
plt.savefig("elbow.png")

# ---- cluster sizes and theme words for the chosen K ----
rec.build_clusters(CHOSEN_K)
sizes = rec.df["cluster"].value_counts().sort_index()
for i, count in sizes.items():
    print(f"Cluster {i:<2} ({count:>4} movies): {rec.cluster_names[i]}")

plt.figure(figsize=(7, 4))
plt.bar(sizes.index, sizes.values)
plt.xlabel("Cluster")
plt.ylabel("Number of movies")
plt.title(f"Cluster sizes (K={CHOSEN_K})")
plt.tight_layout()
plt.savefig("cluster_sizes.png")
plt.show()