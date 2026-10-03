"""Step 2: the recommender class.
TF-IDF turns text into numbers, KNN finds the closest movies, k-means finds movie themes."""
import random

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from preprocess import preprocess


class NoMatchError(Exception):
    """Raised when no movie shares any word with the query."""


# which cleaned column (made by cleandata.py) belongs to which raw text column
PRECOMPUTED = {"overview": "clean_overview", "full_text": "clean_full"}


class MovieRecommender:
    def __init__(self, df, mode="lemma", text_column="full_text", **tfidf_params):
        """text_column: 'overview' or 'full_text' (overview + keywords + tagline).
        tfidf_params: extra TfidfVectorizer settings, e.g. sublinear_tf=True, ngram_range=(1, 2)."""
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.text_column = text_column

        # 1. text -> numbers (TF-IDF)
        ready = PRECOMPUTED.get(text_column)
        if mode == "lemma" and ready in self.df.columns:
            cleaned = self.df[ready]                     # already cleaned by cleandata.py
        else:
            cleaned = self.df[text_column].apply(lambda text: preprocess(text, mode))
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.matrix = self.vectorizer.fit_transform(cleaned)

        # 2. KNN: "training" only stores the movie vectors
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn.fit(self.matrix)

        # 3. k-means is built separately with build_clusters()
        self.kmeans = None
        self.cluster_names = {}

    # ---------- KNN search ----------
    def recommend(self, query, top_n=5):
        if not query.strip():
            raise ValueError("Please type a description first.")

        query_vec = self.vectorizer.transform([preprocess(query, self.mode)])
        if query_vec.nnz == 0:                           # no query word exists in any movie
            raise NoMatchError("None of your words appear in the movie descriptions.")

        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=min(top_n, len(self.df)))
        results = self.df.iloc[indices[0]].copy()
        results["similarity"] = 1 - distances[0]         # cosine distance -> similarity
        results = results[results["similarity"] > 0]
        if results.empty:
            raise NoMatchError("No close matches found. Try different words.")
        return results[["title", "similarity", "overview"]]

    def similarities(self, query):
        """Cosine similarity between the query and every movie (used by the hybrid model)."""
        query_vec = self.vectorizer.transform([preprocess(query, self.mode)])
        return cosine_similarity(query_vec, self.matrix)[0]

    # ---------- k-means themes ----------
    def build_clusters(self, n_clusters=15, top_words=5):
        n_clusters = min(n_clusters, len(self.df))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["cluster"] = self.kmeans.fit_predict(self.matrix)

        terms = self.vectorizer.get_feature_names_out()
        for i, center in enumerate(self.kmeans.cluster_centers_):
            best = center.argsort()[::-1][:top_words]    # highest-weighted words of the cluster
            self.cluster_names[i] = ", ".join(terms[j] for j in best)

    def theme_of(self, query):
        """Name of the theme (cluster) the query belongs to, or None."""
        if self.kmeans is None:
            return None
        query_vec = self.vectorizer.transform([preprocess(query, self.mode)])
        if query_vec.nnz == 0:
            return None
        return self.cluster_names[self.kmeans.predict(query_vec)[0]]

    # ---------- evaluation ----------
    def evaluate(self, sample_size=500, top_n=5, seed=42):
        """Top-5 accuracy (self-retrieval test).
        For random movies, build a query from a random half of the words of its description.
        It counts as correct if that same movie appears in the top 5 results."""
        rng = random.Random(seed)
        n = min(sample_size, len(self.df))
        picked = rng.sample(range(len(self.df)), n)

        queries = []
        for i in picked:
            words = self.df.loc[i, "overview"].split()
            queries.append(" ".join(rng.sample(words, max(1, len(words) // 2))))

        vectors = self.vectorizer.transform([preprocess(q, self.mode) for q in queries])
        _, indices = self.knn.kneighbors(vectors, n_neighbors=min(top_n, len(self.df)))
        hits = sum(1 for i, row in zip(picked, indices) if i in row)
        return hits / n


if __name__ == "__main__":
    from cleandata import load_movies

    rec = MovieRecommender(load_movies())
    rec.build_clusters()
    query = "space exploration with a twist ending"
    print(rec.recommend(query))
    print("Theme:", rec.theme_of(query))
    print(f"Top-5 accuracy: {rec.evaluate():.0%}")