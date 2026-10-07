"""Models 2 and 3: spaCy word vectors, and a hybrid of TF-IDF + spaCy.

spaCy vectors know that words with similar meaning are close together
("dinosaur" is near "extinct animal"), which TF-IDF cannot do.
"""
import os

import numpy as np
import spacy

from recommender import NoMatchError


class SimilarityRecommender:
    """Base class. A subclass only has to provide similarities(query) and self.df."""

    def similarities(self, query):
        raise NotImplementedError("Subclasses must implement similarities()")

    def recommend(self, query, top_n=5):
        if not query.strip():
            raise ValueError("Please type a description first.")
        scores = self.similarities(query)
        if scores.max() <= 0:
            raise NoMatchError("None of your words are known to the model. Try different words.")
        top = scores.argsort()[::-1][:top_n]
        results = self.df.iloc[top].copy()
        results["similarity"] = scores[top]
        return results[["title", "similarity", "overview"]]


class SpacyRecommender(SimilarityRecommender):
    def __init__(self, df, text_column="full_text", model="en_core_web_md"):
        self.df = df.reset_index(drop=True)
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' is not installed. Run:  python -m spacy download {model}"
            ) from None
        if self.nlp.vocab.vectors_length == 0:
            raise RuntimeError("This spaCy model has no word vectors. Use en_core_web_md.")

        # Turning 4,800 texts into vectors is slow, so the result is saved to a file.
        # If you change the cleaning or the text column, delete this file.
        cache_file = f"data/spacy_vectors_{model}_{len(self.df)}.npy"
        if os.path.exists(cache_file):
            self.matrix = np.load(cache_file)
        else:
            self.matrix = np.array([self._vector(text) for text in self.df[text_column]])
            np.save(cache_file, self.matrix)

    def _vector(self, text):
        """Average the vectors of the meaningful words, then scale the result to length 1."""
        doc = self.nlp.make_doc(text)            # only splits into words, no heavy pipeline
        vectors = [t.vector for t in doc if t.is_alpha and not t.is_stop and t.has_vector]
        if not vectors:
            return np.zeros(self.nlp.vocab.vectors_length)
        average = np.mean(vectors, axis=0)
        return average / (np.linalg.norm(average) or 1.0)

    def similarities(self, query):
        return self.matrix @ self._vector(query)     # cosine similarity (all vectors have length 1)


class HybridRecommender(SimilarityRecommender):
    """Mix the exact-word score (TF-IDF) and the meaning score (spaCy)."""

    def __init__(self, tfidf_model, spacy_model, weight=0.5):
        self.df = tfidf_model.df
        self.tfidf_model = tfidf_model
        self.spacy_model = spacy_model
        self.weight = weight                       # share of the TF-IDF score

    @staticmethod
    def _scale(scores):
        """Rescale scores to 0..1 so both models count equally."""
        spread = scores.max() - scores.min()
        return (scores - scores.min()) / spread if spread > 0 else np.zeros_like(scores)

    def similarities(self, query):
        tfidf_scores = self._scale(self.tfidf_model.similarities(query))
        spacy_scores = self._scale(self.spacy_model.similarities(query))
        return self.weight * tfidf_scores + (1 - self.weight) * spacy_scores