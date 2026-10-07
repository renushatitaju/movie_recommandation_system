"""Preprocessing with a choice of mode ('none', 'stem', 'lemma').
Used by recommender.py. The cleaning steps live in cleandata.py, so there is only one copy."""
from nltk.stem import PorterStemmer

from cleandata import regex_clean, tokenize, remove_stopwords, lemmatize_tokens

stemmer = PorterStemmer()


def stem(tokens):
    return [stemmer.stem(t) for t in tokens]


def preprocess(text, mode="lemma"):
    """mode: 'none' (no stem/lemma), 'stem', or 'lemma'. Returns one cleaned string."""
    tokens = remove_stopwords(tokenize(regex_clean(text)))
    if mode == "stem":
        tokens = stem(tokens)
    elif mode == "lemma":
        tokens = lemmatize_tokens(tokens)
    elif mode != "none":
        raise ValueError("mode must be 'none', 'stem' or 'lemma'")
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "The astronauts were traveling through wormholes, searching for new planets!"
    for m in ("none", "stem", "lemma"):
        print(f"{m:>5}: {preprocess(sample, m)}")