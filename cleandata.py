"""Step 1: load the raw TMDB file, clean the text, and save a cleaned CSV.

Cleaning pipeline: regex cleaning -> tokenization -> stopword removal -> lemmatization
Each movie's searchable text = overview + keywords + tagline.
"""
import json
import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_PATH = "data/tmdb_5000_movies.csv"
CLEAN_PATH = "data/cleaned_movies.csv"


# ---------- NLTK setup (downloads only if missing) ----------
def _ensure(package, path):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)


_ensure("stopwords", "corpora/stopwords")
_ensure("wordnet", "corpora/wordnet")

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------- Cleaning steps ----------
def regex_clean(text):
    """Use regular expressions to remove noise from the text."""
    text = text.lower()                        # 1. lowercase
    text = re.sub(r"http\S+", " ", text)       # 2. remove URLs
    text = re.sub(r"<[^>]+>", " ", text)       # 3. remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)      # 4. keep only letters (removes digits, punctuation)
    text = re.sub(r"\s+", " ", text).strip()   # 5. collapse extra spaces
    return text


def tokenize(text):
    """Split the text into a list of words."""
    return text.split()


def remove_stopwords(tokens):
    """Remove common words (the, is, of...) and very short leftovers."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def lemmatize_tokens(tokens):
    """Reduce each word to its dictionary form: verb first, then noun."""
    return [lemmatizer.lemmatize(lemmatizer.lemmatize(t, "v"), "n") for t in tokens]


def clean_text(text):
    """Full pipeline: regex -> tokenize -> stopwords -> lemmatize -> one string."""
    tokens = tokenize(regex_clean(text))
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


# ---------- Data cleaning ----------
def parse_names(text):
    """TMDB stores keywords as JSON text -> return the names joined by spaces."""
    try:
        return " ".join(item["name"] for item in json.loads(text))
    except (TypeError, ValueError, KeyError):
        return ""


def clean_movies(raw_path=RAW_PATH, out_path=CLEAN_PATH):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Could not find {raw_path}. Put the TMDB csv in the data/ folder.")

    df = pd.read_csv(raw_path)
    for column in ("overview", "tagline", "keywords"):
        if column not in df.columns:
            df[column] = ""
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["keywords"] = df["keywords"].apply(parse_names)
    df = df[df["overview"].str.strip() != ""]                 # drop movies with no description

    # raw text used for matching = overview + keywords + tagline
    df["full_text"] = df["overview"] + " " + df["keywords"] + " " + df["tagline"]

    df["clean_overview"] = df["overview"].apply(clean_text)   # overview only
    df["clean_full"] = df["full_text"].apply(clean_text)      # overview + keywords + tagline
    df = df[df["clean_full"] != ""]

    # keep the original overview for display
    df = df[["id", "title", "overview", "full_text", "clean_overview", "clean_full"]]
    df = df.drop_duplicates(subset="title")

    df.to_csv(out_path, index=False)
    return df


def load_movies():
    """Used by the other files. Cleans the data first if needed."""
    if not os.path.exists(CLEAN_PATH):
        clean_movies()
    df = pd.read_csv(CLEAN_PATH)
    if "clean_full" not in df.columns:              # old file from an earlier version -> rebuild
        df = clean_movies()
    for column in ("overview", "full_text", "clean_overview", "clean_full"):
        df[column] = df[column].fillna("")
    return df


if __name__ == "__main__":
    data = clean_movies()
    print("Cleaned data saved to", CLEAN_PATH)
    print(data[["title", "clean_full"]].head())