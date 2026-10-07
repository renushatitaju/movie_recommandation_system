"""Train the TF-IDF model, print train and test accuracy, and save the model.
Run:  python train.py
"""
import json

import joblib
from sklearn.model_selection import train_test_split

from cleandata import load_movies
from recommender import MovieRecommender

MODEL_PATH = "data/tfidf_model.pkl"
ACCURACY_PATH = "data/accuracy.json"
SETTINGS = dict(mode="lemma", text_column="full_text", sublinear_tf=True)

df = load_movies()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 1. learn only from the training movies (80%)
model = MovieRecommender(train_df, **SETTINGS)
train_accuracy = model.evaluate_on(train_df)     # movies the model has seen
test_accuracy = model.evaluate_on(test_df)       # movies the model has never seen
print(f"Train movies: {len(train_df)}   Test movies: {len(test_df)}")
print(f"Train accuracy (top 5): {train_accuracy:.1%}")
print(f"Test accuracy  (top 5): {test_accuracy:.1%}")

with open(ACCURACY_PATH, "w") as file:
    json.dump({"train": train_accuracy, "test": test_accuracy}, file)

# 2. the saved model uses all movies, so the app can recommend every movie
final = MovieRecommender(df, **SETTINGS)
final.build_clusters(15)
joblib.dump(final, MODEL_PATH)
print("Model saved to", MODEL_PATH)