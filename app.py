import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data (runs once when app starts)
df = pd.read_csv("data/cleaned_movies.csv")
df['overview'] = df['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

def recommend(query, top_n=5):
    query_vec = tfidf.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarity_scores[top_indices]
    return results[['title', 'similarity', 'overview']]

# ---- UI starts here ----
st.title("🎬 CineMatch")
st.write("Describe the plot you're looking for, and get the top 5 closest matches.")

query = st.text_input("What kind of movie are you in the mood for?", placeholder="e.g. space exploration with a twist ending")

if query:
    results = recommend(query)
    for _, row in results.iterrows():
        st.subheader(f"{row['title']}  —  {row['similarity']:.2f}")
        st.write(row['overview'])
        st.divider()