import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load the already-cleaned data
df = pd.read_csv("data/cleaned_movies.csv")
df['overview'] = df['overview'].fillna('')  # safety net, in case of re-saved empty strings read back as NaN

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

print(tfidf_matrix.shape)



def recommend(query, top_n=5):
    query_vec = tfidf.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    results = df.iloc[top_indices].copy()
    results['similarity'] = similarity_scores[top_indices]
    return results[['title', 'similarity', 'overview']]
def print_recommendations(query, top_n=5):
    results = recommend(query, top_n)
    print(f"\nTop {top_n} matches for: \"{query}\"\n")
    for _, row in results.iterrows():
        print(f"{row['title']}  (score: {row['similarity']:.2f})")
        print(f"  {row['overview'][:150]}...")
        print()

print_recommendations("space exploration with a twist ending")
