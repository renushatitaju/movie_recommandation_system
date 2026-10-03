import pandas as pd
df = pd.read_csv("data/tmdb_5000_movies.csv")
df['overview'] = df['overview'].fillna('')

