"""Compare different settings and see which recommends best."""
from cleandata import load_movies
from recommender import MovieRecommender, NoMatchError

# (query, movie that should appear in the results). Edit or add your own!
# The last few are written in different words than the overview on purpose.
TESTS = [
    ("marines sent to a moon to protect an alien civilization", "Avatar"),
    ("toys living in a boy's room until a birthday brings a new toy", "Toy Story"),
    ("worried father fish and a forgetful friend travel to bring a clownfish home", "Finding Nemo"),
    ("dinosaurs escaping from a theme park after the security fails", "Jurassic Park"),
    ("an old woman tells the story of a sinking ocean liner", "Titanic"),
    ("hackers fighting the computers that rule the earth", "The Matrix"),
    ("an ogre rescues a princess guarded by a dragon", "Shrek"),
    ("a shark terrorizing the swimmers of an island town", "Jaws"),
    ("explorers travelling through a wormhole to conquer space", "Interstellar"),
    ("a thief infiltrating the subconscious to implant an idea", "Inception"),
    ("a poor artist and a rich girl fall in love on a doomed ship", "Titanic"),
    ("a cowboy doll is jealous of a new space action figure", "Toy Story"),
    ("scientists bring back extinct animals and they get loose", "Jurassic Park"),
]

# (label, MovieRecommender settings)
CONFIGS = [
    ("overview | none",      dict(mode="none", text_column="overview")),
    ("overview | stem",      dict(mode="stem", text_column="overview")),
    ("overview | lemma",     dict(mode="lemma", text_column="overview")),
    ("full text | lemma",    dict(mode="lemma", text_column="full_text")),
    ("+ sublinear tf",       dict(mode="lemma", text_column="full_text", sublinear_tf=True)),
    ("+ bigrams",            dict(mode="lemma", text_column="full_text", sublinear_tf=True, ngram_range=(1, 2))),
]


def rank_of(recommender, query, expected):
    """Position (1-10) of the expected movie in the results, or None."""
    try:
        titles = list(recommender.recommend(query, top_n=10)["title"])
    except NoMatchError:
        return None
    return titles.index(expected) + 1 if expected in titles else None


def hand_test_score(recommender, top_n=5):
    """(hits, total) for the hand-written test queries whose movie exists in the data."""
    titles = set(recommender.df["title"])
    tests = [t for t in TESTS if t[1] in titles]
    hits = 0
    for query, title in tests:
        rank = rank_of(recommender, query, title)
        if rank and rank <= top_n:
            hits += 1
    return hits, len(tests)


def build_models(df):
    """Build every model to compare: a list of (label, model)."""
    models = [(label, MovieRecommender(df, **params)) for label, params in CONFIGS]
    try:
        from spacy_recommender import SpacyRecommender, HybridRecommender
        spacy_model = SpacyRecommender(df)
    except (ImportError, RuntimeError) as error:
        print(f"\nspaCy models skipped: {error}")
        return models
    best_tfidf = dict(models)["+ sublinear tf"]
    models.append(("spaCy vectors", spacy_model))
    models.append(("hybrid tfidf+spacy", HybridRecommender(best_tfidf, spacy_model)))
    return models


def main():
    df = load_movies()
    tests = [t for t in TESTS if (df["title"] == t[1]).any()]
    for query, title in TESTS:
        if (query, title) not in tests:
            print(f"Skipped (not in dataset): {title}")
    if not tests:
        print("None of the test movies are in the dataset. Edit TESTS.")
        return

    print("\nBuilding models (the first spaCy run takes a few minutes)...")
    models = build_models(df)
    print(f"\nRunning {len(tests)} test queries on {len(models)} settings\n")

    all_ranks = []
    print(f"{'#':<3}{'Setting':<22}{'Vocab':>8}{'Hit@5':>8}{'Hit@10':>8}{'MRR':>8}")
    for number, (label, model) in enumerate(models, start=1):
        ranks = [rank_of(model, q, t) for q, t in tests]
        all_ranks.append(ranks)
        hit5 = sum(1 for r in ranks if r and r <= 5)
        hit10 = sum(1 for r in ranks if r)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        vocab = len(model.vectorizer.vocabulary_) if hasattr(model, "vectorizer") else "-"
        print(f"{number:<3}{label:<22}{vocab:>8}{hit5:>8}{hit10:>8}{mrr:>8.2f}")

    print("\nRank of the expected movie (- = not in top 10), columns = setting numbers")
    print(f"{'Query':<20}" + "".join(f"{n:>5}" for n in range(1, len(models) + 1)))
    for i, (_, title) in enumerate(tests):
        row = "".join(f"{(ranks[i] or '-'):>5}" for ranks in all_ranks)
        print(f"{(str(i + 1) + '. ' + title)[:19]:<20}{row}")


if __name__ == "__main__":
    main()