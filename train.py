"""
train.py
--------
Trains the recommender and saves it to models/recommender.joblib.
Run it with:  python train.py

You don't HAVE to run this - the app trains automatically the first time.
Run it again after changing settings in src/config.py.
"""

from src.data_loader import search_movies
from src.model_store import train_and_save

movies, recommender = train_and_save(verbose=True)

# Quick sanity check: recommend for three well-known movies.
examples = ["The Matrix", "Inception", "Interstellar"]
liked_ids = [search_movies(movies, name, limit=1).iloc[0]["movieId"] for name in examples]

print(f"\nSample recommendations for: {', '.join(examples)}")
for i, rec in enumerate(recommender.recommend(liked_ids).itertuples(), start=1):
    print(f"{i:>2}. {rec.title}")
    print(f"    {rec.reason}")
