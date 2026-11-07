"""
evaluate.py
-----------
How GOOD are the recommendations? This script measures it with numbers.
Run it with:  python evaluate.py   (takes about a minute)

The test (called "hold-out evaluation"):
  1. Take a real user who loved (rated 4+) at least 10 movies.
  2. Give the recommender 5 of those loved movies, like the app does.
  3. HIDE the rest of their loved movies.
  4. Get 10 recommendations. How many of the hidden loved movies did it find?

To be fair, the user being tested is removed from the training data first
(otherwise collaborative filtering would "cheat" by seeing their ratings).
We do this with 5-fold cross-validation: split users into 5 groups, and for
each group, train on the other 4 groups and test on this one.

Metrics:
  - Hit rate @10    : % of users where at least 1 of the 10 picks was a hidden loved movie
  - Precision @10   : on average, how many of the 10 picks were hidden loved movies (%)
  - Coverage        : % of all recommendable movies that the method EVER suggests.
                      Low coverage = it shows everyone the same few blockbusters.

We also test a simple "most popular movies" baseline. A recommender that
can't beat "just show the most popular movies" isn't worth much!
"""

import numpy as np
import pandas as pd

from src.config import LOVE_THRESHOLD, MIN_RATINGS
from src.data_loader import load_data
from src.recommender import HybridRecommender

N_INPUT = 5     # movies we give the recommender (like a user in the app)
TOP_N = 10      # recommendations we check
MIN_LOVED = 10  # only test users who loved at least this many movies
N_FOLDS = 5
SEED = 42       # fixed random seed = same result every run

# name -> content_weight (None = popularity baseline)
METHODS = {
    "Popularity baseline": None,
    "Content-based only": 1.0,
    "Collaborative only": 0.0,
    "Hybrid (0.4 / 0.6)": 0.4,
}


def main():
    rng = np.random.default_rng(SEED)
    movies, ratings = load_data()

    # Users with enough loved movies, shuffled and split into 5 groups.
    loved = ratings[ratings["rating"] >= LOVE_THRESHOLD]
    loved_by_user = loved.groupby("userId")["movieId"].apply(list)
    eligible = loved_by_user[loved_by_user.apply(len) >= MIN_LOVED].index.to_numpy()
    test_users = rng.permutation(eligible)  # shuffled copy
    folds = np.array_split(test_users, N_FOLDS)
    print(f"Testing on {len(test_users)} users, {N_FOLDS} folds...\n")

    results = {name: {"hits": [], "precision": [], "picked": set()} for name in METHODS}
    n_recommendable = (movies["num_ratings"] >= MIN_RATINGS).sum()

    for fold_number, fold_users in enumerate(folds, start=1):
        # Train WITHOUT this fold's users.
        train_ratings = ratings[~ratings["userId"].isin(fold_users)]
        model = HybridRecommender().fit(movies, train_ratings)

        # Popularity baseline: most-rated movies in the training data.
        popular = train_ratings["movieId"].value_counts()
        popular = popular[popular >= MIN_RATINGS].index.tolist()

        for user in fold_users:
            user_loved = rng.permutation(loved_by_user[user])
            inputs, hidden = list(user_loved[:N_INPUT]), set(user_loved[N_INPUT:])

            for name, weight in METHODS.items():
                if weight is None:
                    picks = [m for m in popular if m not in inputs][:TOP_N]
                else:
                    picks = model.rank(inputs, content_weight=weight).index[:TOP_N]
                found = len(hidden.intersection(picks))
                results[name]["hits"].append(found > 0)
                results[name]["precision"].append(found / TOP_N)
                results[name]["picked"].update(picks)

        print(f"  fold {fold_number}/{N_FOLDS} done")

    table = pd.DataFrame({
        name: {
            "Hit rate @10": f"{np.mean(r['hits']):.1%}",
            "Precision @10": f"{np.mean(r['precision']):.1%}",
            "Coverage": f"{len(r['picked']) / n_recommendable:.1%}",
        }
        for name, r in results.items()
    }).T
    print("\n" + table.to_string())
    print("\nHigher is better for all three. How to read this:")
    print("- Users mostly rate popular movies, so the 'hidden' movies are usually popular too.")
    print("  That makes the popularity baseline look strong on accuracy...")
    print("- ...but its coverage shows it recommends the same few movies to everyone.")
    print("- The hybrid gives up a little accuracy for the widest, most personal choice.")
    print("- A 'miss' may just be a movie the user never saw, so real quality is higher.")


if __name__ == "__main__":
    main()
