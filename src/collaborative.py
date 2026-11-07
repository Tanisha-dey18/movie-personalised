"""
collaborative.py
----------------
TECHNIQUE 2: Collaborative Filtering
"People who loved the movies you like ALSO loved these."
It never looks at genres or tags - only ratings.

We use "item-item" collaborative filtering:
  1. Build a big table: one row per movie, one column per user,
     each cell = how that user rated that movie (most cells are empty).
  2. Two movies are "similar" if the SAME people rate them in the
     SAME way (both high, or both low). We measure that with cosine
     similarity between their rows.
  3. Score every movie by how similar it is to the movies you liked.
  4. Return the best ones, with a reason based on real fan numbers.

Why item-item and not user-user? A new user has only 3-5 movies, which
isn't enough to find "people like you" - but it IS enough to find
"movies like these". (Amazon used the same idea for "customers also bought".)

Strength: finds genuinely great movies the same audience loved.
Weakness: needs lots of ratings; very popular movies show up everywhere.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.config import LOVE_THRESHOLD, MIN_RATINGS, SHRINKAGE


class CollaborativeRecommender:
    def __init__(self, min_ratings_to_use=5, min_ratings=MIN_RATINGS,
                 shrinkage=SHRINKAGE, love_threshold=LOVE_THRESHOLD):
        # Movies with fewer ratings than this are left out of the table entirely
        # (too little data to say who they're similar to).
        self.min_ratings_to_use = min_ratings_to_use
        # Only RECOMMEND movies with at least this many ratings.
        self.min_ratings = min_ratings
        # Shrinkage: how much to distrust similarities based on only a few people.
        self.shrinkage = shrinkage
        # A rating of 4+ stars counts as "loved it" (used for the reasons).
        self.love_threshold = love_threshold

    def fit(self, movies, ratings):
        """Build the movie x user ratings table. Call this once at the start."""
        self.movies = movies.set_index("movieId")  # look movies up by id

        # Keep only movies with enough ratings.
        counts = ratings.groupby("movieId")["rating"].count()
        keep = counts[counts >= self.min_ratings_to_use].index
        r = ratings[ratings["movieId"].isin(keep)].copy()

        # Mean-centering: some people give everything 4-5 stars, others are harsh.
        # Subtract each user's average so a value means "liked MORE than usual (+)"
        # or "LESS than usual (-)". This makes users comparable.
        r["centered"] = r["rating"] - r.groupby("userId")["rating"].transform("mean")

        # Give every movie a row number and every user a column number.
        self.movie_ids = np.sort(r["movieId"].unique())
        self.row_of = {mid: i for i, mid in enumerate(self.movie_ids)}
        rows = r["movieId"].map(self.row_of).to_numpy()
        cols, _ = pd.factorize(r["userId"])
        shape = (len(self.movie_ids), cols.max() + 1)

        # A "sparse" matrix only stores the filled-in cells. Our table is 98% empty,
        # so this saves a LOT of memory.
        self.centered = csr_matrix((r["centered"], (rows, cols)), shape=shape)
        # 1 where a user rated the movie at all.
        self.watched = csr_matrix((np.ones(len(r)), (rows, cols)), shape=shape)
        # 1 where a user rated the movie 4+ stars.
        loved = (r["rating"] >= self.love_threshold).astype(float)
        self.loved = csr_matrix((loved, (rows, cols)), shape=shape)

        # Number of ratings per row, to filter out obscure movies later.
        self.num_ratings = np.asarray(self.watched.sum(axis=1)).ravel()
        return self

    def _rows(self, movie_ids):
        """Turn movieIds into row numbers (skipping any not in our table)."""
        return [self.row_of[mid] for mid in movie_ids if mid in self.row_of]

    def _similarities(self, liked_rows, other_rows=None):
        """Similarity of each liked movie to other movies (all movies by default).

        Returns a grid: one row per liked movie, one column per other movie.
        """
        others = self.centered if other_rows is None else self.centered[other_rows]
        watched_others = self.watched if other_rows is None else self.watched[other_rows]

        sims = cosine_similarity(self.centered[liked_rows], others)

        # How many people rated BOTH movies? (the @ sign is matrix multiplication:
        # it counts the users the two rows have in common).
        overlap = (self.watched[liked_rows] @ watched_others.T).toarray()

        # Shrinkage: if only 3 people rated both movies, a high similarity
        # is probably luck. overlap / (overlap + 10) is close to 0 for small
        # overlaps and close to 1 for big ones, so it shrinks the unreliable ones.
        return sims * overlap / (overlap + self.shrinkage)

    def score_all(self, liked_ids):
        """Score every movie in the table against the liked movies.

        Returns a pandas Series: movieId -> score (higher = better match).
        """
        liked_rows = self._rows(liked_ids)
        if not liked_rows:
            return pd.Series(dtype=float)

        # Final score = average similarity to all the liked movies.
        scores = self._similarities(liked_rows).mean(axis=0)

        # Don't recommend what the user gave us, or obscure movies.
        scores[liked_rows] = -np.inf
        scores[self.num_ratings < self.min_ratings] = -np.inf

        return pd.Series(scores, index=self.movie_ids)

    def recommend(self, liked_ids, n=10):
        """Return the top-n movies that fans of the liked movies also loved."""
        scores = self.score_all(liked_ids)
        if scores.empty:
            return pd.DataFrame(columns=["movieId", "title", "score", "reason"])

        top_ids = scores.nlargest(n).index
        return pd.DataFrame([{
            "movieId": mid,
            "title": self.movies.at[mid, "display_title"],
            "score": round(float(scores[mid]), 3),
            "reason": self.reason_for(mid, liked_ids),
        } for mid in top_ids])

    def reason_for(self, movie_id, liked_ids):
        """E.g. 'Fans of The Matrix loved it too: 61 of 80 rated it 4+ stars'."""
        row = self.row_of[movie_id]
        liked_rows = self._rows(liked_ids)
        if not liked_rows:
            return "Popular with viewers"

        # Which liked movie is this one most similar to?
        sims = self._similarities(liked_rows, [row]).ravel()
        best = liked_rows[int(np.argmax(sims))]
        best_title = self.movies.at[self.movie_ids[best], "clean_title"]

        # Users who loved the liked movie (a row of 1s and 0s).
        fans = self.loved[best]
        # .multiply = "in both": fans who watched / loved this recommendation.
        fans_watched = int(fans.multiply(self.watched[row]).sum())
        fans_loved = int(fans.multiply(self.loved[row]).sum())

        # Only say "loved it too" when at least half the fans really did - keep reasons honest.
        if fans_watched > 0 and fans_loved / fans_watched >= 0.5:
            return (f"Fans of {best_title} loved it too: "
                    f"{fans_loved} of {fans_watched} rated it 4+ stars")
        return f"Often watched and rated the same way as {best_title}"
