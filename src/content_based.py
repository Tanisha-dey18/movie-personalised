"""
content_based.py
----------------
TECHNIQUE 1: Content-Based Filtering
"You liked these movies, so here are movies that LOOK similar"
(same genres, similar user tags). It never looks at ratings.

How it works, in 4 steps:
  1. Turn each movie's genres and tags into a list of numbers (a "vector")
     using TF-IDF.
  2. Average the vectors of the movies the user liked -> a "taste profile".
  3. Compare the taste profile to every movie with cosine similarity
     (1 = pointing the same way, 0 = nothing in common).
  4. Return the highest-scoring movies, with a reason for each.

Strength: works for any movie, even ones few people rated.
Weakness: can't tell a great sci-fi movie from a bad one - many ties.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config import GENRE_WEIGHT, MIN_RATINGS, TAG_WEIGHT


def split_pipes(text):
    """'Action|Sci-Fi' -> ['Action', 'Sci-Fi'].  '' -> []"""
    return text.split("|") if text else []


class ContentRecommender:
    def __init__(self, genre_weight=GENRE_WEIGHT, tag_weight=TAG_WEIGHT, min_ratings=MIN_RATINGS):
        # How much genres vs. tags count. Try changing these in config.py!
        self.genre_weight = genre_weight
        self.tag_weight = tag_weight
        # Ignore very obscure movies when recommending (fewer than this many ratings).
        self.min_ratings = min_ratings

    def fit(self, movies):
        """Build one vector per movie. Call this once at the start."""
        self.movies = movies.reset_index(drop=True)

        # Remember which row each movieId is on, so we can look movies up fast.
        self.row_of = {mid: i for i, mid in enumerate(self.movies["movieId"])}

        # TF-IDF = "Term Frequency - Inverse Document Frequency".
        # It gives each genre/tag a weight: rare ones (e.g. "time travel") get a
        # HIGH weight because they say a lot; common ones (e.g. "Drama") get a LOW one.
        # tokenizer=split_pipes means each "|"-separated item is one word,
        # so a multi-word tag like "dark comedy" stays together.
        genre_vec = TfidfVectorizer(tokenizer=split_pipes, token_pattern=None, lowercase=False)
        tag_vec = TfidfVectorizer(tokenizer=split_pipes, token_pattern=None, min_df=2)
        # min_df=2: drop tags used on only one movie - they can't link two movies anyway.

        genre_matrix = genre_vec.fit_transform(self.movies["genres"])
        tag_matrix = tag_vec.fit_transform(self.movies["tags"])

        # Stick the two side by side: [genre numbers | tag numbers],
        # then scale each row to length 1 so long tag lists don't dominate.
        combined = hstack([genre_matrix * self.genre_weight, tag_matrix * self.tag_weight])
        self.matrix = normalize(combined.tocsr())

        # Keep each movie's genres and tags as Python sets for building reasons.
        self.genre_sets = [set(split_pipes(g)) for g in self.movies["genres"]]
        self.tag_sets = [set(split_pipes(t)) for t in self.movies["tags"]]
        return self

    def _rows(self, movie_ids):
        """Turn movieIds into row numbers (skipping any we don't know)."""
        return [self.row_of[mid] for mid in movie_ids if mid in self.row_of]

    def score_all(self, liked_ids):
        """Score EVERY movie against the liked movies.

        Returns a pandas Series: movieId -> score (higher = more similar).
        """
        liked_rows = self._rows(liked_ids)
        if not liked_rows:
            return pd.Series(dtype=float)

        # Taste profile = average of the liked movies' vectors.
        profile = np.asarray(self.matrix[liked_rows].mean(axis=0))

        # Compare the profile with every movie. One score per movie.
        scores = cosine_similarity(profile, self.matrix).ravel()

        # Don't recommend movies the user already gave us,
        # or movies too obscure to trust.
        scores[liked_rows] = -1
        scores[self.movies["num_ratings"].to_numpy() < self.min_ratings] = -1

        return pd.Series(scores, index=self.movies["movieId"])

    def recommend(self, liked_ids, n=10):
        """Return the top-n movies similar to the liked movies (a DataFrame)."""
        scores = self.score_all(liked_ids)
        if scores.empty:
            return pd.DataFrame(columns=["movieId", "title", "score", "reason"])

        # Many movies have the exact same genres, so they tie. To break ties,
        # sort by score first and then by number of ratings (more popular wins).
        table = pd.DataFrame({
            "score": scores.round(3).to_numpy(),
            "popularity": self.movies["num_ratings"].to_numpy(),
        }, index=scores.index)
        top_ids = table.sort_values(["score", "popularity"], ascending=False).index[:n]

        return pd.DataFrame([{
            "movieId": mid,
            "title": self.movies.at[self.row_of[mid], "display_title"],
            "score": round(float(scores[mid]), 3),
            "reason": self.reason_for(mid, liked_ids),
        } for mid in top_ids])

    def reason_for(self, movie_id, liked_ids):
        """Short reason: which liked movie it's closest to, and what they share."""
        row = self.row_of[movie_id]
        liked_rows = self._rows(liked_ids)

        # Find the single liked movie this one is MOST similar to.
        sims = cosine_similarity(self.matrix[row], self.matrix[liked_rows]).ravel()
        best = liked_rows[int(np.argmax(sims))]
        best_title = self.movies.at[best, "clean_title"]

        shared_genres = sorted(self.genre_sets[row] & self.genre_sets[best])  # & = in both
        shared_tags = sorted(self.tag_sets[row] & self.tag_sets[best])[:3]

        parts = []
        if shared_genres:
            parts.append(", ".join(shared_genres))
        if shared_tags:
            parts.append("tags: " + ", ".join(shared_tags))
        detail = "; ".join(parts) if parts else "similar overall style"
        return f"Similar to {best_title} ({detail})"
