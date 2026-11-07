"""
recommender.py
--------------
The HYBRID recommender: combines content-based and collaborative
filtering into one list of 10 movies.

Why combine?
  - Content-based finds movies of the same KIND, but can't tell a great
    sci-fi movie from a bad one (lots of tied scores).
  - Collaborative finds movies the same AUDIENCE loved, but sometimes
    drifts off-topic (popular movies show up everywhere).
  Together: "the same kind of movie, AND loved by the same people".

How it works:
  1. Get a score for every movie from each recommender.
  2. Rescale both to 0..1 (they use different scales: content tops out
     around 0.6, collaborative around 0.3), so neither one dominates.
  3. Final score = weighted average of the two.
  4. Build the reason from whichever side(s) the movie scored well on.
"""

import pandas as pd

from src.collaborative import CollaborativeRecommender
from src.config import CONTENT_WEIGHT, N_RECOMMENDATIONS
from src.content_based import ContentRecommender

RESULT_COLUMNS = ["movieId", "title", "score", "content_score", "collab_score", "source", "reason"]


def rescale(scores):
    """Rescale so the best movie gets 1.0 and anything <= 0 becomes 0."""
    scores = scores.clip(lower=0)  # negative / -inf (= "excluded") -> 0
    top = scores.max()
    return scores / top if top > 0 else scores


class HybridRecommender:
    def __init__(self, content_weight=CONTENT_WEIGHT):
        # How much the content score counts (0..1). The rest goes to collaborative.
        # 0.4 = a bit more trust in real ratings than in genres/tags.
        self.content_weight = content_weight

    def fit(self, movies, ratings):
        """Prepare both recommenders. Takes a few seconds - do it once."""
        self.movies = movies.set_index("movieId")
        self.content = ContentRecommender().fit(movies)
        self.collab = CollaborativeRecommender().fit(movies, ratings)
        return self

    def rank(self, liked_ids, content_weight=None):
        """Score every movie and return a table sorted best-first (no reasons, so it's fast).

        content_weight: optional override (0 = only ratings, 1 = only genres/tags).
        """
        w = self.content_weight if content_weight is None else content_weight

        # Step 1 + 2: scores from both sides, each rescaled to 0..1.
        table = pd.DataFrame({
            "content": rescale(self.content.score_all(liked_ids)),
            "collab": rescale(self.collab.score_all(liked_ids)),
        }).fillna(0)  # a movie missing from one side scores 0 there
        if table.empty:
            return table

        # Step 3: weighted average.
        table["score"] = w * table["content"] + (1 - w) * table["collab"]

        # Ties (common when w is high) are broken by popularity.
        table["popularity"] = self.movies["num_ratings"].reindex(table.index).fillna(0)
        table = table[table["score"] > 0]  # drop excluded / unrelated movies
        return table.sort_values(["score", "popularity"], ascending=False)

    def recommend(self, liked_ids, n=N_RECOMMENDATIONS, content_weight=None):
        """Return the top-n movies with a score and a short reason each (a DataFrame)."""
        w = self.content_weight if content_weight is None else content_weight
        top = self.rank(liked_ids, w).head(n)

        # Step 4: reasons.
        results = []
        for movie_id, row in top.iterrows():
            source, reason = self.reason_for(movie_id, liked_ids, row, w)
            results.append({
                "movieId": movie_id,
                "title": self.movies.at[movie_id, "display_title"],
                "score": round(row["score"], 3),
                "content_score": round(row["content"], 3),
                "collab_score": round(row["collab"], 3),
                "source": source,
                "reason": reason,
            })
        return pd.DataFrame(results, columns=RESULT_COLUMNS)

    def reason_for(self, movie_id, liked_ids, row, w, strong=0.6):
        """Pick the reason from the side(s) that liked this movie.

        'strong' = how high a rescaled score must be to count as a real match.
        Returns (source, reason) where source is 'both', 'content' or 'ratings'.
        """
        # A side with zero weight can't be the reason a movie was picked.
        content_ok = row["content"] >= strong and w > 0
        collab_ok = row["collab"] >= strong and w < 1

        if content_ok and collab_ok:
            # Both agree: use the fan numbers + the shared genres.
            fan_reason = self.collab.reason_for(movie_id, liked_ids)
            genre_reason = self.content.reason_for(movie_id, liked_ids)
            return "both", f"{fan_reason}. {genre_reason}"

        # Otherwise: whichever side added more to the final score explains it.
        if row["collab"] * (1 - w) > row["content"] * w:
            return "ratings", self.collab.reason_for(movie_id, liked_ids)
        return "content", self.content.reason_for(movie_id, liked_ids)
