"""
eda.py
------
Exploratory Data Analysis (EDA): look at the data BEFORE building models.
Prints key numbers and saves 6 charts to reports/figures/.
Run it with:  python eda.py

What each chart teaches us (and how it shaped the recommender):
  1. Rating distribution   -> people mostly give 3-4 stars, so 4+ = "loved it"
  2. Ratings per movie     -> a "long tail": most movies have very few ratings,
                              so we ignore movies with < 20 ratings when recommending
  3. Ratings per user      -> every user rated 20+ movies, some rated thousands
  4. Top genres            -> Drama/Comedy are everywhere, so TF-IDF gives them low weight
  5. Most-rated movies     -> the "popular" movies collaborative filtering loves
  6. Popularity vs rating  -> popular movies tend to be rated higher
"""

import matplotlib

matplotlib.use("Agg")  # draw charts to files, without opening windows

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, MIN_RATINGS
from src.data_loader import load_data

sns.set_theme(style="whitegrid", palette="viridis")
COLOR = "#3b6ea8"


def save(fig, name):
    """Save a chart as a PNG and close it (frees memory)."""
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path.relative_to(FIGURES_DIR.parent.parent)}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    movies, ratings = load_data()

    # ---------- Key numbers ----------
    n_users, n_movies = ratings["userId"].nunique(), len(movies)
    print("=== Dataset summary ===")
    print(f"Movies:            {n_movies:,}")
    print(f"Users:             {n_users:,}")
    print(f"Ratings:           {len(ratings):,}")
    print(f"Movies with tags:  {(movies['tags'] != '').sum():,}")
    print(f"Average rating:    {ratings['rating'].mean():.2f}")
    # Sparsity: what share of all (user, movie) pairs has a rating?
    print(f"Table filled in:   {len(ratings) / (n_users * n_movies):.1%}  (the rest is empty)")
    print(f"Movies with {MIN_RATINGS}+ ratings (can be recommended): "
          f"{(movies['num_ratings'] >= MIN_RATINGS).sum():,}")
    print("\nSaving charts...")

    # ---------- 1. Rating distribution ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(x=ratings["rating"], color=COLOR, ax=ax)
    ax.set(title="How often each rating is given", xlabel="Stars", ylabel="Number of ratings")
    save(fig, "1_rating_distribution.png")

    # ---------- 2. Ratings per movie (long tail) ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(movies["num_ratings"].clip(upper=100), bins=50, color=COLOR, ax=ax)
    ax.axvline(MIN_RATINGS, color="crimson", linestyle="--",
               label=f"min. {MIN_RATINGS} ratings to be recommended")
    ax.set(title="Ratings per movie (the 'long tail')",
           xlabel="Number of ratings (100+ grouped together)", ylabel="Number of movies")
    ax.legend()
    save(fig, "2_ratings_per_movie.png")

    # ---------- 3. Ratings per user ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    per_user = ratings.groupby("userId").size()
    sns.histplot(per_user, bins=50, log_scale=True, color=COLOR, ax=ax)
    ax.set(title="How many movies each user rated",
           xlabel="Movies rated (log scale)", ylabel="Number of users")
    save(fig, "3_ratings_per_user.png")

    # ---------- 4. Top genres ----------
    genres = movies["genres"].str.split("|").explode()
    top_genres = genres[genres != ""].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(x=top_genres.values, y=top_genres.index, color=COLOR, ax=ax)
    ax.set(title="Most common genres", xlabel="Number of movies", ylabel="")
    save(fig, "4_top_genres.png")

    # ---------- 5. Most-rated movies ----------
    top_movies = movies.nlargest(15, "num_ratings")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_movies["num_ratings"], y=top_movies["display_title"], color=COLOR, ax=ax)
    ax.set(title="15 most-rated movies", xlabel="Number of ratings", ylabel="")
    save(fig, "5_most_rated_movies.png")

    # ---------- 6. Popularity vs average rating ----------
    rated = movies[movies["num_ratings"] >= 5]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=rated, x="num_ratings", y="avg_rating", alpha=0.35,
                    s=18, color=COLOR, edgecolor=None, ax=ax)
    ax.set_xscale("log")
    ax.set(title="Do popular movies get better ratings?",
           xlabel="Number of ratings (log scale)", ylabel="Average rating")
    save(fig, "6_popularity_vs_rating.png")

    print("\nDone! Open the reports/figures/ folder to see the charts.")


if __name__ == "__main__":
    main()
