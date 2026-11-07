"""
data_loader.py
--------------
Loads the MovieLens CSV files and cleans them so the rest of the
project can use them easily.

The main function is load_data(). It returns two tables:
  - movies  : one row per movie (title, year, genres, tags, rating stats, tmdbId)
  - ratings : one row per rating (userId, movieId, rating)
"""

import pandas as pd

from src.config import DATA_DIR, REQUIRED_FILES


def check_data_files(data_dir=DATA_DIR):
    """Stop with a clear message if the dataset hasn't been downloaded yet."""
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir}: {', '.join(missing)}\n"
            "Download them by running:  python download_data.py"
        )


def load_raw(data_dir=DATA_DIR):
    """Read the four CSV files exactly as they are on disk."""
    check_data_files(data_dir)
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    links = pd.read_csv(data_dir / "links.csv")
    return movies, ratings, tags, links


def clean_movies(movies):
    """Split the year out of the title and tidy up titles and genres."""
    movies = movies.copy()  # never change the original table by accident

    # "Toy Story (1995)" -> year = 1995
    # The pattern looks for 4 digits inside brackets at the END of the title.
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    movies["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")

    # "Toy Story (1995)" -> "Toy Story"
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )

    # MovieLens writes "Matrix, The" - flip it to "The Matrix" so search feels natural.
    movies["clean_title"] = movies["clean_title"].str.replace(
        r"^(.*), (The|A|An)$", r"\2 \1", regex=True
    )

    # "The Matrix (1999)" - a nice title for showing to users.
    movies["display_title"] = movies["clean_title"] + movies["year"].map(
        lambda y: f" ({int(y)})" if pd.notna(y) else ""
    )

    # A few movies have "(no genres listed)" - treat that as no genres.
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    # "IMAX" is listed as a genre, but it's a screen format, not a kind of story.
    # Left in, it makes every IMAX blockbuster look "similar". Remove it.
    movies["genres"] = (
        movies["genres"].str.split("|")
        .apply(lambda gs: "|".join(g for g in gs if g and g != "IMAX"))
    )

    return movies


def build_tag_text(tags):
    """Combine all user tags for each movie into one string.

    Example result for a movie: "dark comedy|nonlinear|quentin tarantino"
    """
    tags = tags.dropna(subset=["tag"]).copy()

    # Lowercase + strip spaces so "Dark Comedy" and "dark comedy " count as the same tag.
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

    # Group by movie, remove duplicates with set(), and join with "|".
    tag_text = (
        tags.groupby("movieId")["tag"]
        .apply(lambda t: "|".join(sorted(set(t))))
        .reset_index()
        .rename(columns={"tag": "tags"})
    )
    return tag_text


def rating_stats(ratings):
    """How many ratings each movie has, and its average rating."""
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(num_ratings="count", avg_rating="mean")
        .reset_index()
    )
    stats["avg_rating"] = stats["avg_rating"].round(2)
    return stats


def load_data(data_dir=DATA_DIR):
    """Load everything and return (movies, ratings), ready to use."""
    movies, ratings, tags, links = load_raw(data_dir)

    movies = clean_movies(movies)

    # Attach extra info to each movie. how="left" keeps every movie,
    # even ones with no tags or no ratings.
    movies = movies.merge(build_tag_text(tags), on="movieId", how="left")
    movies = movies.merge(rating_stats(ratings), on="movieId", how="left")
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    # Fill the gaps the merges left behind.
    movies["tags"] = movies["tags"].fillna("")
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)
    movies["tmdbId"] = movies["tmdbId"].astype("Int64")  # whole numbers, blanks allowed

    # We don't need the timestamp column for this project.
    ratings = ratings.drop(columns=["timestamp"])

    return movies, ratings


def search_movies(movies, query, limit=10):
    """Find movies whose title contains the query (ignoring upper/lower case).

    The most-rated matches come first, so "matrix" finds The Matrix
    before its sequels.
    """
    mask = movies["clean_title"].str.contains(query, case=False, regex=False)
    return movies[mask].sort_values("num_ratings", ascending=False).head(limit)
