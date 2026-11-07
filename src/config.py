"""
config.py
---------
All settings and file paths in ONE place, so you never have to hunt
through the code to change something.
"""

from pathlib import Path

# ---------- Paths ----------
# ROOT = the movie-recommender folder. Worked out from this file's location,
# so the code works no matter which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"                    # MovieLens CSV files go here
MODELS_DIR = ROOT / "models"                # the trained model is saved here
FIGURES_DIR = ROOT / "reports" / "figures"  # charts from eda.py
ENV_FILE = ROOT / ".env"                    # your TMDB API key

MODEL_PATH = MODELS_DIR / "recommender.joblib"

# ---------- Dataset ----------
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
REQUIRED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]

# ---------- Recommender settings (try changing these!) ----------
MIN_LIKED = 3        # the user must pick at least this many movies
MAX_LIKED = 5        # ...and at most this many
N_RECOMMENDATIONS = 10

MIN_RATINGS = 20     # never recommend movies with fewer ratings than this (too obscure)
CONTENT_WEIGHT = 0.4 # hybrid mix: 0.4 content + 0.6 collaborative
GENRE_WEIGHT = 1.0   # content-based: how much genres count...
TAG_WEIGHT = 1.5     # ...versus user tags
SHRINKAGE = 10       # collaborative: distrust similarities based on few people
LOVE_THRESHOLD = 4.0 # a rating of 4+ stars counts as "loved it"

# ---------- TMDB (posters) ----------
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w342{path}"  # w342 = 342 px wide
