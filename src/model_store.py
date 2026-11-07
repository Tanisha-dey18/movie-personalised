"""
model_store.py
--------------
Saves the trained recommender to disk with joblib, and loads it back.

Why? Training takes a few seconds every time. Saving the result to
models/recommender.joblib means the app can start almost instantly.

joblib is like Python's pickle (it turns objects into a file and back),
but faster for objects that contain big NumPy arrays - like ours.
"""

import time

import joblib

from src import config
from src.config import DATA_DIR, MODEL_PATH, MODELS_DIR, REQUIRED_FILES
from src.data_loader import check_data_files, load_data
from src.recommender import HybridRecommender

# Bump this number if you change how the model is built, to force a retrain.
MODEL_VERSION = 1

# Settings from config.py that change the trained model.
MODEL_SETTINGS = ["MIN_RATINGS", "CONTENT_WEIGHT", "GENRE_WEIGHT", "TAG_WEIGHT",
                  "SHRINKAGE", "LOVE_THRESHOLD"]


def data_signature():
    """A 'fingerprint' of the dataset files (size + last-modified time) and the
    model settings. If either changes, the fingerprint changes and we retrain."""
    check_data_files()
    files = tuple(
        (name, (DATA_DIR / name).stat().st_size, int((DATA_DIR / name).stat().st_mtime))
        for name in REQUIRED_FILES
    )
    settings = tuple((name, getattr(config, name)) for name in MODEL_SETTINGS)
    return files + settings


def train_and_save(verbose=True):
    """Load data, train the hybrid recommender, save it. Returns (movies, recommender)."""
    start = time.time()
    if verbose:
        print("Loading data...")
    movies, ratings = load_data()

    if verbose:
        print("Training content-based + collaborative models...")
    recommender = HybridRecommender().fit(movies, ratings)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": MODEL_VERSION,
        "data_signature": data_signature(),
        "movies": movies,
        "recommender": recommender,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)  # compress=3: smaller file, still fast

    if verbose:
        size_mb = MODEL_PATH.stat().st_size / 1_000_000
        print(f"Saved model to {MODEL_PATH} ({size_mb:.1f} MB) in {time.time() - start:.1f}s")
    return movies, recommender


def load_or_train(verbose=False):
    """Load the saved model if it's up to date, otherwise train a new one."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if (bundle.get("version") == MODEL_VERSION
                    and bundle.get("data_signature") == data_signature()):
                return bundle["movies"], bundle["recommender"]
            if verbose:
                print("Saved model is out of date - retraining.")
        except Exception as error:  # e.g. file from an older library version
            if verbose:
                print(f"Couldn't load saved model ({error}) - retraining.")
    return train_and_save(verbose=verbose)
