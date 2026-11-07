"""
posters.py
----------
Gets a movie's poster and short description from TMDB (The Movie Database).

How it works:
  1. links.csv gave every movie a tmdbId (e.g. The Matrix = 603).
  2. We ask TMDB's API: "tell me about movie 603".
  3. TMDB answers with JSON that includes "poster_path": "/abc123.jpg".
  4. The full image address is  https://image.tmdb.org/t/p/w342/abc123.jpg

If anything goes wrong (no key, no internet, movie missing) we return
no poster, and the app shows a placeholder image instead of crashing.
"""

import os
import textwrap
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from src.config import ENV_FILE, TMDB_API_URL, TMDB_IMAGE_URL

PLACEHOLDER_KEY = "paste_your_key_here"  # the value in .env.example


def get_api_key():
    """Read TMDB_API_KEY from the .env file (or from your system environment)."""
    load_dotenv(ENV_FILE)  # copies the lines of .env into os.environ
    key = (os.environ.get("TMDB_API_KEY") or "").strip()
    if not key or key == PLACEHOLDER_KEY:
        return None
    return key


def check_api_key(api_key):
    """Try the key once and return (ok, message)."""
    if not api_key:
        return False, f"No key found. Copy .env.example to .env and add your key ({ENV_FILE})"
    try:
        # 603 = The Matrix, a movie we know exists.
        resp = requests.get(TMDB_API_URL.format(tmdb_id=603),
                            params={"api_key": api_key}, timeout=5)
    except requests.RequestException as error:
        return False, f"Couldn't reach TMDB (internet problem?): {error}"
    if resp.status_code == 200:
        return True, "Key works!"
    if resp.status_code == 401:
        return False, ("TMDB rejected the key. Check you copied the short 'API Key', "
                       "not the long 'Read Access Token'.")
    return False, f"TMDB answered with error {resp.status_code}"


@lru_cache(maxsize=2000)
def _fetch(tmdb_id, api_key):
    """Ask TMDB about one movie. @lru_cache remembers answers, so asking about
    the same movie twice doesn't call the internet twice (faster, and polite to TMDB)."""
    resp = requests.get(TMDB_API_URL.format(tmdb_id=tmdb_id),
                        params={"api_key": api_key}, timeout=5)
    resp.raise_for_status()  # turn errors like 404 into a Python exception
    return resp.json()


def get_movie_details(tmdb_id, api_key):
    """Return {'poster_url', 'overview', 'tmdb_url', 'vote'} for one movie.

    Values can be None/empty if TMDB has nothing - the caller should handle that.
    """
    details = {"poster_url": None, "overview": "", "tmdb_url": None, "vote": None}
    if tmdb_id is None or pd.isna(tmdb_id):  # 8 movies have no TMDB id
        return details

    tmdb_id = int(tmdb_id)
    details["tmdb_url"] = f"https://www.themoviedb.org/movie/{tmdb_id}"
    if not api_key:
        return details
    try:
        data = _fetch(tmdb_id, api_key)
    except requests.RequestException:
        return details  # no internet, bad key, or movie not found: just no poster

    if data.get("poster_path"):
        details["poster_url"] = TMDB_IMAGE_URL.format(path=data["poster_path"])
    details["overview"] = data.get("overview") or ""
    details["vote"] = data.get("vote_average")
    return details


@lru_cache(maxsize=500)
def placeholder_poster(title, width=342, height=513):
    """Draw a simple poster-shaped image with the title on it (using Pillow).

    Used when a movie has no poster, or when there's no API key.
    """
    image = Image.new("RGB", (width, height), color=(38, 42, 58))
    draw = ImageDraw.Draw(image)

    # A thin border and a film-reel style strip at the top and bottom.
    draw.rectangle([8, 8, width - 9, height - 9], outline=(90, 96, 120), width=2)
    for x in range(20, width - 20, 28):
        draw.rectangle([x, 18, x + 14, 30], fill=(70, 76, 98))
        draw.rectangle([x, height - 31, x + 14, height - 19], fill=(70, 76, 98))

    # Wrap the title onto several lines and centre it.
    try:
        font = ImageFont.truetype("arial.ttf", 28)  # available on Windows
    except OSError:
        font = ImageFont.load_default()
    lines = textwrap.wrap(title, width=16)[:6]
    line_height = 36
    y = (height - line_height * len(lines)) // 2
    for line in lines:
        text_width = draw.textlength(line, font=font)
        draw.text(((width - text_width) / 2, y), line, fill=(225, 228, 240), font=font)
        y += line_height
    return image
