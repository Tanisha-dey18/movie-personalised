"""
app.py
------
The web app. Run it with:  streamlit run app.py

How Streamlit works (the one thing to understand):
  Every time you click or type something, Streamlit re-runs THIS WHOLE
  FILE from top to bottom. That's why:
    - slow things (loading the model, calling TMDB) are wrapped in
      @st.cache_... so they only really run once;
    - things we want to remember between re-runs go in st.session_state.
"""

import pandas as pd
import streamlit as st

from src.config import CONTENT_WEIGHT, FIGURES_DIR, MAX_LIKED, MIN_LIKED, MIN_RATINGS, N_RECOMMENDATIONS
from src.model_store import load_or_train
from src.posters import get_api_key, get_movie_details, placeholder_poster

COLUMNS = 5  # posters per row

# How each recommendation source is shown: (badge text, badge colour, icon)
SOURCE_BADGES = {
    "both": ("Genres + fans", "green", ":material/verified:"),
    "ratings": ("Fans' ratings", "blue", ":material/group:"),
    "content": ("Genres & tags", "violet", ":material/category:"),
}

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# =====================================================================
# Cached helpers - these run once, then Streamlit reuses the result
# =====================================================================

@st.cache_resource(show_spinner="Loading the recommender (first time: training, ~5 seconds)...")
def get_model():
    """Load the saved model, or train it if there isn't one yet."""
    return load_or_train()


@st.cache_data(show_spinner=False)
def details_for(tmdb_id, api_key):
    """Poster + overview from TMDB, cached so each movie is fetched only once."""
    return get_movie_details(tmdb_id, api_key)


@st.cache_data(show_spinner=False)
def recommendations_for(liked_ids, content_weight):
    """Cached so moving the slider back and forth is instant.
    (liked_ids is a tuple because cache keys can't be lists.)"""
    return recommender.recommend(list(liked_ids), content_weight=content_weight)


# =====================================================================
# Setup
# =====================================================================

try:
    movies, recommender = get_model()
except FileNotFoundError as error:
    # Friendly message instead of a crash if the dataset isn't downloaded yet.
    st.error(f"**Dataset not found.**\n\n{error}")
    st.stop()

api_key = get_api_key()

# Quick lookups: movieId -> title / tmdbId
title_of = dict(zip(movies["movieId"], movies["display_title"]))
tmdb_of = {
    mid: (None if pd.isna(tid) else int(tid))  # None = no TMDB id for this movie
    for mid, tid in zip(movies["movieId"], movies["tmdbId"])
}


def show_poster(movie_id):
    """Draw the movie's poster (or a Pillow-made placeholder). Returns the TMDB details."""
    details = details_for(tmdb_of[movie_id], api_key)
    if details["poster_url"]:
        st.image(details["poster_url"], width="stretch")
    else:
        st.image(placeholder_poster(title_of[movie_id]), width="stretch")
    return details


# =====================================================================
# Sidebar: settings and info
# =====================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    content_weight = st.slider(
        "What should matter more?",
        min_value=0.0, max_value=1.0, value=CONTENT_WEIGHT, step=0.1,
        help="0 = only what fans rated highly (collaborative filtering). "
             "1 = only genres and tags (content-based filtering). "
             f"The default {CONTENT_WEIGHT} mixes both.",
    )
    left, right = st.columns(2)
    left.caption("← Fans' ratings")
    right.markdown("<div style='text-align:right'><small>Genres & tags →</small></div>",
                   unsafe_allow_html=True)

    st.divider()
    st.header("📊 Dataset")
    ratings_total = int(movies["num_ratings"].sum())
    st.metric("Movies", f"{len(movies):,}")
    st.metric("Ratings", f"{ratings_total:,}")
    st.metric("Recommendable", f"{(movies['num_ratings'] >= MIN_RATINGS).sum():,}",
              help=f"Movies with at least {MIN_RATINGS} ratings")

    st.divider()
    if api_key:
        st.success("TMDB connected: posters on", icon=":material/image:")
    else:
        st.warning("No TMDB key: showing placeholder posters. "
                   "Copy `.env.example` to `.env` and add your key, then restart.",
                   icon=":material/key_off:")


# =====================================================================
# Main page
# =====================================================================

st.title("🎬 Movie Recommender")
st.markdown(
    f"Pick **{MIN_LIKED}–{MAX_LIKED} movies you liked** and get "
    f"**{N_RECOMMENDATIONS} personalised recommendations**, each with a reason."
)

tab_recommend, tab_how, tab_data = st.tabs(["🍿 Recommend", "🧠 How it works", "📈 Data insights"])

# ---------------------------------------------------------------------
with tab_recommend:
    # Movie search. The multiselect box has search built in: just start typing.
    # The most-rated movies are listed first, so famous titles appear at the top.
    popular_first = movies.sort_values("num_ratings", ascending=False)["movieId"].tolist()
    liked_ids = st.multiselect(
        "Movies you liked",
        options=popular_first,
        format_func=lambda mid: title_of[mid],  # show titles, but give us movieIds
        max_selections=MAX_LIKED,
        placeholder="Start typing a title, e.g. The Matrix",
    )

    # Small posters of the chosen movies.
    if liked_ids:
        cols = st.columns(COLUMNS * 2)  # twice as many columns = smaller posters
        for col, mid in zip(cols, liked_ids):
            with col:
                show_poster(mid)
                st.caption(title_of[mid])

    needed = MIN_LIKED - len(liked_ids)
    if st.button(
        "Recommend movies" if needed <= 0 else f"Add {needed} more to get recommendations",
        type="primary", icon=":material/auto_awesome:", disabled=needed > 0,
    ):
        st.session_state["submitted"] = list(liked_ids)

    # Show results only if they belong to the CURRENT selection.
    # (Change your movies and the old results disappear until you click again.
    #  Moving the slider updates the results straight away.)
    if st.session_state.get("submitted") == liked_ids and liked_ids:
        with st.spinner("Finding movies for you..."):
            recs = recommendations_for(tuple(liked_ids), content_weight)

        st.subheader(f"Your {len(recs)} recommendations")
        if recs.empty:
            st.info("No recommendations found - try different movies.")

        for start in range(0, len(recs), COLUMNS):
            cols = st.columns(COLUMNS)
            row_recs = recs.iloc[start:start + COLUMNS].itertuples()
            for number, (col, rec) in enumerate(zip(cols, row_recs), start=start + 1):
                with col:
                    details = show_poster(rec.movieId)
                    st.markdown(f"**{number}. {rec.title}**")
                    label, color, icon = SOURCE_BADGES[rec.source]
                    st.badge(label, color=color, icon=icon)
                    st.caption(rec.reason)
                    with st.expander("Details"):
                        st.progress(min(rec.score, 1.0), text=f"Match score: {rec.score:.0%}")
                        st.caption(f"Genres & tags: {rec.content_score:.0%} · "
                                   f"Fans' ratings: {rec.collab_score:.0%}")
                        if details["vote"]:
                            st.caption(f"TMDB rating: ⭐ {details['vote']:.1f}/10")
                        if details["overview"]:
                            st.write(details["overview"])
                        if details["tmdb_url"]:
                            st.markdown(f"[View on TMDB ↗]({details['tmdb_url']})")

# ---------------------------------------------------------------------
with tab_how:
    st.markdown(f"""
### Two techniques, combined

**1. Content-based filtering: "movies that look similar"**
- Each movie's **genres** and **user tags** are turned into numbers with **TF-IDF**.
  Rare words like *"time travel"* get more weight than common ones like *"Drama"*.
- Your movies are averaged into a **taste profile**. Every movie is compared to it with
  **cosine similarity** (1 = identical, 0 = nothing in common).
- ✅ Works for any movie · ❌ Can't tell a great sci-fi film from a bad one.

**2. Collaborative filtering: "fans of your movies also loved..."**
- A huge table of **610 users × 9,700 movies** (98% empty) of ratings.
- Two movies are similar when **the same people rate them the same way**
  (item-item filtering). Ratings are **mean-centred** (harsh and generous raters
  become comparable) and **shrunk** when only a few people rated both movies.
- ✅ Finds great movies the same audience loved · ❌ Popular movies show up everywhere.

**3. Hybrid: the best of both**
- Both scores are rescaled to 0–1 and mixed: **{CONTENT_WEIGHT:.0%} content + {1 - CONTENT_WEIGHT:.0%} collaborative**
  (change it with the slider in the sidebar).
- Each recommendation's reason comes from whichever technique picked it.
  When both agree, you get a **Genres + fans** badge.

Only movies with at least **{MIN_RATINGS} ratings** are recommended, to avoid obscure flukes.
Run `python evaluate.py` to measure how well each technique works.
""")

# ---------------------------------------------------------------------
with tab_data:
    charts = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if not charts:
        st.info("No charts yet. Run `python eda.py` in the terminal, then refresh this page.")
    else:
        st.caption("Charts made by `eda.py` from the MovieLens data.")
        cols = st.columns(2)
        for i, chart in enumerate(charts):
            with cols[i % 2]:
                st.image(str(chart), width="stretch")

# TMDB asks every app using its API to show this notice.
st.divider()
st.caption(
    "Data: [MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research. "
    "Posters: [TMDB](https://www.themoviedb.org/). "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
