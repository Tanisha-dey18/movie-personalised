"""
recommend_cli.py
----------------
The recommender in the terminal: type 3-5 movies you liked,
get 10 recommendations with a reason for each.
Run it with:  python recommend_cli.py
"""

from src.config import MAX_LIKED, MIN_LIKED
from src.data_loader import search_movies
from src.model_store import load_or_train


def pick_movie(movies):
    """Ask for a title, show matches, let the user choose one. Returns a row or None."""
    query = input("\nMovie title (or Enter to finish): ").strip()
    if not query:
        return None

    matches = search_movies(movies, query, limit=5)
    if matches.empty:
        print("  No match - try a shorter or different spelling.")
        return pick_movie(movies)  # ask again

    # Show numbered choices: 1. The Matrix (1999)
    for i, title in enumerate(matches["display_title"], start=1):
        print(f"  {i}. {title}")
    choice = input("  Which one? [1]: ").strip() or "1"

    if not choice.isdigit() or not 1 <= int(choice) <= len(matches):
        print("  Please type one of the numbers shown.")
        return pick_movie(movies)
    return matches.iloc[int(choice) - 1]


def main():
    print("Loading the recommender...")
    movies, recommender = load_or_train(verbose=True)

    print(f"\nTell me {MIN_LIKED}-{MAX_LIKED} movies you liked.")
    liked = []  # list of (movieId, title)
    while len(liked) < MAX_LIKED:
        movie = pick_movie(movies)
        if movie is None:  # user pressed Enter
            if len(liked) >= MIN_LIKED:
                break
            print(f"  Please add at least {MIN_LIKED - len(liked)} more.")
            continue
        if movie["movieId"] in [mid for mid, _ in liked]:
            print("  You already added that one.")
            continue
        liked.append((movie["movieId"], movie["display_title"]))
        print(f"  Added: {movie['display_title']}  ({len(liked)}/{MAX_LIKED})")

    print("\nBecause you liked: " + ", ".join(title for _, title in liked))
    recs = recommender.recommend([mid for mid, _ in liked])

    print("\n=== Your 10 recommendations ===")
    for i, rec in enumerate(recs.itertuples(), start=1):
        print(f"\n{i:>2}. {rec.title}")
        print(f"    {rec.reason}")


if __name__ == "__main__":  # only run main() when this file is run directly
    main()
