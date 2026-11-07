"""
download_data.py
----------------
Downloads the MovieLens "ml-latest-small" dataset into the data/ folder.
Run it with:  python download_data.py

(You can also download it by hand from https://grouplens.org/datasets/movielens/latest/
 and copy movies.csv, ratings.csv, tags.csv and links.csv into data/.)
"""

import io
import zipfile

import requests

from src.config import DATA_DIR, DATASET_URL, REQUIRED_FILES


def main():
    if all((DATA_DIR / name).exists() for name in REQUIRED_FILES):
        print(f"Dataset already in {DATA_DIR} - nothing to do.")
        return

    print(f"Downloading {DATASET_URL} (about 1 MB)...")
    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The zip has a folder inside: ml-latest-small/movies.csv etc.
    # We copy just the files we need straight into data/.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        for name in REQUIRED_FILES + ["README.txt"]:
            with archive.open(f"ml-latest-small/{name}") as src:
                (DATA_DIR / name).write_bytes(src.read())
            print(f"  saved data/{name}")

    print("Done! Next: python check_setup.py")


if __name__ == "__main__":
    main()
