# 🎬 Movie Recommendation System

A beginner-friendly movie recommender built with Python and machine learning.
Pick **3–5 movies you like** and get **10 personalised recommendations**, each with a short reason.
It runs as a **Streamlit web app** and shows movie posters from **TMDB**.

It uses two recommendation techniques and combines them:

| Technique | Idea | Built with |
|---|---|---|
| **Content-based filtering** | "Movies that *look* similar": same genres and user tags | TF-IDF + cosine similarity |
| **Collaborative filtering** | "Fans of your movies *also loved* these": based on 100,836 real ratings | Item-item similarity on a sparse user × movie matrix |
| **Hybrid** (what the app uses) | 40% content + 60% collaborative (adjustable in the app) | Score rescaling + weighted average |

Data: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) (9,742 movies, 610 users, 100,836 ratings).

---

## 🚀 Quick start (5 commands)

You need **Python 3.11** (3.10 or newer works). Open a terminal **inside this folder**.

**1. Create and turn on a virtual environment** (keeps this project's libraries separate)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
You should now see `(venv)` at the start of the line.
If PowerShell says *"running scripts is disabled"*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
On macOS/Linux use `source venv/bin/activate` instead.

**2. Install the libraries**
```powershell
pip install -r requirements.txt
```

**3. Download the dataset** (about 1 MB, saved into `data/`)
```powershell
python download_data.py
```

**4. Add your TMDB API key** (for posters; the app also works without it, showing placeholder posters)
1. Get a free key at <https://www.themoviedb.org/settings/api> (sign up → Settings → API → request a key → choose "Developer").
2. Copy `.env.example` and name the copy `.env`:
   ```powershell
   copy .env.example .env
   ```
3. Open `.env` and replace `paste_your_key_here` with your key. Use the **short "API Key"**, not the long "API Read Access Token" that starts with `eyJ...`.

**5. Check everything and run the app**
```powershell
python check_setup.py
streamlit run app.py
```
Your browser opens at <http://localhost:8501>. The first start takes about 5 seconds while the model trains and saves itself. After that it starts almost instantly. Press **Ctrl+C** in the terminal to stop the app.

---

## 📁 Project structure

```
movie-recommender/
├── app.py                 ← the Streamlit web app (start here!)
├── requirements.txt       ← libraries to install
├── .env.example           ← template for your TMDB key (copy to .env)
├── .streamlit/config.toml ← app colours and settings
│
├── src/                   ← the recommender's logic (one job per file)
│   ├── config.py          ← ALL settings and paths in one place
│   ├── data_loader.py     ← load + clean the CSV files, search movies
│   ├── content_based.py   ← technique 1: genres & tags → TF-IDF → cosine similarity
│   ├── collaborative.py   ← technique 2: ratings → item-item similarity
│   ├── recommender.py     ← hybrid: combines both + writes the reasons
│   ├── model_store.py     ← saves/loads the trained model with joblib
│   └── posters.py         ← TMDB posters (requests + dotenv) and placeholders (Pillow)
│
├── download_data.py       ← downloads MovieLens into data/
├── check_setup.py         ← checks Python, libraries, data and API key
├── train.py               ← trains + saves the model, prints sample results
├── eda.py                 ← data exploration: prints stats, saves 6 charts
├── evaluate.py            ← measures recommendation quality with numbers
├── recommend_cli.py       ← the recommender in the terminal (no browser)
│
├── data/                  ← MovieLens CSV files (after download_data.py)
├── models/                ← recommender.joblib (created automatically)
└── reports/figures/       ← charts from eda.py
```

---

## 🧰 Other scripts

| Command | What it does |
|---|---|
| `python eda.py` | Prints dataset statistics and saves 6 charts (matplotlib + seaborn) to `reports/figures/`. They also appear in the app's **Data insights** tab. |
| `python train.py` | Retrains the model and saves it to `models/recommender.joblib`. Run this after changing `src/config.py`. |
| `python evaluate.py` | Tests how good the recommendations are (takes about 1 minute). |
| `python recommend_cli.py` | Type movies in the terminal and get recommendations, without the web app. |

---

## 🧠 How it works

### 1. Data cleaning (`data_loader.py`)
- The year is split out of titles: `"Matrix, The (1999)"` → `"The Matrix"` + `1999`.
- All user tags for each movie are combined into one string.
- **"IMAX" is removed from genres.** It's a screen format, not a genre, and left in it made every IMAX blockbuster look "similar".
- Each movie gets its rating count and average rating.

### 2. Content-based filtering (`content_based.py`)
1. Each movie's genres and tags are turned into a vector with **TF-IDF**. Rare tags (*"time travel"*) get high weight and common genres (*"Drama"*) get low weight.
2. The vectors of your 3–5 movies are averaged into a **taste profile**.
3. Every movie is compared to the profile with **cosine similarity** (1 = identical, 0 = nothing in common).

### 3. Collaborative filtering (`collaborative.py`)
1. A **sparse matrix** of 9,700 movies × 610 users. Only 1.7% of it is filled in, so a sparse matrix saves a lot of memory.
2. **Mean-centring:** each user's average rating is subtracted, so harsh and generous raters become comparable.
3. Two movies are similar if **the same people rate them the same way** (cosine similarity between movie rows). This is **item-item** filtering, which works better than user-user for new users with only a few movies.
4. **Shrinkage:** similarity × `overlap / (overlap + 10)`. If only 3 people rated both movies, the similarity is probably luck, so it's reduced.

### 4. Hybrid (`recommender.py`)
- Both scores are rescaled to 0–1 so neither one dominates, then mixed **40/60**.
- Each reason comes from the technique that picked the movie:
  - *"Fans of The Matrix loved it too: 122 of 144 rated it 4+ stars"* (collaborative)
  - *"Similar to Interstellar (Sci-Fi; tags: sci-fi)"* (content-based)
- Only movies with **20+ ratings** are recommended, to avoid obscure flukes.

---

## 📊 Evaluation results (`evaluate.py`)

**Method:** for 579 users who loved 10+ movies, the recommender is given 5 of their loved movies, and the rest are hidden. Does the top 10 contain the hidden ones?
To keep the test fair, **5-fold cross-validation** removes the tested users from the training data.

| Method | Hit rate @10 | Precision @10 | Coverage |
|---|---|---|---|
| Popularity baseline | 83.6% | 32.0% | **1.2%** |
| Content-based only | 41.1% | 7.4% | 50.1% |
| Collaborative only | **86.9%** | 31.8% | 55.1% |
| **Hybrid (app default)** | 84.6% | 28.8% | **58.7%** |

**How to read this:**
- *Hit rate:* % of users who got at least one hidden loved movie in their 10 picks.
- *Precision:* average share of the 10 picks that were hidden loved movies.
- *Coverage:* % of recommendable movies the method ever suggests.

**What it shows:**
- The **popularity baseline is surprisingly accurate** because users mostly rate popular movies. But it shows **everyone the same ~15 blockbusters** (1.2% coverage), so it isn't personal at all.
- **Collaborative filtering is the most accurate.** Content-based alone is weak on accuracy, but it keeps recommendations on-topic.
- The **hybrid gives up a little accuracy for the widest coverage** and on-topic results. For example, it stops recommending *The Dark Knight* to Pixar fans.
- These numbers *underestimate* real quality, because a "miss" might simply be a good movie the user never watched.

---

## ⚙️ Settings to experiment with (`src/config.py`)

| Setting | Default | Try... |
|---|---|---|
| `CONTENT_WEIGHT` | `0.4` | `0.0` (only ratings) or `1.0` (only genres). This is also the slider in the app. |
| `TAG_WEIGHT` | `1.5` | `0.5` or `3.0` to see how much tags matter |
| `MIN_RATINGS` | `20` | `5` to allow more obscure movies |
| `SHRINKAGE` | `10` | `0` (flukier picks) or `50` (safer, more popular picks) |

After changing settings, restart the app. It notices the change and retrains automatically. Run `python evaluate.py` to measure the effect.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Missing dataset files...` | Run `python download_data.py` |
| `ModuleNotFoundError: No module named ...` | Turn on the venv (`.\venv\Scripts\Activate.ps1`) and run `pip install -r requirements.txt` |
| Placeholder posters instead of real ones | Check that `.env` exists (not just `.env.example`) and contains your key, then run `python check_setup.py` |
| `TMDB rejected the key` | Use the short **API Key**, not the long **Read Access Token** |
| Strange characters like `D�j� Vu` in the terminal | Only the Windows terminal display is affected. The web app shows them correctly. |
| App shows old results after editing the code in `src/` | Stop it (Ctrl+C), run `python train.py`, and start it again |

---

## 🙏 Credits

- **Data:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5, 4. <https://grouplens.org/datasets/movielens/>
- **Posters:** [TMDB](https://www.themoviedb.org/). *This product uses the TMDB API but is not endorsed or certified by TMDB.*
