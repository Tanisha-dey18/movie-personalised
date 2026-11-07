"""
check_setup.py
--------------
Checks that everything is ready, and tells you exactly what to fix if not.
Run it with:  python check_setup.py
"""

import importlib
import sys

from src.config import DATA_DIR, ENV_FILE, REQUIRED_FILES

OK, FAIL, WARN = "[OK]  ", "[FIX] ", "[WARN]"
problems = 0

# 1. Python version
version = ".".join(map(str, sys.version_info[:3]))
if sys.version_info >= (3, 10):
    print(f"{OK} Python {version}")
else:
    print(f"{FAIL} Python {version} is too old - install Python 3.11")
    problems += 1

# 2. Libraries
# (import name, pip name) - they differ for a few libraries
libraries = [("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
             ("scipy", "scipy"), ("streamlit", "streamlit"), ("requests", "requests"),
             ("dotenv", "python-dotenv"), ("PIL", "pillow"), ("matplotlib", "matplotlib"),
             ("seaborn", "seaborn"), ("joblib", "joblib")]
missing = []
for import_name, pip_name in libraries:
    try:
        importlib.import_module(import_name)
    except ImportError:
        missing.append(pip_name)
if missing:
    print(f"{FAIL} Missing libraries: {', '.join(missing)}")
    print("       Run:  pip install -r requirements.txt")
    problems += 1
else:
    print(f"{OK} All {len(libraries)} libraries installed")

# 3. Dataset
missing_files = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
if missing_files:
    print(f"{FAIL} Dataset missing in data/: {', '.join(missing_files)}")
    print("       Run:  python download_data.py")
    problems += 1
else:
    print(f"{OK} Dataset found in data/")

# 4. TMDB key (optional - the app works without it, just no posters)
if not missing:
    from src.posters import check_api_key, get_api_key

    if not ENV_FILE.exists():
        print(f"{WARN} No .env file - posters will be hidden.")
        print("       Copy .env.example to .env and paste your TMDB key into it.")
    else:
        ok, message = check_api_key(get_api_key())
        print(f"{OK if ok else WARN} TMDB key: {message}")

print()
if problems:
    print(f"Fix the {problems} item(s) marked [FIX], then run this again.")
else:
    print("All set! Start the app with:  streamlit run app.py")
