# notebooks/

Three notebooks that must be executed in order. Each saves its outputs so the
next notebook can load them without re-computation.

## Execution order

```text
01_eda.ipynb  →  02_features.ipynb  →  03_train_evaluate.ipynb
```

All notebooks add `../src` to `sys.path` so the `hybrid_recsys` package is
importable without a prior `pip install`.

---

## 01_eda.ipynb — Exploratory Data Analysis & Preprocessing

**Reads:** `data/raw/*.csv`  
**Writes:** `data/processed/movies.parquet`, `ratings.parquet`, `split_train/val/test.parquet`  
**Figures:** `artifacts/figures/01_*` through `05_*`

- Loads all six MovieLens 25M CSV files.
- Cleans ratings: drops duplicates, enforces half-star scale.
- Enriches the movie table: extracts release year from title, aggregates top user tags.
- Applies a user-wise temporal split (80/10/10) via `temporal_split()`, dropping
  users with fewer than 5 ratings.
- Produces distribution charts for ratings, users, movies, time, and genres.

---

## 02_features.ipynb — Item Feature Engineering

**Reads:** `data/processed/movies.parquet`  
**Writes:** `data/processed/item_features.npz`, `movie_index.parquet`  
**Figures:** `artifacts/figures/06_*`, `07_*`

- Builds a **multi-hot genre matrix** (20 binary columns).
- Builds a **TF-IDF matrix** over each movie's aggregated tag text,
  then reduces to 256 dimensions with Truncated SVD (LSA).
- Horizontally stacks both matrices into a single sparse item feature matrix.
- Visualises genre co-occurrence and explained variance of LSA components.

---

## 03_train_evaluate.ipynb — Model Training & Evaluation

**Reads:** `data/processed/*`, `data/processed/item_features.npz`  
**Writes:** `artifacts/models/*.joblib`, `artifacts/metrics/all_metrics.json`  
**Figures:** `artifacts/figures/08_*` through `10_*`

Trains and evaluates eight models under a leak-free protocol:

| Section | Model |
|---|---|
| §3 | Global Mean baseline |
| §3 | Popularity baseline |
| §4 | Content-Based (cosine similarity) |
| §5 | User-Based k-NN |
| §6 | Item-Based k-NN |
| §7 | SVD (with 5-fold GridSearchCV hyperparameter tuning) |
| §8 | Weighted Hybrid (α tuned on validation RMSE) |
| §9 | Stacked Hybrid (Ridge meta-learner on 5-fold OOF predictions) |

Results are summarised in a styled DataFrame and three Plotly charts.

> **Runtime note:** Section §7 (SVD grid search) and §9 (OOF stacking loop)
> are the most compute-intensive. On a modern laptop with MovieLens 25M, expect
> 20–60 minutes total. Set `OOF_SAMPLE_FRAC = 0.2` near the top of §9 to reduce
> the OOF training subset for a faster (approximate) run.

---

## generate.py

Developer utility that regenerates all three `.ipynb` files from source using
`nbformat`. Run from the project root:

```bash
python notebooks/generate.py
```

Use this after editing notebook cell content in `generate.py` — do not edit
the `.ipynb` files directly as those changes will be overwritten.
