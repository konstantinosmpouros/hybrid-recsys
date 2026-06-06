# notebooks/

Four notebooks that must be executed in order. Each saves its outputs so the
next notebook can load them without re-computation.

## Execution order

```text
01_eda.ipynb  →  02_features.ipynb  →  03_train.ipynb  →  04_evaluation.ipynb
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

## 03_train.ipynb — Model Training

**Reads:** `data/processed/*`, `data/processed/item_features.npz`  
**Writes:** `artifacts/models/*.joblib`

Fits and **persists** the six trainable models — no evaluation here:

| Section | Model |
|---|---|
| §2 | Content-Based (cosine similarity) |
| §3 | User-Based k-NN |
| §4 | Item-Based k-NN |
| §5 | SVD (with 5-fold GridSearchCV hyperparameter tuning) |
| §6 | Weighted Hybrid (α tuned on validation RMSE) |
| §7 | Stacked Hybrid (Ridge meta-learner on 5-fold OOF predictions) |

> **Runtime note:** §5 (SVD grid search) and §7 (OOF stacking loop) are the
> most compute-intensive — expect 20–60 minutes total on MovieLens 25M. Set
> `OOF_SAMPLE_FRAC = 0.2` near the top of §7 for a faster (approximate) run.

---

## 04_evaluation.ipynb — Model Evaluation

**Reads:** `artifacts/models/*.joblib`, `data/processed/*`  
**Writes:** `artifacts/metrics/all_metrics.json`  
**Figures:** `artifacts/figures/08_*` through `10_*`

Loads the trained models and scores all **eight** (the two naive baselines are
recomputed inline) under a leak-free protocol: RMSE/MAE on the full test set, and
Precision/Recall/F1@K via the sampled-negatives protocol. Results are summarised
in a styled DataFrame and three Plotly charts.

---

## generate.py

Developer utility that regenerates all four `.ipynb` files from source using
`nbformat`. Run from the project root:

```bash
python notebooks/generate.py
```

Use this after editing notebook cell content in `generate.py` — do not edit
the `.ipynb` files directly as those changes will be overwritten.
