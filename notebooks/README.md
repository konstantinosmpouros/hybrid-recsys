# notebooks/

Fourteen notebooks executed in order. The first two prepare data/features; then one
notebook per model trains **and** evaluates it; three extension notebooks follow; the
last aggregates the comparison.

## Execution order

```text
01_eda → 02_features → 03_baselines → 04_content_based → 05_user_knn
→ 06_item_knn → 07_svd → 08_weighted_hybrid → 09_stacked_hybrid
→ 10_content_genome → 11_lightgcn → 12_dual_head_hybrid → 13_semantic_content
→ 14_advanced_eval   (final: deep evaluation + comparison)
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

## 03–09 — One notebook per model (train **and** evaluate)

Each model gets its own notebook that **trains + saves** the model, then **evaluates** it
(RMSE/MAE on the full test set + Precision/Recall/F1@K via sampled-negatives, written to
`all_metrics.json`), plus example recommendations and model-specific plots.

| Notebook | Model | Notable extras |
|---|---|---|
| `03_baselines.ipynb` | Global Mean, Popularity | top-10 popular movies |
| `04_content_based.ipynb` | Content-Based | "why this?" content neighbours of a liked movie |
| `05_user_knn.ipynb` | User-Based k-NN | nearest-users **graph**, similarity distribution |
| `06_item_knn.ipynb` | Item-Based k-NN | nearest-movies **graph** |
| `07_svd.ipynb` | SVD | 5-fold CV; PCA of learned item factors |
| `08_weighted_hybrid.ipynb` | Weighted Hybrid | α-sweep curve (loads SVD + CB) |
| `09_stacked_hybrid.ipynb` | Stacked Hybrid | OOF stacking; Ridge coefficient bar |

Shared eval boilerplate lives in `hybrid_recsys.evaluation.report` (`full_metrics`,
`save_metric`, `top_n`). Run in order — 08 loads the models saved by 04 & 07; 09 loads 04–07.

> **Runtime note:** 07 (SVD grid search) and 09 (OOF stacking loop) are the most
> compute-intensive — set `OOF_SAMPLE_FRAC = 0.2` in 09 for a faster approximate run.

---

## 10–14 — Extensions (additive; the frozen models 03–09 are untouched)

| Notebook | What |
|---|---|
| `10_content_genome.ipynb` | A 2nd content model on the **tag genome** (`genre ⊕ SVD(genome)`); measures the lift vs the TF-IDF content model. |
| `11_lightgcn.ipynb` | **LightGCN** graph CF (PyTorch, BPR loss). Ranking-only (embedding scores aren't ratings). Trains on a user subsample. |
| `12_dual_head_hybrid.ipynb` | **Dual-head hybrid**: a Ridge rating head (RMSE/MAE) + a logistic rank head (P/R/F1) blended on validation over all base models incl. genome & LightGCN. |
| `13_semantic_content.ipynb` | A 3rd content model on **sentence-transformer embeddings** (`all-MiniLM-L6-v2`); meaning-aware similarity vs TF-IDF/genome. |
| `14_advanced_eval.ipynb` | **FINAL** notebook — folds in the comparison **and** the deep eval (see below). |

---

## 14_advanced_eval.ipynb — Advanced Evaluation & Comparison (final)

Loads **all** trained models and runs the full battery (re-scoring frozen models, no
re-training):

- **A. Comparison leaderboard** from `all_metrics.json` — full P/R/F1@K table, RMSE/MAE &
  F1@10 charts, the **rating-vs-ranking scatter**, and F1@K curves.
- **B. NDCG@K & AUC** — robust ranking metrics.
- **C. Segmented RMSE** by user-activity / item-popularity buckets.
- **D. Beyond-accuracy** — coverage, diversity, novelty.
- **E. Bootstrap CIs** on RMSE.
- **F. Cold-start** simulation (content models with 3 ratings).
- **G. Full-catalogue** ranking sanity pass.

Deep sections run on bounded samples (config constants at the top of each cell).

---

## generate.py

Developer utility that regenerates all `.ipynb` files from source using
`nbformat`. Run from the project root:

```bash
python notebooks/generate.py
```

Use this after editing notebook cell content in `generate.py` — do not edit
the `.ipynb` files directly as those changes will be overwritten.
