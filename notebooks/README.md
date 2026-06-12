# notebooks/

Sixteen notebooks executed in order. The first two prepare data/features; then one
notebook per model trains **and** evaluates it; four extension notebooks follow; notebook 14
aggregates the comparison (deep eval), notebook 15 is a practical user-by-user case study, and
notebook 16 is the read-only consolidated evaluation report (all results + every figure).

## Execution order

```text
01_eda → 02_features → 03_baselines → 04_content_based → 05_user_knn
→ 06_item_knn → 07_svd → 08_weighted_hybrid → 09_stacked_hybrid
→ 10_content_genome → 11_lightgcn → 12_dual_head_hybrid → 13_semantic_content
→ 14_advanced_eval     (aggregate deep evaluation + comparison)
→ 15_case_study        (practical user-centric case study)
→ 16_evaluation_report (consolidated read-only results report)
```

All notebooks add `..` (the repo root) to `sys.path` so the `hybrid_recsys`
package is importable without a prior `pip install`.

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

## 10–16 — Extensions, deep eval, case study & report (additive; the frozen models 03–09 are untouched)

| Notebook | What |
|---|---|
| `10_content_genome.ipynb` | A 2nd content model on the **tag genome** (`genre ⊕ SVD(genome)`); measures the lift vs the TF-IDF content model. |
| `11_lightgcn.ipynb` | **LightGCN** graph CF (PyTorch, BPR loss). Ranking-only (embedding scores aren't ratings). Trains on a user subsample. |
| `12_dual_head_hybrid.ipynb` | **Dual-head hybrid**: a Ridge rating head (RMSE/MAE) + a logistic rank head (P/R/F1) blended on validation over all base models incl. genome & LightGCN. |
| `13_semantic_content.ipynb` | A 3rd content model on **sentence-transformer embeddings** (`all-MiniLM-L6-v2`); meaning-aware similarity vs TF-IDF/genome. |
| `14_advanced_eval.ipynb` | **Aggregate deep evaluation** + comparison leaderboard (see below). |
| `15_case_study.ipynb` | **Practical user-centric case study** — CB vs CF vs Hybrid on real archetype users (see below). |
| `16_evaluation_report.ipynb` | **Consolidated read-only report** — all result tables (from `all_metrics.json`) + every valuable figure with commentary (see below). |

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

## 15_case_study.ipynb — Practical case study (CB vs CF vs Hybrid)

The **practical** companion to notebook 14: instead of aggregate metrics, it shows — for real
users — that the hybrid combines content coherence with collaborative accuracy. Uses the
strongest representative of each family (**CB = Content-Genome, CF = Item-kNN, Hybrid =
Dual-Head**), all derived from a single model-load set (the Dual-Head pulls in the CB/CF bases).

- **§1–2** picks four reproducible archetype users (mainstream-heavy / niche-specialist /
  eclectic / light-sparse) from their train profiles + shows their tastes.
- **§3** side-by-side top-10 from CB / CF / Hybrid over a shared candidate pool, each rec
  annotated with held-out hit ✅ / popularity-percentile / genre-overlap.
- **§4** rigorous hit-rate: per-archetype Precision/Recall/F1/NDCG@10 + AUC (sampled negatives,
  same protocol as nb14) and RMSE/MAE on held-out ratings.
- **§5** beyond-accuracy (novelty / diversity / coverage), **§6** the blend mechanism (CB vs CF
  vs Hybrid scores + the Dual-Head's learned coefficients), **§7** verdict + cross-check vs nb14.

> **Memory:** loads the Dual-Head's five base models (~14.5 GB RAM). Run with ~15 GB free.
> Figures are saved with the `15_cs_*` prefix.

---

## 16_evaluation_report.ipynb — Consolidated evaluation report

The single notebook that **gathers every result and figure** from notebooks 01–15 and explains
each one. **Read-only and lightweight** — it loads `all_metrics.json` for the tables and
**references the saved figures** (`artifacts/figures/*.png`); it loads **no model** and recomputes
nothing, so it runs in seconds with no special memory. Sections: §0 protocol, metric definitions and the master results table · §1–2 data & features ·
§3 rating accuracy (RMSE/MAE) · §4 ranking (P/R/F1/NDCG/AUC@K) · §5 the rating-vs-ranking
trade-off · §6 beyond-accuracy · §7 **how each hybrid fuses CB+CF** · §8 per-model diagnostic
gallery · §9 the nb15 case study · §10 verdict · §11 a figure-integrity check (verifies all 41
referenced figures exist). This is the
presentation-ready results writeup; run notebooks 03–15 first so the figures exist.

---

## generate.py

Developer utility that regenerates all `.ipynb` files from source using
`nbformat`. Run from the project root:

```bash
python notebooks/generate.py
```

Use this after editing notebook cell content in `generate.py` — do not edit
the `.ipynb` files directly as those changes will be overwritten.
