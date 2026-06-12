# Notebooks — The Experimental Pipeline (in depth)

The fifteen notebooks **are** the experiment: they ingest the data, build features, then
train **and** evaluate one model per notebook, aggregate everything into the comparison and
deep-evaluation study, and finish with a practical user-by-user case study. This document
explains what each notebook does, what it reads and writes, and what to look at in its output. For the *model internals* (maths,
hyperparameters, trade-offs) see [`models.md`](models.md); for the honest results assessment
see [`project-walkthrough.md`](project-walkthrough.md).

> **Notebooks are generated, not hand-edited.** Every `.ipynb` is produced by
> [`notebooks/generate.py`](../notebooks/generate.py) (via `nbformat`). **Edit the cell source
> in `generate.py`, then re-run it** — editing an `.ipynb` directly is unsafe because the next
> `generate.py` run overwrites it *and wipes its cell outputs*. (The one exception was the
> `src/`→root migration, which patched the `sys.path` line in place precisely to preserve
> already-computed outputs.)

---

## Execution order

```text
01_eda → 02_features → 03_baselines → 04_content_based → 05_user_knn
→ 06_item_knn → 07_svd → 08_weighted_hybrid → 09_stacked_hybrid
→ 10_content_genome → 11_lightgcn → 12_dual_head_hybrid → 13_semantic_content
→ 14_advanced_eval   (aggregate deep evaluation + comparison)
→ 15_case_study      (practical user-centric case study)
```

Four phases:

| Phase | Notebooks | Produces |
|---|---|---|
| **Prepare** | 01–02 | processed parquets, the train/val/test split, item-feature matrices |
| **Train + evaluate (one per model)** | 03–09 (core) · 10–13 (extensions) | `artifacts/models/*.joblib`, incremental `all_metrics.json` |
| **Aggregate + deep eval** | 14 | leaderboard, NDCG/AUC, segmented/cold-start/diversity studies, figures |
| **Practical case study** | 15 | per-user CB-vs-CF-vs-Hybrid recommendations, hit-rates, `15_cs_*` figures |

Each notebook adds the repo root to `sys.path` (`sys.path.insert(0, "..")`) so it can
`import hybrid_recsys` without a prior `pip install -e .`. All randomness is pinned to
`RANDOM_STATE = 42` (in [`config.py`](../hybrid_recsys/config.py)).

**Contract that makes "one notebook per model" work.** Every model exposes the same
`predict(user, movie) → score` operation, and a shared helper
[`evaluation/report.full_metrics(predict_fn, …)`](../hybrid_recsys/evaluation/report.py)
turns that single function into *all* metrics (RMSE/MAE over the full test set; P/R/F1@K via
sampled negatives). Results are **checkpointed to `all_metrics.json` after every model**, so a
crash mid-run never loses completed results.

---

## Phase 1 — Data & features

### `01_eda.ipynb` — Exploratory analysis & the split

- **Reads:** `data/raw/*.csv` (all six MovieLens 25M files).
- **Writes:** `data/processed/movies.parquet`, `ratings.parquet`,
  `split_train.parquet`, `split_val.parquet`, `split_test.parquet`.
- **Figures:** `artifacts/figures/01_*`–`05_*`.

What it does and why it matters:

- Loads and **downcasts** the 25M-row ratings frame (`int32`/`float32`) to keep memory sane.
- **Cleans + enriches** the movie table ([`pipeline/data.build_movies_table`](../hybrid_recsys/pipeline/data.py)):
  regex-extracts the release `year` from each title into `clean_title`, and aggregates each
  movie's free-text user tags into a `tags_text` column (the raw material for the TF-IDF
  content features).
- Profiles the dataset — the four EDA findings that justify later design choices:
  - **Positivity bias** (modal rating 4.0) → **relevance threshold ≥ 4.0** is justified.
  - **Power-law long tail** (most users/movies have few ratings) → sparsity > 99.8% is the
    core CF challenge and the reason content helps.
  - **Genre dominance** (Drama/Comedy/Thriller).
  - **Rating volume grows over time** → a **temporal** split is the honest protocol (a random
    split would leak the future into training).
- Produces the **user-wise temporal 80/10/10 split**
  ([`pipeline/splits.temporal_split`](../hybrid_recsys/pipeline/splits.py)): each user's
  ratings sorted by timestamp, then sliced. Users with < 5 ratings dropped; a guard steals one
  row from train when needed so every retained user keeps ≥ 1 test row.
- **Genome note:** `genome-scores`/`genome-tags` are loaded here **only** for a coverage stat.
  They are *not* a feature of the core content model (that uses free-text tags). The genome
  becomes a feature only in notebook 10.

### `02_features.ipynb` — Item feature engineering

- **Reads:** `data/processed/movies.parquet`.
- **Writes:** `data/processed/item_features.npz`, `movie_index.parquet`; transformers
  `artifacts/models/tfidf.joblib`, `svd_text.joblib`.
- **Figures:** `artifacts/figures/06_*`, `07_*`.

Turns each movie into a **276-dim vector** ([`pipeline/features.build_item_features`](../hybrid_recsys/pipeline/features.py)):

1. **Genre block (~20 dims):** `genres.str.get_dummies("|")` → multi-hot, `float32`.
2. **Text block (256 dims):** `TfidfVectorizer(min_df=5, ngram_range=(1,2), sublinear_tf=True)`
   over `clean_title + tags_text`, then `TruncatedSVD(256)` (Latent Semantic Analysis).
3. `hstack` → a sparse `(n_movies × 276)` matrix, saved with the fitted transformers so serving
   never re-fits.

Visualises genre co-occurrence and the LSA explained-variance curve. (Known no-op: the text
matrix is fit twice — once for the variance plot, once inside `build_item_features` — identical
output because `random_state` is fixed; only the second is saved.)

---

## Phase 2 — One notebook per model (train + evaluate)

Each notebook below **trains the model, saves it to `artifacts/models/`, evaluates it once**
(appending to `all_metrics.json`), and shows example recommendations + a model-specific plot.
The eval protocol is identical across all of them (see [Evaluation protocol](#evaluation-protocol-shared)).

### `03_baselines.ipynb` — Global Mean & Popularity

- **Global Mean:** predicts `μ = mean(train.rating) ≈ 3.53` for everything → the RMSE floor.
- **Popularity:** scores by training interaction count, mapped monotonically onto [0.5, 5.0].
  A strong *ranking* baseline, a meaningless *rating* one.
- Both defined inline (no saved artifact). Also lists the top-10 most-rated movies.

### `04_content_based.ipynb` — Content-Based (TF-IDF)

- Loads `item_features.npz`, fits [`ContentBasedRecommender`](../hybrid_recsys/models/content.py)
  → `content_model.joblib`.
- Extra: a **"why this?"** demo — the content-nearest neighbours of a movie a sample user liked.
- This notebook also **defines the sampled-negatives ranking evaluator** the others reuse
  (`EVAL_USERS=1000`, `N_NEGATIVES=100`, K∈{5,10,20}, threshold 4.0).

### `05_user_knn.ipynb` — User-Based k-NN

- Surprise `KNNWithMeans` (Pearson-baseline, `k=80`, `min_k=5`, `user_based=True`).
  **Samples 20K users** before fitting (full 162K user-user matrix ≈ 98 GB) → `user_knn_model.joblib`.
- Extra: a **nearest-users graph** and the similarity distribution.

### `06_item_knn.ipynb` — Item-Based k-NN

- Same `KNNWithMeans` but `user_based=False`. **Caps to the 15K most-rated items**
  (full 62K item-item matrix ≈ 15 GB → ~900 MB) → `item_knn_model.joblib`.
- Extra: a **nearest-movies graph**. The item cap rarely bites (ratings concentrate on popular
  titles), which is why item-kNN's RMSE (0.83) far beats user-kNN's (1.04).

### `07_svd.ipynb` — SVD (matrix factorisation)

- Surprise `SVD` (`μ + b_u + b_i + q_iᵀp_u`), tuned by **5-fold `GridSearchCV`** over
  `n_factors ∈ {50,100,200}`, `n_epochs ∈ {20,40}`, `lr_all ∈ {0.002,0.005}`,
  `reg_all ∈ {0.02,0.05}`; best-RMSE estimator kept (`random_state` pinned per candidate for
  reproducibility) → `svd_model.joblib`.
- Extra: a **PCA of the learned item factors**. Most compute-intensive notebook.

### `08_weighted_hybrid.ipynb` — Weighted Hybrid

- Loads the saved SVD + content models, then `tune_alpha` sweeps `α ∈ [0,1]` step 0.05 to
  minimise **validation** RMSE (converges near α ≈ 0.9 → mostly SVD) → `weighted_hybrid.joblib`.
- Extra: the **α-sweep curve**.

### `09_stacked_hybrid.ipynb` — Stacked Hybrid

- **Leak-free OOF stacking:** 5-fold out-of-fold base predictions on train, then a
  `Ridge(α=1)` meta-learner over `[content, user_knn, item_knn, svd, item_pop, user_cnt,
  item_cnt]`; base models retrained on full train for serving → `stacked_hybrid.joblib`.
- Extra: the **Ridge coefficient bar** (SVD + Item-kNN dominate; User-kNN ≈ 0).
- **Runtime knob:** `OOF_SAMPLE_FRAC = 0.2` for a faster approximate run.

---

## Phase 2b — Extensions (additive; 03–09 stay frozen)

These add models *without* re-training anything above — each saves to its own artifact files.

### `10_content_genome.ipynb` — Content-Based (Tag Genome)

- Builds a **second** feature matrix from the **tag genome** (genre ⊕ `TruncatedSVD(256)` of
  the 1,128-dim relevance vectors), via
  [`build_item_features_genome`](../hybrid_recsys/pipeline/features.py) → `item_features_genome.npz`,
  `svd_genome.joblib`. Re-uses `ContentBasedRecommender` → `content_genome_model.joblib`.
- **The big content lift:** RMSE 1.046 → **0.967**, F1@10 0.207 → **0.376**, no other model touched.

### `11_lightgcn.ipynb` — LightGCN (graph CF)

- [`LightGCNRecommender`](../hybrid_recsys/models/lightgcn.py) — PyTorch, BPR ranking loss,
  full-batch propagation, trained on a **10K-user subsample** → `lightgcn_model.joblib`.
- **Ranking-only** (embedding dot-products aren't ratings → `rmse`/`mae` null). **Best F1@10
  (≈ 0.62)** with a candidate-coverage caveat (scores only in-graph items).

### `12_dual_head_hybrid.ipynb` — Dual-Head Hybrid

- [`DualHeadHybrid`](../hybrid_recsys/models/hybrid.py) — a **Ridge rating head** + a **logistic
  rank head** over 8 features `[content_genome, user_knn, item_knn, svd, lightgcn, item_pop,
  user_cnt, item_cnt]`, blended on validation (frozen bases, NaN imputed by per-feature median)
  → `dual_head_hybrid.joblib`.
- Targets **all five** metrics with task-specific heads. **Best RMSE in the project (0.8028)**;
  the app's default recommender.

### `13_semantic_content.ipynb` — Content-Based (Embeddings)

- A **third** content matrix from a **sentence-transformer** (`all-MiniLM-L6-v2`) over
  `title | genres | tags`, L2-normalised, via
  [`build_item_features_embedding`](../hybrid_recsys/pipeline/features.py) →
  `content_embed_model.joblib`. Meaning-aware similarity; lands between TF-IDF and genome.

---

## Phase 3 — Aggregate & deep evaluation

### `14_advanced_eval.ipynb` — Advanced Evaluation & Comparison *(final)*

Loads **all** trained models and runs the full battery — **no re-training**, deep sections run
on bounded samples (config constants atop each cell). The standalone comparison notebook was
folded in here:

- **A. Comparison leaderboard** from `all_metrics.json` — the full P/R/F1@K table, RMSE/MAE &
  F1@10 bars, the **rating-vs-ranking scatter** (shows RMSE-optimal ≠ ranking-optimal), F1@K curves.
- **B. NDCG@K & AUC** — robust ranking metrics ([`metrics.evaluate_ranking_extended`](../hybrid_recsys/evaluation/metrics.py)).
- **C. Segmented RMSE** by user-activity / item-popularity buckets (who each model serves well).
- **D. Beyond-accuracy** — catalogue coverage, intra-list diversity, novelty.
- **E. Bootstrap confidence intervals** on RMSE (are the RMSE gaps significant?).
- **F. Cold-start simulation** — content models given only 3 ratings.
- **G. Full-catalogue sanity pass** — ranking against the *entire* catalogue, not just sampled
  negatives, as a cross-check on the sampled protocol.

The figures it exports under `artifacts/figures/` are the same PNGs the app's **Comparison** tab
serves.

---

## Phase 4 — Practical case study

### `15_case_study.ipynb` — CB vs CF vs Hybrid, user by user *(final)*

The **practical** counterpart to notebook 14. Where 14 answers *"which model is best on
average?"*, this answers *"does the hybrid actually combine content coherence with
collaborative accuracy for real users?"* — the heart of the assignment, shown as a story
rather than a metrics table. Strictly scoped to **CB vs CF vs Hybrid**, using the strongest
representative of each family: **CB = Content-Genome, CF = Item-kNN, Hybrid = Dual-Head**.

- **Loads once, gets all three.** The Dual-Head's feature vector is built from five base models
  (`genome-CB, user-kNN, item-kNN, SVD, LightGCN`), so loading it brings the CB and CF baselines
  into memory too — the comparison is apples-to-apples. **~14.5 GB RAM; run with ~15 GB free.**
- **§1–2 — four archetype users**, picked reproducibly from their train profiles: *mainstream
  heavy* (popular taste), *niche specialist* (low genre entropy), *eclectic cinephile* (high
  entropy), *light/sparse*. Each is required to have held-out test likes as ground truth.
- **§3 — side-by-side top-10** from CB / CF / Hybrid over a shared candidate pool, each rec
  annotated: held-out **hit ✅**, **popularity percentile** (exposes CF's popularity bias),
  **genre overlap** (exposes CB's over-specialisation).
- **§4 — "would they like it?"**: per-archetype **Precision/Recall/F1/NDCG@10 + AUC** (sampled
  negatives, same protocol as nb14) on stratified samples, plus per-archetype **RMSE/MAE** on
  held-out ratings, plus the four named users individually.
- **§5 — beyond-accuracy** (novelty / intra-list diversity / catalogue coverage per family).
- **§6 — the blend mechanism**: for the eclectic user, candidate movies where CB and CF disagree
  (off-taste blockbuster vs niche gem), showing every base signal and the Hybrid's output, plus
  the Dual-Head's learned **rating-head coefficients** (which signals it leans on).
- **§7 — verdict + cross-check** against `all_metrics.json` (practical ≈ aggregate ordering).

Read-only: it loads frozen artifacts and does **not** modify `all_metrics.json`. Figures use the
`15_cs_*` prefix. (See also the app's **Side-by-side** tab for an interactive version of §3.)

---

## Evaluation protocol (shared)

The same protocol is applied by every train+eval notebook (03–13) and re-checked in 14:

| Setting | Value |
|---|---|
| Split | user-wise **temporal** 80/10/10 (no future leakage) |
| Hyperparameter tuning | on **val / train-CV only**; test touched **once** |
| Rating metrics | **RMSE, MAE** over the **full** test set (NaN predictions masked) |
| Ranking metrics | **Precision/Recall/F1@K**, K ∈ {5, 10, 20}, relevance ≥ 4.0 |
| Ranking sample | **1,000 stratified test users** (macro-averaged) |
| Ranking protocol | **sampled negatives** — each user's relevant test items vs **100 random non-rated negatives**, per-user-seeded so every model sees the identical pool |
| Tie handling | candidates **shuffled** before the stable sort (so constant-output models can't ride the sort order) |
| F1 definition | `F1 = harmonic(macro-P, macro-R)` (always lies between P and R) |
| Primary selection metric | **F1@10** |

Why sampled negatives: the CF models have a **restricted vocabulary** (Item-kNN caps to 15K
items, User-kNN to 20K users — see [`models.md`](models.md)), so ranking against the full 62K
catalogue would unfairly punish them for items they structurally can't score. The
NCF/BPR-style sampled-negatives protocol stays meaningful for every model. Notebook 14's
full-catalogue pass (§G) confirms the sampled protocol doesn't distort the ordering.

---

## Reproducing from scratch

```bash
pip install -e .                      # installs hybrid_recsys + deps
# place the MovieLens 25M CSVs in data/raw/
jupyter notebook                      # run 01 → 02 → 03…09 → 10…13 → 14, in order
```

Outputs land in `data/processed/` (parquets), `artifacts/models/` (`.joblib`),
`artifacts/metrics/all_metrics.json`, and `artifacts/figures/` — exactly what the
[backend](backend.md) loads to serve the [app](app.md).
