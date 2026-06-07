# Models — How Each Recommender Works (in depth)

Complete reference for every model: philosophy, data, training mechanics (with the
actual maths), prediction formula, hyperparameters, complexity/memory, strengths &
weaknesses, and back-stage caveats. Verified against [`src/hybrid_recsys/`](../src/hybrid_recsys/).

---

## 0. Notation & the unifying contract

| Symbol | Meaning |
|---|---|
| `u`, `i`/`j` | a user, an item (movie) |
| `r_{u,i}` | observed rating of user *u* on item *i* (0.5–5.0) |
| `r̂(u,i)` | the model's **predicted** rating |
| `r̄_u`, `r̄_i` | mean rating of user *u* / item *i* |
| `μ` | global mean rating over train (≈ 3.53) |
| `sim(a,b)` | similarity between two users or two items |
| `N^k(...)` | the *k* nearest neighbours |

Every model — heuristic or learned — exposes exactly one operation:

```
predict(user, movie) -> a single score, clipped to [0.5, 5.0]
```

From that one scalar we derive **both** metric families:

- **Rating accuracy (RMSE / MAE):** call `predict` on each true test pair `(u, i, r)`, compare `r̂` to `r`.
- **Ranking (Precision / Recall / F1@K):** call `predict` on many candidate movies for a user, **sort descending**, take the top-K. A candidate counts as *relevant* if the user actually rated it ≥ 4.0.

So **ranking is never produced directly — it is just "sort the candidates by `predict`."** This is why the shared `full_metrics(predict_fn, ...)` helper (in `evaluation/report.py`, called by each per-model notebook) takes only a `predict_fn` and computes everything.

### Heuristic vs. learned

| Camp | Models | "fit" means |
|---|---|---|
| **Memory-based / heuristic** (no optimised parameters) | Global Mean, Popularity, Content-Based, User-kNN, Item-kNN | precompute and store something: a mean, counts, normalised feature vectors, or a similarity matrix |
| **Learned** (parameters fit by minimising a loss) | SVD, Weighted Hybrid, Stacked Hybrid | optimise an objective — latent factors via SGD, the blend weight α, or Ridge coefficients |

---

## Shared substrate

- **Interactions** — `split_train/val/test.parquet`: a user-wise **temporal** 80/10/10 split of the 25M ratings (each user's history sorted by time, then sliced → no future leakage).
- **Item features** (content model only) — a **276-dim vector per movie** = 20-dim multi-hot genres ⊕ 256-dim TruncatedSVD of TF-IDF over `clean_title + tags_text`. Built in notebook 02, stored as `item_features.npz` + `movie_index.parquet`.
- **`user_ratings_map`** — `{userId: {movieId: rating}}` from **train**; the history the content model and hybrids read at prediction time.

### Two similarity functions used throughout

- **Cosine** (content model): for L2-normalised vectors `a, b`, `sim = aᵀb`. Range [−1, 1]; only the *direction* of the feature vector matters, not its magnitude.
- **Pearson-baseline** (Surprise kNN): a Pearson correlation computed on **baseline-centred** ratings (`r_{u,i} − (μ + b_u + b_i)`) with **shrinkage** toward 0 for pairs with few co-ratings. The baseline removes "this user rates high / this movie is loved" effects so the correlation reflects genuine taste agreement; shrinkage stops two users who share only one movie from looking perfectly similar.

---

## 1. Global Mean — naive baseline

- **Philosophy.** The dumbest non-trivial guess: ignore *who* and *what*, predict one number.
- **Data.** Training ratings only.
- **Training.** `μ = mean(train.rating)` (≈ 3.53). Stores a single float — O(1) memory.
- **Prediction.** `r̂(u,i) = μ` for every pair.
- **Why it matters.** It's the **RMSE/MAE floor**: any real model must beat predicting the average. RMSE here (≈ 1.06) is essentially the standard deviation of ratings.
- **Caveats.** Useless as a *ranking* model — every score is identical, so the top-K is arbitrary. Defined inline in notebook 03 (baselines).

## 2. Popularity — naive baseline

- **Philosophy.** "Recommend what everyone watches." A blunt but surprisingly strong **ranking** baseline (popular items really are more often relevant).
- **Data.** Training interaction **counts** per movie.
- **Training.** `count(i)` per `movieId`; store the dict + `max_count`. O(#items) memory.
- **Prediction.** `r̂(u,i) = 0.5 + 4.5 · count(i) / max_count` — a **monotonic** map of popularity onto the rating scale.
- **Why the mapping.** It lets us compute RMSE/MAE at all; being monotonic, it leaves the *ranking* unchanged. RMSE is meaningless (most movies are rare → predicts ≈ 0.5 vs true ≈ 4), but ranking is informative.
- **Caveats.** No personalisation — every user gets the same order.

## 3. Content-Based — CB, memory-based

[`models/content.py`](../src/hybrid_recsys/models/content.py)

- **Philosophy.** "You liked these; here are movies that *look similar by their attributes*." Recommends from **item content**, independent of other users → it can score brand-new (cold) items that CF cannot.
- **Data.** The 276-dim feature matrix + the target user's own training history.
- **Training (`fit`) — no learning.** Three precomputations:
  1. **Densify + L2-normalise** every item vector once (~70 MB float32). Then cosine similarity between any two items is a single dot product, and similarity of one item against *all* items is one **matrix–vector multiply** (BLAS, ~3 ms) instead of the slow sparse path (~100 ms).
  2. **Top-L selection via `argpartition`** — O(N) to grab the 50 nearest, vs O(N log N) for a full sort.
  3. An **LRU cache** (20K items) memoises neighbour lookups so repeated queries are free.
- **Prediction (mean-centred weighted average).**

  ```
            r̄_u + Σ_{i ∈ N^L_u(j)} sim(i,j) · (r_{u,i} − r̄_u)
  r̂(u,j) = ────────────────────────────────────────────────────
                     Σ_{i ∈ N^L_u(j)} |sim(i,j)| + ε
  ```

  where `N^L_u(j)` = the **top-50 content-neighbours of *j* that user *u* has rated**, `r̄_u` = the user's mean rating. *Mean-centring* matters: it predicts *how much above/below your average* you'll rate *j*, so a generous and a harsh rater are handled correctly. Result clipped to [0.5, 5.0].
- **Hyperparameters.** `n_neighbors = 50`, `cache_max = 20000`.
- **Complexity.** Per prediction: one (62K × 276) matmul + a partial sort = a few ms.
- **Strengths.** Cold-item friendly; interpretable ("similar because same genres / tags"); needs no other users.
- **Weaknesses / caveats.** Returns `r̄_u` when the user rated **none** of *j*'s neighbours (frequent → it often behaves like a per-user-mean predictor, which is why its RMSE only just beats Global Mean). Returns `NaN` if the movie is unknown or the user has no history — `NaN` triggers the hybrid fallback and is dropped from ranking. Tends to **over-specialise** (recommends more of the same).

## 4. User-Based k-NN — CF, memory-based

[`models/collaborative.py`](../src/hybrid_recsys/models/collaborative.py)

- **Philosophy.** "People whose taste matches yours rated *i* like this." Pure collaborative filtering — uses only the rating matrix, no content.
- **Data.** Training ratings (ids cast to strings; `Reader(rating_scale=(0.5, 5.0))`).
- **Training (`fit`) — memorisation, not optimisation.** Surprise `KNNWithMeans` computes a **user × user similarity matrix** (Pearson-baseline). No gradients — "fitting" = building and storing similarities.
- **Prediction.**

  ```
  r̂(u,i) = r̄_u + Σ_{v ∈ N^k_i(u)} sim(u,v) · (r_{v,i} − r̄_v) / Σ |sim(u,v)|
  ```

  the *k* most-similar users *who rated i*, their deviations from their own means, weighted by similarity.
- **Hyperparameters.** `k = 80`, `min_k = 5` (need ≥ 5 neighbours or it backs off toward the mean).
- **Complexity / memory caveat.** A full 162K-user similarity matrix ≈ **98 GB**, so the model **samples 20K users (seeded)** before fitting (→ ~1.6 GB). Consequence: a test user outside that sample is *unknown*, and Surprise returns the **baseline mean**. Empirically ~90% of test predictions are this fallback — so user-kNN's reported numbers largely reflect the baseline, not genuine user-CF. (This also inflates its ranking via constant-output ties; see the tie-break note in the evaluation.)

## 5. Item-Based k-NN — CF, memory-based

[`models/collaborative.py`](../src/hybrid_recsys/models/collaborative.py)

- **Philosophy.** "Items similar to ones you already liked." Same idea as user-kNN but over the **item** space — more stable in movie domains (the catalogue changes slower than the user population, and item-item co-rating is denser).
- **Data / Training.** Same `KNNWithMeans` (Pearson-baseline, `k = 80`, `min_k = 5`) but `user_based=False` → an **item × item similarity matrix**.
- **Prediction.**

  ```
  r̂(u,i) = r̄_i + Σ_{j ∈ N^k_u(i)} sim(i,j) · (r_{u,j} − r̄_j) / Σ |sim(i,j)|
  ```

- **Memory caveat.** Full 62K-item matrix ≈ **15 GB**, so it **caps to the 15K most-rated items** (→ ~900 MB). Because ratings concentrate on popular titles, this cap barely reduces coverage — which is exactly why item-kNN's RMSE (≈ 0.83) is far better than user-kNN's (≈ 1.04): the item cap rarely bites, the user cap bites hard.
- **Strengths.** Robust, strong RMSE; the most reliable pure-CF model here.

## 6. SVD — CF, matrix factorisation (the one genuinely *trained* model)

[`models/collaborative.py`](../src/hybrid_recsys/models/collaborative.py)

- **Philosophy.** Every user and movie gets a short **latent vector** in a shared space; their dot product reconstructs the rating. The factors discover patterns ("likes dark sci-fi", "is a crowd-pleaser") that no single neighbour reveals.
- **Data.** Training ratings.
- **Training (`fit`) — real optimisation by SGD.** Minimise regularised squared error over observed ratings:

  ```
  min  Σ_{(u,i)} ( r_{u,i} − μ − b_u − b_i − q_iᵀp_u )²  +  λ(b_u² + b_i² + ‖p_u‖² + ‖q_i‖²)
  ```

  via stochastic gradient descent — for each rating, with error `e = r − r̂`:

  ```
  b_u ← b_u + lr·(e − λ·b_u)        p_u ← p_u + lr·(e·q_i − λ·p_u)
  b_i ← b_i + lr·(e − λ·b_i)        q_i ← q_i + lr·(e·p_u − λ·q_i)
  ```

  Hyperparameters chosen by **5-fold cross-validation** (`GridSearchCV`, scoring RMSE):
  `n_factors ∈ {50,100,200}`, `n_epochs ∈ {20,40}`, `lr_all ∈ {0.002,0.005}`, `reg_all ∈ {0.02,0.05}`. `random_state` is **pinned on every candidate** so results are reproducible.
- **Prediction.** `r̂(u,i) = μ + b_u + b_i + q_iᵀp_u`. Unknown user/item → a bias-only fallback.
- **Complexity.** Training ≈ O(`n_epochs` × `#ratings` × `n_factors`); prediction is one dot product.
- **Strengths / weaknesses.** Best single model on RMSE/MAE (learns global structure, generalises to sparse users). Weaker on *ranking* — it minimises rating error, not the goal of pushing a few relevant items above many random ones.

## 7. Weighted Hybrid — CB + CF by fixed blend

[`models/hybrid.py`](../src/hybrid_recsys/models/hybrid.py)

- **Philosophy.** The simplest fusion: trust SVD for accuracy, lean on CB for cold-start/content coverage, mix them linearly.
- **Data.** The already-trained SVD and Content-Based models (held by reference).
- **Training (learns exactly **one** number).** `tune_alpha` sweeps `α ∈ [0, 1]` in steps of 0.05 and keeps the value with the lowest **validation** RMSE. (Here it converges near α ≈ 0.9 → mostly SVD, because CB is the weaker signal.)
- **Prediction.**

  ```
  r̂(u,i) = α · r̂_SVD(u,i) + (1 − α) · r̂_CB(u,i)
  ```

- **Back-stage.** If CB returns `NaN` (cold item / no history) → falls back to the pure SVD prediction, so the hybrid never fails where SVD succeeds.
- **Trade-off.** Transparent and safe, but a *single global* weight can't adapt per user/item — which motivates stacking.

## 8. Stacked Hybrid — CB + CF by learned meta-model

[`models/hybrid.py`](../src/hybrid_recsys/models/hybrid.py)

- **Philosophy.** Don't guess the weights — **learn** them, and over richer signals, so the combiner can down-weight weak base models automatically and use context (popularity, how active the user is) to decide whom to trust.
- **Features (7 per pair).** `[pred_content, pred_user_knn, pred_item_knn, pred_svd, item_popularity, user_rating_count, item_rating_count]` — the four base predictions plus three side features.
- **Training (learns Ridge weights) — leak-free via OOF stacking.**

  ```
  Split train into 5 folds.
  For each fold f:
      fit CB, User-kNN, Item-kNN, SVD on the OTHER 4 folds
      predict fold f  → out-of-fold (OOF) predictions     (no model sees its own training rows)
  Assemble the 7-feature matrix from the OOF predictions + side features.
  Fit Ridge(α=1):   min ‖Xw − y‖² + α‖w‖²   → 7 coefficients.
  ```

  OOF is the crucial trick: training the meta-model on *in-sample* base predictions would leak (each base model is overconfident on data it trained on), inflating the meta-weights.
- **Prediction.** Base models retrained on full train → build the 7 features for `(u,i)` → push through Ridge → clip to [0.5, 5.0]. Side features (`item_popularity`, counts, `global_mean`) are stored on the object so it scores standalone at serving time. Returns `global_mean` if any base prediction is `NaN`.
- **Result / interpretability.** The learned coefficients lean on **SVD (~0.61)** and **Item-kNN (~0.40)**, and drive the weak **User-kNN toward 0** — i.e. the meta-model *discovered* which signals to trust. This is why it tends to be the best model on **both** RMSE and ranking.

---

## Summary

| # | Model | Type | Learns? | Core mechanism | Prediction |
|---|---|---|---|---|---|
| 1 | Global Mean | baseline | ❌ | training average | `μ` |
| 2 | Popularity | baseline | ❌ | interaction counts | `0.5 + 4.5·count/max` |
| 3 | Content-Based | CB | ❌ memory | cosine sim on 276-dim features | mean-centred weighted avg over similar items |
| 4 | User-kNN | CF | ❌ memory | user–user Pearson-baseline sim | weighted avg of similar users |
| 5 | Item-kNN | CF | ❌ memory | item–item Pearson-baseline sim | weighted avg over similar items |
| 6 | SVD | CF | ✅ SGD | latent factors + biases | `μ + b_u + b_i + q_iᵀp_u` |
| 7 | Weighted Hybrid | hybrid | ✅ (α) | linear blend | `α·SVD + (1−α)·CB` |
| 8 | Stacked Hybrid | hybrid | ✅ (Ridge) | meta-learner over base preds | Ridge over 7 features |

**Assignment mapping.** The required fusion of **content-based (CB)** + **collaborative filtering (SVD / kNN)** is realised two ways: a **fixed-weight** blend (Weighted Hybrid) and a **learned** meta-model (Stacked Hybrid). The six single models + two baselines exist so the report can show the hybrids beating every CB-only and CF-only baseline on RMSE, MAE, and P/R/F1@K.

---

## Where each lives in code

| Model | File |
|---|---|
| Global Mean, Popularity | inline in `03_baselines.ipynb` |
| Content-Based | [`models/content.py`](../src/hybrid_recsys/models/content.py) |
| User-kNN, Item-kNN, SVD | [`models/collaborative.py`](../src/hybrid_recsys/models/collaborative.py) |
| Weighted & Stacked Hybrid | [`models/hybrid.py`](../src/hybrid_recsys/models/hybrid.py) |
| Feature matrix (CB input) | [`pipeline/features.py`](../src/hybrid_recsys/pipeline/features.py) |
| Evaluation (both metric families) | [`evaluation/metrics.py`](../src/hybrid_recsys/evaluation/metrics.py) |
| Serving (loads all, exposes `predict`) | [`serving.py`](../src/hybrid_recsys/serving.py) |
