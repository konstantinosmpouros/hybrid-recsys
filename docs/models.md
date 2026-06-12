# Models — How Each Recommender Works (in depth)

Complete reference for every model: philosophy, data, training mechanics (with the
actual maths), prediction formula, hyperparameters, complexity/memory, strengths &
weaknesses, and back-stage caveats. Verified against [`hybrid_recsys/`](../hybrid_recsys/).

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

### The metrics, defined

The five required metrics plus the extended/beyond-accuracy ones used in notebooks 14–16:

| Metric | Family | Definition | Direction |
|---|---|---|---|
| **RMSE** | rating | √(mean of (true − predicted)²) over the full test set | lower better |
| **MAE** | rating | mean of \|true − predicted\| over the full test set | lower better |
| **Precision@K** | ranking | fraction of the top-K recommendations that are relevant (rating ≥ 4) | higher better |
| **Recall@K** | ranking | fraction of the user's relevant items that land in the top-K | higher better |
| **F1@K** | ranking | harmonic mean of **macro** Precision@K and Recall@K (so F1 lies between them) | higher better |
| **NDCG@K** | ranking | discounted cumulative gain — like F1 but rewards ranking relevant items *higher* in the list | higher better |
| **AUC** | ranking | P(a relevant item is scored above a random non-relevant one), via Mann–Whitney | higher better |
| **Coverage** | beyond-acc. | fraction of the catalogue that ever appears in some user's top-K | higher = explores more |
| **Diversity** | beyond-acc. | mean (1 − cosine similarity) within a list, in the content space | higher = less repetitive |
| **Novelty** | beyond-acc. | mean −log₂ p(item) of recommendations (rare items score high) | higher = less blockbuster-y |

All of these are implemented in [`evaluation/metrics.py`](../hybrid_recsys/evaluation/metrics.py); the rating + P/R/F1@K set is persisted to `all_metrics.json`, while NDCG/AUC/coverage/diversity/novelty are computed live in notebook 14 and shown as figures. The **consolidated results report** ([`notebooks/16_evaluation_report.ipynb`](../notebooks/16_evaluation_report.ipynb)) tabulates and visualises all of them with commentary.

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
- **Pearson-baseline** (Surprise kNN): a Pearson correlation computed on **baseline-centred** ratings with **shrinkage** toward 0 for pairs with few co-ratings. Concretely, with the baseline `b_{u,i} = μ + b_u + b_i` and (for the item–item case) `U_{ij}` = users who rated both *i* and *j*:

  ```
              Σ_{u∈U_ij} (r_{u,i} − b_{u,i})(r_{u,j} − b_{u,j})
  ρ̂(i,j) = ─────────────────────────────────────────────────────────
            √Σ_{u∈U_ij}(r_{u,i} − b_{u,i})² · √Σ_{u∈U_ij}(r_{u,j} − b_{u,j})²

                       |U_ij| − 1
  sim(i,j) = ────────────────────────── · ρ̂(i,j)        (shrinkage = 100, Surprise default)
              |U_ij| − 1 + shrinkage
  ```

  Two effects: the baseline-centring removes "this user rates high / this movie is loved" offsets so the correlation reflects genuine taste agreement, and the shrinkage factor `(n−1)/(n−1+100)` crushes similarities supported by few co-ratings — two users who share a single movie can no longer look perfectly similar. (The user–user case is symmetric, with `I_{uv}` = items co-rated by users *u* and *v*.)

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

[`models/content.py`](../hybrid_recsys/models/content.py)

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

[`models/collaborative.py`](../hybrid_recsys/models/collaborative.py)

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

[`models/collaborative.py`](../hybrid_recsys/models/collaborative.py)

- **Philosophy.** "Items similar to ones you already liked." Same idea as user-kNN but over the **item** space — more stable in movie domains (the catalogue changes slower than the user population, and item-item co-rating is denser).
- **Data / Training.** Same `KNNWithMeans` (Pearson-baseline, `k = 80`, `min_k = 5`) but `user_based=False` → an **item × item similarity matrix**.
- **Prediction.**

  ```
  r̂(u,i) = r̄_i + Σ_{j ∈ N^k_u(i)} sim(i,j) · (r_{u,j} − r̄_j) / Σ |sim(i,j)|
  ```

- **Memory caveat.** Full 62K-item matrix ≈ **15 GB**, so it **caps to the 15K most-rated items** (→ ~900 MB). Because ratings concentrate on popular titles, this cap barely reduces coverage — which is exactly why item-kNN's RMSE (≈ 0.83) is far better than user-kNN's (≈ 1.04): the item cap rarely bites, the user cap bites hard.
- **Strengths.** Robust, strong RMSE; the most reliable pure-CF model here.

### What Surprise actually does at predict time (both kNN models)

Calling `model.predict(u, i)` on a Surprise `KNNWithMeans` runs this pipeline in the background:

1. **Id translation.** Raw ids (cast to `str` by our wrapper) are mapped to Surprise's inner
   integer ids. If the user **or** the item was not in the trainset — e.g. outside the 20K-user
   sample (User-kNN) or the 15K-item cap (Item-kNN) — Surprise raises `PredictionImpossible`
   *internally*, catches it, and returns the **default estimate = the global train mean μ**.
   Never an exception, never NaN.
2. **Neighbour retrieval.** Otherwise it reads the precomputed similarity row and keeps the
   `k = 80` most-similar users/items **that actually rated the target** (co-rated support, not
   just globally similar ones).
3. **`min_k` guard.** If fewer than `min_k = 5` such neighbours exist, the weighted-deviation
   term is dropped and the prediction collapses to the plain mean (`r̄_u` user-based, `r̄_i`
   item-based).
4. **Aggregation + clipping.** Else the mean-centred weighted average above is computed and the
   result is clipped to the 0.5–5.0 scale.

Two consequences worth internalising: **(a)** User-kNN's 20K-user sample means roughly 90% of
test calls exit at step 1 with the global-mean fallback — which is exactly why its metrics
hover near Global Mean; **(b)** the Surprise models **never return NaN**, unlike the content
models — a distinction that drives the hybrid fallback logic (see the
[fallback-semantics table](#fallback--edge-case-semantics-all-12-models) below).

## 6. SVD — CF, matrix factorisation (the one genuinely *trained* model)

[`models/collaborative.py`](../hybrid_recsys/models/collaborative.py)

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

[`models/hybrid.py`](../hybrid_recsys/models/hybrid.py)

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

[`models/hybrid.py`](../hybrid_recsys/models/hybrid.py)

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
- **Result / interpretability.** The learned coefficients lean on **SVD** and **Item-kNN**, and drive the weak **User-kNN toward 0** — i.e. the meta-model *discovered* which signals to trust. On the test set it is the **best of the eight core models on RMSE (0.8054)** and a strong all-rounder on ranking (F1@10 ≈ 0.42); only the extension Dual-Head edges its RMSE.

---

# Extension models (notebooks 10–14)

The eight models above are **frozen** — their saved artifacts and `all_metrics.json`
entries are never re-trained. The four models below are **additive**: each is built and
evaluated under the identical protocol (same split, same sampled-negatives ranking) so it
drops straight into the comparison. Two raise the **content** ceiling (a richer feature
representation), one brings a **graph** learner, and one is a second hybrid that targets
*all* metrics with task-specific heads.

## 9. Content-Based (Tag Genome) — CB, memory-based · *extension*

[`models/content.py`](../hybrid_recsys/models/content.py) · [`pipeline/features.py`](../hybrid_recsys/pipeline/features.py) (`build_item_features_genome`)

- **Philosophy.** Same content model as #3, but fed a **far better description of each
  movie**. Instead of free-text user tags, it uses MovieLens's **tag genome**: 1,128
  human-curated *relevance scores* per movie (how strongly each of 1,128 tags — "dark",
  "based on a book", "visually stunning" — applies). This is a dense, low-noise content
  signal, where TF-IDF over free text is sparse and noisy.
- **Data.** A **second** 276-dim feature matrix = 20-dim genre multi-hot ⊕ **TruncatedSVD(256)
  of the 1,128-dim genome relevance vectors** (`item_features_genome.npz`). Built and saved
  separately so the original TF-IDF features (and the trained core models that depend on
  them) are never touched. Movies with no genome coverage get a **zero genome block** → they
  fall back to genre-only similarity.
- **Training / Prediction.** Identical mechanics to the TF-IDF content model (densify +
  L2-normalise once, cosine, top-L=50, mean-centred weighted average). Only the input vectors
  differ.
- **Result.** A **big lift** over TF-IDF content: **RMSE 1.046 → 0.967**, **F1@10 0.207 →
  0.376**, with no change to any other model. This is the cleanest demonstration in the
  project that *content representation*, not the algorithm, was the content model's
  bottleneck — and it makes the genome content model the default "why this?" engine in the app.
- **Caveats.** Genome coverage is incomplete (older/obscure titles lack scores) — those
  movies revert to genre-only. Still cold-item friendly and interpretable.

## 10. LightGCN — graph CF, learned (BPR) · *extension*

[`models/lightgcn.py`](../hybrid_recsys/models/lightgcn.py)

- **Philosophy.** Treat the ratings as a **bipartite user–item graph** and learn embeddings
  by **message passing**: a user's vector becomes the smoothed average of the items they
  touched, which are themselves averages of the users who touched them, repeated over a few
  hops. LightGCN strips the usual GCN of feature transforms and non-linearities — just
  neighbourhood aggregation — which is exactly what helps collaborative filtering. It is
  trained to **rank**, not to predict ratings, so it attacks the metric (F1@K) where the
  rating-optimal models (SVD) are weakest.
- **Data.** The train interaction graph (implicit — *which* (u,i) pairs exist, not the rating
  value). On a CPU laptop it **subsamples to `max_users = 10,000`** — a reduced-scale
  demonstration of the method, not a full-25M SOTA run.
- **Training (`fit`) — real optimisation, PyTorch.**
  1. Build the symmetric-normalised bipartite adjacency `Â = D^(−1/2) · A · D^(−1/2)`.
  2. Initialise a `(n_users + n_items) × 64` embedding table.
  3. Each epoch: **propagate the whole graph once** (`E_k = Â·E_{k−1}`, `k = 0..3`), take the
     **mean over layers** as the final embedding, then take a **BPR** gradient step — for each
     observed pair `(u, i⁺)` and a random unobserved `i⁻`, push the score of the positive above
     the negative: `loss = −log σ(eᵤ·eᵢ₊ − eᵤ·eᵢ₋) + reg`. The whole-graph-once-per-epoch design
     (vs once per minibatch) is the key efficiency trick — `epochs` propagations total.
  4. After training, the propagated embeddings are cached as numpy, so **serving needs only
     numpy** (no torch at predict time).
- **Hyperparameters.** `dim = 64`, `n_layers = 3`, `epochs = 200`, `lr = 5e-3`, `reg = 1e-4`,
  `max_users = 10,000`.
- **Prediction.** `score(u,i) = eᵤ · eᵢ` (dot product of cached embeddings). A **relevance
  score, not a rating** → `ranking_only = True`, so RMSE/MAE are reported as **null** and it
  is ranked by raw score only.
- **Result.** **Best F1@10 in the project (≈ 0.62)** — the ranking-trained graph model does
  exactly what it's designed to. **Caveat:** it can only score items inside its training
  subgraph, so its candidate vocabulary is restricted; in the app, heavy models like this use
  **two-stage retrieval** (popularity-pooled top-3000, then re-rank).

## 11. Dual-Head Hybrid — CB + CF, two learned heads · *extension*

[`models/hybrid.py`](../hybrid_recsys/models/hybrid.py) (`DualHeadHybrid`)

- **Philosophy.** A single hybrid that is honest about the fact that **RMSE and ranking are
  different objectives**. Rather than one regressor scored two ways, it trains **two heads on
  the same feature vector**: a regressor for rating accuracy and a classifier for "will the
  user like this?" The rating head drives RMSE/MAE; the rank head drives P/R/F1@K.
- **Features (8 per pair).** `[pred_content_genome, pred_user_knn, pred_item_knn, pred_svd,
  pred_lightgcn, item_popularity, user_rating_count, item_rating_count]` — the **five** strongest
  base models (note it uses the *genome* content model and LightGCN, the best of each family)
  plus three side features.
- **Training — blending on validation (no OOF re-run).** Base models are **frozen** (loaded
  from disk, never retrained), so unlike the Stacked Hybrid this trains by *blending on the
  validation split*: build the 8-feature matrix on val, then fit
  - `rating_head = Ridge(α=1)` on the true ratings, and
  - `rank_head  = LogisticRegression` on the binary label `rating ≥ 4`.
  NaN base predictions (e.g. LightGCN on an out-of-graph user) are **imputed with the
  per-feature median** learned at fit time, so the heads always see a complete vector — meaning
  it can **score every candidate**, unlike raw LightGCN.
- **Prediction (two heads, one feature vector `x ∈ ℝ⁸`).** After median-imputing any NaN
  entries of `x`:

  ```
  rating head :  r̂(u,i)    = clip( w_rᵀx + b_r , 0.5, 5.0 )        Ridge — drives RMSE/MAE
  rank head   :  P(like)   = σ( w_kᵀx + b_k ) = 1 / (1 + e^−(w_kᵀx + b_k))
                                                                    logistic — drives P/R/F1@K
  ```

  **Top-N lists are sorted by the rank head's probability; star ratings come from the rating
  head.** The two heads share the same inputs but their weight vectors `w_r`, `w_k` are fit to
  *different objectives* (squared error vs log-loss on `rating ≥ 4`) — that separation is the
  whole point: it removes the compromise a single regressor must make between calibration and
  separation (see the rating-vs-ranking trade-off in [`evaluation.md`](evaluation.md) §6).
- **Result.** **Best RMSE in the whole project (0.8028)** and a strong F1@10 (≈ 0.42). It
  targets *all five* required metrics with one model and is the app's default recommender.
- **Trade-off.** More moving parts than the Weighted/Stacked hybrids and depends on five base
  models being loadable — which is why it is the most memory-hungry model to serve.

## 12. Content-Based (Semantic Embeddings) — CB, memory-based · *extension*

[`models/content.py`](../hybrid_recsys/models/content.py) · [`pipeline/features.py`](../hybrid_recsys/pipeline/features.py) (`build_item_features_embedding`)

- **Philosophy.** A **third** content representation that captures **meaning**, not word
  overlap. A pretrained **sentence-transformer** (`all-MiniLM-L6-v2`) encodes a per-movie text
  profile into a dense semantic vector, so two movies described differently but *meaning* the
  same thing land close together — something bag-of-words TF-IDF cannot do.
- **Data.** For each movie, a profile string `"<title> | genres: … | tags: …"` is encoded and
  **L2-normalised** into a 384-dim vector (`content_embed_model.joblib`). Drop-in replacement
  for the content feature matrix.
- **Training / Prediction.** Identical to the other content models (the `ContentBasedRecommender`
  class is reused unchanged) — only the feature matrix differs.
- **Result.** **RMSE 1.029, F1@10 0.320** — better than TF-IDF content but **below the genome
  model**. The honest finding: a curated relevance signal (genome) beats a general-purpose
  embedding *here*, because the genome was built specifically to describe these movies. The
  three content spaces are compared side-by-side in the app's **Movie Explorer**.

---

## Fallback & edge-case semantics (all 12 models)

What every model does **in the background** when the happy path fails — the behaviour that
drives the hybrid fallback logic, the NaN-filtering in the ranking evaluators, and the serving
layer's guards. ("Unknown" = not present at fit time.)

| Model | Unknown movie | Unknown user / empty history | Too little evidence | Can return NaN? |
|---|---|---|---|---|
| Global Mean | `μ` | `μ` | — | no |
| Popularity | maps count 0 → 0.5 | (not personalised) | — | no |
| Content (TF-IDF / Genome / Embed) | **NaN** (not in `movie_index`) | **NaN** (empty ratings dict) | returns the **user mean `r̄_u`** when none of the top-L neighbours were rated | **yes** |
| User-kNN | global mean `μ` | global mean `μ` (outside the 20K sample — ~90% of test users) | `r̄_u` when < `min_k` = 5 neighbours | no |
| Item-kNN | global mean `μ` (outside the 15K cap) | global mean `μ` | `r̄_i` when < `min_k` = 5 neighbours | no |
| SVD | bias-only `μ + b_u` | bias-only `μ + b_i` (both unknown → `μ`) | — | no |
| Weighted Hybrid | **falls back to pure SVD** whenever CB returns NaN | SVD side handles it | — | no |
| Stacked Hybrid | returns its stored **`global_mean`** if *any* base prediction is NaN | same | — | no |
| LightGCN | **NaN** (item outside the training subgraph) | **NaN** (user outside the 10K subsample) | — | **yes** |
| Dual-Head Hybrid | NaN bases **median-imputed** (per-feature medians learned at fit) → always scores | same | — | no |

Three design consequences:

1. **NaN is a signal, not a bug.** The content models and LightGCN return NaN precisely where
   they have *no basis* for a prediction; the ranking evaluators drop NaN candidates, and the
   hybrids convert NaN into a fallback (SVD / global mean / median imputation) — so a hybrid
   never fails where its strongest parent succeeds.
2. **The Surprise models silently degrade rather than fail** — useful for robustness, but it
   means a "prediction" from User-kNN is very often just the global mean (its evaluation
   numbers say as much).
3. **Only the Dual-Head can score literally every (user, movie) pair**, because imputation
   guarantees a complete feature vector — which is why it's the app's default recommender.

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
| 9 | Content (Genome) | CB · *ext* | ❌ memory | cosine on genre ⊕ SVD(tag-genome) | mean-centred weighted avg |
| 10 | LightGCN | graph CF · *ext* | ✅ BPR | message passing on the user–item graph | `eᵤ · eᵢ` (ranking-only) |
| 11 | Dual-Head Hybrid | hybrid · *ext* | ✅ (Ridge + Logit) | two heads over 8 features | rating head + `P(r≥4)` rank head |
| 12 | Content (Embeddings) | CB · *ext* | ❌ memory | cosine on sentence-transformer vectors | mean-centred weighted avg |

**Assignment mapping.** The required fusion of **content-based (CB)** + **collaborative filtering (SVD / kNN)** is realised — at minimum — two ways: a **fixed-weight** blend (Weighted Hybrid) and a **learned** meta-model (Stacked Hybrid). The single models + two baselines exist so the report can show the hybrids beating every CB-only and CF-only baseline. The four extensions go further: a richer content signal (Genome, Embeddings), a ranking-trained graph learner (LightGCN), and a hybrid with task-specific heads (Dual-Head).

### Results on the held-out test set

From [`artifacts/metrics/all_metrics.json`](../artifacts/metrics/) — RMSE/MAE over the full
test set, P/R/F1@10 via the sampled-negatives protocol (1,000 users × 100 negatives, relevance ≥ 4.0).

| Model | RMSE ↓ | MAE ↓ | P@10 | R@10 | F1@10 ↑ |
|---|---|---|---|---|---|
| **Dual-Head Hybrid** *(ext)* | **0.8028** | 0.6031 | 0.348 | 0.542 | 0.424 |
| **Stacked Hybrid** | 0.8054 | **0.6029** | 0.343 | 0.533 | 0.418 |
| Weighted Hybrid | 0.8095 | 0.6074 | 0.292 | 0.451 | 0.354 |
| SVD | 0.8108 | 0.608 | 0.290 | 0.448 | 0.352 |
| Item-kNN | 0.8336 | 0.6223 | 0.354 | 0.558 | 0.433 |
| Content (Genome) *(ext)* | 0.9670 | 0.7091 | 0.317 | 0.460 | 0.376 |
| Content (Embeddings) *(ext)* | 1.0292 | 0.7647 | 0.281 | 0.373 | 0.320 |
| User-kNN | 1.0401 | 0.8119 | 0.103 | 0.150 | 0.122 |
| Content-Based (TF-IDF) | 1.0462 | 0.7831 | 0.183 | 0.239 | 0.207 |
| Global Mean | 1.0609 | 0.8339 | 0.066 | 0.089 | 0.076 |
| LightGCN *(ext)* | — | — | 0.489 | 0.849 | **0.621** |
| Popularity | 2.7120 | 2.4856 | 0.479 | 0.836 | 0.609 |

**Reading the table.** On **rating accuracy** the two learned hybrids lead, exactly as the
assignment wants: both **Stacked and Dual-Head beat every CB-only and CF-only baseline** on
RMSE *and* MAE. On **ranking**, **LightGCN wins F1@10** because it is the only model trained
on a ranking loss — but it carries a candidate-coverage caveat (it scores only in-graph items)
and reports no RMSE/MAE. **Popularity's** high F1@10 is an artifact of the sampled-negatives
protocol (popular titles really are relevant more often) and is paired with a meaningless RMSE
of 2.71 — it personalises nothing. Among models that do *both* well, the hybrids and Item-kNN
are the honest top tier. (Notebook 14 adds NDCG@K, AUC, segmented RMSE, coverage/diversity/novelty
and bootstrap CIs on top of this table.)

---

## Where each lives in code

| Model | File |
|---|---|
| Global Mean, Popularity | inline in `03_baselines.ipynb` |
| Content-Based (TF-IDF, Genome, Embeddings) | [`models/content.py`](../hybrid_recsys/models/content.py) — one class, three feature matrices |
| User-kNN, Item-kNN, SVD | [`models/collaborative.py`](../hybrid_recsys/models/collaborative.py) |
| Weighted, Stacked & Dual-Head Hybrid | [`models/hybrid.py`](../hybrid_recsys/models/hybrid.py) |
| LightGCN | [`models/lightgcn.py`](../hybrid_recsys/models/lightgcn.py) |
| Feature matrices (TF-IDF / genome / embedding) | [`pipeline/features.py`](../hybrid_recsys/pipeline/features.py) |
| Evaluation (both metric families + NDCG/AUC/coverage) | [`evaluation/metrics.py`](../hybrid_recsys/evaluation/metrics.py) |
| Serving (loads all 12, exposes `predict`) | [`serving.py`](../backend/serving.py) |

> **A note on the three content models.** They are the *same* `ContentBasedRecommender`
> class — only the item-feature matrix differs (TF-IDF/LSA vs SVD-of-genome vs
> sentence-transformer embeddings). This is deliberate: it isolates the effect of the
> *representation* from the algorithm, and the result (genome ≫ embeddings > TF-IDF) is the
> headline content finding.

---

## Diagnostic figures (per model)

Each model's notebook saves a characteristic figure to [`artifacts/figures/`](../artifacts/figures/)
(referenced and explained in the consolidated report,
[`notebooks/16_evaluation_report.ipynb`](../notebooks/16_evaluation_report.ipynb) §7–8). The most
informative per model:

| Model | Notebook | Key figure(s) | What it shows |
|---|---|---|---|
| Popularity | 03 | `eval_popularity_ranking` | strong ranking from popularity alone (the baseline to beat) |
| Content (TF-IDF) | 04 | `eval_content_error`, `eval_content_ranking` | rating-error spread; weak ranking |
| Content (Genome) | 10 | `eval_content_genome_error`, `eval_content_genome_ranking` | the genome lift over TF-IDF |
| Content (Embeddings) | 13 | `eval_content_embed_error`, `eval_content_embed_ranking` | semantic-embedding variant |
| User-kNN | 05 | `eval_userknn_neighbors`, `eval_userknn_simdist` | neighbour structure + the similarity distribution that explains its heavy fallback |
| Item-kNN | 06 | `eval_itemknn_graph`, `eval_itemknn_ranking` | item–item neighbour graph; strong classic-CF ranking |
| SVD | 07 | `eval_svd_factors`, `eval_svd_error` | learned latent item-factor space; tight rating error |
| Weighted Hybrid | 08 | `eval_weighted_alpha` | the α-sweep (how the fixed blend is chosen) |
| Stacked Hybrid | 09 | `eval_stacked_coefficients` | learned Ridge weights (SVD + Item-kNN dominate) |
| LightGCN | 11 | `eval_lightgcn_ranking` | best F1@K (ranking-trained graph CF) |
| Dual-Head Hybrid | 12 | `eval_dualhead_weights`, `eval_dualhead_ranking` | rating-head coefficients — *how it fuses content + collaborative* |

The aggregate/headline figures (RMSE/MAE bars, F1@10, the rating-vs-ranking scatter, NDCG/AUC,
segmented RMSE, bootstrap CIs, beyond-accuracy) come from notebook 14 (`08_*`–`21_*`), and the
practical case-study figures from notebook 15 (`15_cs_*`). All are collected in notebook 16.
