# Project Walkthrough — Hybrid Movie Recommender

A complete, honest explanation of what we built, why, whether it works, and what else
we could do. Written against the actual source and the results in
`artifacts/metrics/all_metrics.json`.

> **Status (updated).** The two issues an earlier draft of this document flagged are now
> **fixed**, and the project has grown from 8 models / 3 notebooks to **12 models / 14
> notebooks**:
> - The **Stacked Hybrid is in `all_metrics.json`** (all 12 models present), and the
>   **ranking tie-ordering artifact is fixed** (candidates are now shuffled before sorting,
>   so a constant predictor no longer "wins"). The §7 results below are the corrected numbers.
> - The **tag genome is now used** as a content feature (notebook 10), which delivered the
>   single biggest content lift in the project.
> - Companion docs: model internals → [`models.md`](models.md); per-notebook detail →
>   [`notebooks.md`](notebooks.md); serving → [`backend.md`](backend.md) · [`app.md`](app.md).

---

## 1. The problem we're solving

**Assignment Θέμα 2:** build a **hybrid recommender** that fuses a **content-based**
method with a **collaborative-filtering** method, then prove it beats each one alone on
RMSE, MAE, Precision@K, Recall@K, F1@K — and wrap it in an app.

A recommender answers: *given what user u has rated, what rating would they give item j,
and which unseen items should we show them?* The two classic families:

- **Collaborative Filtering (CF)** — "people who rated like you also liked X." Uses only
  the rating matrix. Powerful with lots of ratings; useless for brand-new items/users
  (cold start).
- **Content-Based (CB)** — "you liked these, here are similar *items by their
  attributes*." Uses item metadata (genres, tags). Handles cold-start items, but
  over-specializes and ignores cross-user taste patterns.

**Hybrid** = combine them to get CF's accuracy *and* CB's cold-start robustness. That is
the whole point of the assignment.

---

## 2. The data — MovieLens 25M

| | |
|---|---|
| Ratings | 25,000,095 |
| Users | 162,541 |
| Movies | 62,423 |
| Scale | 0.5–5.0 (half-star) |
| Tag applications | ~1.09M |
| Sparsity | **> 99.8%** |

Six raw CSVs: `ratings`, `movies`, `tags`, `genome-scores`, `genome-tags`, `links`.

**Key EDA findings (notebook 01):**

- **Positivity bias** — modal rating is 4.0, low ratings rare → justifies the
  **relevance threshold ≥ 4.0** for "did the user like it."
- **Power-law / long tail** — most users rate few films, most films get few ratings →
  sparsity is the core CF challenge and the reason CB helps.
- **Genre dominance** — Drama / Comedy / Thriller dominate.
- **Temporal trend** — rating volume grows over time → a **temporal split** is the honest
  protocol (a random split would leak the future into training).

> The **tag genome** (1,128 relevance scores/movie) is profiled in EDA. The *core* content
> model (notebook 04) uses free-text tags only; the genome was later wired into a **second
> content model** (notebook 10), which gave the biggest content lift in the project
> (RMSE 1.046 → 0.967). So the genome signal is now used — see §5 and [`models.md`](models.md) §9.

---

## 3. The notebooks

Fourteen notebooks, run in order — full per-notebook detail in [`notebooks.md`](notebooks.md).
The shape: **prepare** (01–02) → **one notebook per model that trains *and* evaluates it**
(03–09 core, 10–13 extensions) → **aggregate + deep eval** (14).

- **`01_eda.ipynb` — Explore & split.** Loads CSVs, profiles everything above, then
  produces the **user-wise temporal 80/10/10 split**: each user's ratings sorted by time,
  first 80% → train, next 10% → val, last 10% → test. Users with < 5 total ratings
  dropped. Saves processed parquets + splits — the foundation everything else loads
  identically.
- **`02_features.ipynb` — Build item features.** Turns each movie into a vector (§4).
  Saves the feature matrix + fitted transformers so serving never re-fits.
- **`03`–`09` — one model per notebook.** Each trains + saves a model, then evaluates it
  exactly once on the untouched test set under a leak-free protocol and appends to
  `all_metrics.json`: baselines (03), Content-Based (04), User-kNN (05), Item-kNN (06),
  SVD (07), Weighted Hybrid (08), Stacked Hybrid (09).
- **`10`–`13` — extensions (additive; 03–09 stay frozen).** Content-Genome (10), LightGCN (11),
  Dual-Head Hybrid (12), Content-Embeddings (13).
- **`14_advanced_eval.ipynb` — the final notebook.** Loads everything and produces the
  comparison leaderboard **plus** deep eval (NDCG/AUC, segmented RMSE, coverage/diversity/novelty,
  bootstrap CIs, cold-start, full-catalogue sanity) — no re-training.

---

## 4. The features (content representation)

Each movie becomes a **276-dimensional vector** = two blocks stacked:

1. **Genre block (~20 dims)** — multi-hot indicator (`Action=1, Comedy=0, …`).
   Interpretable, zero-cost.
2. **Text block (256 dims)** — TF-IDF over `clean_title + aggregated_tags` (bigrams,
   `min_df=5`, sublinear TF), compressed with **TruncatedSVD → 256 dims** (Latent Semantic
   Analysis: turns a huge sparse vocabulary into a dense semantic embedding).

Similarity between two movies = **cosine similarity** of these vectors.

---

## 5. The models

We built **12** (the assignment requires 3: one CB, one CF, one hybrid — so this is well
over-spec). The first 8 are the core; 9–12 are additive extensions. Full internals in
[`models.md`](models.md).

| # | Model | Family | One-liner |
|---|---|---|---|
| 1 | Global Mean | baseline | predict the training average for everything |
| 2 | Popularity | baseline | score by how many ratings an item has |
| 3 | **Content-Based** | CB | rate j by your ratings on its content-nearest neighbors |
| 4 | User-kNN | CF | "users like you rated j as…" |
| 5 | Item-kNN | CF | "items like j that you rated…" |
| 6 | **SVD** | CF | matrix factorization with bias terms |
| 7 | **Weighted Hybrid** | hybrid | `α·SVD + (1−α)·CB` |
| 8 | **Stacked Hybrid** | hybrid | Ridge meta-learner over all base predictions |
| 9 | Content (Genome) | CB · *ext* | content model on the tag genome (the big content lift) |
| 10 | LightGCN | graph CF · *ext* | message passing on the user–item graph, BPR loss (ranking-only) |
| 11 | **Dual-Head Hybrid** | hybrid · *ext* | Ridge rating head + logistic rank head over 5 base models |
| 12 | Content (Embeddings) | CB · *ext* | content model on sentence-transformer embeddings |

**Content-Based** — predicts via a *mean-centred weighted average*:
`r̂(u,j) = r̄_u + Σ sim(i,j)·(r_{u,i} − r̄_u) / Σ|sim|`, over the top-50 content-similar
items you've rated. (Engineering note: it densifies + L2-normalizes the matrix once so
cosine is a single fast BLAS matmul, with an LRU cache.)

**User-kNN / Item-kNN** — Surprise `KNNWithMeans`, Pearson-baseline similarity, k=80. To
fit in RAM, Item-kNN keeps only the 15K most-rated movies and User-kNN samples 20K users
(full similarity matrices would be ~15 GB / ~98 GB).

**SVD** — learns latent user/item factors + biases: `r̂ = μ + b_u + b_i + q_iᵀp_u`.
Grid-searched by 5-fold CV.

### How the hybrid combines the two (the part the report must explain)

- **Weighted Hybrid:** linear blend `α·SVD + (1−α)·CB`, α found by sweeping 0→1 on the
  validation set. If CB can't score an item (cold start), falls back to SVD. Simple,
  interpretable.
- **Stacked Hybrid:** a **Ridge meta-learner** that *learns from data* how to weight
  `[content, user-kNN, item-kNN, SVD]` predictions plus side features (popularity, rating
  counts). Trained on **5-fold out-of-fold** predictions so the meta-model never sees a
  base model predicting on its own training data (no leakage). More powerful, less
  interpretable.

---

## 6. Evaluation protocol

- **Rating metrics (RMSE, MAE):** over the full test set.
- **Ranking metrics (P/R/F1@K):** for 1,000 sampled test users, rank each user's relevant
  items against **100 random "negative" items**, K ∈ {5, 10, 20}, relevance ≥ 4.0
  (standard NCF/BPR-style sampled-negatives protocol).
- Hyperparameters tuned only on val/CV; test touched once.

---

## 7. Is it good? — the honest assessment

*(Numbers below are the current, corrected results from `all_metrics.json` — after the
tie-break fix and with all 12 models present. The full P/R/F1@K table is in
[`models.md`](models.md#results-on-the-held-out-test-set).)*

### Rating accuracy (RMSE/MAE) — ✅ clean, sensible, the hybrids lead

| Model | RMSE | MAE |
|---|---|---|
| **Dual-Head Hybrid** *(ext)* | **0.8028** | 0.6031 |
| **Stacked Hybrid** | 0.8054 | **0.6029** |
| Weighted Hybrid | 0.8095 | 0.6074 |
| SVD | 0.8108 | 0.608 |
| Item-kNN | 0.8336 | 0.6223 |
| Content (Genome) *(ext)* | 0.9670 | 0.7091 |
| Content (Embeddings) *(ext)* | 1.0292 | 0.7647 |
| User-kNN | 1.0401 | 0.8119 |
| Content-Based (TF-IDF) | 1.0462 | 0.7831 |
| Global Mean | 1.0609 | 0.8339 |
| Popularity | 2.712 | 2.4856 |

This is exactly the story the assignment wants: **both learned hybrids beat every CB-only
and CF-only baseline** on RMSE *and* MAE. The ordering is textbook — hybrids > SVD > Item-kNN
> content > user-kNN > naive baselines.

Two honest caveats: (a) the **Weighted Hybrid** beats SVD by only ~0.001 RMSE, because α tuned
almost entirely toward SVD (the TF-IDF content signal is weak), so it is ~90% SVD — it doesn't
*hurt* and adds cold-start coverage, but the lift is marginal. (b) The **Stacked and Dual-Head**
hybrids open a clearer gap precisely because they bring in *better* signals (Item-kNN, the genome
content model, LightGCN) and *learn* the weights — the meta-learner is what makes the hybrid
genuinely better than its best single base, not just a tie.

### Ranking (F1@K) — ✅ now sensible (tie-break fixed)

| Model | F1@10 |
|---|---|
| **LightGCN** *(ext, ranking-only)* | **0.621** |
| Popularity | 0.609 |
| Item-kNN | 0.433 |
| Dual-Head Hybrid *(ext)* | 0.424 |
| Stacked Hybrid | 0.418 |
| Content (Genome) *(ext)* | 0.376 |
| Weighted Hybrid | 0.354 |
| SVD | 0.352 |
| Content (Embeddings) *(ext)* | 0.320 |
| Content-Based (TF-IDF) | 0.207 |
| User-kNN | 0.122 |
| Global Mean | 0.076 |

After the fix this ranks the way it should: the **ranking-trained graph model (LightGCN) wins**,
the personalised CF/hybrid models cluster in the strong middle, and the constant predictor
(Global Mean) is correctly at the **bottom** (0.076, not the inflated 0.58 an earlier buggy run
reported). Two things to keep stating honestly:

1. **Popularity's high F1 is an artifact of the protocol, not personalisation.** In sampled
   negatives, popular titles genuinely are relevant more often, so a popularity sort scores
   well — but its RMSE (2.71) shows it predicts nothing user-specific. It's a baseline to beat
   on *both* axes, which the hybrids do.
2. **LightGCN's win comes with a coverage caveat.** It only scores items inside its training
   subgraph (10K-user subsample), so its candidate pool is narrower than the rating models'.
   It's the right tool for ranking, but not directly comparable on RMSE (it has none).
3. **RMSE-optimal ≠ ranking-optimal** remains the real lesson: SVD minimises squared error, not
   "push the few relevant items above 100 randoms," which is why a ranking-loss model (LightGCN)
   and the dual-head's dedicated rank head do better on F1. (Notebook 14 corroborates with
   NDCG@K and AUC.)

### What was fixed since the first draft

All three action items an earlier draft flagged are **done** (see [`CLAUDE.md`](../CLAUDE.md)
"Known bugs → Fixed"): the **Stacked Hybrid is in the results** (all 12 models present), the
**ranking tie-break is fixed** (candidates shuffled before the stable sort), and **NDCG@K + AUC**
are reported alongside F1@K in notebook 14.

### Overall verdict

The *engineering* is genuinely strong — a clean installable library, a leak-free protocol,
memory-aware CF, four hybrid/extension strategies, and a working two-tier app, well beyond the
3-model minimum. The *rating-accuracy* results are solid and tell the exact story the assignment
asks for (hybrids beat every single-family baseline). The *ranking* evaluation, after the
tie-break fix, is now defensible and even pedagogically nice (it shows why ranking-first methods
win on ranking). This is a strong submission.

---

## 8. What other approaches exist — graph / Neo4j / etc.

The repo is named `knowledge_graphs_ass`, so this is a natural question. Note: graph
methods aren't *required* by Θέμα 2 (they're closer to Θέμα 1, link prediction on a
bipartite graph) — treat these as **extensions / alternatives**, not gaps.

### Graph-based recommendation

Model the data as a graph: `(User)-[RATED]->(Movie)-[HAS_GENRE]->(Genre)`,
`(Movie)-[TAGGED]->(Tag)`. Recommendation becomes **link prediction** on the user–item
bipartite graph:

- **Heuristics:** Common Neighbors, Jaccard, Adamic-Adar (exactly Θέμα 1's option 2).
- **Node embeddings:** Node2Vec / DeepWalk → train a classifier on node-pair features.
- **Graph Neural Networks (current SOTA for CF):** **LightGCN** (✅ **implemented** — notebook
  11), NGCF, PinSage, GraphSAGE — learn user/item embeddings by message-passing over the
  interaction graph and optimize a *ranking* loss (BPR). These typically **beat plain matrix
  factorization on ranking metrics** — and indeed our LightGCN posts the **best F1@10 in the
  project (≈ 0.62)**, confirming the thesis. Remaining headroom: train it on the full user set
  (we subsample 10K) on a GPU.

### Neo4j specifically

Neo4j is a graph database with a **Graph Data Science (GDS)** library shipping
recommender-relevant algorithms:

- Store the data as a property graph (users, movies, genres, tags as nodes).
- **Node Similarity** (Jaccard/cosine over neighborhoods) → item-item or user-user CF
  directly in Cypher.
- **Link Prediction ML pipelines** — GDS has a built-in pipeline: generate
  `FastRP`/`GraphSAGE`/`Node2Vec` embeddings, train a link predictor, evaluate.
  Essentially Θέμα 1 turnkey.
- **PageRank / personalized PageRank** for popularity- and proximity-based recs.

### Knowledge Graph Embeddings (KGE)

Embed a richer movie knowledge graph (genres, tags, actors, directors via `links.csv` →
IMDb/TMDb) with **TransE / DistMult / ComplEx**, then score `(user, likes, movie)`
triples. Content + structure fused — a "knowledge-graph hybrid," very on-theme for the
course name.

### Non-graph improvements (smaller effort, on-topic for Θέμα 2)

- **Use the tag genome** — ✅ **done** (notebook 10). The 1,128 dense relevance scores/movie
  gave a much richer content signal than free-text tags (RMSE 1.046 → 0.967) and feed the
  Dual-Head Hybrid as its content base.
- **Optimize for ranking, not RMSE** — switch SVD → **BPR** (Bayesian Personalized
  Ranking) or add a learning-to-rank meta-model. Directly addresses the poor F1.
- **Neural CF / autoencoders** — NCF, Mult-VAE, or Factorization Machines (libFM) as extra
  CF bases.
- **Switching hybrid** — route to CB when the item is cold (few ratings) and to CF when
  it's well-observed, instead of a fixed blend.

---

## 9. Suggested next steps (priority order)

**Already done** (were the top-3 in an earlier draft): ✅ ranking tie-break fixed + all 12
models in the results · ✅ tag genome wired into content features (notebook 10, the biggest
content lift) · ✅ a ranking-optimised graph learner built (LightGCN, notebook 11) — plus
NDCG/AUC and a deep-eval study (notebook 14).

Remaining ideas, if the project were taken further:

1. **(lift)** Replace SVD with **BPR** (a ranking loss) as a base model, or add a
   learning-to-rank meta-head — directly targets the metric the rating models are weakest on.
2. **(scale)** Train LightGCN on the **full** user set (not the 10K subsample) on a GPU, to
   remove its candidate-coverage caveat and see its true ceiling.
3. **(extension)** A **Neo4j / GDS** knowledge-graph version (Node Similarity, Link-Prediction
   pipelines, personalised PageRank) — turnkey for the graph framing in §8.
4. **(serving)** Slim the Surprise models for serving (drop the retained trainset post-fit) so
   the big CF models fit alongside each other instead of one-at-a-time.
