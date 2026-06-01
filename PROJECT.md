# Project Brief — Hybrid Movie Recommender System

MSc AI — Εφαρμογές Τεχνητής Νοημοσύνης · Θέμα 2  
Konstantinos Mpouros · 2026

---

## Assignment goal

Design, implement, evaluate, and demonstrate a **hybrid recommender system** that combines
at least one content-based method with at least one collaborative-filtering method on a
real-world dataset, and expose the results through a web application.

---

## Dataset

**MovieLens 25M** (stable benchmark, not the "latest" variant):

| Statistic | Value |
| --- | --- |
| Ratings | 25,000,095 |
| Users | 162,541 |
| Movies | 62,423 |
| Rating scale | 0.5 – 5.0 (half-star increments) |
| Tag applications | 1,093,360 |
| Tag genome entries | ~15 M (1,128 tags × 13,176 movies) |

Files placed in `data/raw/`: `ratings.csv`, `movies.csv`, `tags.csv`,
`genome-scores.csv`, `genome-tags.csv`, `links.csv`.

---

## Architecture

### Package structure (`src/hybrid_recsys/`)

```text
config.py               global paths and constants (RATING_SCALE, RELEVANCE_THRESHOLD, K_VALUES)
serving.py              RecommenderBundle — loads all artifacts, exposes get_recommendations()

pipeline/
  data.py               raw CSV ingestion, genre / year / tag text extraction, save processed parquet
  splits.py             user-wise temporal split (80/10/10), min_train_ratings guard
  features.py           multi-hot genre matrix + TF-IDF tag text → TruncatedSVD(256) → hstack

models/
  content.py            ContentBasedRecommender — cosine similarity, mean-centred prediction
  collaborative.py      SVDModel (GridSearchCV tuning), ItemKNNModel, UserKNNModel (Surprise)
  hybrid.py             WeightedHybrid (alpha tuned on val RMSE), StackedHybrid (Ridge on OOF)

evaluation/
  metrics.py            rmse, mae, precision_at_k, recall_at_k, f1_at_k, evaluate_ranking
```

### Application (`src/app/app.py`)

Three-tab Streamlit application:

| Tab | Description |
| --- | --- |
| Existing User | Select a userId, choose a model, get top-K recommendations with history |
| New User | Rate 12 seed movies via slider, content model recommends immediately |
| Model Comparison | RMSE/MAE table + F1@10 bar chart from persisted metrics |

---

## Modeling decisions

### Content-Based

- **Features:** multi-hot genre vector (20 dims) concatenated with a 256-dim TruncatedSVD
  reduction of the TF-IDF matrix built from each movie's aggregated user tags.
- **Similarity:** cosine similarity between item feature vectors (sparse × dense).
- **Prediction:** mean-centred weighted average over the top-L similar items the user has rated.

### Collaborative Filtering

All CF models use [Surprise](https://surpriselib.com/) with `KNNWithMeans`
(Pearson baseline similarity) and `SVD` (with bias terms):

- **User-kNN:** k=80, min_k=5, user_based=True
- **Item-kNN:** k=80, min_k=5, user_based=False
- **SVD:** n_factors ∈ {50,100,200}, n_epochs ∈ {20,40}, lr ∈ {0.002,0.005},
  reg ∈ {0.02,0.05} — selected by 5-fold CV on train RMSE.

### Hybrid Fusion

| Model | Method |
| --- | --- |
| Weighted Hybrid | α·SVD + (1−α)·CB; α exhaustively searched on validation RMSE in steps of 0.05 |
| Stacked Hybrid | Ridge(α=1) meta-learner; features = OOF predictions of all 4 base models + item popularity + user/item rating counts; 5-fold OOF on train to prevent leakage |

---

## Evaluation protocol

- **Split:** user-wise temporal — each user's ratings sorted by timestamp,
  then divided 80/10/10. Users with fewer than 5 ratings are dropped.
- **Rating metrics:** RMSE, MAE (computed on the full test set, NaN predictions ignored).
- **Ranking metrics:** macro-averaged Precision@K, Recall@K, F1@K over a stratified
  sample of 1,000 test users, at K ∈ {5, 10, 20}. Relevance threshold: rating ≥ 4.0.
- **Primary selection metric:** F1@10.
- **No data leakage:** hyperparameters tuned on val/train-CV only; test evaluated once.

---

## Infrastructure

| Component | Tool |
| --- | --- |
| Package management | pyproject.toml + requirements.txt |
| Serialisation | joblib (.joblib) |
| Web app | Streamlit ≥ 1.30 |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions (pytest on push/PR) |
| Charts | Plotly (HTML + PNG via kaleido) |

---

## Reproducibility

Run the following in order to reproduce all results from scratch:

```bash
pip install -e .
# place MovieLens 25M CSVs in data/raw/
jupyter notebook  # run 01, 02, 03 in sequence
streamlit run src/app/app.py
```

All random seeds are fixed via `RANDOM_STATE = 42` in `config.py`.
