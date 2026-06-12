# Hybrid Movie Recommender System

MSc AI — Εφαρμογές Τεχνητής Νοημοσύνης · Θέμα 2  
Konstantinos Mpouros

---

## Overview

A hybrid recommender system built on the **MovieLens 25M** dataset (25 M ratings, 162 K users, 62 K movies).
It combines **content-based filtering** (cosine similarity on genre + TF-IDF item features) with
**collaborative filtering** (User-kNN, Item-kNN, SVD) into two fusion models — a weighted ensemble and
a Ridge meta-learner — plus four extension models (Content-Genome, Content-Embeddings, LightGCN, a
Dual-Head Hybrid). All 12 models are served through a **FastAPI** backend and a **Streamlit** front-end
(a thin HTTP client).

---

## Repository layout

```text
knowledge_graphs_ass/
├── hybrid_recsys/                   # installable ML library
│   ├── config.py                    # paths & global constants
│   ├── pipeline/
│   │   ├── data.py                  # raw CSV loading & preprocessing
│   │   ├── splits.py                # user-wise temporal train/val/test split
│   │   └── features.py              # item features (genres + TF-IDF/LSA; + genome, embeddings)
│   ├── models/
│   │   ├── content.py               # content-based item-item recommender
│   │   ├── collaborative.py         # SVD, ItemKNN, UserKNN (Surprise wrappers)
│   │   ├── hybrid.py                # WeightedHybrid, StackedHybrid, DualHeadHybrid
│   │   └── lightgcn.py              # LightGCN graph CF (PyTorch)
│   └── evaluation/
│       ├── metrics.py               # RMSE, MAE, P/R/F1@K, NDCG, AUC, coverage…
│       └── report.py                # shared notebook eval helpers
├── backend/                         # FastAPI REST service
│   ├── main.py                      # endpoints (recommend/compare/predict/explain/search/…)
│   ├── serving.py                   # RecommenderBundle — loads artifacts, serves all 12 models
│   └── schemas.py                   # pydantic request models
├── app/
│   └── app.py                       # Streamlit UI — thin HTTP client (6 tabs)
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis & preprocessing
│   ├── 02_features.ipynb            # item feature engineering
│   ├── 03_baselines.ipynb           # Global Mean & Popularity
│   ├── 04_content_based.ipynb       # content model: train + evaluate
│   ├── 05_user_knn.ipynb            # user-kNN: train + evaluate (+ neighbour graph)
│   ├── 06_item_knn.ipynb            # item-kNN: train + evaluate (+ neighbour graph)
│   ├── 07_svd.ipynb                 # SVD: train + evaluate (+ latent factor space)
│   ├── 08_weighted_hybrid.ipynb     # weighted hybrid: tune α + evaluate
│   ├── 09_stacked_hybrid.ipynb      # stacked hybrid: OOF train + evaluate
│   ├── 10_content_genome.ipynb      # extension: content model on the tag genome
│   ├── 11_lightgcn.ipynb            # extension: LightGCN (graph CF, PyTorch)
│   ├── 12_dual_head_hybrid.ipynb    # extension: dual-head hybrid (rating + ranking)
│   ├── 13_semantic_content.ipynb    # extension: content model on sentence-transformer embeddings
│   └── 14_advanced_eval.ipynb       # FINAL: comparison + NDCG/AUC + segmented + diversity
│                                    #        + bootstrap CIs + cold-start + full-catalogue
├── tests/
│   ├── test_splits.py
│   └── test_metrics.py
├── data/
│   ├── raw/                         # ← place MovieLens 25M CSVs here (gitignored)
│   └── processed/                   # generated parquet files & splits (gitignored)
├── artifacts/
│   ├── models/                      # saved .joblib model files (gitignored)
│   ├── metrics/                     # all_metrics.json (gitignored)
│   └── figures/                     # exported Plotly charts
├── docs/                            # design documents & plan
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Models

| Model | Type | Library |
| --- | --- | --- |
| Global Mean | Naive baseline | — |
| Popularity | Naive baseline | — |
| Content-Based | Cosine similarity on genre + TF-IDF/LSA | scikit-learn |
| User-Based k-NN | CF, Pearson baseline similarity | Surprise |
| Item-Based k-NN | CF, Pearson baseline similarity | Surprise |
| SVD | Matrix factorisation with bias terms | Surprise |
| **Weighted Hybrid** | α·SVD + (1−α)·CB, α tuned on validation | custom |
| **Stacked Hybrid** | Ridge meta-learner on out-of-fold predictions | scikit-learn |

**Extension models** (additive — the 8 above stay frozen):

| Model | Type | Library |
| --- | --- | --- |
| Content — Tag Genome | Content on genre ⊕ SVD(tag-genome) | scikit-learn |
| Content — Embeddings | Content on sentence-transformer embeddings | sentence-transformers |
| LightGCN | Graph CF, BPR loss (ranking-only) | PyTorch |
| **Dual-Head Hybrid** | Ridge rating head + logistic rank head over all base models | scikit-learn |

---

## Evaluation protocol

| Setting | Value |
| --- | --- |
| Split strategy | User-wise temporal (80 / 10 / 10) |
| Rating metrics | RMSE, MAE |
| Ranking metrics | Precision@K, Recall@K, F1@K |
| K values | 5, 10, 20 |
| Relevance threshold | rating ≥ 4.0 |
| Primary metric | F1@10 |

---

## Quick start

```bash
# 1. Install (requires Python ≥ 3.10 and Microsoft C++ Build Tools on Windows)
pip install -e .

# 2. Download MovieLens 25M and place the CSVs in data/raw/
#    https://grouplens.org/datasets/movielens/25m/

# 3. Run notebooks in order
jupyter notebook
#    01_eda → 02_features → 03_baselines → 04…09 (one per model)
#    → 10_content_genome → 11_lightgcn → 12_dual_head_hybrid → 13_semantic_content
#    → 14_advanced_eval (final: deep eval + comparison)

# 4. Launch the app — TWO processes (backend API + Streamlit UI)
#    Easiest on Windows: ./run_local.ps1   (starts both)
#    Or manually, in two terminals:
python -m uvicorn backend.main:app --port 8000     # backend  → http://localhost:8000/docs
python -m streamlit run app/app.py                 # UI       → http://localhost:8501
```

> Use `python -m uvicorn` / `python -m streamlit` (not the bare `uvicorn`/`streamlit`
> commands) so they run under the same interpreter that has `scikit-surprise` installed.

**Memory — on-demand loading.** Models load **only when you click "Load"** (sidebar
or inline) — the backend boots with just the catalogue (~1 GB) and no models. This
matters because the Surprise models are huge in RAM: **SVD ≈ 6.8 GB**, the k-NN
models larger still (they retain the full 20M-rating trainset), while each content
model is only ~0.3 GB. Load just the models you want to demo; recommend/predict
endpoints return `409 {status: not_loaded}` until the chosen model is loaded.

Optional startup preload via `RECSYS_MODELS` (default: none):

```bash
RECSYS_MODELS=          # default — preload nothing, load on demand
RECSYS_MODELS=lite      # preload content models + LightGCN (~2 GB)
RECSYS_MODELS=all       # preload everything (needs ~20 GB free)
RECSYS_MODELS=svd,content_genome   # preload a custom set
```

The Comparison tab always shows metrics for all 12 (from `all_metrics.json`,
independent of what's loaded).

### API

The backend exposes a typed REST API (interactive docs at `/docs`):
`/api/recommend` · `/api/recommend/compare` · `/api/predict` · `/api/explain` ·
`/api/movies/search` · `/api/movies/{id}/similar` · `/api/users/{id}/profile` ·
`/api/metrics` · `/api/figures`.

### Docker (Linux / Mac)

```bash
# Build and serve both services: backend (:8000) + Streamlit app (:8501)
docker compose up --build
```

> The Docker image handles all compilation automatically via `build-essential`.
> You still need to run the notebooks locally first to generate `data/processed/` and `artifacts/`.

---

## Tests

```bash
pytest tests/ -v
```

21 tests covering temporal split correctness and all metric functions.

---

## Dependencies

Key packages — full list in [`requirements.txt`](requirements.txt):

- `scikit-learn` — TF-IDF, TruncatedSVD, cosine similarity, Ridge
- `scikit-surprise` — KNNWithMeans, SVD, GridSearchCV (requires `numpy<2`)
- `pandas` / `numpy` / `scipy` — data manipulation and sparse matrices
- `fastapi` / `uvicorn` — REST backend serving all 12 models
- `streamlit` / `requests` — web UI (HTTP client)
- `torch` — LightGCN · `sentence-transformers` — embedding content model
- `plotly` — interactive charts
- `joblib` — model serialisation

> **Windows note:** `scikit-surprise` requires Microsoft C++ Build Tools.
> Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and
> select "Desktop development with C++".
> NumPy must be `< 2.0` due to a binary compatibility constraint in scikit-surprise.

---

## Dataset

MovieLens 25M — F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems, 5(4):19.
