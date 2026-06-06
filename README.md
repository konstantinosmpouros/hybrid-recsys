# Hybrid Movie Recommender System

MSc AI — Εφαρμογές Τεχνητής Νοημοσύνης · Θέμα 2  
Konstantinos Mpouros

---

## Overview

A hybrid recommender system built on the **MovieLens 25M** dataset (25 M ratings, 162 K users, 62 K movies).
It combines **content-based filtering** (cosine similarity on genre + TF-IDF item features) with
**collaborative filtering** (User-kNN, Item-kNN, SVD) into two fusion models — a weighted ensemble and
a Ridge meta-learner — and exposes everything through a **Streamlit** web application.

---

## Repository layout

```text
knowledge_graphs_ass/
├── src/
│   ├── hybrid_recsys/               # installable library package
│   │   ├── config.py                # paths & global constants
│   │   ├── serving.py               # RecommenderBundle — loads artifacts for the app
│   │   ├── pipeline/
│   │   │   ├── data.py              # raw CSV loading & preprocessing
│   │   │   ├── splits.py            # user-wise temporal train/val/test split
│   │   │   └── features.py          # item feature matrix (genres + TF-IDF + LSA)
│   │   ├── models/
│   │   │   ├── content.py           # content-based item-item recommender
│   │   │   ├── collaborative.py     # SVD, ItemKNN, UserKNN (Surprise wrappers)
│   │   │   └── hybrid.py            # WeightedHybrid & StackedHybrid
│   │   └── evaluation/
│   │       └── metrics.py           # RMSE, MAE, Precision/Recall/F1@K
│   └── app/
│       └── app.py                   # Streamlit app (3 tabs)
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis & preprocessing
│   ├── 02_features.ipynb            # item feature engineering
│   ├── 03_train.ipynb               # model training (fits & saves all models)
│   └── 04_evaluation.ipynb          # model evaluation (RMSE/MAE/P/R/F1@K)
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
#    01_eda.ipynb  →  02_features.ipynb  →  03_train.ipynb  →  04_evaluation.ipynb

# 4. Launch the app
streamlit run src/app/app.py
```

### Docker (Linux / Mac)

```bash
# Build and serve the app (mount pre-built artifacts from the host)
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
- `streamlit` — web application
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
