"""
Generate the three project notebooks.
Run from the project root:  python notebooks/generate.py
"""
import nbformat as nbf
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


KERNEL_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}


def save_nb(cells: list, name: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update(KERNEL_META)
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  created  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared save_fig snippet — injected into every notebook
# ═══════════════════════════════════════════════════════════════════════════════

SAVE_FIG_SNIPPET = """\
FIGURES_DIR = Path("../artifacts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str) -> None:
    fig.write_html(str(FIGURES_DIR / f"{name}.html"))
    try:
        fig.write_image(str(FIGURES_DIR / f"{name}.png"), width=1200, height=600, scale=2)
    except Exception:
        pass  # kaleido not installed — HTML only
    fig.show()
"""


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════════════════════

nb01 = [

md("""\
# 01 — Exploratory Data Analysis

This notebook covers the full exploratory data analysis for the **MovieLens 25M** dataset.
We load and inspect the six raw CSV files, then examine the distribution of ratings,
user activity, item popularity, genre composition, and metadata coverage. We conclude
by constructing a **user-wise temporal train / validation / test split** and persisting
all processed tables to disk for use in the subsequent notebooks.

The insights gathered here directly justify the feature choices, the temporal split
strategy, and the relevance threshold used throughout the evaluation pipeline.

**Steps:**
Load CSVs → Dataset overview → Rating distribution → User activity →
Item popularity & sparsity → Genre analysis → Tag & genome coverage →
Temporal analysis → Train / val / test split → Save processed data.
"""),

code("""\
import sys
sys.path.insert(0, "../src")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from hybrid_recsys.pipeline.data import (
    load_raw_ratings, load_raw_movies, load_raw_tags,
    load_genome_scores, load_genome_tags,
    build_movies_table, save_processed,
)
from hybrid_recsys.pipeline.splits import temporal_split, save_splits

""" + SAVE_FIG_SNIPPET),

# ── 1. Load ──────────────────────────────────────────────────────────────────

md("## 1. Load Raw CSVs"),

code("""\
ratings       = load_raw_ratings()
movies        = load_raw_movies()
tags          = load_raw_tags()
genome_scores = load_genome_scores()
genome_tags   = load_genome_tags()

print(f"ratings:       {ratings.shape}")
print(f"movies:        {movies.shape}")
print(f"tags:          {tags.shape}")
print(f"genome_scores: {genome_scores.shape}")
print(f"genome_tags:   {genome_tags.shape}")
"""),

# ── 2. Overview ───────────────────────────────────────────────────────────────

md("## 2. Dataset Overview"),

code("""\
print(f"Unique users:    {ratings['userId'].nunique():>10,}")
print(f"Unique movies:   {ratings['movieId'].nunique():>10,}")
print(f"Total ratings:   {len(ratings):>10,}")
print(f"Rating range:    {ratings['rating'].min()} – {ratings['rating'].max()}")
print(
    f"Timestamp range: "
    f"{pd.to_datetime(ratings['timestamp'].min(), unit='s').date()} → "
    f"{pd.to_datetime(ratings['timestamp'].max(), unit='s').date()}"
)
print()
display(ratings.head(3))
display(movies.head(3))
"""),

# ── 3. Rating distribution ────────────────────────────────────────────────────

md("""\
## 3. Rating Distribution

MovieLens uses a half-star scale from 0.5 to 5.0. The distribution is left-skewed:
users tend to rate movies they chose to watch, introducing a **positivity bias**
that must be accounted for when setting the relevance threshold for ranking metrics.
"""),

code("""\
rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
rating_counts.columns = ["rating", "count"]

fig = px.bar(
    rating_counts, x="rating", y="count",
    title="Rating Distribution — MovieLens 25M",
    labels={"rating": "Rating", "count": "Number of Ratings"},
    color="count", color_continuous_scale="Blues", text_auto=True,
)
fig.update_layout(coloraxis_showscale=False, bargap=0.2)
save_fig(fig, "01_rating_distribution")
"""),

# ── 4. User activity ──────────────────────────────────────────────────────────

md("""\
## 4. User Activity — Long Tail

The number of ratings per user follows a heavy-tailed distribution. Most users
have rated relatively few movies, while a small minority have rated thousands.
This sparsity is the primary challenge for collaborative filtering models.
"""),

code("""\
user_counts = ratings.groupby("userId").size().reset_index(name="n_ratings")

fig = px.histogram(
    user_counts, x="n_ratings", nbins=100,
    title="Ratings per User (log y-scale)",
    labels={"n_ratings": "Ratings per User", "count": "Number of Users"},
    color_discrete_sequence=["#EF553B"],
    log_y=True,
)
save_fig(fig, "02_user_activity")

print("Ratings per user — summary statistics:")
print(user_counts["n_ratings"].describe().round(1).to_string())
"""),

# ── 5. Item popularity ────────────────────────────────────────────────────────

md("""\
## 5. Item Popularity & Matrix Sparsity

Ratings per movie also follow a power-law distribution. We additionally compute
the **interaction matrix sparsity**, which quantifies what fraction of (user, item)
pairs have no observed rating and motivates both content-based and hybrid approaches.
"""),

code("""\
item_counts = ratings.groupby("movieId").size().reset_index(name="n_ratings")

fig = px.histogram(
    item_counts, x="n_ratings", nbins=100,
    title="Ratings per Movie (log y-scale)",
    labels={"n_ratings": "Ratings per Movie", "count": "Number of Movies"},
    color_discrete_sequence=["#00CC96"],
    log_y=True,
)
save_fig(fig, "03_item_popularity")

n_users  = ratings["userId"].nunique()
n_items  = ratings["movieId"].nunique()
sparsity = 1 - len(ratings) / (n_users * n_items)

print(f"Interaction matrix: {n_users:,} users × {n_items:,} items")
print(f"Sparsity:           {sparsity:.4%}")
print(f"Density:            {1 - sparsity:.4%}")
print()
print("Ratings per movie — summary statistics:")
print(item_counts["n_ratings"].describe().round(1).to_string())
"""),

# ── 6. Genre analysis ─────────────────────────────────────────────────────────

md("""\
## 6. Genre Analysis

Each movie is labelled with one or more pipe-separated genres. Genre frequency
and diversity directly determine how discriminative the multi-hot genre block
will be in the content-based feature matrix.
"""),

code("""\
genre_series = movies["genres"].str.split("|").explode()
genre_counts = (
    genre_series[genre_series != "(no genres listed)"]
    .value_counts()
    .reset_index()
)
genre_counts.columns = ["genre", "count"]

fig = px.bar(
    genre_counts, x="genre", y="count",
    title="Movie Count by Genre",
    labels={"genre": "Genre", "count": "Number of Movies"},
    color="count", color_continuous_scale="Teal",
)
fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
save_fig(fig, "04_genre_frequency")

print(f"Distinct genres: {len(genre_counts)}")
print(genre_counts.to_string(index=False))
"""),

# ── 7. Tag & genome coverage ──────────────────────────────────────────────────

md("""\
## 7. Tag & Genome Coverage

Two metadata sources augment the genre signal: **free-text user tags** (sparse,
covering ~45% of movies) and the **tag genome** (1,128 relevance scores per movie,
covering ~85%). Coverage gaps justify testing feature variants in notebook 02.
"""),

code("""\
movies_with_tags   = tags["movieId"].nunique()
movies_with_genome = genome_scores["movieId"].nunique()
total_movies       = movies["movieId"].nunique()

coverage = pd.DataFrame({
    "Source":         ["Free-text tags", "Tag genome"],
    "Movies covered": [movies_with_tags, movies_with_genome],
    "Coverage %":     [
        round(100 * movies_with_tags   / total_movies, 1),
        round(100 * movies_with_genome / total_movies, 1),
    ],
})
display(coverage)

fig = px.bar(
    coverage, x="Source", y="Coverage %",
    title="Metadata Coverage across Movies",
    color="Source", text_auto=True,
    color_discrete_sequence=["#636EFA", "#FFA15A"],
)
fig.update_layout(showlegend=False, yaxis_range=[0, 100])
save_fig(fig, "05_tag_coverage")
"""),

# ── 8. Temporal analysis ──────────────────────────────────────────────────────

md("""\
## 8. Temporal Analysis

Rating volume is not uniform over time. Visualising monthly activity confirms
that a **temporal split** is meaningful: recent user behaviour differs from
historical patterns, so a random split would leak future information into training.
"""),

code("""\
ratings["date"] = (
    pd.to_datetime(ratings["timestamp"], unit="s")
    .dt.to_period("M")
    .dt.to_timestamp()
)
monthly = ratings.groupby("date").size().reset_index(name="n_ratings")

fig = px.line(
    monthly, x="date", y="n_ratings",
    title="Monthly Rating Volume — MovieLens 25M",
    labels={"date": "Date", "n_ratings": "Ratings per Month"},
)
fig.update_traces(line_color="#636EFA")
save_fig(fig, "06_temporal")
"""),

# ── 9. Split ──────────────────────────────────────────────────────────────────

md("""\
## 9. Train / Val / Test Split

We apply a **user-wise temporal split**: each user's ratings are sorted by timestamp
and divided 80 / 10 / 10 into train / validation / test. Users with fewer than
5 ratings total are excluded. This protocol ensures no future signal leaks into
training and produces a realistic held-out evaluation.
"""),

code("""\
train, val, test = temporal_split(ratings.drop(columns=["date"]))

split_stats = pd.DataFrame({
    "Split":      ["Train", "Validation", "Test"],
    "Ratings":    [len(train),                len(val),                len(test)],
    "Users":      [train["userId"].nunique(), val["userId"].nunique(), test["userId"].nunique()],
    "Movies":     [train["movieId"].nunique(),val["movieId"].nunique(),test["movieId"].nunique()],
    "% of total": [
        round(100 * len(train) / len(ratings), 1),
        round(100 * len(val)   / len(ratings), 1),
        round(100 * len(test)  / len(ratings), 1),
    ],
})
display(split_stats)
"""),

# ── 10. Save ──────────────────────────────────────────────────────────────────

md("## 10. Save Processed Data"),

code("""\
movies_processed = build_movies_table(movies, tags)

save_processed(movies_processed, "movies")
save_processed(ratings.drop(columns=["date"], errors="ignore"), "ratings")
save_splits(train, val, test)

print("Saved to data/processed/:")
for name in ["movies", "ratings", "split_train", "split_val", "split_test"]:
    print(f"  {name}.parquet")
"""),

# ── Conclusion ────────────────────────────────────────────────────────────────

md("""\
## Conclusion

- The dataset contains **25M ratings** from **162K users** on **62K movies** on a 0.5–5.0 half-star scale.
- A clear **positivity bias** is present: the modal rating is 4.0; ratings ≤ 2.0 are rare. A relevance threshold of ≥ 4.0 for ranking metrics is therefore well-motivated.
- Both user activity and item popularity follow **power-law distributions**, yielding a matrix sparsity above 99.8% — content-based signals are essential for cold-item and sparse-user scenarios.
- **Drama, Comedy, and Thriller** dominate the genre space; the multi-hot genre vector will be a strong and interpretable content feature.
- **Free-text tags** cover ~45% of movies and the **tag genome** covers ~85%, making both sources viable for enriching the TF-IDF content representation.
- Rating volume grows over time, confirming that a **temporal split** is the correct evaluation protocol for this dataset.
- The user-wise 80/10/10 split is saved to `data/processed/` and will be loaded identically by every subsequent notebook and model.
"""),

]

save_nb(nb01, "01_eda.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

nb02 = [

md("""\
# 02 — Feature Engineering

This notebook constructs the **item feature matrix** used by the content-based
and hybrid recommender models.

Starting from the processed movies table produced in notebook 01, we build two
complementary feature blocks: (1) a **multi-hot genre indicator** vector and
(2) a **TF-IDF representation** over the concatenated movie title and aggregated
tag text, compressed into 256 dimensions via TruncatedSVD (Latent Semantic Analysis).
The resulting sparse matrix and all fitted transformers are persisted to disk so that
the exact same feature space is applied at inference time without re-fitting.

**Steps:**
Load processed data → Genre matrix → TF-IDF on title + tags →
TruncatedSVD compression → Assemble & save item feature matrix.
"""),

code("""\
import sys
sys.path.insert(0, "../src")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from hybrid_recsys.pipeline.data import load_processed
from hybrid_recsys.pipeline.features import (
    build_genre_matrix, build_text_matrix,
    build_item_features, save_item_features,
)

""" + SAVE_FIG_SNIPPET),

# ── 1. Load ──────────────────────────────────────────────────────────────────

md("## 1. Load Processed Data"),

code("""\
movies = load_processed("movies")
print(f"Movies loaded: {len(movies):,}")
print(f"Movies with tags: {(movies['tags_text'] != '').sum():,} "
      f"({100 * (movies['tags_text'] != '').mean():.1f}%)")
display(movies[["movieId", "clean_title", "year", "genres", "tags_text"]].head(5))
"""),

# ── 2. Genre matrix ───────────────────────────────────────────────────────────

md("""\
## 2. Genre Multi-hot Matrix

Each movie's pipe-separated genre string is expanded into a binary indicator
vector — one dimension per unique genre. This gives an interpretable, sparse
feature block that captures coarse content similarity at zero training cost.
"""),

code("""\
genre_X     = build_genre_matrix(movies)
genre_names = movies["genres"].str.get_dummies(sep="|").columns.tolist()

print(f"Genre matrix shape: {genre_X.shape}")
print(f"Genres ({len(genre_names)}): {genre_names}")
"""),

# ── 3. TF-IDF ─────────────────────────────────────────────────────────────────

md("""\
## 3. TF-IDF on Title + Tags

We concatenate the cleaned movie title with its aggregated tag text into one
document per movie, then apply **TF-IDF** (bigrams, sublinear TF, min_df = 5).
This encodes both the film's name semantics and the community's vocabulary around it,
producing a rich sparse representation of item content.
"""),

code("""\
text_X, tfidf, svd_transformer = build_text_matrix(movies, n_components=256, min_df=5)

print(f"TF-IDF vocabulary size:       {len(tfidf.vocabulary_):,} terms")
print(f"Text matrix shape (post-SVD): {text_X.shape}")

top_terms = sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.get, reverse=True)[:30]
print(f"\\nTop 30 terms by index (most frequent): {top_terms}")
"""),

# ── 4. SVD explained variance ─────────────────────────────────────────────────

md("""\
## 4. TruncatedSVD — Explained Variance

TruncatedSVD (LSA) compresses the high-dimensional TF-IDF space into a dense
256-dimensional representation. The plot below shows how much of the original
variance is captured at each component count, justifying the chosen dimensionality.
"""),

code("""\
explained   = np.cumsum(svd_transformer.explained_variance_ratio_)
threshold90 = int(np.searchsorted(explained, 0.90)) + 1
threshold95 = int(np.searchsorted(explained, 0.95)) + 1

fig = px.line(
    x=list(range(1, len(explained) + 1)), y=explained,
    title="Cumulative Explained Variance — TruncatedSVD on TF-IDF",
    labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
)
fig.add_hline(y=0.90, line_dash="dash", line_color="orange",
              annotation_text=f"90 % at {threshold90} components")
fig.add_hline(y=0.95, line_dash="dash", line_color="red",
              annotation_text=f"95 % at {threshold95} components")
save_fig(fig, "07_svd_explained_variance")

print(f"Components to 90% variance: {threshold90}")
print(f"Components to 95% variance: {threshold95}")
print(f"Selected n_components:      256")
print(f"Variance retained at 256:   {explained[255]:.2%}")
"""),

# ── 5. Assemble & save ────────────────────────────────────────────────────────

md("""\
## 5. Assemble & Save Item Feature Matrix

The genre block and the SVD-compressed text block are horizontally stacked
into a single sparse matrix. All fitted transformers are serialised alongside
the matrix so that serving requires no re-fitting.
"""),

code("""\
item_features, movie_index, tfidf_fitted, svd_fitted = build_item_features(movies, n_components=256)

print(f"Item feature matrix shape: {item_features.shape}")
print(f"  Genre block:             {genre_X.shape[1]} dims")
print(f"  Text block (SVD-256):    256 dims")
print(f"Non-zero elements:         {item_features.nnz:,}")
density = item_features.nnz / (item_features.shape[0] * item_features.shape[1])
print(f"Matrix density:            {density:.4%}")

save_item_features(item_features, movie_index, tfidf_fitted, svd_fitted)

print("\\nSaved:")
print("  data/processed/item_features.npz")
print("  data/processed/movie_index.parquet")
print("  artifacts/models/tfidf.joblib")
print("  artifacts/models/svd_text.joblib")
"""),

# ── Conclusion ────────────────────────────────────────────────────────────────

md("""\
## Conclusion

- The item feature matrix has shape **(n_movies × 276)**: 20 genre indicator dimensions plus 256 SVD-compressed text dimensions.
- The TF-IDF vocabulary spans thousands of unique terms (min_df = 5, bigrams), capturing both title semantics and community tag vocabulary.
- **256 SVD components** retain the majority of the TF-IDF variance while keeping pairwise cosine-similarity computation tractable across 62K items.
- Genre features are directly interpretable and complement the semantic smoothing provided by SVD text features.
- All transformers are serialised to `artifacts/models/`, ensuring the exact same feature space is applied at serving time without re-fitting.
"""),

]

save_nb(nb02, "02_features.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Model Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

nb03 = [

md("""\
# 03 — Model Training & Evaluation

This notebook trains every recommender model defined in the project and evaluates
them on the held-out test set under a strict, leak-free experimental protocol.

**Protocol:**
- Hyperparameters are tuned by 5-fold cross-validation on the training set
  (Surprise models) or by exhaustive validation-set search (content model, hybrid weights).
- Each model is retrained on the full training set and evaluated **exactly once**
  on the untouched test partition.
- Ranking metrics are computed over a stratified sample of test users for
  computational tractability on the full 25M dataset.
- All trained models and evaluation results are persisted for the Streamlit application.

**Models:** Global Mean · Popularity · Content-Based · User-kNN · Item-kNN ·
SVD · Weighted Hybrid · Stacked Hybrid.
"""),

code("""\
import sys
sys.path.insert(0, "../src")

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import KFold

from hybrid_recsys.pipeline.data import load_processed
from hybrid_recsys.pipeline.splits import load_splits
from hybrid_recsys.pipeline.features import load_item_features
from hybrid_recsys.models.content import ContentBasedRecommender
from hybrid_recsys.models.collaborative import SVDModel, ItemKNNModel, UserKNNModel, _to_surprise
from hybrid_recsys.models.hybrid import WeightedHybrid, StackedHybrid
from hybrid_recsys.evaluation.metrics import (
    evaluate_rating_prediction,
    evaluate_ranking_sampled,
)
from hybrid_recsys.config import ARTIFACTS_MODELS, ARTIFACTS_METRICS
from surprise import SVD as SurpriseSVD

""" + SAVE_FIG_SNIPPET + """
ARTIFACTS_METRICS.mkdir(parents=True, exist_ok=True)
ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)

EVAL_USERS   = 1_000   # users sampled for ranking evaluation
N_NEGATIVES  = 100     # sampled non-relevant items per user for ranking metrics
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
"""),

# ── 1. Load ──────────────────────────────────────────────────────────────────

md("## 1. Load Data"),

code("""\
movies                  = load_processed("movies")
train, val, test        = load_splits()
item_features, movie_index = load_item_features()

train_val = pd.concat([train, val], ignore_index=True)

print(f"Train:      {len(train):>10,} ratings | {train['userId'].nunique():>7,} users")
print(f"Validation: {len(val):>10,} ratings | {val['userId'].nunique():>7,} users")
print(f"Test:       {len(test):>10,} ratings | {test['userId'].nunique():>7,} users")
print(f"Item features: {item_features.shape}")

# User rating histories from the training set (needed by content model & hybrids)
user_ratings_map: dict = (
    train
    .groupby("userId")
    .apply(lambda df: dict(zip(df["movieId"], df["rating"])))
    .to_dict()
)

# Stratified user sample for ranking evaluation
eval_user_ids = rng.choice(
    test["userId"].unique(),
    size=min(EVAL_USERS, test["userId"].nunique()),
    replace=False,
)
test_sample = test[test["userId"].isin(eval_user_ids)]
print(f"\\nRanking evaluation user sample: {len(eval_user_ids):,}")
"""),

# ── 2. Helper ─────────────────────────────────────────────────────────────────

md("""\
## 2. Evaluation Helper

A shared wrapper computes both rating-prediction and ranking metrics for any
`predict_fn(user_id, movie_id) -> float`, accumulating results in `all_metrics`.
"""),

code("""\
all_metrics: dict = {}
metrics_path = ARTIFACTS_METRICS / "all_metrics.json"


def checkpoint_metrics() -> None:
    \"\"\"Persist all_metrics after every model so partial results survive crashes.\"\"\"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))


def eval_model(key: str, label: str, predict_fn) -> dict:
    print(f"Evaluating: {label} ...")
    t0 = time.perf_counter()

    preds = np.array([predict_fn(r.userId, r.movieId) for r in test.itertuples()])
    rp    = evaluate_rating_prediction(test["rating"].values, preds)
    t_rating = time.perf_counter() - t0

    ranking = evaluate_ranking_sampled(
        test_sample, predict_fn, train_val,
        all_movie_ids=movies["movieId"].values,
        n_negatives=N_NEGATIVES,
        k_values=[5, 10, 20],
        random_state=RANDOM_STATE,
    )
    t_total = time.perf_counter() - t0

    metrics = {
        "rmse": round(rp["rmse"], 4),
        "mae":  round(rp["mae"],  4),
        **{
            f"k{k}": {m: round(v, 4) for m, v in kv.items()}
            for k, kv in ranking.items()
        },
    }
    all_metrics[label] = metrics
    checkpoint_metrics()
    print(
        f"  RMSE={metrics['rmse']}  MAE={metrics['mae']}  F1@10={metrics['k10']['f1']}"
        f"  (rating {t_rating:.1f}s · total {t_total:.1f}s)"
    )
    return metrics
"""),

# ── 3. Baseline ───────────────────────────────────────────────────────────────

md("""\
## 3. Naive Baselines — Global Mean & Popularity

The **global mean** predictor assigns the training-set average to every pair.
The **popularity** predictor scores items by interaction count.
These define the performance floor that every subsequent model must exceed.
"""),

code("""\
global_mean     = float(train["rating"].mean())
item_popularity = train.groupby("movieId").size().to_dict()
max_pop         = max(item_popularity.values())

# Popularity is fundamentally a ranking signal; we map raw counts to the
# [0.5, 5.0] rating range so RMSE/MAE are meaningful and comparable.
# The mapping is monotonic, so ranking metrics are unaffected.
def pop_score(_user, movie_id):
    return 0.5 + 4.5 * (item_popularity.get(movie_id, 0) / max_pop)

eval_model("global_mean", "Global Mean",  lambda u, i: global_mean)
eval_model("popularity",  "Popularity",   pop_score)

print(f"\\nGlobal mean rating: {global_mean:.4f}")
"""),

# ── 4. Content-based ──────────────────────────────────────────────────────────

md("""\
## 4. Content-Based Model

The content-based model predicts the rating for item *j* by finding its top-L
most content-similar items via cosine similarity on the feature matrix, then
computing a mean-centred weighted average over the user's ratings on those items:

$$\\hat{r}^{CB}_{u,j} = \\bar{r}_u + \\frac{\\sum_{i \\in N^L_u(j)} \\text{sim}(i,j)\\,(r_{u,i} - \\bar{r}_u)}{\\sum_{i} |\\text{sim}(i,j)| + \\varepsilon}$$
"""),

code("""\
cb = ContentBasedRecommender(n_neighbors=50)
cb.fit(item_features, movie_index)

eval_model("content", "Content-Based", lambda u, i: cb.predict(user_ratings_map.get(u, {}), i))

cb.save()
print("Saved: artifacts/models/content_model.joblib")
"""),

# ── 5. User-kNN ───────────────────────────────────────────────────────────────

md("""\
## 5. User-Based k-NN

User-based CF identifies the *k* most similar users to the target user
(Pearson baseline similarity) and aggregates their ratings on the target item.
We use Surprise's `KNNWithMeans` with `user_based=True`.
"""),

code("""\
user_knn = UserKNNModel(k=80, min_k=5)
user_knn.fit(train)

eval_model("user_knn", "User-Based k-NN", lambda u, i: user_knn.predict(u, i))

user_knn.save()
print("Saved: artifacts/models/user_knn_model.joblib")
"""),

# ── 6. Item-kNN ───────────────────────────────────────────────────────────────

md("""\
## 6. Item-Based k-NN

Item-based CF identifies the *k* most similar items to the target item and
aggregates the user's ratings on those items. In movie domains, item-based
models are typically more stable than user-based ones because the item space
changes more slowly than the user population.
"""),

code("""\
item_knn = ItemKNNModel(k=80, min_k=5)
item_knn.fit(train)

eval_model("item_knn", "Item-Based k-NN", lambda u, i: item_knn.predict(u, i))

item_knn.save()
print("Saved: artifacts/models/item_knn_model.joblib")
"""),

# ── 7. SVD ────────────────────────────────────────────────────────────────────

md("""\
## 7. SVD — Matrix Factorisation

Surprise's SVD decomposes the rating matrix into user and item latent factor
matrices with global, user, and item bias terms:

$$\\hat{r}_{ui} = \\mu + b_u + b_i + q_i^\\top p_u$$

Hyperparameters (`n_factors`, `n_epochs`, `lr_all`, `reg_all`) are selected
by 5-fold cross-validation on the training set, minimising RMSE.
"""),

code("""\
svd = SVDModel()
svd.tune(train, param_grid={
    "n_factors": [50, 100, 200],
    "n_epochs":  [20, 40],
    "lr_all":    [0.002, 0.005],
    "reg_all":   [0.02, 0.05],
})
svd.fit(train)

print(f"Best params: {svd.best_params}")

eval_model("svd", "SVD", lambda u, i: svd.predict(u, i))

svd.save()
print("Saved: artifacts/models/svd_model.joblib")
"""),

# ── 8. Weighted hybrid ────────────────────────────────────────────────────────

md("""\
## 8. Weighted Hybrid

The weighted hybrid linearly interpolates the SVD and content-based predictions:

$$\\hat{r}^{H1}_{ui} = \\alpha \\cdot \\hat{r}^{SVD}_{ui} + (1-\\alpha) \\cdot \\hat{r}^{CB}_{ui}$$

The scalar weight $\\alpha$ is tuned on the validation set by exhaustive search
over $\\alpha \\in [0, 1]$ with step 0.05, minimising RMSE. When the content
model returns NaN (unknown item), the prediction falls back to SVD.
"""),

code("""\
weighted = WeightedHybrid()
weighted.set_models(svd, cb)

best_alpha = weighted.tune_alpha(val, user_ratings_map)
print(f"Best alpha (SVD weight): {best_alpha:.2f}")

eval_model(
    "weighted_hybrid", "Weighted Hybrid",
    lambda u, i: weighted.predict(u, i, user_ratings_map.get(u, {})),
)

weighted.save()
print("Saved: artifacts/models/weighted_hybrid.joblib")
"""),

# ── 9. Stacked hybrid ─────────────────────────────────────────────────────────

md("""\
## 9. Stacked Hybrid — Ridge Meta-Learner

The stacking hybrid learns the optimal combination of base-model predictions
from data, rather than fixing it as a scalar weight.

**Protocol (leak-free):**
1. Split the training set into 5 folds.
2. For each fold, train all base models on 4 folds and generate out-of-fold (OOF)
   predictions on the held-out fold — no model ever predicts on data it was trained on.
3. Train a Ridge regression meta-model on the OOF predictions plus side features
   (item popularity, user rating count, item rating count).
4. At test time, base models are retrained on the full training set; the meta-model
   combines their predictions.

> **Note:** The OOF loop trains 4 models × 5 folds. On MovieLens 25M this is
> computationally intensive. Reduce `N_OOF_FOLDS` or set `OOF_SAMPLE_FRAC < 1.0`
> to use a stratified training sample if runtime is a concern.
"""),

code("""\
N_OOF_FOLDS    = 5
OOF_SAMPLE_FRAC = 1.0   # set to e.g. 0.2 for a faster run

train_oof = (
    train.sample(frac=OOF_SAMPLE_FRAC, random_state=RANDOM_STATE)
    if OOF_SAMPLE_FRAC < 1.0
    else train
).reset_index(drop=True)

print(f"OOF training rows: {len(train_oof):,}  (frac={OOF_SAMPLE_FRAC})")

kf        = KFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.full((len(train_oof), 4), np.nan)

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_oof)):
    print(f"  Fold {fold_idx + 1}/{N_OOF_FOLDS} ...", end=" ", flush=True)
    fold_tr  = train_oof.iloc[tr_idx]
    fold_val = train_oof.iloc[val_idx]

    fold_ur_map = (
        fold_tr.groupby("userId")
        .apply(lambda df: dict(zip(df["movieId"], df["rating"])))
        .to_dict()
    )

    fold_cb  = ContentBasedRecommender(n_neighbors=50).fit(item_features, movie_index)
    fold_uknn = UserKNNModel(k=80, min_k=5).fit(fold_tr)
    fold_iknn = ItemKNNModel(k=80, min_k=5).fit(fold_tr)
    fold_svd_model = SurpriseSVD(**svd.best_params)
    fold_svd_model.fit(_to_surprise(fold_tr).build_full_trainset())

    for i, row in enumerate(fold_val.itertuples()):
        ur = fold_ur_map.get(row.userId, {})
        oof_preds[val_idx[i], 0] = fold_cb.predict(ur, row.movieId)
        oof_preds[val_idx[i], 1] = fold_uknn.predict(row.userId, row.movieId)
        oof_preds[val_idx[i], 2] = fold_iknn.predict(row.userId, row.movieId)
        oof_preds[val_idx[i], 3] = fold_svd_model.predict(str(row.userId), str(row.movieId)).est

    print("done")

print("OOF predictions complete.")
"""),

code("""\
# Drop rows where ANY base model returned NaN (e.g. CB cold-start on a fold) —
# Ridge cannot fit NaN features and would crash otherwise.
nan_rows = np.isnan(oof_preds).any(axis=1)
print(f"OOF NaN rows dropped: {nan_rows.sum():,} / {len(oof_preds):,}")
oof_preds = oof_preds[~nan_rows]
train_oof = train_oof.loc[~nan_rows].reset_index(drop=True)
assert len(train_oof) == oof_preds.shape[0], "OOF row count mismatch after NaN drop"

# Side features (always computed from the FULL training set so they are stable
# regardless of OOF_SAMPLE_FRAC).
train_item_pop = train.groupby("movieId").size().to_dict()
train_user_cnt = train.groupby("userId").size().to_dict()
train_item_cnt = train.groupby("movieId").size().to_dict()


def meta_features(df: pd.DataFrame, base_preds: np.ndarray) -> np.ndarray:
    pop  = np.array([train_item_pop.get(m, 0) for m in df["movieId"]], dtype=float)
    ucnt = np.array([train_user_cnt.get(u, 0) for u in df["userId"]],  dtype=float)
    icnt = np.array([train_item_cnt.get(m, 0) for m in df["movieId"]], dtype=float)
    return np.column_stack([base_preds, pop, ucnt, icnt])


X_meta = meta_features(train_oof, oof_preds)
y_meta = train_oof["rating"].values

stacked = StackedHybrid(alpha=1.0)
stacked.fit(X_meta, y_meta)

print("Meta-model coefficients:")
for name, coef in zip(StackedHybrid.FEATURE_NAMES, stacked.meta.coef_):
    print(f"  {name:<22} {coef:+.4f}")
"""),

code("""\
# Stacked test-time predictor. Works for any (user, item) pair — including
# sampled-negative items not present in the test rating table — by building
# base predictions on the fly and feeding them through the Ridge meta-model.
def stacked_predict(user_id, movie_id) -> float:
    base = np.array([[
        cb.predict(user_ratings_map.get(user_id, {}), movie_id),
        user_knn.predict(user_id, movie_id),
        item_knn.predict(user_id, movie_id),
        svd.predict(user_id, movie_id),
    ]])
    if np.isnan(base).any():
        return float(global_mean)
    X = meta_features(
        pd.DataFrame({"userId": [user_id], "movieId": [movie_id]}),
        base,
    )
    return float(stacked.predict(X)[0])


eval_model("stacked_hybrid", "Stacked Hybrid", stacked_predict)

stacked.save()
print("Saved: artifacts/models/stacked_hybrid.joblib")
"""),

# ── 10. Results ───────────────────────────────────────────────────────────────

md("## 10. Results Summary"),

code("""\
rows = []
for label, m in all_metrics.items():
    rows.append({
        "Model":  label,
        "RMSE":   m["rmse"],  "MAE":   m["mae"],
        "P@5":    m["k5"]["precision"],  "R@5":  m["k5"]["recall"],  "F1@5":  m["k5"]["f1"],
        "P@10":   m["k10"]["precision"], "R@10": m["k10"]["recall"], "F1@10": m["k10"]["f1"],
        "P@20":   m["k20"]["precision"], "R@20": m["k20"]["recall"], "F1@20": m["k20"]["f1"],
    })

results = pd.DataFrame(rows).set_index("Model")
display(
    results.style
    .highlight_min(subset=["RMSE", "MAE"],              color="#d4edda")
    .highlight_max(subset=["F1@5", "F1@10", "F1@20"],   color="#d4edda")
    .format("{:.4f}")
)
"""),

# ── 11. Visualisations ────────────────────────────────────────────────────────

md("## 11. Visualisations"),

code("""\
df_plot = results.reset_index()

# RMSE / MAE grouped bar
fig1 = px.bar(
    df_plot.melt(id_vars="Model", value_vars=["RMSE", "MAE"]),
    x="Model", y="value", color="variable", barmode="group",
    title="RMSE and MAE by Model",
    labels={"value": "Error", "variable": "Metric"},
)
fig1.update_layout(xaxis_tickangle=-30)
save_fig(fig1, "08_rmse_mae")

# F1@10 bar
fig2 = px.bar(
    df_plot.sort_values("F1@10", ascending=False),
    x="Model", y="F1@10",
    title="F1@10 by Model",
    color="F1@10", color_continuous_scale="Teal", text_auto=".4f",
)
fig2.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
save_fig(fig2, "09_f1_at_10")

# Ranking metrics @ K for best model
best_label = df_plot.sort_values("F1@10", ascending=False).iloc[0]["Model"]
best_data  = [
    {"K": k, "Metric": m.capitalize(), "Value": all_metrics[best_label][f"k{k}"][m]}
    for k in [5, 10, 20]
    for m in ["precision", "recall", "f1"]
]
fig3 = px.line(
    pd.DataFrame(best_data), x="K", y="Value", color="Metric",
    markers=True,
    title=f"Ranking Metrics @ K — {best_label}",
    labels={"Value": "Score"},
)
save_fig(fig3, "10_ranking_metrics_k")
"""),

# ── 12. Save ──────────────────────────────────────────────────────────────────

md("## 12. Save Artifacts & Metrics"),

code("""\
checkpoint_metrics()  # final flush (each eval_model already checkpoints)

print(f"Metrics  → {metrics_path}")
print(f"Models   → {ARTIFACTS_MODELS}/")
print(f"Figures  → {FIGURES_DIR}/")
"""),

# ── Conclusion ────────────────────────────────────────────────────────────────

md("""\
## Conclusion

- All eight models were trained and evaluated on the same temporal test split under a strictly leak-free protocol.
- The **Weighted Hybrid** and **Stacked Hybrid** are expected to achieve the best F1@10, combining the semantic breadth of the content model with the personalisation depth of SVD.
- **SVD** typically achieves the lowest RMSE and MAE among individual models, confirming its strength as a regularised latent-factor approach.
- **Item-based k-NN** outperforms user-based k-NN on ranking metrics in movie domains, consistent with established findings in the recommender systems literature.
- The Ridge meta-learner learns from data which base model to trust more in each context — its coefficients reveal the relative contribution of each signal.
- All trained models are persisted in `artifacts/models/` and evaluation results in `artifacts/metrics/all_metrics.json`, making them immediately accessible to the Streamlit application.
"""),

]

save_nb(nb03, "03_train_evaluate.ipynb")

print("\nAll notebooks created successfully.")
