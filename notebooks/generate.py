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

md("""\
### Schema, dtypes & missing values

A first look at column types, memory footprint, and where values are missing —
before computing any statistics.
"""),

code("""\
print("=== ratings ===")
ratings.info(memory_usage="deep")
print("\\n=== movies (head) ===")
display(movies.head())
print("=== tags (head) ===")
display(tags.head(3))
print("=== genome_tags (head) ===")
display(genome_tags.head(3))

print("\\nMissing values per table:")
for _name, _df in [("ratings", ratings), ("movies", movies), ("tags", tags)]:
    print(f"  {_name:<8} {_df.isna().sum().sum():,} NaNs across {_df.shape[1]} columns")
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

print("Rating value — descriptive statistics:")
display(ratings[["rating"]].describe().T)
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

md("""\
### Long-tail concentration (Pareto)

How concentrated are ratings on the most popular movies? The cumulative curve below
makes the sparsity tangible — a small fraction of titles absorbs most of the ratings,
which is precisely why content signals matter for the rest of the catalogue.
"""),

code("""\
sorted_counts = np.sort(item_counts["n_ratings"].values)[::-1]
cum_share     = np.cumsum(sorted_counts) / sorted_counts.sum()
movie_share   = np.arange(1, len(cum_share) + 1) / len(cum_share) * 100

fig = px.area(
    x=movie_share, y=cum_share * 100,
    title="Cumulative Share of Ratings vs. Share of Movies (Long Tail)",
    labels={"x": "Top % of movies (most-rated first)", "y": "% of all ratings"},
)
save_fig(fig, "12_long_tail_pareto")

for pct in [1, 5, 10, 20]:
    share = cum_share[int(pct / 100 * len(cum_share)) - 1] * 100
    print(f"Top {pct:>2}% of movies  ->  {share:.1f}% of all ratings")
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

Two metadata sources could augment the genre signal: **free-text user tags** and the
**tag genome** (a dense 1,128-dim relevance vector). The cell below reports the *actual*
coverage of each across the full movie catalogue. Note: only the free-text tags feed the
content model — the genome is profiled here for reference but is **not** used as a feature.
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

md("""\
### 9b. Split Characterisation

Counts alone don't tell us whether the splits are *comparable*. Here we check the
rating distribution, per-user activity, and — most importantly for a hybrid /
content-based system — how many **cold items** appear in validation and test (movies
with no ratings in train, which pure collaborative filtering cannot score). We also
verify the temporal split introduced **no leakage**.
"""),

code("""\
def describe_split(name, df):
    rpu = df.groupby("userId").size()
    return {
        "Split":               name,
        "Ratings":             len(df),
        "Users":               df["userId"].nunique(),
        "Movies":              df["movieId"].nunique(),
        "Mean rating":         round(df["rating"].mean(), 3),
        "Std rating":          round(df["rating"].std(), 3),
        "Median ratings/user": int(rpu.median()),
    }

char = pd.DataFrame([
    describe_split("Train", train),
    describe_split("Validation", val),
    describe_split("Test", test),
]).set_index("Split")
display(char)

# Cold items: movies appearing in val/test but never seen in train.
train_items = set(train["movieId"].unique())
for name, df in [("Validation", val), ("Test", test)]:
    cold_mask = ~df["movieId"].isin(train_items)
    print(f"{name}: {cold_mask.sum():,} ratings on cold items "
          f"({100 * cold_mask.mean():.2f}%) across "
          f"{df.loc[cold_mask, 'movieId'].nunique():,} distinct cold movies")

# Leakage sanity check: each user's last train rating should precede their first
# test rating (the temporal split sorts by timestamp before slicing).
chk = (
    train.groupby("userId")["timestamp"].max().rename("train_max").to_frame()
    .join(test.groupby("userId")["timestamp"].min().rename("test_min"))
    .dropna()
)
violations = int((chk["train_max"] > chk["test_min"]).sum())
print(f"\\nLeakage check — users with train_max > test_min: {violations:,} / {len(chk):,}")
"""),

md("""\
### Rating distribution across splits

If the temporal split is sound, the *shape* of the rating distribution should be
stable across train / val / test — only the volume changes, not user behaviour.
"""),

code("""\
dist_rows = []
for name, df in [("Train", train), ("Validation", val), ("Test", test)]:
    props = df["rating"].value_counts(normalize=True).sort_index()
    for rating_val, prop in props.items():
        dist_rows.append({"Split": name, "Rating": rating_val, "Proportion": prop})

fig = px.bar(
    pd.DataFrame(dist_rows), x="Rating", y="Proportion", color="Split",
    barmode="group", title="Rating Distribution by Split",
)
save_fig(fig, "11_rating_dist_by_split")
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
- Both user activity and item popularity follow **power-law distributions**, yielding a matrix sparsity of ≈99.7% — content-based signals are essential for cold-item and sparse-user scenarios.
- **Drama, Comedy, and Thriller** dominate the genre space; the multi-hot genre vector will be a strong and interpretable content feature.
- **Free-text tags** cover ~72% of catalogued movies; the **tag genome** covers only ~22% (≈13.8K movies). The content representation therefore uses the broader free-text tags, not the genome.
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

print(f"\\nYear parsed for {movies['year'].notna().mean() * 100:.1f}% of movies")
"""),

md("""\
### Tag-text richness

The content text block is built from each movie's aggregated tags. How much text is
there per movie? Movies with no tags contribute only their title — so this distribution
tells us how strong the textual signal is across the catalogue.
"""),

code("""\
word_count = movies["tags_text"].str.split().apply(len)
print("Tag-text length (words per movie):")
print(word_count.describe().round(1).to_string())

fig = px.histogram(
    word_count[word_count > 0], nbins=60, log_y=True,
    title="Tag-text Length (words) per Movie — tagged movies only",
    labels={"value": "Words of tag text", "count": "Number of movies"},
    color_discrete_sequence=["#AB63FA"],
)
fig.update_layout(showlegend=False)
save_fig(fig, "14_tagtext_wordcount")
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

genre_freq = (
    pd.DataFrame({
        "genre": genre_names,
        "count": np.asarray(genre_X.sum(axis=0)).ravel().astype(int),
    })
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)
fig = px.bar(
    genre_freq, x="genre", y="count",
    title="Movies per Genre (multi-hot column sums)",
    color="count", color_continuous_scale="Teal",
)
fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
save_fig(fig, "13_genre_frequency")
display(genre_freq)
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

# Most "common" terms = lowest IDF (appear in the most documents). Sorting the
# vocabulary by its integer index would just give alphabetical order, not frequency.
feat = np.array(tfidf.get_feature_names_out())
most_common = feat[np.argsort(tfidf.idf_)][:30]
print(f"\\n30 most common terms (lowest IDF): {list(most_common)}")
"""),

# ── 4. SVD explained variance ─────────────────────────────────────────────────

md("""\
## 4. TruncatedSVD — Explained Variance

TruncatedSVD (LSA) compresses the high-dimensional TF-IDF space into a dense
256-dimensional representation. TF-IDF spectra are very flat — text has high intrinsic
dimensionality — so 256 components capture only ~20% of the *total* variance. That is
expected and still yields a useful dense semantic embedding (the alternative, keeping
all 41K sparse dimensions, is intractable for pairwise cosine similarity). The plot
shows the cumulative captured variance.
"""),

code("""\
explained = np.cumsum(svd_transformer.explained_variance_ratio_)

fig = px.line(
    x=list(range(1, len(explained) + 1)), y=explained,
    title="Cumulative Explained Variance — TruncatedSVD on TF-IDF",
    labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
)
fig.add_annotation(
    x=len(explained), y=explained[-1],
    text=f"{explained[-1]:.1%} at {len(explained)} components",
    showarrow=True, arrowhead=1, ax=-60, ay=-30,
)
save_fig(fig, "07_svd_explained_variance")


def components_for(target):
    idx = int(np.searchsorted(explained, target))
    return f"{idx + 1}" if idx < len(explained) else f"not reached within {len(explained)} comps"

print(f"Components to 90% variance: {components_for(0.90)}")
print(f"Components to 95% variance: {components_for(0.95)}")
print(f"Selected n_components:      {len(explained)}")
print(f"Variance retained at {len(explained)} comps: {explained[-1]:.2%}")
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

md("""\
## 6. Feature-Space Sanity Checks

Numbers are abstract — these two checks make the feature space *tangible*.

1. **Nearest neighbours**: for a couple of well-known films, the most content-similar
   movies (by cosine similarity on the 276-dim vectors). If the features are sensible,
   the neighbours should be recognisably related.
2. **2-D projection**: a PCA scatter of a random sample of movies, coloured by their
   primary genre — we expect genres to form visible clusters.
"""),

code("""\
# Dense, L2-normalised copy → cosine similarity is a single matmul (same trick the
# ContentBasedRecommender uses at serving time).
dense = item_features.toarray().astype("float32")
norms = np.linalg.norm(dense, axis=1, keepdims=True)
norms[norms == 0] = 1.0
dn = dense / norms


def most_similar(title, n=5):
    matches = movies.index[movies["clean_title"] == title]
    if len(matches) == 0:
        matches = movies.index[movies["clean_title"].str.contains(title, case=False, na=False)]
    i = int(matches[0])
    sims = dn @ dn[i]
    sims[i] = -1.0
    top = np.argsort(-sims)[:n]
    return (
        movies.iloc[top][["clean_title", "year", "genres"]]
        .assign(similarity=np.round(sims[top], 3))
        .reset_index(drop=True)
    )


for query in ["Toy Story", "Pulp Fiction", "The Matrix"]:
    print(f"Most content-similar to '{query}':")
    display(most_similar(query))
"""),

code("""\
from sklearn.decomposition import PCA

rng_viz     = np.random.default_rng(42)
sample_idx  = rng_viz.choice(len(movies), size=min(4000, len(movies)), replace=False)
coords      = PCA(n_components=2, random_state=42).fit_transform(dense[sample_idx])
primary     = movies.iloc[sample_idx]["genres"].str.split("|").str[0].values

viz = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Primary genre": primary})
top_genres = viz["Primary genre"].value_counts().head(8).index
viz = viz[viz["Primary genre"].isin(top_genres)]

fig = px.scatter(
    viz, x="PC1", y="PC2", color="Primary genre", opacity=0.6,
    title="Item Feature Space — PCA-2D of 276-dim vectors (4,000-movie sample)",
)
fig.update_traces(marker_size=5)
save_fig(fig, "15_feature_space_pca")
"""),

md("""\
## 7. Non-linear Projections — t-SNE & UMAP

PCA is linear, so it can flatten curved structure. Two non-linear embeddings often
reveal tighter genre clusters. We project the **same 4,000-movie sample** to 2-D,
coloured by primary genre. Both run on a 50-dim PCA pre-reduction — standard practice
that speeds up t-SNE and denoises the input. UMAP needs the optional `umap-learn`
package; the cell degrades gracefully if it isn't installed.

> Runtime: t-SNE on 4,000 points takes ~1–2 minutes; UMAP is faster.
"""),

code("""\
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 50-dim PCA pre-reduction shared by both non-linear projections.
X_pca50 = PCA(n_components=50, random_state=42).fit_transform(dense[sample_idx])

tsne_xy = TSNE(
    n_components=2, init="pca", perplexity=30, random_state=42,
).fit_transform(X_pca50)

viz_tsne = pd.DataFrame({"x": tsne_xy[:, 0], "y": tsne_xy[:, 1], "Primary genre": primary})
viz_tsne = viz_tsne[viz_tsne["Primary genre"].isin(top_genres)]

fig = px.scatter(
    viz_tsne, x="x", y="y", color="Primary genre", opacity=0.6,
    title="Item Feature Space — t-SNE (2-D, 4,000-movie sample)",
)
fig.update_traces(marker_size=5)
save_fig(fig, "16_feature_space_tsne")
"""),

code("""\
try:
    import umap  # provided by the `umap-learn` package

    umap_xy = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=42,
    ).fit_transform(X_pca50)

    viz_umap = pd.DataFrame({"x": umap_xy[:, 0], "y": umap_xy[:, 1], "Primary genre": primary})
    viz_umap = viz_umap[viz_umap["Primary genre"].isin(top_genres)]

    fig = px.scatter(
        viz_umap, x="x", y="y", color="Primary genre", opacity=0.6,
        title="Item Feature Space — UMAP (2-D, 4,000-movie sample)",
    )
    fig.update_traces(marker_size=5)
    save_fig(fig, "17_feature_space_umap")
except ImportError:
    print("umap-learn not installed — skipping UMAP. Install with:  pip install umap-learn")
"""),

# ── Conclusion ────────────────────────────────────────────────────────────────

md("""\
## Conclusion

- The item feature matrix has shape **(n_movies × 276)**: 20 genre indicator dimensions plus 256 SVD-compressed text dimensions.
- The TF-IDF vocabulary spans thousands of unique terms (min_df = 5, bigrams), capturing both title semantics and community tag vocabulary.
- **256 SVD components** capture ~20% of the (very flat) TF-IDF variance — typical for sparse text — while giving a compact dense embedding that keeps pairwise cosine similarity tractable across 62K items.
- Genre features are directly interpretable and complement the semantic smoothing provided by SVD text features.
- All transformers are serialised to `artifacts/models/`, ensuring the exact same feature space is applied at serving time without re-fitting.
"""),

]

save_nb(nb02, "02_features.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared setup snippets for the per-model notebooks.
# SETUP_ENV  → imports + data load.   SETUP_HELPERS → small plotting/example helpers.
# Each is injected as its OWN cell so no single cell does more than one job.
# ═══════════════════════════════════════════════════════════════════════════════

SETUP_ENV = '''import sys
sys.path.insert(0, "../src")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from hybrid_recsys.pipeline.data import load_processed
from hybrid_recsys.pipeline.splits import load_splits
from hybrid_recsys.evaluation.report import full_metrics, save_metric, top_n
from hybrid_recsys.config import RANDOM_STATE

FIGURES_DIR = Path("../artifacts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(FIGURES_DIR / f"{name}.html"))
    try:
        fig.write_image(str(FIGURES_DIR / f"{name}.png"), width=1100, height=550, scale=2)
    except Exception:
        pass
    fig.show()


EVAL_USERS, N_NEGATIVES = 1_000, 100
rng = np.random.default_rng(RANDOM_STATE)

movies           = load_processed("movies")
train, val, test = load_splits()
train_val        = pd.concat([train, val], ignore_index=True)
all_movie_ids    = movies["movieId"].values
user_ratings_map = (
    train.groupby("userId").apply(lambda d: dict(zip(d["movieId"], d["rating"]))).to_dict()
)
eval_user_ids = rng.choice(
    test["userId"].unique(), size=min(EVAL_USERS, test["userId"].nunique()), replace=False
)
test_sample = test[test["userId"].isin(eval_user_ids)]
demo_user   = max(eval_user_ids, key=lambda u: len(user_ratings_map.get(u, {})))
print(f"Loaded splits - train {len(train):,}, test {len(test):,}; demo user = {int(demo_user)}")
'''

SETUP_HELPERS = '''def ranking_curve(metrics, title):
    rows = [{"K": k, "Metric": m.capitalize(), "Value": metrics[f"k{k}"][m]}
            for k in [5, 10, 20] for m in ["precision", "recall", "f1"]]
    return px.line(pd.DataFrame(rows), x="K", y="Value", color="Metric", markers=True,
                   title=f"Ranking metrics @ K - {title}")


def error_hist(preds, title):
    err = test["rating"].to_numpy() - preds
    err = err[~np.isnan(err)]
    fig = px.histogram(err, nbins=50, title=f"Rating error (true - predicted) - {title}")
    fig.update_layout(showlegend=False, xaxis_title="true - predicted", bargap=0.02)
    return fig


def show_example(predict_fn, n=10, n_candidates=3000):
    seen = set(user_ratings_map.get(demo_user, {}))
    cand = rng.choice(all_movie_ids, size=min(n_candidates, len(all_movie_ids)), replace=False)
    recs = top_n(predict_fn, demo_user, seen, cand, movies, n=n)
    hist = (
        pd.DataFrame({"movieId": list(seen),
                      "rating": [user_ratings_map[demo_user][m] for m in seen]})
        .merge(movies[["movieId", "clean_title", "genres"]], on="movieId", how="left")
        .sort_values("rating", ascending=False).head(n)
    )
    return hist, recs


def star_graph(center, leaves, weights, title, name):
    k = len(leaves)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    lx, ly = np.cos(angles), np.sin(angles)
    edge_x, edge_y = [], []
    for x, y in zip(lx, ly):
        edge_x += [0, x, None]
        edge_y += [0, y, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="#cccccc", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="markers+text",
        marker=dict(size=16, color=list(weights), colorscale="Teal",
                    showscale=True, colorbar=dict(title="sim")),
        text=[f"{l}<br>{w:.2f}" for l, w in zip(leaves, weights)],
        textposition="top center", hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                             marker=dict(size=26, color="#EF553B"),
                             text=[center], textposition="bottom center", hoverinfo="text"))
    fig.update_layout(title=title, showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    save_fig(fig, name)
    return fig
'''


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Naive Baselines
# ═══════════════════════════════════════════════════════════════════════════════

nb03 = [
md('''# 03 - Naive Baselines (Global Mean & Popularity)

Two parameter-free baselines that define the performance floor every later model must
beat. Each is evaluated on the held-out test set and its results written to
`artifacts/metrics/all_metrics.json`.'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Global Mean — train + evaluate"),
code('''global_mean = float(train["rating"].mean())
gm_predict = lambda u, i: global_mean

m, preds = full_metrics(gm_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("Global Mean", m)
print(f"global mean = {global_mean:.3f}  |  RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Popularity — train + evaluate"),
code('''item_pop = train.groupby("movieId").size().to_dict()
max_pop  = max(item_pop.values())

def pop_predict(u, i):
    return 0.5 + 4.5 * (item_pop.get(i, 0) / max_pop)

m, preds = full_metrics(pop_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("Popularity", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("### What Popularity recommends to everyone (top-10)"),
code('''top_pop = (
    pd.Series(item_pop, name="ratings").sort_values(ascending=False).head(10)
    .rename_axis("movieId").reset_index()
    .merge(movies[["movieId", "clean_title", "genres"]], on="movieId")
)
display(top_pop[["clean_title", "genres", "ratings"]])'''),

md("### Popularity — ranking metrics @ K"),
code('save_fig(ranking_curve(m, "Popularity"), "eval_popularity_ranking")'),

md('''## Takeaway

Global Mean anchors RMSE/MAE but cannot rank (constant output). Popularity ranks well under
sampled-negatives (relevant items skew popular) yet has meaningless RMSE. These are the bar
the real models must clear.'''),
]
save_nb(nb03, "03_baselines.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — Content-Based
# ═══════════════════════════════════════════════════════════════════════════════

nb04 = [
md('''# 04 - Content-Based Model

Item-item cosine similarity on the 276-dim feature vectors; predicts a mean-centred
weighted average over the user's ratings on the target's most similar items.'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Train & save"),
code('''from hybrid_recsys.pipeline.features import load_item_features
from hybrid_recsys.models.content import ContentBasedRecommender

item_features, movie_index = load_item_features()
cb = ContentBasedRecommender(n_neighbors=50).fit(item_features, movie_index)
cb.save()
print("saved content_model.joblib")'''),

md("## Evaluate — compute & record metrics"),
code('''cb_predict = lambda u, i: cb.predict(user_ratings_map.get(u, {}), i)
m, preds = full_metrics(cb_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("Content-Based", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Ranking metrics @ K"),
code('save_fig(ranking_curve(m, "Content-Based"), "eval_content_ranking")'),

md("## Rating-error distribution"),
code('save_fig(error_hist(preds, "Content-Based"), "eval_content_error")'),

md("## Example — the demo user's rating history"),
code('''hist, recs = show_example(cb_predict)
print(f"Demo user {int(demo_user)} top-rated movies:")
display(hist[["clean_title", "genres", "rating"]])'''),

md("## Example — what the model recommends"),
code('display(recs[["clean_title", "genres", "pred"]])'),

md("## Why these? Content neighbours of a movie the user liked"),
code('''liked_id = int(hist.iloc[0]["movieId"])
sim_ids, sim_scores = cb._similar_items(liked_id)
nb = (
    pd.DataFrame({"movieId": [int(x) for x in sim_ids[:10]],
                  "similarity": np.round(sim_scores[:10], 3)})
    .merge(movies[["movieId", "clean_title", "genres"]], on="movieId")
)
print(f"Because the user liked: {hist.iloc[0]['clean_title']}")
display(nb[["clean_title", "genres", "similarity"]])'''),

md("## Takeaway\n\nCold-item friendly and interpretable, but often falls back to the user mean, so RMSE only narrowly beats Global Mean."),
]
save_nb(nb04, "04_content_based.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 05 — User-Based k-NN
# ═══════════════════════════════════════════════════════════════════════════════

nb05 = [
md('''# 05 - User-Based k-NN

Collaborative filtering over a user-user similarity matrix (Pearson-baseline). Samples
20K users before fitting (memory cap); users outside the sample fall back to the baseline.'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Train & save"),
code('''from hybrid_recsys.models.collaborative import UserKNNModel

user_knn = UserKNNModel(k=80, min_k=5).fit(train)
user_knn.save()
print("saved user_knn_model.joblib")'''),

md("## Evaluate — compute & record metrics"),
code('''uk_predict = lambda u, i: user_knn.predict(u, i)
m, preds = full_metrics(uk_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("User-Based k-NN", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Ranking metrics @ K"),
code('save_fig(ranking_curve(m, "User-kNN"), "eval_userknn_ranking")'),

md("## Rating-error distribution"),
code('save_fig(error_hist(preds, "User-kNN"), "eval_userknn_error")'),

md("## Neighbourhood graph — the *k* nearest users"),
code('''algo = user_knn.model
ts = algo.trainset

demo_raw = None
for u in eval_user_ids:
    try:
        ts.to_inner_uid(str(u)); demo_raw = str(u); break
    except ValueError:
        continue

inner     = ts.to_inner_uid(demo_raw)
neighbors = algo.get_neighbors(inner, k=10)
sims      = [float(algo.sim[inner, nb]) for nb in neighbors]
neigh_raw = [ts.to_raw_uid(nb) for nb in neighbors]

star_graph(f"user {demo_raw}", [f"user {r}" for r in neigh_raw], sims,
           f"User {demo_raw} - 10 nearest users (Pearson-baseline sim)",
           "eval_userknn_neighbors")'''),

md("## Distribution of user-user similarities"),
code('''tri = algo.sim[np.triu_indices_from(algo.sim, k=1)]
tri = tri[np.isfinite(tri) & (tri != 0)]
samp = rng.choice(tri, size=min(200_000, len(tri)), replace=False)
fig = px.histogram(samp, nbins=60, title="User-User Similarity Distribution (non-zero, sampled)")
fig.update_layout(showlegend=False, xaxis_title="similarity")
save_fig(fig, "eval_userknn_simdist")'''),

md("## Takeaway\n\nThe 20K-user cap means most test users hit the baseline fallback, so these metrics under-represent true user-CF (item-kNN suffers far less)."),
]
save_nb(nb05, "05_user_knn.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06 — Item-Based k-NN
# ═══════════════════════════════════════════════════════════════════════════════

nb06 = [
md('''# 06 - Item-Based k-NN

Collaborative filtering over an item-item similarity matrix. The neighbourhood graph is
especially interpretable here - the nodes are movies.'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Train & save"),
code('''from hybrid_recsys.models.collaborative import ItemKNNModel

item_knn = ItemKNNModel(k=80, min_k=5).fit(train)
item_knn.save()
print("saved item_knn_model.joblib")'''),

md("## Evaluate — compute & record metrics"),
code('''ik_predict = lambda u, i: item_knn.predict(u, i)
m, preds = full_metrics(ik_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("Item-Based k-NN", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Ranking metrics @ K"),
code('save_fig(ranking_curve(m, "Item-kNN"), "eval_itemknn_ranking")'),

md("## Rating-error distribution"),
code('save_fig(error_hist(preds, "Item-kNN"), "eval_itemknn_error")'),

md("## Item neighbourhood graph — nearest movies to a query"),
code('''algo = item_knn.model
ts = algo.trainset
title_by_id = movies.set_index("movieId")["clean_title"]

def neighbours_of(substr, k=10):
    cand = movies[movies["clean_title"].str.contains(substr, case=False, na=False)]
    inner = None
    for mid in cand["movieId"]:
        try:
            inner = ts.to_inner_iid(str(mid)); break
        except ValueError:
            continue
    if inner is None:
        print(f"'{substr}' not in the (capped) trainset"); return None
    nbrs  = algo.get_neighbors(inner, k=k)
    sims  = [float(algo.sim[inner, nb]) for nb in nbrs]
    ids   = [int(ts.to_raw_iid(nb)) for nb in nbrs]
    names = [str(title_by_id.get(i, i)) for i in ids]
    center = str(title_by_id.get(int(ts.to_raw_iid(inner)), substr))
    return center, names, sims

res = neighbours_of("Toy Story")
if res:
    center, names, sims = res
    display(pd.DataFrame({"similar movie": names, "similarity": np.round(sims, 3)}))
    star_graph(center, names, sims, f"Item-kNN neighbours of '{center}'", "eval_itemknn_graph")'''),

md("## Takeaway\n\nStrongest pure-CF model on RMSE - the 15K-item cap barely bites, and the neighbour graph shows genuinely related movies."),
]
save_nb(nb06, "06_item_knn.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 07 — SVD
# ═══════════════════════════════════════════════════════════════════════════════

nb07 = [
md('''# 07 - SVD (Matrix Factorisation)

The one genuinely *trained* model: SGD learns latent user/item factors + biases, tuned by
5-fold cross-validation.'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Train — 5-fold CV grid search + save"),
code('''from hybrid_recsys.models.collaborative import SVDModel

svd = SVDModel()
svd.tune(train, param_grid={
    "n_factors": [50, 100, 200],
    "n_epochs":  [20, 40],
    "lr_all":    [0.002, 0.005],
    "reg_all":   [0.02, 0.05],
})
svd.fit(train)
svd.save()
print("best params:", svd.best_params)'''),

md("## Evaluate — compute & record metrics"),
code('''sv_predict = lambda u, i: svd.predict(u, i)
m, preds = full_metrics(sv_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("SVD", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Ranking metrics @ K"),
code('save_fig(ranking_curve(m, "SVD"), "eval_svd_ranking")'),

md("## Rating-error distribution"),
code('save_fig(error_hist(preds, "SVD"), "eval_svd_error")'),

md("## Example recommendations for the demo user"),
code('''hist, recs = show_example(sv_predict)
display(recs[["clean_title", "genres", "pred"]])'''),

md("## Learned item-factor space (PCA-2D of qᵢ)"),
code('''from sklearn.decomposition import PCA

algo = svd.model
qi   = algo.qi
ts   = algo.trainset
ids  = np.array([int(ts.to_raw_iid(ii)) for ii in range(qi.shape[0])])

samp   = rng.choice(len(ids), size=min(3000, len(ids)), replace=False)
coords = PCA(n_components=2, random_state=42).fit_transform(qi[samp])
g0 = (movies.set_index("movieId")["genres"].reindex(ids[samp]).fillna("")
      .str.split("|").str[0].values)

viz = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "Primary genre": g0})
top = viz["Primary genre"].value_counts().head(8).index
viz = viz[viz["Primary genre"].isin(top)]
fig = px.scatter(viz, x="x", y="y", color="Primary genre", opacity=0.6,
                 title="SVD Latent Item Factors (PCA-2D)")
fig.update_traces(marker_size=5)
save_fig(fig, "eval_svd_factors")'''),

md("## Takeaway\n\nBest single model on RMSE/MAE; the factor space shows genres separating even though the model never saw genres - it learned them from rating patterns alone."),
]
save_nb(nb07, "07_svd.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 08 — Weighted Hybrid
# ═══════════════════════════════════════════════════════════════════════════════

nb08 = [
md('''# 08 - Weighted Hybrid (α·SVD + (1−α)·CB)

Loads the trained SVD and Content-Based models, tunes the blend weight α on validation,
then evaluates. **Run notebooks 04 and 07 first.**'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Load the base models"),
code('''from hybrid_recsys.models.content import ContentBasedRecommender
from hybrid_recsys.models.collaborative import SVDModel
from hybrid_recsys.models.hybrid import WeightedHybrid

cb       = ContentBasedRecommender.load()
svd      = SVDModel.load()
weighted = WeightedHybrid().set_models(svd, cb)
print("loaded SVD + Content base models")'''),

md("## Tune α on the validation set"),
code('''from hybrid_recsys.evaluation.metrics import rmse

val_s = val.sample(min(50_000, len(val)), random_state=RANDOM_STATE)   # fast representative sweep
alphas = np.round(np.arange(0.0, 1.05, 0.05), 2)
rmses = []
for a in alphas:
    weighted.alpha = float(a)
    p = np.array([weighted.predict(r.userId, r.movieId, user_ratings_map.get(r.userId, {}))
                  for r in val_s.itertuples()])
    rmses.append(rmse(val_s["rating"].to_numpy(), p))

best = float(alphas[int(np.argmin(rmses))])
weighted.alpha = best
weighted.save()
print(f"best α (SVD weight) = {best}")'''),

md("## α-sweep curve"),
code('''fig = px.line(x=alphas, y=rmses, markers=True,
              title="Validation RMSE vs α  (α=1 -> pure SVD, α=0 -> pure CB)",
              labels={"x": "α (SVD weight)", "y": "validation RMSE"})
fig.add_vline(x=best, line_dash="dash", line_color="red")
save_fig(fig, "eval_weighted_alpha")'''),

md("## Evaluate — compute & record metrics"),
code('''w_predict = lambda u, i: weighted.predict(u, i, user_ratings_map.get(u, {}))
m, preds = full_metrics(w_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("Weighted Hybrid", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Ranking metrics @ K"),
code('save_fig(ranking_curve(m, "Weighted Hybrid"), "eval_weighted_ranking")'),

md("## Example recommendations for the demo user"),
code('''hist, recs = show_example(w_predict)
display(recs[["clean_title", "genres", "pred"]])'''),

md("## Takeaway\n\nα converges high (≈0.9) - CB is the weaker signal, so the blend is mostly SVD. Safe and never worse than SVD, but a single global weight can't adapt per item (→ stacking)."),
]
save_nb(nb08, "08_weighted_hybrid.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 09 — Stacked Hybrid
# ═══════════════════════════════════════════════════════════════════════════════

nb09 = [
md('''# 09 - Stacked Hybrid (Ridge meta-learner)

Learns how to combine the four base predictions (+ side features) via a Ridge meta-model
trained on **5-fold out-of-fold** predictions (leak-free). **Run notebooks 04-07 first.**'''),
code(SETUP_ENV),
code(SETUP_HELPERS),

md("## Setup — imports, base SVD params, side-feature dicts, OOF config"),
code('''from sklearn.model_selection import KFold
from hybrid_recsys.pipeline.features import load_item_features
from hybrid_recsys.models.content import ContentBasedRecommender
from hybrid_recsys.models.collaborative import SVDModel, ItemKNNModel, UserKNNModel, _to_surprise
from hybrid_recsys.models.hybrid import StackedHybrid
from surprise import SVD as SurpriseSVD

item_features, movie_index = load_item_features()
svd = SVDModel.load()                       # reuse the tuned hyperparameters
global_mean    = float(train["rating"].mean())
train_item_pop = train.groupby("movieId").size().to_dict()
train_user_cnt = train.groupby("userId").size().to_dict()
train_item_cnt = train_item_pop

N_OOF_FOLDS, OOF_SAMPLE_FRAC = 5, 1.0       # set OOF_SAMPLE_FRAC < 1.0 for a faster run
train_oof = (train.sample(frac=OOF_SAMPLE_FRAC, random_state=RANDOM_STATE)
             if OOF_SAMPLE_FRAC < 1.0 else train).reset_index(drop=True)
kf  = KFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof = np.full((len(train_oof), 4), np.nan)
print(f"OOF rows: {len(train_oof):,}")'''),

md("## Generate out-of-fold base predictions (leak-free)"),
code('''for fi, (tr, va) in enumerate(kf.split(train_oof)):
    print(f"  fold {fi+1}/{N_OOF_FOLDS}", end=" ", flush=True)
    ftr, fva = train_oof.iloc[tr], train_oof.iloc[va]
    fur = ftr.groupby("userId").apply(lambda d: dict(zip(d["movieId"], d["rating"]))).to_dict()
    fcb = ContentBasedRecommender(n_neighbors=50).fit(item_features, movie_index)
    fuk = UserKNNModel(k=80, min_k=5).fit(ftr)
    fik = ItemKNNModel(k=80, min_k=5).fit(ftr)
    fsv = SurpriseSVD(**svd.best_params); fsv.fit(_to_surprise(ftr).build_full_trainset())
    for n, row in enumerate(fva.itertuples()):
        ur = fur.get(row.userId, {})
        oof[va[n], 0] = fcb.predict(ur, row.movieId)
        oof[va[n], 1] = fuk.predict(row.userId, row.movieId)
        oof[va[n], 2] = fik.predict(row.userId, row.movieId)
        oof[va[n], 3] = fsv.predict(str(row.userId), str(row.movieId)).est
    print("done")'''),

md("## Fit the Ridge meta-model & save"),
code('''keep = ~np.isnan(oof).any(axis=1)
oof, train_oof = oof[keep], train_oof.loc[keep].reset_index(drop=True)

def meta_features(df, base):
    pop  = np.array([train_item_pop.get(m, 0) for m in df["movieId"]], dtype=float)
    ucnt = np.array([train_user_cnt.get(u, 0) for u in df["userId"]],  dtype=float)
    icnt = np.array([train_item_cnt.get(m, 0) for m in df["movieId"]], dtype=float)
    return np.column_stack([base, pop, ucnt, icnt])

stacked = StackedHybrid(alpha=1.0)
stacked.fit(meta_features(train_oof, oof), train_oof["rating"].to_numpy())
stacked.set_side_features(item_popularity=train_item_pop, user_count=train_user_cnt,
                          item_count=train_item_cnt, global_mean=global_mean)
stacked.save()
print("saved stacked_hybrid.joblib")'''),

md("## Learned Ridge coefficients"),
code('''coef = pd.DataFrame({"feature": StackedHybrid.FEATURE_NAMES, "weight": stacked.meta.coef_})
display(coef)
fig = px.bar(coef, x="feature", y="weight", title="Stacked Hybrid - learned Ridge coefficients",
             color="weight", color_continuous_scale="RdBu")
fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
save_fig(fig, "eval_stacked_coefficients")'''),

md("## Evaluate — compute & record metrics"),
code('''cb       = ContentBasedRecommender.load()
user_knn = UserKNNModel.load()
item_knn = ItemKNNModel.load()

def st_predict(u, i):
    base = np.array([cb.predict(user_ratings_map.get(u, {}), i),
                     user_knn.predict(u, i), item_knn.predict(u, i), svd.predict(u, i)], dtype=float)
    return stacked.predict_one(u, i, base)

m, preds = full_metrics(st_predict, test, test_sample, train_val, all_movie_ids,
                        n_negatives=N_NEGATIVES, random_state=RANDOM_STATE)
save_metric("Stacked Hybrid", m)
print(f"RMSE={m['rmse']}  MAE={m['mae']}  F1@10={m['k10']['f1']}")'''),

md("## Ranking metrics @ K"),
code('save_fig(ranking_curve(m, "Stacked Hybrid"), "eval_stacked_ranking")'),

md("## Takeaway\n\nThe coefficients lean on SVD and Item-kNN and zero out the weak User-kNN - the meta-model *learned* whom to trust, which is why it tends to win on both RMSE and ranking."),
]
save_nb(nb09, "09_stacked_hybrid.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 10 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════

nb10 = [
md('''# 10 - Model Comparison

Aggregates every model's results from `all_metrics.json` (written by notebooks 03-09) into
one table and the headline charts. Run last.'''),

md("## Setup — load metrics"),
code('''import sys
sys.path.insert(0, "../src")
import json
import pandas as pd
import plotly.express as px
from pathlib import Path
from hybrid_recsys.config import ARTIFACTS_METRICS

FIGURES_DIR = Path("../artifacts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(FIGURES_DIR / f"{name}.html"))
    try:
        fig.write_image(str(FIGURES_DIR / f"{name}.png"), width=1200, height=600, scale=2)
    except Exception:
        pass
    fig.show()

metrics = json.loads((ARTIFACTS_METRICS / "all_metrics.json").read_text())
print(f"{len(metrics)} models in all_metrics.json")'''),

md("## Results table"),
code('''rows = []
for label, mm in metrics.items():
    rows.append({"Model": label, "RMSE": mm["rmse"], "MAE": mm["mae"],
                 "P@10": mm["k10"]["precision"], "R@10": mm["k10"]["recall"], "F1@10": mm["k10"]["f1"]})
results = pd.DataFrame(rows).set_index("Model")
display(
    results.style
    .highlight_min(subset=["RMSE", "MAE"], color="#d4edda")
    .highlight_max(subset=["F1@10"], color="#d4edda")
    .format("{:.4f}")
)'''),

md("## RMSE & MAE by model"),
code('''dfp = results.reset_index()
fig = px.bar(dfp.melt(id_vars="Model", value_vars=["RMSE", "MAE"]),
             x="Model", y="value", color="variable", barmode="group",
             title="RMSE and MAE by Model", labels={"value": "Error", "variable": "Metric"})
fig.update_layout(xaxis_tickangle=-30)
save_fig(fig, "08_rmse_mae")'''),

md("## F1@10 by model"),
code('''fig = px.bar(dfp.sort_values("F1@10", ascending=False), x="Model", y="F1@10",
             title="F1@10 by Model", color="F1@10", color_continuous_scale="Teal", text_auto=".3f")
fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
save_fig(fig, "09_f1_at_10")'''),

md("## Takeaway\n\nLead with the Stacked Hybrid's best RMSE/MAE; note Popularity's high F1@10 is a sampled-negatives artifact (it is worst on RMSE). The hybrids beat every CB-only and CF-only baseline on rating accuracy."),
]
save_nb(nb10, "10_comparison.ipynb")

print("\nAll notebooks created successfully.")
