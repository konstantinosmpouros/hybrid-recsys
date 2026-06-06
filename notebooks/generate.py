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
# NOTEBOOK 03 — Model Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

nb03 = [

md("""\
# 03 — Model Training

This notebook **fits and persists** every trainable recommender model. No evaluation
happens here — that lives in `04_evaluation.ipynb`, which loads these saved artifacts
and scores them on the held-out test set. Separating training from evaluation keeps the
expensive fitting apart from the cheap, frequently re-run metric computation.

**Protocol:**
- Surprise models (SVD) are tuned by 5-fold cross-validation on the training set;
  the weighted hybrid's α is chosen by exhaustive validation-set search.
- Every model is fit on the full training set and serialised to `artifacts/models/`.
- All random seeds are pinned (`RANDOM_STATE = 42`) so results are reproducible.

**Models trained:** Content-Based · User-kNN · Item-kNN · SVD ·
Weighted Hybrid · Stacked Hybrid. (The two naive baselines — Global Mean and
Popularity — are trivial to recompute, so they are defined directly in the
evaluation notebook rather than persisted here.)
"""),

code("""\
import sys
sys.path.insert(0, "../src")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

from hybrid_recsys.pipeline.data import load_processed
from hybrid_recsys.pipeline.splits import load_splits
from hybrid_recsys.pipeline.features import load_item_features
from hybrid_recsys.models.content import ContentBasedRecommender
from hybrid_recsys.models.collaborative import SVDModel, ItemKNNModel, UserKNNModel, _to_surprise
from hybrid_recsys.models.hybrid import WeightedHybrid, StackedHybrid
from hybrid_recsys.config import ARTIFACTS_MODELS, RANDOM_STATE
from surprise import SVD as SurpriseSVD

ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
"""),

# ── 1. Load ──────────────────────────────────────────────────────────────────

md("## 1. Load Data"),

code("""\
movies                     = load_processed("movies")
train, val, test           = load_splits()
item_features, movie_index = load_item_features()

print(f"Train:      {len(train):>10,} ratings | {train['userId'].nunique():>7,} users")
print(f"Validation: {len(val):>10,} ratings | {val['userId'].nunique():>7,} users")
print(f"Item features: {item_features.shape}")

# User rating histories from the training set (needed by content model & hybrids)
user_ratings_map: dict = (
    train
    .groupby("userId")
    .apply(lambda df: dict(zip(df["movieId"], df["rating"])))
    .to_dict()
)

# Training-set global mean — reused by the stacked hybrid's NaN fallback.
global_mean = float(train["rating"].mean())
print(f"Global mean rating: {global_mean:.4f}")
"""),

# ── 2. Helper ─────────────────────────────────────────────────────────────────

md("""\
## 2. Content-Based Model

The content-based model predicts the rating for item *j* by finding its top-L
most content-similar items via cosine similarity on the feature matrix, then
computing a mean-centred weighted average over the user's ratings on those items:

$$\\hat{r}^{CB}_{u,j} = \\bar{r}_u + \\frac{\\sum_{i \\in N^L_u(j)} \\text{sim}(i,j)\\,(r_{u,i} - \\bar{r}_u)}{\\sum_{i} |\\text{sim}(i,j)| + \\varepsilon}$$
"""),

code("""\
cb = ContentBasedRecommender(n_neighbors=50)
cb.fit(item_features, movie_index)
cb.save()
print("Saved: artifacts/models/content_model.joblib")
"""),

# ── 5. User-kNN ───────────────────────────────────────────────────────────────

md("""\
## 3. User-Based k-NN

User-based CF identifies the *k* most similar users to the target user
(Pearson baseline similarity) and aggregates their ratings on the target item.
We use Surprise's `KNNWithMeans` with `user_based=True`. To bound memory, the
model samples 20K users before fitting — non-sampled users fall back to the
baseline prediction at scoring time.
"""),

code("""\
user_knn = UserKNNModel(k=80, min_k=5)
user_knn.fit(train)
user_knn.save()
print("Saved: artifacts/models/user_knn_model.joblib")
"""),

# ── 6. Item-kNN ───────────────────────────────────────────────────────────────

md("""\
## 4. Item-Based k-NN

Item-based CF identifies the *k* most similar items to the target item and
aggregates the user's ratings on those items. Caps to the 15K most-rated items
before fitting; because ratings concentrate on popular titles, this cap barely
reduces coverage (unlike the user-side cap).
"""),

code("""\
item_knn = ItemKNNModel(k=80, min_k=5)
item_knn.fit(train)
item_knn.save()
print("Saved: artifacts/models/item_knn_model.joblib")
"""),

# ── 7. SVD ────────────────────────────────────────────────────────────────────

md("""\
## 5. SVD — Matrix Factorisation

Surprise's SVD decomposes the rating matrix into user and item latent factor
matrices with global, user, and item bias terms:

$$\\hat{r}_{ui} = \\mu + b_u + b_i + q_i^\\top p_u$$

Hyperparameters (`n_factors`, `n_epochs`, `lr_all`, `reg_all`) are selected
by 5-fold cross-validation on the training set, minimising RMSE. `random_state`
is pinned inside `SVDModel.tune`, so the factor initialisation is reproducible.
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

svd.save()
print("Saved: artifacts/models/svd_model.joblib")
"""),

# ── 8. Weighted hybrid ────────────────────────────────────────────────────────

md("""\
## 6. Weighted Hybrid

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

weighted.save()
print("Saved: artifacts/models/weighted_hybrid.joblib")
"""),

# ── 9. Stacked hybrid ─────────────────────────────────────────────────────────

md("""\
## 7. Stacked Hybrid — Ridge Meta-Learner

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

# Attach side features + global mean so the saved model scores standalone
# (the serving bundle and the evaluation notebook never reload the training frame).
stacked.set_side_features(
    item_popularity=train_item_pop,
    user_count=train_user_cnt,
    item_count=train_item_cnt,
    global_mean=global_mean,
)
stacked.save()
print("\\nSaved: artifacts/models/stacked_hybrid.joblib")
"""),

md("""\
## Conclusion

All six trainable models are fitted and persisted to `artifacts/models/`:
`content_model.joblib`, `user_knn_model.joblib`, `item_knn_model.joblib`,
`svd_model.joblib`, `weighted_hybrid.joblib`, `stacked_hybrid.joblib`.

The two naive baselines (Global Mean, Popularity) are trivial to recompute and
are defined directly in the evaluation notebook. Proceed to
**`04_evaluation.ipynb`** to score every model on the held-out test set.
"""),

]

save_nb(nb03, "03_train.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

nb04 = [

md("""\
# 04 — Model Evaluation

Loads the models trained in `03_train.ipynb` and evaluates **all eight** of them on the
untouched test set under a strict, leak-free protocol.

**Metrics:**
- **Rating prediction** — RMSE, MAE over the full test set.
- **Ranking** — Precision@K, Recall@K, F1@K under the *sampled-negatives* protocol:
  each user's relevant test items (rating ≥ 4.0) are ranked against 100 random
  non-relevant items, macro-averaged over a 1,000-user sample, K ∈ {5, 10, 20}.
  Candidate order is shuffled so tied predictions break randomly, and F1 is the
  harmonic mean of the macro-averaged precision and recall (so F1 always lies
  between the two).

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

from hybrid_recsys.pipeline.data import load_processed
from hybrid_recsys.pipeline.splits import load_splits
from hybrid_recsys.models.content import ContentBasedRecommender
from hybrid_recsys.models.collaborative import SVDModel, ItemKNNModel, UserKNNModel
from hybrid_recsys.models.hybrid import WeightedHybrid, StackedHybrid
from hybrid_recsys.evaluation.metrics import (
    evaluate_rating_prediction,
    evaluate_ranking_sampled,
)
from hybrid_recsys.config import ARTIFACTS_METRICS

""" + SAVE_FIG_SNIPPET + """
ARTIFACTS_METRICS.mkdir(parents=True, exist_ok=True)

EVAL_USERS   = 1_000   # users sampled for ranking evaluation
N_NEGATIVES  = 100     # sampled non-relevant items per user for ranking metrics
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
"""),

# ── 1. Load data & models ─────────────────────────────────────────────────────

md("## 1. Load Data & Trained Models"),

code("""\
movies           = load_processed("movies")
train, val, test = load_splits()
train_val        = pd.concat([train, val], ignore_index=True)

print(f"Test: {len(test):>10,} ratings | {test['userId'].nunique():>7,} users")

# User histories from train (content model & weighted hybrid need these).
user_ratings_map: dict = (
    train.groupby("userId")
    .apply(lambda df: dict(zip(df["movieId"], df["rating"])))
    .to_dict()
)

# Load every model trained in notebook 03.
cb       = ContentBasedRecommender.load()
user_knn = UserKNNModel.load()
item_knn = ItemKNNModel.load()
svd      = SVDModel.load()
weighted = WeightedHybrid.load()
stacked  = StackedHybrid.load()
print("Loaded 6 trained models.")

# Stratified user sample for ranking evaluation.
eval_user_ids = rng.choice(
    test["userId"].unique(),
    size=min(EVAL_USERS, test["userId"].nunique()),
    replace=False,
)
test_sample = test[test["userId"].isin(eval_user_ids)]
print(f"Ranking evaluation user sample: {len(eval_user_ids):,}")
"""),

# ── 2. Helper ─────────────────────────────────────────────────────────────────

md("""\
## 2. Evaluation Helper

A shared wrapper computes both rating-prediction and ranking metrics for any
`predict_fn(user_id, movie_id) -> float`, checkpointing `all_metrics` after every model.
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

# ── 3. Baselines ──────────────────────────────────────────────────────────────

md("""\
## 3. Naive Baselines — Global Mean & Popularity

Recomputed from the training set (trivial, so not persisted as artifacts). The
**global mean** assigns the training average to every pair; **popularity** scores
items by interaction count. These define the floor every model must beat.
"""),

code("""\
global_mean     = float(train["rating"].mean())
item_popularity = train.groupby("movieId").size().to_dict()
max_pop         = max(item_popularity.values())

# Popularity is fundamentally a ranking signal; we map raw counts to the
# [0.5, 5.0] rating range so RMSE/MAE are computable. The map is monotonic,
# so ranking metrics are unaffected.
def pop_score(_user, movie_id):
    return 0.5 + 4.5 * (item_popularity.get(movie_id, 0) / max_pop)

eval_model("global_mean", "Global Mean", lambda u, i: global_mean)
eval_model("popularity",  "Popularity",  pop_score)
"""),

# ── 4-8. Single models ────────────────────────────────────────────────────────

md("## 4. Content-Based"),
code("""\
eval_model("content", "Content-Based", lambda u, i: cb.predict(user_ratings_map.get(u, {}), i))
"""),

md("## 5. User-Based k-NN"),
code("""\
eval_model("user_knn", "User-Based k-NN", lambda u, i: user_knn.predict(u, i))
"""),

md("## 6. Item-Based k-NN"),
code("""\
eval_model("item_knn", "Item-Based k-NN", lambda u, i: item_knn.predict(u, i))
"""),

md("## 7. SVD"),
code("""\
eval_model("svd", "SVD", lambda u, i: svd.predict(u, i))
"""),

md("## 8. Weighted Hybrid"),
code("""\
eval_model(
    "weighted_hybrid", "Weighted Hybrid",
    lambda u, i: weighted.predict(u, i, user_ratings_map.get(u, {})),
)
"""),

# ── 9. Stacked hybrid ─────────────────────────────────────────────────────────

md("""\
## 9. Stacked Hybrid

Builds the four base predictions on the fly and feeds them through the saved Ridge
meta-model via `predict_one` — exactly the path the Streamlit serving bundle uses.
Returns the global mean if any base prediction is NaN.
"""),

code("""\
def stacked_predict(user_id, movie_id) -> float:
    base = np.array([
        cb.predict(user_ratings_map.get(user_id, {}), movie_id),
        user_knn.predict(user_id, movie_id),
        item_knn.predict(user_id, movie_id),
        svd.predict(user_id, movie_id),
    ], dtype=float)
    return stacked.predict_one(user_id, movie_id, base)

eval_model("stacked_hybrid", "Stacked Hybrid", stacked_predict)
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

md("## 12. Save Metrics"),

code("""\
checkpoint_metrics()  # final flush (each eval_model already checkpoints)
print(f"Metrics → {metrics_path}")
"""),

# ── Conclusion ────────────────────────────────────────────────────────────────

md("""\
## Conclusion

- All eight models were evaluated on the same temporal test split under a strictly leak-free protocol.
- Rating accuracy (RMSE/MAE) and ranking quality (P/R/F1@K) are persisted to `artifacts/metrics/all_metrics.json` for the Streamlit comparison tab.
- Ranking uses **random tie-breaking** and a **consistent harmonic-mean F1**, so constant-output models (e.g. Global Mean) no longer score artificially high.
"""),

]

save_nb(nb04, "04_evaluation.ipynb")

print("\nAll notebooks created successfully.")
