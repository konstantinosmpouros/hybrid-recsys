import pandas as pd
import joblib
from scipy.sparse import csr_matrix, coo_matrix, hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from ..config import DATA_PROCESSED, ARTIFACTS_MODELS, RANDOM_STATE


def build_genre_matrix(movies: pd.DataFrame) -> csr_matrix:
    genre_dummies = movies["genres"].str.get_dummies(sep="|").astype("float32")
    return csr_matrix(genre_dummies.values)


def build_text_matrix(
    movies: pd.DataFrame,
    n_components: int = 256,
    min_df: int = 5,
) -> tuple[csr_matrix, TfidfVectorizer, TruncatedSVD]:
    corpus = movies["clean_title"].fillna("") + " " + movies["tags_text"].fillna("")
    tfidf = TfidfVectorizer(min_df=min_df, ngram_range=(1, 2), sublinear_tf=True)
    text_X = tfidf.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    reduced = csr_matrix(svd.fit_transform(text_X))
    return reduced, tfidf, svd


def build_item_features(
    movies: pd.DataFrame,
    n_components: int = 256,
) -> tuple[csr_matrix, pd.Series, TfidfVectorizer, TruncatedSVD]:
    genre_X = build_genre_matrix(movies)
    text_X, tfidf, svd = build_text_matrix(movies, n_components=n_components)
    item_features = hstack([genre_X, text_X]).tocsr()
    movie_index = pd.Series(range(len(movies)), index=movies["movieId"].values)
    return item_features, movie_index, tfidf, svd


def save_item_features(
    item_features: csr_matrix,
    movie_index: pd.Series,
    tfidf: TfidfVectorizer,
    svd: TruncatedSVD,
) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
    save_npz(DATA_PROCESSED / "item_features.npz", item_features)
    movie_index.to_frame("position").to_parquet(DATA_PROCESSED / "movie_index.parquet")
    joblib.dump(tfidf, ARTIFACTS_MODELS / "tfidf.joblib")
    joblib.dump(svd, ARTIFACTS_MODELS / "svd_text.joblib")


def load_item_features() -> tuple[csr_matrix, pd.Series]:
    item_features = load_npz(DATA_PROCESSED / "item_features.npz")
    movie_index = pd.read_parquet(DATA_PROCESSED / "movie_index.parquet")["position"]
    return item_features, movie_index


# ─────────────────────────────────────────────────────────────────────────────
# Genome feature variant (extension). Kept entirely separate from the functions
# above so the original item_features.npz — used by the trained content model and
# hybrids — is never touched. This builds a SECOND representation: genre block ⊕
# a TruncatedSVD reduction of the dense 1,128-dim tag-genome relevance vectors.
# Movies without genome coverage get a zero genome block (genre signal only).
# ─────────────────────────────────────────────────────────────────────────────

def build_genome_matrix(
    movies: pd.DataFrame,
    genome_scores: pd.DataFrame,
    n_components: int = 256,
) -> tuple[csr_matrix, TruncatedSVD]:
    """Pivot genome-scores (movieId, tagId, relevance) into a movies × n_tags
    sparse matrix aligned to `movies` row order, then reduce with TruncatedSVD."""
    movie_pos = {mid: i for i, mid in enumerate(movies["movieId"].to_numpy())}
    pos = genome_scores["movieId"].map(movie_pos)
    mask = pos.notna()
    rows = pos[mask].astype(int).to_numpy()
    n_tags = int(genome_scores["tagId"].max())
    cols = genome_scores.loc[mask, "tagId"].to_numpy() - 1  # tagId is 1-indexed
    vals = genome_scores.loc[mask, "relevance"].to_numpy().astype("float32")

    genome = coo_matrix((vals, (rows, cols)), shape=(len(movies), n_tags)).tocsr()
    svd = TruncatedSVD(n_components=min(n_components, n_tags - 1), random_state=RANDOM_STATE)
    reduced = csr_matrix(svd.fit_transform(genome))
    return reduced, svd


def build_item_features_genome(
    movies: pd.DataFrame,
    genome_scores: pd.DataFrame,
    n_components: int = 256,
) -> tuple[csr_matrix, pd.Series, TruncatedSVD]:
    """Genre multi-hot ⊕ SVD-reduced tag-genome block (the genome feature variant)."""
    genre_X = build_genre_matrix(movies)
    genome_X, svd_genome = build_genome_matrix(movies, genome_scores, n_components)
    item_features = hstack([genre_X, genome_X]).tocsr()
    movie_index = pd.Series(range(len(movies)), index=movies["movieId"].values)
    return item_features, movie_index, svd_genome


def save_item_features_genome(item_features, movie_index, svd_genome) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
    save_npz(DATA_PROCESSED / "item_features_genome.npz", item_features)
    movie_index.to_frame("position").to_parquet(DATA_PROCESSED / "movie_index_genome.parquet")
    joblib.dump(svd_genome, ARTIFACTS_MODELS / "svd_genome.joblib")


def load_item_features_genome() -> tuple[csr_matrix, pd.Series]:
    item_features = load_npz(DATA_PROCESSED / "item_features_genome.npz")
    movie_index = pd.read_parquet(DATA_PROCESSED / "movie_index_genome.parquet")["position"]
    return item_features, movie_index


# ─────────────────────────────────────────────────────────────────────────────
# Semantic-embedding feature variant (extension). Encodes a per-movie text profile
# with a pretrained sentence-transformer → dense semantic vectors. Captures meaning
# that bag-of-words TF-IDF / genome cannot. Kept separate from the originals.
# ─────────────────────────────────────────────────────────────────────────────

def build_item_features_embedding(
    movies: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
):
    """L2-normalised sentence-transformer embeddings of `title | genres | tags`.

    Returns (item_features csr, movie_index, model_name). Drop-in replacement for
    the feature matrix consumed by ContentBasedRecommender.
    """
    from sentence_transformers import SentenceTransformer

    profile = (
        movies["clean_title"].fillna("")
        + " | genres: " + movies["genres"].fillna("").str.replace("|", " ", regex=False)
        + " | tags: " + movies["tags_text"].fillna("")
    ).tolist()
    model = SentenceTransformer(model_name)
    emb = model.encode(
        profile, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")
    item_features = csr_matrix(emb)
    movie_index = pd.Series(range(len(movies)), index=movies["movieId"].values)
    return item_features, movie_index, model_name
