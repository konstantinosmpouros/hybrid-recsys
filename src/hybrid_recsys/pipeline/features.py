import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
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
