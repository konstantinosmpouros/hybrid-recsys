import pandas as pd
from ..config import DATA_RAW, DATA_PROCESSED


def load_raw_ratings() -> pd.DataFrame:
    return pd.read_csv(
        DATA_RAW / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )


def load_raw_movies() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "movies.csv", dtype={"movieId": "int32"})


def load_raw_tags() -> pd.DataFrame:
    return pd.read_csv(
        DATA_RAW / "tags.csv",
        dtype={"userId": "int32", "movieId": "int32"},
    )


def load_genome_scores() -> pd.DataFrame:
    return pd.read_csv(
        DATA_RAW / "genome-scores.csv",
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    )


def load_genome_tags() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "genome-tags.csv", dtype={"tagId": "int32"})


def _extract_year(title: pd.Series) -> pd.Series:
    return title.str.extract(r"\((\d{4})\)$")[0].astype("Int16")


def _clean_title(title: pd.Series) -> pd.Series:
    return title.str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()


def build_movies_table(movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    movies = movies_df.copy()
    movies["year"] = _extract_year(movies["title"])
    movies["clean_title"] = _clean_title(movies["title"])

    tags_agg = (
        tags_df.dropna(subset=["tag"])
        .assign(tag=lambda df: df["tag"].str.lower().str.strip())
        .groupby("movieId")["tag"]
        .apply(" ".join)
        .rename("tags_text")
    )
    movies = movies.merge(tags_agg, on="movieId", how="left")
    movies["tags_text"] = movies["tags_text"].fillna("")
    return movies


def save_processed(df: pd.DataFrame, name: str) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / f"{name}.parquet", index=False)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / f"{name}.parquet")
