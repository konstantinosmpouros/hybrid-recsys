import pandas as pd
import numpy as np
from ..config import DATA_PROCESSED


def temporal_split(
    ratings: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    min_train_ratings: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """User-wise temporal split.

    Each user's ratings are sorted by timestamp, then divided 80/10/10
    (train/val/test). Users with fewer than min_train_ratings + 2 are dropped.
    """
    ratings = ratings.sort_values(["userId", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []

    for _, user_df in ratings.groupby("userId", sort=False):
        n = len(user_df)
        if n < min_train_ratings + 2:
            continue

        n_train = max(min_train_ratings, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        # Guarantee at least one test row by stealing from train if needed.
        # Edge case: users with exactly min_train_ratings + 2 ratings would
        # otherwise produce an empty test partition.
        if n_train + n_val >= n:
            n_train = max(min_train_ratings, n - n_val - 1)

        train_rows.append(user_df.iloc[:n_train])
        val_rows.append(user_df.iloc[n_train : n_train + n_val])
        test_rows.append(user_df.iloc[n_train + n_val :])

    empty = ratings.iloc[:0]
    train = pd.concat(train_rows) if train_rows else empty
    val   = pd.concat(val_rows)   if val_rows   else empty
    test  = pd.concat(test_rows)  if test_rows  else empty

    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(DATA_PROCESSED / "split_train.parquet", index=False)
    val.to_parquet(DATA_PROCESSED / "split_val.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / "split_test.parquet", index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_PROCESSED / "split_train.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "split_val.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "split_test.parquet")
    return train, val, test
