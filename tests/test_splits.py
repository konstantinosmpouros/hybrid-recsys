import pandas as pd
import pytest
from hybrid_recsys.pipeline.splits import temporal_split


def _make_ratings(n_users: int = 10, n_per_user: int = 20) -> pd.DataFrame:
    rows = []
    for uid in range(n_users):
        for i in range(n_per_user):
            rows.append({"userId": uid, "movieId": i % 100, "rating": 4.0, "timestamp": i})
    return pd.DataFrame(rows)


def test_split_produces_three_non_empty_parts():
    ratings = _make_ratings()
    train, val, test = temporal_split(ratings)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_train_larger_than_val_and_test():
    ratings = _make_ratings()
    train, val, test = temporal_split(ratings)
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_val_and_test_users_subset_of_train():
    ratings = _make_ratings()
    train, val, test = temporal_split(ratings)
    train_users = set(train["userId"])
    assert set(val["userId"]).issubset(train_users)
    assert set(test["userId"]).issubset(train_users)


def test_no_rating_appears_in_two_splits():
    ratings = _make_ratings()
    train, val, test = temporal_split(ratings)
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_temporal_order_per_user():
    """All val/test timestamps must be >= the last train timestamp for each user."""
    ratings = _make_ratings()
    train, val, test = temporal_split(ratings)
    for uid in train["userId"].unique():
        max_train_ts = train[train["userId"] == uid]["timestamp"].max()
        user_val = val[val["userId"] == uid]
        if not user_val.empty:
            assert user_val["timestamp"].min() >= max_train_ts
        user_test = test[test["userId"] == uid]
        if not user_test.empty:
            assert user_test["timestamp"].min() >= max_train_ts


def test_users_with_too_few_ratings_are_dropped():
    tiny = pd.DataFrame([
        {"userId": 0, "movieId": 1, "rating": 4.0, "timestamp": 1},
        {"userId": 0, "movieId": 2, "rating": 3.0, "timestamp": 2},
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
    ])
    train, val, test = temporal_split(tiny, min_train_ratings=3)
    assert 0 not in set(train["userId"])
    assert 1 not in set(train["userId"])
