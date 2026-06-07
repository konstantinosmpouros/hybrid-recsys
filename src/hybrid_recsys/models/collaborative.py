import numpy as np
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise.model_selection import GridSearchCV
from ..config import ARTIFACTS_MODELS, RATING_SCALE, RANDOM_STATE


def _to_surprise(ratings_df: pd.DataFrame):
    reader = Reader(rating_scale=RATING_SCALE)
    df = ratings_df[["userId", "movieId", "rating"]].copy()
    df["userId"] = df["userId"].astype(str)
    df["movieId"] = df["movieId"].astype(str)
    return Dataset.load_from_df(df, reader)


class SVDModel:
    """Surprise SVD wrapper with grid-search tuning."""

    def __init__(self):
        self.model: SVD | None = None
        self.best_params: dict = {}

    def tune(self, train_df: pd.DataFrame, param_grid: dict | None = None) -> "SVDModel":
        if param_grid is None:
            param_grid = {
                "n_factors": [50, 100, 200],
                "n_epochs": [20, 40],
                "lr_all": [0.002, 0.005],
                "reg_all": [0.02, 0.05],
            }
        # Pin the random_state on every grid candidate so the SVD factor
        # initialisation is deterministic (otherwise Surprise seeds from the OS
        # RNG and RMSE/MAE drift run-to-run). Copied so the caller's dict is
        # left untouched; best_params then carries random_state into the OOF
        # refits in the stacked-hybrid notebook (OOF folds) as well.
        param_grid = {**param_grid}
        param_grid.setdefault("random_state", [RANDOM_STATE])
        data = _to_surprise(train_df)
        gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=-1)
        gs.fit(data)
        self.best_params = gs.best_params["rmse"]
        self.model = gs.best_estimator["rmse"]
        return self

    def fit(self, train_df: pd.DataFrame) -> "SVDModel":
        data = _to_surprise(train_df)
        self.model.fit(data.build_full_trainset())
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.model.predict(str(user_id), str(movie_id)).est

    def save(self) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ARTIFACTS_MODELS / "svd_model.joblib")

    @classmethod
    def load(cls) -> "SVDModel":
        return joblib.load(ARTIFACTS_MODELS / "svd_model.joblib")


class ItemKNNModel:
    """Surprise KNNWithMeans item-based CF wrapper.

    max_items: retain only the top-N most-rated items before fitting.
    A full 62K-item similarity matrix requires ~15 GB RAM; capping at 15K
    reduces this to ~900 MB while preserving coverage for popular titles.
    """

    def __init__(self, k: int = 80, min_k: int = 5, max_items: int = 15_000):
        self.k = k
        self.min_k = min_k
        self.max_items = max_items
        self.model: KNNWithMeans | None = None

    def fit(self, train_df: pd.DataFrame) -> "ItemKNNModel":
        df = train_df
        if self.max_items and df["movieId"].nunique() > self.max_items:
            top_items = df["movieId"].value_counts().head(self.max_items).index
            df = df[df["movieId"].isin(top_items)]
        data = _to_surprise(df)
        sim_options = {"name": "pearson_baseline", "user_based": False}
        self.model = KNNWithMeans(k=self.k, min_k=self.min_k, sim_options=sim_options)
        self.model.fit(data.build_full_trainset())
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.model.predict(str(user_id), str(movie_id)).est

    def save(self) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ARTIFACTS_MODELS / "item_knn_model.joblib")

    @classmethod
    def load(cls) -> "ItemKNNModel":
        return joblib.load(ARTIFACTS_MODELS / "item_knn_model.joblib")


class UserKNNModel:
    """Surprise KNNWithMeans user-based CF wrapper.

    max_users: randomly sample this many users before fitting.
    A full 162K-user similarity matrix requires ~98 GB RAM; capping at 20K
    reduces this to ~1.6 GB while retaining a representative user population.
    """

    def __init__(self, k: int = 80, min_k: int = 5, max_users: int = 20_000,
                 random_state: int = 42):
        self.k = k
        self.min_k = min_k
        self.max_users = max_users
        self.random_state = random_state
        self.model: KNNWithMeans | None = None

    def fit(self, train_df: pd.DataFrame) -> "UserKNNModel":
        df = train_df
        if self.max_users and df["userId"].nunique() > self.max_users:
            rng = np.random.default_rng(self.random_state)
            sampled_users = rng.choice(
                df["userId"].unique(), self.max_users, replace=False
            )
            df = df[df["userId"].isin(sampled_users)]
        data = _to_surprise(df)
        sim_options = {"name": "pearson_baseline", "user_based": True}
        self.model = KNNWithMeans(k=self.k, min_k=self.min_k, sim_options=sim_options)
        self.model.fit(data.build_full_trainset())
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.model.predict(str(user_id), str(movie_id)).est

    def save(self) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ARTIFACTS_MODELS / "user_knn_model.joblib")

    @classmethod
    def load(cls) -> "UserKNNModel":
        return joblib.load(ARTIFACTS_MODELS / "user_knn_model.joblib")
