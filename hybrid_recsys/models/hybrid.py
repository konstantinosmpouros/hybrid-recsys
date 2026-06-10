import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, LogisticRegression
from ..config import ARTIFACTS_MODELS
from ..evaluation.metrics import rmse


class WeightedHybrid:
    """Weighted ensemble: alpha * SVD + (1 - alpha) * ContentBased.

    alpha is tuned on the validation set by RMSE.
    Falls back to SVD prediction when the content model returns NaN.
    """

    def __init__(self, alpha: float = 0.75):
        self.alpha = alpha
        self.svd_model = None
        self.cb_model = None

    def set_models(self, svd_model, cb_model) -> "WeightedHybrid":
        self.svd_model = svd_model
        self.cb_model = cb_model
        return self

    def predict(self, user_id: int, movie_id: int, user_ratings: dict[int, float]) -> float:
        pred_svd = self.svd_model.predict(user_id, movie_id)
        pred_cb = self.cb_model.predict(user_ratings, movie_id)
        if np.isnan(pred_cb):
            return pred_svd
        return float(self.alpha * pred_svd + (1.0 - self.alpha) * pred_cb)

    def tune_alpha(
        self,
        val_df: pd.DataFrame,
        user_ratings_map: dict[int, dict[int, float]],
        alphas: np.ndarray | None = None,
    ) -> float:
        if alphas is None:
            alphas = np.arange(0.0, 1.05, 0.05)
        best_alpha, best_rmse = 0.5, float("inf")
        for a in alphas:
            self.alpha = a
            preds = np.array([
                self.predict(row.userId, row.movieId, user_ratings_map.get(row.userId, {}))
                for row in val_df.itertuples()
            ])
            r = rmse(val_df["rating"].values, preds)
            if r < best_rmse:
                best_rmse, best_alpha = r, a
        self.alpha = best_alpha
        return best_alpha

    def save(self) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ARTIFACTS_MODELS / "weighted_hybrid.joblib")

    @classmethod
    def load(cls) -> "WeightedHybrid":
        return joblib.load(ARTIFACTS_MODELS / "weighted_hybrid.joblib")


class StackedHybrid:
    """Ridge meta-learner trained on out-of-fold base-model predictions.

    Expected feature columns (in order):
        pred_content, pred_user_knn, pred_item_knn, pred_svd,
        item_popularity, user_rating_count, item_rating_count
    """

    FEATURE_NAMES = [
        "pred_content", "pred_user_knn", "pred_item_knn", "pred_svd",
        "item_popularity", "user_rating_count", "item_rating_count",
    ]

    def __init__(self, alpha: float = 1.0):
        self.meta = Ridge(alpha=alpha, random_state=42)
        # Side features attached at training time so serving doesn't need
        # to re-load the training data to score a (user, movie) pair.
        self.item_popularity: dict = {}
        self.user_count: dict = {}
        self.item_count: dict = {}
        self.global_mean: float = 3.5

    def set_side_features(
        self,
        item_popularity: dict,
        user_count: dict,
        item_count: dict,
        global_mean: float,
    ) -> "StackedHybrid":
        self.item_popularity = item_popularity
        self.user_count = user_count
        self.item_count = item_count
        self.global_mean = float(global_mean)
        return self

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "StackedHybrid":
        self.meta.fit(X_meta, y)
        return self

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return np.clip(self.meta.predict(X_meta), 0.5, 5.0)

    def predict_one(self, user_id, movie_id, base_preds: np.ndarray) -> float:
        """Score a single (user, movie) given the 4 base-model predictions.

        Used at serving time so the bundle doesn't need to keep the training
        DataFrame in memory. `base_preds` must be ordered [cb, user_knn,
        item_knn, svd]. Returns global_mean if any base prediction is NaN.
        """
        if np.isnan(base_preds).any():
            return float(self.global_mean)
        X = np.array([[
            *base_preds,
            self.item_popularity.get(movie_id, 0),
            self.user_count.get(user_id, 0),
            self.item_count.get(movie_id, 0),
        ]], dtype=float)
        return float(self.predict(X)[0])

    def save(self) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ARTIFACTS_MODELS / "stacked_hybrid.joblib")

    @classmethod
    def load(cls) -> "StackedHybrid":
        return joblib.load(ARTIFACTS_MODELS / "stacked_hybrid.joblib")


class DualHeadHybrid:
    """Two-head blending hybrid that targets *all* metrics at once.

    Both heads consume the SAME feature vector — the base-model predictions plus
    side features — but optimise different objectives:

      * rating_head : Ridge regression → predicts the rating  (drives RMSE/MAE)
      * rank_head   : LogisticRegression → predicts P(rating ≥ 4)  (drives P/R/F1@K)

    Trained by *blending* on the validation split (frozen base models, no OOF
    re-run). NaN base predictions (e.g. LightGCN on out-of-graph users) are imputed
    with the per-feature median learned at fit time.
    """

    FEATURE_NAMES = [
        "pred_content_genome", "pred_user_knn", "pred_item_knn", "pred_svd",
        "pred_lightgcn", "item_popularity", "user_rating_count", "item_rating_count",
    ]

    def __init__(self, ridge_alpha: float = 1.0, random_state: int = 42):
        self.rating_head = Ridge(alpha=ridge_alpha, random_state=random_state)
        self.rank_head = LogisticRegression(max_iter=1000, random_state=random_state)
        self.impute_: np.ndarray | None = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        if self.impute_ is not None:
            nan = np.isnan(X)
            if nan.any():
                idx = np.where(nan)
                X[idx] = np.take(self.impute_, idx[1])
        return X

    def fit(self, X: np.ndarray, ratings: np.ndarray) -> "DualHeadHybrid":
        X = np.asarray(X, dtype=float)
        med = np.nanmedian(X, axis=0)
        self.impute_ = np.where(np.isnan(med), 0.0, med)
        Xi = self._prep(X)
        ratings = np.asarray(ratings, dtype=float)
        self.rating_head.fit(Xi, ratings)
        self.rank_head.fit(Xi, (ratings >= 4.0).astype(int))
        return self

    def predict_rating(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.rating_head.predict(self._prep(X)), 0.5, 5.0)

    def rank_score(self, X: np.ndarray) -> np.ndarray:
        return self.rank_head.predict_proba(self._prep(X))[:, 1]

    def predict_rating_one(self, feats) -> float:
        return float(self.predict_rating(np.asarray(feats, dtype=float).reshape(1, -1))[0])

    def rank_score_one(self, feats) -> float:
        return float(self.rank_score(np.asarray(feats, dtype=float).reshape(1, -1))[0])

    def save(self, path=None) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path or ARTIFACTS_MODELS / "dual_head_hybrid.joblib")

    @classmethod
    def load(cls, path=None) -> "DualHeadHybrid":
        return joblib.load(path or ARTIFACTS_MODELS / "dual_head_hybrid.joblib")
