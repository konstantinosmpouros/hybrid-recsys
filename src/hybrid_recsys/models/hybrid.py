import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
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

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "StackedHybrid":
        self.meta.fit(X_meta, y)
        return self

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return np.clip(self.meta.predict(X_meta), 0.5, 5.0)

    def save(self) -> None:
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ARTIFACTS_MODELS / "stacked_hybrid.joblib")

    @classmethod
    def load(cls) -> "StackedHybrid":
        return joblib.load(ARTIFACTS_MODELS / "stacked_hybrid.joblib")
