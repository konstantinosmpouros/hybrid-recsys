import json
import numpy as np
import pandas as pd
from .config import DATA_PROCESSED, ARTIFACTS_METRICS
from .models.content import ContentBasedRecommender
from .models.collaborative import SVDModel, ItemKNNModel, UserKNNModel
from .models.hybrid import WeightedHybrid, StackedHybrid


class RecommenderBundle:
    """Loads all trained models and exposes a single recommendation interface."""

    def __init__(self):
        self.cb: ContentBasedRecommender | None = None
        self.svd: SVDModel | None = None
        self.item_knn: ItemKNNModel | None = None
        self.user_knn: UserKNNModel | None = None
        self.weighted: WeightedHybrid | None = None
        self.stacked: StackedHybrid | None = None
        self.movies_df: pd.DataFrame | None = None

    def load(self) -> "RecommenderBundle":
        self.cb = ContentBasedRecommender.load()
        self.svd = SVDModel.load()
        self.item_knn = ItemKNNModel.load()
        self.user_knn = UserKNNModel.load()
        self.weighted = WeightedHybrid.load()
        self.stacked = StackedHybrid.load()
        self.movies_df = pd.read_parquet(DATA_PROCESSED / "movies.parquet")
        return self

    def _predict_fn(self, model: str, user_id: int, user_ratings: dict[int, float]):
        def stacked_predict(m):
            base = np.array([
                self.cb.predict(user_ratings, m),
                self.user_knn.predict(user_id, m),
                self.item_knn.predict(user_id, m),
                self.svd.predict(user_id, m),
            ], dtype=float)
            return self.stacked.predict_one(user_id, m, base)

        dispatch = {
            "content":  lambda m: self.cb.predict(user_ratings, m),
            "svd":      lambda m: self.svd.predict(user_id, m),
            "item_knn": lambda m: self.item_knn.predict(user_id, m),
            "user_knn": lambda m: self.user_knn.predict(user_id, m),
            "weighted": lambda m: self.weighted.predict(user_id, m, user_ratings),
            "stacked":  stacked_predict,
        }
        if model not in dispatch:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(dispatch)}")
        return dispatch[model]

    def get_recommendations(
        self,
        user_id: int,
        user_ratings: dict[int, float],
        model: str = "weighted",
        n: int = 10,
        exclude: set | None = None,
    ) -> pd.DataFrame:
        seen = set(user_ratings) | (exclude or set())
        candidates = [m for m in self.movies_df["movieId"].values if m not in seen]

        predict = self._predict_fn(model, user_id, user_ratings)
        scored = [(m, predict(m)) for m in candidates]
        scored = sorted([(m, s) for m, s in scored if not np.isnan(s)], key=lambda x: x[1], reverse=True)

        top = pd.DataFrame(scored[:n], columns=["movieId", "predicted_rating"])
        top = top.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
        top["predicted_rating"] = top["predicted_rating"].round(2)
        return top.reset_index(drop=True)

    def load_metrics(self) -> dict:
        path = ARTIFACTS_METRICS / "all_metrics.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}
