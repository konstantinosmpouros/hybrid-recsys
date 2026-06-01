from collections import OrderedDict

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from ..config import ARTIFACTS_MODELS

_SIM_CACHE_MAX = 20_000


class ContentBasedRecommender:
    """Item-item content-based recommender using cosine similarity on item features.

    Prediction formula (mean-centred):
        r_hat(u, j) = r_mean(u) + sum_i[ sim(i,j) * (r(u,i) - r_mean(u)) ]
                                    / sum_i[ |sim(i,j)| ] + eps
    where i ranges over the top-L content-similar items that user u has rated.
    """

    def __init__(self, n_neighbors: int = 50, cache_max: int = _SIM_CACHE_MAX):
        self.n_neighbors = n_neighbors
        self.cache_max = cache_max
        self.item_features: csr_matrix | None = None
        self.movie_index: pd.Series | None = None
        self._movie_ids: np.ndarray | None = None
        self._sim_cache: OrderedDict = OrderedDict()

    def fit(self, item_features: csr_matrix, movie_index: pd.Series) -> "ContentBasedRecommender":
        self.item_features = item_features
        self.movie_index = movie_index
        self._movie_ids = movie_index.index.values
        self._sim_cache = OrderedDict()
        return self

    def _similar_items(self, movie_id: int) -> tuple[np.ndarray, np.ndarray]:
        if movie_id in self._sim_cache:
            self._sim_cache.move_to_end(movie_id)
            return self._sim_cache[movie_id]
        pos = self.movie_index[movie_id]
        sims = cosine_similarity(self.item_features[pos], self.item_features).flatten()
        sims[pos] = 0.0  # exclude self
        top_idx = np.argsort(sims)[::-1][: self.n_neighbors]
        result = self._movie_ids[top_idx], sims[top_idx]
        self._sim_cache[movie_id] = result
        if len(self._sim_cache) > self.cache_max:
            self._sim_cache.popitem(last=False)
        return result

    def predict(self, user_ratings: dict[int, float], target_movie_id: int) -> float:
        """Return predicted rating for target_movie_id given user's rating history."""
        if target_movie_id not in self.movie_index.index:
            return np.nan
        if not user_ratings:
            return np.nan

        sim_movies, sim_scores = self._similar_items(target_movie_id)
        user_mean = float(np.mean(list(user_ratings.values())))
        num, denom = 0.0, 0.0

        for m, s in zip(sim_movies, sim_scores):
            if m in user_ratings:
                num += s * (user_ratings[m] - user_mean)
                denom += abs(s)

        if denom < 1e-9:
            return user_mean
        return float(np.clip(user_mean + num / denom, 0.5, 5.0))

    def recommend(
        self,
        user_ratings: dict[int, float],
        n: int = 10,
        exclude: set | None = None,
    ) -> list[tuple[int, float]]:
        """Return top-n (movieId, predicted_rating) for unseen items."""
        seen = set(user_ratings) | (exclude or set())
        candidates = [m for m in self._movie_ids if m not in seen]

        scored = [(m, self.predict(user_ratings, m)) for m in candidates]
        scored = [(m, p) for m, p in scored if not np.isnan(p)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def save(self, path=None) -> None:
        path = path or ARTIFACTS_MODELS / "content_model.joblib"
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None) -> "ContentBasedRecommender":
        path = path or ARTIFACTS_MODELS / "content_model.joblib"
        return joblib.load(path)
