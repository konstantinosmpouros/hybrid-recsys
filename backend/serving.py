"""Serving layer for the hybrid recommender.

`RecommenderBundle` owns the trained models **and** read access to the processed
data (movie catalogue, train/test ratings). It exposes every operation the API
needs: recommendations (with genre/year filters), cold-start recommendations from
an in-session ratings dict, per-model prediction inspection, "why this?"
explanations, movie search and item-to-item similarity.
"""
import gc
import json
import threading
import numpy as np
import pandas as pd
from hybrid_recsys.config import DATA_PROCESSED, ARTIFACTS_METRICS, ARTIFACTS_MODELS
from hybrid_recsys.models.content import ContentBasedRecommender
from hybrid_recsys.models.collaborative import SVDModel, ItemKNNModel, UserKNNModel
from hybrid_recsys.models.hybrid import WeightedHybrid, StackedHybrid, DualHeadHybrid
from hybrid_recsys.models.lightgcn import LightGCNRecommender


# Model registry — single source of truth shared with the API/app.
# key -> (display label, family, ranking_only, cold_start)
#   ranking_only : emits a relevance score, not a calibrated rating (don't show
#                  as a star value, don't use for RMSE/MAE; rank by raw score).
#   cold_start   : can recommend for a brand-new user from an in-session ratings
#                  dict alone (content models + the hybrids that consume ratings).
#                  Pure CF (SVD/kNN/LightGCN) needs a user present at train time.
MODEL_REGISTRY: dict[str, tuple[str, str, bool, bool]] = {
    "dual":           ("Dual-Head Hybrid",        "Hybrid",        False, True),
    "stacked":        ("Stacked Hybrid",          "Hybrid",        False, True),
    "weighted":       ("Weighted Hybrid",         "Hybrid",        False, True),
    "svd":            ("SVD (Matrix Factoris.)",  "Collaborative", False, False),
    "item_knn":       ("Item-Based k-NN",         "Collaborative", False, False),
    "user_knn":       ("User-Based k-NN",         "Collaborative", False, False),
    "lightgcn":       ("LightGCN (graph CF)",     "Collaborative", True,  False),
    "content_genome": ("Content — Tag Genome",    "Content-Based", False, True),
    "content":        ("Content — TF-IDF",        "Content-Based", False, True),
    "content_embed":  ("Content — Embeddings",    "Content-Based", False, True),
}

# Heavy at serve time (many base-model calls per candidate, or a restricted graph
# vocabulary): retrieve a popularity-ranked candidate pool first, then re-rank.
_POOLED_MODELS = {"dual", "lightgcn"}
_DEFAULT_POOL = 3000

# Artifact file(s) each model needs — lets the API report availability WITHOUT
# loading the ~8 GB bundle (so `/api/models` answers instantly at startup).
_MODEL_FILES: dict[str, list[str]] = {
    "dual":           ["dual_head_hybrid.joblib", "content_genome_model.joblib", "lightgcn_model.joblib"],
    "stacked":        ["stacked_hybrid.joblib"],
    "weighted":       ["weighted_hybrid.joblib"],
    "svd":            ["svd_model.joblib"],
    "item_knn":       ["item_knn_model.joblib"],
    "user_knn":       ["user_knn_model.joblib"],
    "lightgcn":       ["lightgcn_model.joblib"],
    "content_genome": ["content_genome_model.joblib"],
    "content":        ["content_model.joblib"],
    "content_embed":  ["content_embed_model.joblib"],
}


def static_model_info() -> list[dict]:
    """Model registry + availability by artifact-file existence (no bundle load)."""
    out = []
    for k, (label, family, ranking_only, cold_start) in MODEL_REGISTRY.items():
        files = _MODEL_FILES.get(k, [])
        avail = bool(files) and all((ARTIFACTS_MODELS / f).exists() for f in files)
        out.append({"key": k, "label": label, "family": family,
                    "ranking_only": ranking_only, "cold_start": cold_start, "available": avail})
    return out


# "lite" set — content models + LightGCN only (~2 GB total). Excludes EVERY
# Surprise model: each keeps the full 20M-rating trainset in RAM as Python
# objects, so the compact .joblib explodes on load — SVD alone ≈ 6.8 GB,
# Weighted ≈ 7.2 GB (it embeds an SVD), and the kNN models are larger still.
# Use RECSYS_MODELS=all only with lots of free RAM (≈ 20 GB for the full set).
_LITE_MODELS = ["content", "content_genome", "content_embed", "lightgcn"]


def _selected_models() -> set[str] | None:
    """Models to PRELOAD at startup. Default is NONE — models load on demand when
    the user clicks "Load" in the UI. `RECSYS_MODELS=all` preloads everything (≈20 GB),
    `lite` preloads the content+LightGCN set, or pass a comma-separated list of keys."""
    import os
    val = os.environ.get("RECSYS_MODELS", "").strip().lower()
    if val in ("", "none"):
        return set()                      # preload nothing — pure on-demand
    if val == "all":
        return None                       # preload everything
    if val == "lite":
        return set(_LITE_MODELS)
    return {k.strip() for k in val.split(",") if k.strip()}


# Each model's atomic dependencies — what must be in memory for it to score.
# (Loading "weighted" needs only its own self-contained joblib; "stacked"/"dual"
# pull their base models in because the bundle builds their feature vectors.)
_MODEL_DEPS: dict[str, list[str]] = {
    "content": ["content"], "content_genome": ["content_genome"],
    "content_embed": ["content_embed"], "svd": ["svd"],
    "item_knn": ["item_knn"], "user_knn": ["user_knn"],
    "lightgcn": ["lightgcn"], "weighted": ["weighted"],
    "stacked": ["stacked", "content", "user_knn", "item_knn", "svd"],
    "dual": ["dual", "content_genome", "user_knn", "item_knn", "svd", "lightgcn"],
}

# Approximate resident RAM per atomic model (GB), measured on this dataset. The
# Surprise models are huge because they retain the full 20M-rating trainset.
_ATOMIC_RAM_GB: dict[str, float] = {
    "content": 0.4, "content_genome": 0.25, "content_embed": 0.45,
    "svd": 6.9, "item_knn": 3.0, "user_knn": 3.5, "weighted": 7.2,
    "stacked": 0.05, "lightgcn": 0.2, "dual": 0.05,
}

# A model is "heavy" if it costs more than this much RAM. This machine fits at
# most ONE heavy model at a time, so loading any model first unloads every other
# heavy model it doesn't itself depend on. Everything ≤ this stays resident and
# coexists freely: the three content models (~0.3 GB each), LightGCN (~0.2 GB),
# and the meta-learner heads. Heavy = the kNNs (~3 GB) and SVD/Weighted (~7 GB).
_HEAVY_GB = 1.0


def release_memory() -> None:
    """Force-reclaim freed memory after unloading a model: run the GC and, if
    torch happens to be loaded with CUDA, clear its cache. Serving is CPU-only,
    so the CUDA call is purely defensive."""
    gc.collect()
    import sys
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def available_gb() -> float:
    """Free RAM in GB, cgroup/container-aware (reads /proc/meminfo on Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024   # kB -> GB
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return float("inf")


class RecommenderBundle:
    """Loads all trained models + processed data and serves every API operation.

    The six core models are required; the four extension models (Dual-Head,
    LightGCN, Content-Genome, Content-Embedding) load gracefully — a missing
    artifact just drops that model from `available_models()`.
    """

    def __init__(self):
        # core (required)
        self.cb: ContentBasedRecommender | None = None
        self.svd: SVDModel | None = None
        self.item_knn: ItemKNNModel | None = None
        self.user_knn: UserKNNModel | None = None
        self.weighted: WeightedHybrid | None = None
        self.stacked: StackedHybrid | None = None
        # extensions (optional)
        self.cb_genome: ContentBasedRecommender | None = None
        self.cb_embed: ContentBasedRecommender | None = None
        self.lightgcn: LightGCNRecommender | None = None
        self.dual: DualHeadHybrid | None = None
        # data
        self.movies_df: pd.DataFrame | None = None
        self.train_ratings: pd.DataFrame | None = None   # indexed by userId
        self.test_ratings: pd.DataFrame | None = None     # indexed by userId
        # derived lookups
        self.item_pop: dict[int, int] = {}    # train rating count per movie
        self.user_cnt: dict[int, int] = {}    # train rating count per user
        self._pop_order: np.ndarray | None = None   # movieIds, most-rated first
        self._movie_year: dict[int, int] = {}
        self._movie_genres: dict[int, set] = {}
        self._meta_cols: list[str] = ["movieId", "title", "genres", "year"]
        # readiness flags — load is split so the catalogue/search UI works within
        # seconds while the ~8 GB of models load separately (in a background thread).
        self.data_ready: bool = False
        self.models_ready: bool = False         # True only when the full set is loaded
        self._load_lock = threading.Lock()       # serialises on-demand model loads

    # ── loading ──────────────────────────────────────────────────────────────
    def load(self) -> "RecommenderBundle":
        self.load_data()
        self.load_models(only=None)
        return self

    def load_data(self) -> "RecommenderBundle":
        """Fast (~seconds): movie catalogue + train/test ratings + lookups.
        Enough to serve search, users, genres, popular, profiles."""
        self.movies_df = pd.read_parquet(DATA_PROCESSED / "movies.parquet")
        self._load_data()
        self.data_ready = True
        return self

    def load_models(self, only: set[str] | None = None) -> "RecommenderBundle":
        """Preload a set of models (used at startup only when RECSYS_MODELS is set).
        `only=None` → everything; otherwise the given keys. Day-to-day the app
        loads each model lazily via `ensure_model` when the user clicks "Load"."""
        keys = list(MODEL_REGISTRY) if only is None else list(only)
        for k in keys:
            self.ensure_model(k)
        if only is None:
            self.models_ready = True
        return self

    def _load_one(self, key: str) -> None:
        """Load a single atomic model object if not already in memory."""
        if key == "content" and self.cb is None:
            self.cb = ContentBasedRecommender.load()
        elif key == "svd" and self.svd is None:
            self.svd = SVDModel.load()
        elif key == "item_knn" and self.item_knn is None:
            self.item_knn = ItemKNNModel.load()
        elif key == "user_knn" and self.user_knn is None:
            self.user_knn = UserKNNModel.load()
        elif key == "weighted" and self.weighted is None:
            self.weighted = WeightedHybrid.load()
        elif key == "stacked" and self.stacked is None:
            self.stacked = StackedHybrid.load()
        elif key == "content_genome" and self.cb_genome is None:
            self.cb_genome = ContentBasedRecommender.load(path=ARTIFACTS_MODELS / "content_genome_model.joblib")
        elif key == "content_embed" and self.cb_embed is None:
            self.cb_embed = ContentBasedRecommender.load(path=ARTIFACTS_MODELS / "content_embed_model.joblib")
        elif key == "lightgcn" and self.lightgcn is None:
            self.lightgcn = LightGCNRecommender.load()
        elif key == "dual" and self.dual is None:
            self.dual = DualHeadHybrid.load()

    def ensure_model(self, key: str) -> bool:
        """Load `key` (and the base models it depends on) on demand. Thread-safe;
        a no-op if already loaded. Returns True if the model is usable afterwards."""
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{key}'.")
        with self._load_lock:
            for dep in _MODEL_DEPS.get(key, [key]):
                self._load_one(dep)
        return key in self.loaded_models()

    def loaded_models(self) -> set[str]:
        """Model keys currently in memory and ready to score."""
        return set(self._build_dispatch(-1, {}).keys())

    _ATTR = {"content": "cb", "svd": "svd", "item_knn": "item_knn", "user_knn": "user_knn",
             "weighted": "weighted", "stacked": "stacked", "content_genome": "cb_genome",
             "content_embed": "cb_embed", "lightgcn": "lightgcn", "dual": "dual"}

    def _atomic_loaded(self, dep: str) -> bool:
        return getattr(self, self._ATTR.get(dep, dep), None) is not None

    def estimate_load_gb(self, key: str) -> float:
        """Extra RAM (GB) that loading `key` would add — sum of its not-yet-loaded
        atomic dependencies. ~0 if already loaded."""
        return round(sum(_ATOMIC_RAM_GB.get(d, 0.0)
                         for d in _MODEL_DEPS.get(key, [key]) if not self._atomic_loaded(d)), 2)

    def total_ram_gb(self, key: str) -> float:
        """Full RAM (GB) the model occupies once loaded (all deps)."""
        return round(sum(_ATOMIC_RAM_GB.get(d, 0.0) for d in _MODEL_DEPS.get(key, [key])), 2)

    def _evictable_for(self, key: str) -> list[str]:
        """Loaded heavy (> _HEAVY_GB) atomic models that `key` does NOT need —
        these get unloaded before `key` loads, so only one heavy model is ever
        resident at a time (the small models, which always stay, are excluded)."""
        required = set(_MODEL_DEPS.get(key, [key]))
        return [d for d in self._ATTR
                if d not in required and self._atomic_loaded(d)
                and _ATOMIC_RAM_GB.get(d, 0) > _HEAVY_GB]

    def freeable_gb(self, key: str) -> float:
        """RAM (GB) we could free for `key` by unloading other large models."""
        return round(sum(_ATOMIC_RAM_GB.get(d, 0) for d in self._evictable_for(key)), 2)

    def evict_for(self, key: str) -> list[str]:
        """Unload every heavy model `key` doesn't need (the small models stay), so
        only one heavy model is ever resident. Returns the labels unloaded."""
        freed = []
        with self._load_lock:
            for d in self._evictable_for(key):
                setattr(self, self._ATTR[d], None)
                freed.append(self.label(d))
            if freed:
                release_memory()
        return freed

    @staticmethod
    def _try(fn):
        try:
            return fn()
        except Exception:
            return None

    def _load_data(self) -> None:
        # Movie metadata lookups for filters / display.
        m = self.movies_df
        years = m["year"].astype("float").to_numpy()
        self._movie_year = {int(mid): (int(y) if not np.isnan(y) else None)
                            for mid, y in zip(m["movieId"].to_numpy(), years)}
        self._movie_genres = {int(mid): set(str(g).split("|")) if g and g != "(no genres listed)" else set()
                              for mid, g in zip(m["movieId"].to_numpy(), m["genres"].to_numpy())}

        # Train ratings: histories + side features + popularity retrieval order.
        try:
            tr = pd.read_parquet(DATA_PROCESSED / "split_train.parquet",
                                 columns=["userId", "movieId", "rating"])
            self.item_pop = tr.groupby("movieId").size().to_dict()
            self.user_cnt = tr.groupby("userId").size().to_dict()
            order = sorted(self.item_pop.items(), key=lambda kv: kv[1], reverse=True)
            self._pop_order = np.array([m for m, _ in order])
            self.train_ratings = tr.set_index("userId").sort_index()
        except Exception:
            self.item_pop, self.user_cnt, self._pop_order = {}, {}, None
            self.train_ratings = None

        # Held-out test ratings: the "true rating" for the prediction inspector.
        self.test_ratings = self._try(
            lambda: pd.read_parquet(DATA_PROCESSED / "split_test.parquet",
                                    columns=["userId", "movieId", "rating"]).set_index("userId").sort_index())

    # ── dispatch ─────────────────────────────────────────────────────────────
    def _dual_feats(self, user_id: int, user_ratings: dict, m: int) -> list:
        # Exact feature order DualHeadHybrid was trained on (notebook 12):
        # [content_genome, user_knn, item_knn, svd, lightgcn,
        #  item_popularity, user_rating_count, item_rating_count]
        return [
            self.cb_genome.predict(user_ratings, m),
            self.user_knn.predict(user_id, m),
            self.item_knn.predict(user_id, m),
            self.svd.predict(user_id, m),
            self.lightgcn.predict(user_id, m),
            self.item_pop.get(m, 0),
            self.user_cnt.get(user_id, 0),
            self.item_pop.get(m, 0),
        ]

    def _build_dispatch(self, user_id: int, user_ratings: dict) -> dict:
        def stacked_predict(m):
            base = np.array([
                self.cb.predict(user_ratings, m),
                self.user_knn.predict(user_id, m),
                self.item_knn.predict(user_id, m),
                self.svd.predict(user_id, m),
            ], dtype=float)
            return self.stacked.predict_one(user_id, m, base)

        def dual_predict(m):
            return self.dual.predict_rating_one(self._dual_feats(user_id, user_ratings, m))

        # Only expose a model whose object(s) actually loaded — so `lite` mode
        # (or any RECSYS_MODELS subset) doesn't advertise models it can't score.
        dispatch = {}
        if self.cb is not None:
            dispatch["content"] = lambda m: self.cb.predict(user_ratings, m)
        if self.svd is not None:
            dispatch["svd"] = lambda m: self.svd.predict(user_id, m)
        if self.item_knn is not None:
            dispatch["item_knn"] = lambda m: self.item_knn.predict(user_id, m)
        if self.user_knn is not None:
            dispatch["user_knn"] = lambda m: self.user_knn.predict(user_id, m)
        if self.weighted is not None:
            dispatch["weighted"] = lambda m: self.weighted.predict(user_id, m, user_ratings)
        if self.stacked is not None and None not in (self.cb, self.user_knn, self.item_knn, self.svd):
            dispatch["stacked"] = stacked_predict
        if self.cb_genome is not None:
            dispatch["content_genome"] = lambda m: self.cb_genome.predict(user_ratings, m)
        if self.cb_embed is not None:
            dispatch["content_embed"] = lambda m: self.cb_embed.predict(user_ratings, m)
        if self.lightgcn is not None:
            dispatch["lightgcn"] = lambda m: self.lightgcn.predict(user_id, m)
        if self.dual is not None and None not in (self.cb_genome, self.lightgcn, self.user_knn, self.item_knn, self.svd):
            dispatch["dual"] = dual_predict
        return dispatch

    def _predict_fn(self, model: str, user_id: int, user_ratings: dict):
        dispatch = self._build_dispatch(user_id, user_ratings)
        if model not in dispatch:
            raise ValueError(f"Unknown / unavailable model '{model}'. Choose from {list(dispatch)}")
        return dispatch[model]

    # ── registry helpers ───────────────────────────────────────────────────────
    def available_models(self) -> list[str]:
        present = set(self._build_dispatch(-1, {}).keys())
        return [k for k in MODEL_REGISTRY if k in present]

    def model_info(self) -> list[dict]:
        """Per-model: `available` = artifact on disk (loadable), `loaded` = in RAM now."""
        loaded = self.loaded_models()
        on_disk = {m["key"]: m["available"] for m in static_model_info()}
        out = []
        for k, (label, family, ranking_only, cold_start) in MODEL_REGISTRY.items():
            out.append({"key": k, "label": label, "family": family,
                        "ranking_only": ranking_only, "cold_start": cold_start,
                        "available": on_disk.get(k, False), "loaded": k in loaded,
                        "ram_gb": self.total_ram_gb(k)})
        return out

    @staticmethod
    def label(model: str) -> str:
        return MODEL_REGISTRY.get(model, (model, "", False, False))[0]

    @staticmethod
    def is_ranking_only(model: str) -> bool:
        return MODEL_REGISTRY.get(model, ("", "", False, False))[2]

    # ── data access ────────────────────────────────────────────────────────────
    def _meta(self, movie_ids) -> pd.DataFrame:
        cols = [c for c in self._meta_cols if c in self.movies_df.columns]
        return self.movies_df[self.movies_df["movieId"].isin(list(movie_ids))][cols]

    def movie_meta(self, movie_id: int) -> dict | None:
        row = self.movies_df[self.movies_df["movieId"] == int(movie_id)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {"movieId": int(r["movieId"]), "title": str(r["title"]),
                "genres": str(r["genres"]), "year": self._movie_year.get(int(movie_id))}

    def genres_list(self) -> list[str]:
        seen = set()
        for gs in self._movie_genres.values():
            seen |= gs
        seen.discard("(no genres listed)")
        return sorted(seen)

    def user_ratings(self, user_id: int) -> dict[int, float]:
        if self.train_ratings is None:
            return {}
        try:
            sub = self.train_ratings.loc[[int(user_id)]]
        except KeyError:
            return {}
        return {int(m): float(r) for m, r in zip(sub["movieId"], sub["rating"])}

    def user_exists(self, user_id: int) -> bool:
        return int(user_id) in self.user_cnt

    def sample_users(self, n: int = 200, min_ratings: int = 20, max_ratings: int = 500,
                     seed: int = 42) -> list[int]:
        pool = [u for u, c in self.user_cnt.items() if min_ratings <= c <= max_ratings]
        if not pool:
            pool = list(self.user_cnt.keys())
        rng = np.random.default_rng(seed)
        n = min(n, len(pool))
        return sorted(int(u) for u in rng.choice(pool, size=n, replace=False))

    def user_profile(self, user_id: int, top: int = 15) -> dict:
        ur = self.user_ratings(user_id)
        if not ur:
            return {"user_id": int(user_id), "n_ratings": 0, "mean_rating": None, "history": []}
        meta = self._meta(ur.keys()).set_index("movieId")
        rows = []
        for mid, r in ur.items():
            if mid in meta.index:
                mr = meta.loc[mid]
                rows.append({"movieId": mid, "title": str(mr["title"]),
                             "genres": str(mr["genres"]), "year": self._movie_year.get(mid),
                             "rating": round(float(r), 2)})
        rows.sort(key=lambda x: x["rating"], reverse=True)
        return {"user_id": int(user_id), "n_ratings": len(ur),
                "mean_rating": round(float(np.mean(list(ur.values()))), 3),
                "history": rows[:top]}

    def true_rating(self, user_id: int, movie_id: int) -> dict | None:
        """Held-out (test) rating if present, else the train rating. None if unseen."""
        for src, frame in (("test", self.test_ratings), ("train", self.train_ratings)):
            if frame is None:
                continue
            try:
                sub = frame.loc[[int(user_id)]]
            except KeyError:
                continue
            hit = sub[sub["movieId"] == int(movie_id)]
            if not hit.empty:
                return {"rating": round(float(hit["rating"].iloc[0]), 2), "source": src}
        return None

    def search_movies(self, query: str, limit: int = 25) -> list[dict]:
        q = (query or "").strip().lower()
        if not q:
            return []
        m = self.movies_df
        col = "clean_title" if "clean_title" in m.columns else "title"
        mask = m[col].str.lower().str.contains(q, regex=False, na=False) | \
            m["title"].str.lower().str.contains(q, regex=False, na=False)
        hits = m[mask].copy()
        hits["_pop"] = hits["movieId"].map(lambda x: self.item_pop.get(int(x), 0))
        hits = hits.sort_values("_pop", ascending=False).head(limit)
        return [{"movieId": int(r["movieId"]), "title": str(r["title"]),
                 "genres": str(r["genres"]), "year": self._movie_year.get(int(r["movieId"])),
                 "n_ratings": int(r["_pop"])} for _, r in hits.iterrows()]

    def similar_movies(self, movie_id: int, space: str = "genome", k: int = 10) -> list[dict]:
        model = {"genome": self.cb_genome, "tfidf": self.cb, "embed": self.cb_embed}.get(space)
        if model is None:
            model = self.cb_genome or self.cb
        neigh = model.neighbors(int(movie_id), k=k) if model is not None else []
        meta = self._meta([m for m, _ in neigh]).set_index("movieId")
        out = []
        for mid, sim in neigh:
            if mid in meta.index:
                mr = meta.loc[mid]
                out.append({"movieId": mid, "title": str(mr["title"]),
                            "genres": str(mr["genres"]), "year": self._movie_year.get(mid),
                            "similarity": round(float(sim), 4)})
        return out

    # ── recommendation ─────────────────────────────────────────────────────────
    def _filter_candidates(self, candidates, genres, year_min, year_max):
        if not genres and year_min is None and year_max is None:
            return candidates
        gset = set(genres) if genres else None
        out = []
        for m in candidates:
            if gset is not None and not (self._movie_genres.get(m, set()) & gset):
                continue
            if year_min is not None or year_max is not None:
                y = self._movie_year.get(m)
                if y is None:
                    continue
                if year_min is not None and y < year_min:
                    continue
                if year_max is not None and y > year_max:
                    continue
            out.append(m)
        return out

    def get_recommendations(
        self,
        user_id: int,
        user_ratings: dict[int, float],
        model: str = "dual",
        n: int = 10,
        exclude: set | None = None,
        pool_size: int | None = None,
        genres: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> pd.DataFrame:
        """Top-n recommendations for `user_id`.

        Heavy models (`_POOLED_MODELS`) retrieve a popularity-ranked candidate
        pool first, then re-rank. Returns columns: movieId, title, genres, year,
        score (+ predicted_rating for rating models).
        """
        seen = set(user_ratings) | (exclude or set())
        if pool_size is None and model in _POOLED_MODELS:
            pool_size = _DEFAULT_POOL

        if pool_size and self._pop_order is not None:
            base = (int(m) for m in self._pop_order if m not in seen)
            base = self._filter_candidates(base, genres, year_min, year_max)
            candidates = base[:pool_size] if isinstance(base, list) else list(base)[:pool_size]
        else:
            base = [m for m in self.movies_df["movieId"].values if m not in seen]
            candidates = self._filter_candidates(base, genres, year_min, year_max)

        predict = self._predict_fn(model, user_id, user_ratings)
        scored = [(m, predict(m)) for m in candidates]
        scored = sorted(
            [(m, s) for m, s in scored if s is not None and not np.isnan(s)],
            key=lambda x: x[1], reverse=True,
        )

        top = pd.DataFrame(scored[:n], columns=["movieId", "score"])
        meta = self._meta(top["movieId"]).drop_duplicates("movieId")
        top = top.merge(meta, on="movieId", how="left")
        top["score"] = top["score"].round(3)
        if not self.is_ranking_only(model):
            top["predicted_rating"] = top["score"].round(2)
        return top.reset_index(drop=True)

    def recommend(self, user_id, user_ratings, model, n=10, exclude=None,
                  genres=None, year_min=None, year_max=None) -> list[dict]:
        """Recommendations as a list of dicts (API-friendly)."""
        df = self.get_recommendations(
            user_id, user_ratings, model=model, n=n, exclude=exclude,
            genres=genres, year_min=year_min, year_max=year_max)
        records = df.to_dict(orient="records")
        for r in records:
            r["movieId"] = int(r["movieId"])
            if r.get("year") is not None and not (isinstance(r["year"], float) and np.isnan(r["year"])):
                r["year"] = int(r["year"])
            else:
                r["year"] = None
        return records

    # ── inspection & explanation ────────────────────────────────────────────────
    def predict_all(self, user_id: int, movie_id: int,
                    user_ratings: dict | None = None) -> dict:
        """Every available model's prediction for one (user, movie) pair."""
        if user_ratings is None:
            user_ratings = self.user_ratings(user_id)
        dispatch = self._build_dispatch(user_id, user_ratings)
        preds = []
        for k in self.available_models():
            try:
                v = dispatch[k](int(movie_id))
                v = None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 3)
            except Exception:
                v = None
            preds.append({"key": k, "label": self.label(k), "value": v,
                          "ranking_only": self.is_ranking_only(k)})
        return {"user_id": int(user_id), "movie": self.movie_meta(movie_id),
                "true_rating": self.true_rating(user_id, movie_id),
                "predictions": preds}

    def explain(self, user_id: int, movie_id: int, model: str,
                user_ratings: dict | None = None, top: int = 5) -> dict:
        """Explain a recommendation: which of the user's rated movies are most
        content-similar to the target, plus per-model agreement on the score."""
        if user_ratings is None:
            user_ratings = self.user_ratings(user_id)
        cb = self.cb_genome or self.cb
        sims = cb.similarity_to(int(movie_id), list(user_ratings.keys())) if cb else {}
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:top]
        meta = self._meta([m for m, _ in ranked]).set_index("movieId")
        because = []
        for mid, s in ranked:
            if mid in meta.index and s > 0:
                mr = meta.loc[mid]
                because.append({"movieId": mid, "title": str(mr["title"]),
                                "genres": str(mr["genres"]),
                                "your_rating": round(float(user_ratings[mid]), 2),
                                "similarity": round(float(s), 4)})
        agreement = self.predict_all(user_id, movie_id, user_ratings)["predictions"]
        return {"movie": self.movie_meta(movie_id), "model": self.label(model),
                "because_you_liked": because, "model_agreement": agreement}

    def popular_movies(self, n: int = 200) -> list[dict]:
        ids = self._pop_order[:n] if self._pop_order is not None else \
            self.movies_df["movieId"].head(n).to_numpy()
        meta = self._meta(ids).set_index("movieId")
        out = []
        for mid in ids:
            mid = int(mid)
            if mid in meta.index:
                mr = meta.loc[mid]
                out.append({"movieId": mid, "title": str(mr["title"]),
                            "genres": str(mr["genres"]), "year": self._movie_year.get(mid),
                            "n_ratings": int(self.item_pop.get(mid, 0))})
        return out

    def load_metrics(self) -> dict:
        path = ARTIFACTS_METRICS / "all_metrics.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}
