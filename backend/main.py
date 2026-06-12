"""FastAPI service for the hybrid movie recommender.

Loading is **non-blocking and on-demand**. On startup a background thread loads
only the catalogue/ratings (fast) — no models — so the server answers instantly.
Each model is loaded **lazily when the user clicks "Load"** in the UI (POST
/api/models/{key}/load); the Surprise models are large in RAM (SVD ≈ 6.8 GB,
kNN larger), so loading everything up front can exhaust memory. Model-dependent
endpoints return 503 (catalogue still loading) or 409 `{status: not_loaded}`
(model not loaded yet) so the front-end can guide the user.

Run:  uvicorn backend.main:app --reload --port 8000
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hybrid_recsys.config import ARTIFACTS_METRICS, ROOT
from backend.serving import RecommenderBundle, MODEL_REGISTRY, _selected_models, available_gb
from backend.schemas import RecommendRequest, CompareRequest, ExplainRequest

_SIMILAR_SPACE_MODEL = {"genome": "content_genome", "tfidf": "content", "embed": "content_embed"}


# ── background loader (data only; models load on demand) ──────────────────────
class _Loader:
    def __init__(self):
        self.bundle = RecommenderBundle()
        self.state = "idle"     # idle -> loading -> ready | error
        self.error: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self.state = "loading"
            self._thread = threading.Thread(target=self._run, name="data-loader", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self.bundle.load_data()                 # fast: catalogue + ratings
            sel = _selected_models()                 # optional startup preload (default: none)
            if sel is None:
                self.bundle.load_models(only=None)   # RECSYS_MODELS=all
            elif sel:
                self.bundle.load_models(only=sel)    # RECSYS_MODELS=lite / explicit list
            self.state = "ready"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"

    def status(self) -> dict:
        return {"status": self.state,
                "data_ready": self.bundle.data_ready,
                "loaded": sorted(self.bundle.loaded_models()),
                "n_models": len(MODEL_REGISTRY),
                "error": self.error}


LOADER = _Loader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOADER.start()
    yield


app = FastAPI(
    title="Hybrid Movie Recommender API",
    description="Content-based + collaborative-filtering hybrid recommender on MovieLens 25M.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FIGURES_DIR = ROOT / "artifacts" / "figures"
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


# ── gates ─────────────────────────────────────────────────────────────────────
def _check_error():
    if LOADER.state == "error":
        raise HTTPException(500, {"status": "error", "message": LOADER.error})


def require_data() -> RecommenderBundle:
    """Catalogue/ratings must be loaded (a few seconds after boot)."""
    _check_error()
    if not LOADER.bundle.data_ready:
        LOADER.start()
        raise HTTPException(503, {"status": "loading",
                                  "message": "Backend is starting up (loading catalogue)…"})
    return LOADER.bundle


def _require_loaded(b: RecommenderBundle, key: str):
    if key not in b.loaded_models():
        raise HTTPException(409, {"status": "not_loaded", "model": key,
                                  "message": f"Model '{b.label(key)}' isn't loaded yet — click Load first."})


def _resolve(b: RecommenderBundle, user_id, ratings):
    if ratings:
        return (user_id if user_id is not None else -1), {int(k): float(v) for k, v in ratings.items()}
    if user_id is not None:
        return int(user_id), b.user_ratings(int(user_id))
    raise HTTPException(400, "Provide either `user_id` or `ratings`.")


# ── meta / lightweight ────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    LOADER.start()
    return LOADER.status()


@app.get("/api/models")
def models():
    # available = artifact on disk · loaded = in RAM now. Works at any time.
    return LOADER.bundle.model_info()


@app.post("/api/models/{key}/load")
def load_model(key: str, b: RecommenderBundle = Depends(require_data)):
    """Load one model (and its base-model dependencies) into memory on demand.
    Synchronous — can take from a few seconds (content) to a minute+ (SVD/kNN)."""
    if key not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model '{key}'.")
    # Check fit BEFORE allocating — otherwise the OOM killer takes down the whole
    # backend mid-load. `freeable` = RAM we can reclaim by unloading other large
    # models this one doesn't need (the small content models are always kept).
    need = b.estimate_load_gb(key)
    free = available_gb()
    freeable = b.freeable_gb(key)
    if need > 0 and free + freeable < need + 0.5:
        raise HTTPException(507, {
            "status": "insufficient_memory", "model": key,
            "needed_gb": need, "free_gb": round(free, 1), "freeable_gb": freeable,
            "message": (f"Loading {b.label(key)} needs ~{need:.1f} GB but only {free:.1f} GB is free"
                        + (f" (+{freeable:.1f} GB reclaimable)" if freeable else "")
                        + ". Free memory (close apps / stop other containers) or load lighter "
                          "models — content models are ~0.3 GB.")})
    # On every model change, free the other large (≥5 GB) models this one doesn't
    # need (small content models are kept), then GC — bounds memory to ~one big
    # model at a time and cleans the previous model's garbage.
    freed = b.evict_for(key)
    try:
        ok = b.ensure_model(key)
    except Exception as e:
        raise HTTPException(500, {"status": "error", "message": f"{type(e).__name__}: {e}"})
    return {"key": key, "loaded": ok, "freed": freed, "models": b.model_info()}


@app.get("/api/metrics")
def metrics():
    path = ARTIFACTS_METRICS / "all_metrics.json"
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text())


@app.get("/api/figures")
def figures():
    if not FIGURES_DIR.exists():
        return {"figures": []}
    return {"figures": sorted(p.stem for p in FIGURES_DIR.glob("*.png")), "base_url": "/figures"}


# ── catalogue / users (need data only) ────────────────────────────────────────
@app.get("/api/genres")
def genres(b: RecommenderBundle = Depends(require_data)):
    return b.genres_list()


@app.get("/api/users/sample")
def users_sample(n: int = Query(200, ge=1, le=2000), b: RecommenderBundle = Depends(require_data)):
    return b.sample_users(n=n)


@app.get("/api/users/{user_id}/profile")
def user_profile(user_id: int, top: int = Query(15, ge=1, le=100),
                 b: RecommenderBundle = Depends(require_data)):
    if not b.user_exists(user_id):
        raise HTTPException(404, f"User {user_id} not found in training data.")
    return b.user_profile(user_id, top=top)


@app.get("/api/movies/search")
def movies_search(q: str, limit: int = Query(25, ge=1, le=100),
                  b: RecommenderBundle = Depends(require_data)):
    return b.search_movies(q, limit=limit)


@app.get("/api/movies/popular")
def movies_popular(n: int = Query(60, ge=1, le=500), b: RecommenderBundle = Depends(require_data)):
    return b.popular_movies(n=n)


@app.get("/api/movies/{movie_id}")
def movie(movie_id: int, b: RecommenderBundle = Depends(require_data)):
    m = b.movie_meta(movie_id)
    if m is None:
        raise HTTPException(404, f"Movie {movie_id} not found.")
    return m


# ── model-dependent endpoints (need the specific model loaded) ────────────────
@app.get("/api/movies/{movie_id}/similar")
def movies_similar(movie_id: int, space: str = "genome", k: int = Query(10, ge=1, le=50),
                   b: RecommenderBundle = Depends(require_data)):
    _require_loaded(b, _SIMILAR_SPACE_MODEL.get(space, "content_genome"))
    return b.similar_movies(movie_id, space=space, k=k)


@app.post("/api/recommend")
def recommend(req: RecommendRequest, b: RecommenderBundle = Depends(require_data)):
    _require_loaded(b, req.model)
    uid, ur = _resolve(b, req.user_id, req.ratings)
    try:
        items = b.recommend(uid, ur, model=req.model, n=req.k,
                            genres=req.genres, year_min=req.year_min, year_max=req.year_max)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"model": req.model, "label": b.label(req.model),
            "ranking_only": b.is_ranking_only(req.model),
            "cold_start": req.ratings is not None, "items": items}


@app.post("/api/recommend/compare")
def recommend_compare(req: CompareRequest, b: RecommenderBundle = Depends(require_data)):
    uid, ur = _resolve(b, req.user_id, req.ratings)
    loaded = b.loaded_models()
    results = {}
    for m in req.models:
        if m not in loaded:
            results[m] = {"label": b.label(m), "error": "not loaded", "items": []}
            continue
        try:
            results[m] = {"label": b.label(m), "ranking_only": b.is_ranking_only(m),
                          "items": b.recommend(uid, ur, model=m, n=req.k, genres=req.genres,
                                               year_min=req.year_min, year_max=req.year_max)}
        except Exception as e:
            results[m] = {"label": b.label(m), "error": str(e), "items": []}
    return {"models": results}


@app.get("/api/predict")
def predict(user_id: int, movie_id: int, b: RecommenderBundle = Depends(require_data)):
    # Shows predictions for whatever models are currently loaded.
    return b.predict_all(user_id, movie_id)


@app.post("/api/explain")
def explain(req: ExplainRequest, b: RecommenderBundle = Depends(require_data)):
    uid, ur = _resolve(b, req.user_id, req.ratings)
    return b.explain(uid, req.movie_id, model=req.model, user_ratings=ur)
