"""FastAPI service for the hybrid movie recommender.

Loading is **non-blocking**: on startup (and on the first `/api/health` hit from
the UI) a background thread loads the catalogue/ratings first (fast) then the
~8 GB of models (slow). The server answers immediately the whole time —
catalogue/search endpoints come online within seconds, and model-dependent
endpoints return HTTP 503 `{"status": "loading"}` until the models are ready, so
the front-end can render fully and degrade gracefully rather than freeze.

Run:  uvicorn backend.main:app --reload --port 8000
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hybrid_recsys.config import ARTIFACTS_METRICS, ROOT
from backend.serving import RecommenderBundle, MODEL_REGISTRY, static_model_info
from backend.schemas import RecommendRequest, CompareRequest, ExplainRequest


# ── background loader ─────────────────────────────────────────────────────────
class _Loader:
    """Loads the bundle in a daemon thread; exposes readiness state. Idempotent
    `start()` so both the startup hook and the UI's health ping can trigger it."""

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
            self._thread = threading.Thread(target=self._run, name="bundle-loader", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self.bundle.load_data()      # fast: catalogue + ratings
            self.bundle.load_models()    # slow: ~8 GB of models
            self.state = "ready"
        except Exception as e:           # surfaced via /api/health and the gates
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"

    def status(self) -> dict:
        return {"status": self.state,
                "data_ready": self.bundle.data_ready,
                "models_ready": self.bundle.models_ready,
                "n_models": len(MODEL_REGISTRY),
                "error": self.error}


LOADER = _Loader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOADER.start()        # begin loading as soon as the server boots
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


# ── readiness gates (FastAPI dependencies) ────────────────────────────────────
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


def require_models() -> RecommenderBundle:
    """All 12 models must be loaded (~a minute after boot)."""
    _check_error()
    if not LOADER.bundle.models_ready:
        LOADER.start()
        raise HTTPException(503, {"status": "loading",
                                  "message": "Models are still loading — try again in a moment."})
    return LOADER.bundle


def _resolve(b: RecommenderBundle, user_id, ratings):
    """Map a request to (user_id, ratings_dict). `ratings` ⇒ cold-start user."""
    if ratings:
        return (user_id if user_id is not None else -1), {int(k): float(v) for k, v in ratings.items()}
    if user_id is not None:
        return int(user_id), b.user_ratings(int(user_id))
    raise HTTPException(400, "Provide either `user_id` or `ratings`.")


# ── meta / lightweight (always available, no bundle) ──────────────────────────
@app.get("/api/health")
def health():
    LOADER.start()        # UI pings this on load → guarantees loading is underway
    return LOADER.status()


@app.get("/api/models")
def models():
    # Availability by artifact-file existence → answers instantly, even mid-load.
    return static_model_info()


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


# ── model-dependent endpoints (need models) ───────────────────────────────────
@app.get("/api/movies/{movie_id}/similar")
def movies_similar(movie_id: int, space: str = "genome", k: int = Query(10, ge=1, le=50),
                   b: RecommenderBundle = Depends(require_models)):
    return b.similar_movies(movie_id, space=space, k=k)


@app.post("/api/recommend")
def recommend(req: RecommendRequest, b: RecommenderBundle = Depends(require_models)):
    uid, ur = _resolve(b, req.user_id, req.ratings)
    items = b.recommend(uid, ur, model=req.model, n=req.k,
                        genres=req.genres, year_min=req.year_min, year_max=req.year_max)
    return {"model": req.model, "label": b.label(req.model),
            "ranking_only": b.is_ranking_only(req.model),
            "cold_start": req.ratings is not None, "items": items}


@app.post("/api/recommend/compare")
def recommend_compare(req: CompareRequest, b: RecommenderBundle = Depends(require_models)):
    uid, ur = _resolve(b, req.user_id, req.ratings)
    results = {}
    for m in req.models:
        try:
            results[m] = {"label": b.label(m), "ranking_only": b.is_ranking_only(m),
                          "items": b.recommend(uid, ur, model=m, n=req.k, genres=req.genres,
                                               year_min=req.year_min, year_max=req.year_max)}
        except Exception as e:
            results[m] = {"label": b.label(m), "error": str(e), "items": []}
    return {"models": results}


@app.get("/api/predict")
def predict(user_id: int, movie_id: int, b: RecommenderBundle = Depends(require_models)):
    return b.predict_all(user_id, movie_id)


@app.post("/api/explain")
def explain(req: ExplainRequest, b: RecommenderBundle = Depends(require_models)):
    uid, ur = _resolve(b, req.user_id, req.ratings)
    return b.explain(uid, req.movie_id, model=req.model, user_ratings=ur)
