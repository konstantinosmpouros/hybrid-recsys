# Backend — FastAPI REST Service (in depth)

The backend is a **stateless-ish REST API** over the trained models. It owns one
`RecommenderBundle` that holds the catalogue/ratings plus whichever models are currently
loaded, and exposes every operation the [app](app.md) needs over HTTP. The app contains **no
model logic** — it is a thin client of this API.

- **Code:** [`backend/main.py`](../backend/main.py) (endpoints + loader),
  [`backend/serving.py`](../backend/serving.py) (`RecommenderBundle` — all the logic),
  [`backend/schemas.py`](../backend/schemas.py) (pydantic request bodies).
- **Interactive docs:** `http://localhost:8000/docs` (Swagger UI, auto-generated).
- **Depends on:** the `hybrid_recsys` library + the artifacts produced by the
  [notebooks](notebooks.md) (`artifacts/models/*.joblib`, `artifacts/metrics/all_metrics.json`,
  `artifacts/figures/*.png`, `data/processed/*.parquet`).

---

## Why it exists (the memory problem it solves)

The Surprise models are **enormous in RAM** even though their `.joblib` files are small,
because they retain the full 20M-rating trainset as Python objects:

| Model | Resident RAM | Model | Resident RAM |
|---|---|---|---|
| SVD | ≈ 6.9 GB | Content (TF-IDF/Genome/Embed) | ~0.3–0.45 GB each |
| Weighted Hybrid | ≈ 7.2 GB (embeds an SVD) | LightGCN | ≈ 0.2 GB |
| User-kNN | ≈ 3.5 GB | Stacked (meta only) | ≈ 0.05 GB |
| Item-kNN | ≈ 3.0 GB | Dual-Head (meta only) | ≈ 0.05 GB |

Loading all twelve at once is ~15–20 GB and will **swap and freeze a host with little free
RAM** (it's host-RAM exhaustion, not a container OOM). So the backend never loads everything
eagerly — it **loads the catalogue at startup and each model on demand**, and actively
**bounds memory to one heavy model at a time**.

---

## Loading lifecycle

```
boot ── lifespan/health ──► _Loader thread ──► load_data()  (catalogue + ratings, ~0.6 GB)
                                                    │
                                                    ▼  state: ready, data_ready=True, loaded=[]
                                       (optional) RECSYS_MODELS preload set
                                                    │
        user clicks "Load" ──► POST /api/models/{key}/load ──► memory guard ──► evict heavies ──► ensure_model()
```

### 1. Background data loader (`_Loader` in [main.py](../backend/main.py))

- A **daemon thread** started by the FastAPI **lifespan** hook **and** by the first
  `/api/health` ping (so it starts even under a server that skips lifespan).
- It runs `bundle.load_data()` only — movie catalogue, train/test ratings, and the derived
  lookups (per-movie year/genres, item popularity, user rating counts, popularity-ranked
  retrieval order). This is the ~0.6 GB, few-seconds step. **No models.**
- If `RECSYS_MODELS` is set it then preloads that set (see below).
- Exposes `status()` → `{status, data_ready, loaded, n_models, error}` (the `/api/health` body).

### 2. On-demand model loading (`RecommenderBundle.ensure_model` in [serving.py](../backend/serving.py))

`POST /api/models/{key}/load` resolves the model's **atomic dependencies** and loads each
under a thread lock (idempotent — a no-op if already resident):

```python
_MODEL_DEPS = {
  "content": ["content"], "svd": ["svd"], "item_knn": ["item_knn"], …,
  "weighted": ["weighted"],                                   # self-contained joblib
  "stacked":  ["stacked", "content", "user_knn", "item_knn", "svd"],
  "dual":     ["dual", "content_genome", "user_knn", "item_knn", "svd", "lightgcn"],
}
```

The composite hybrids (`stacked`, `dual`) pull their base models in because the bundle rebuilds
their feature vectors at serve time.

### 3. The memory guard + eviction (the heart of it)

Before allocating, `load_model` does two things so a too-big load can't OOM-kill the backend:

1. **Pre-load guard (HTTP 507).** Compares `estimate_load_gb(key)` (sum of the not-yet-loaded
   deps' RAM) against `available_gb()` (free RAM from `/proc/meminfo` `MemAvailable`,
   container/cgroup-aware) **plus** `freeable_gb(key)` (RAM reclaimable by evicting heavy models
   this one doesn't need). If it still won't fit, it returns **507 `insufficient_memory`**
   *before* allocating — a clean refusal instead of a crash.
2. **Eviction (one heavy model at a time).** On *every* load it unloads each **heavy** model
   (`> _HEAVY_GB = 1 GB`) the new model doesn't depend on, then calls `release_memory()`
   (`gc.collect()` + a defensive `torch.cuda.empty_cache()` — serving is CPU-only). The **small**
   models (the 3 content models, LightGCN, the meta heads) are never evicted and **coexist
   freely**. So at most one heavy CF/hybrid model is ever resident.

```python
# POST /api/models/{key}/load  (abridged)
need, free, freeable = b.estimate_load_gb(key), available_gb(), b.freeable_gb(key)
if need > 0 and free + freeable < need + 0.5:
    raise HTTPException(507, {"status": "insufficient_memory", …})
freed = b.evict_for(key)        # unload heavy models this one doesn't need + GC
ok    = b.ensure_model(key)     # load key + its deps
return {"key": key, "loaded": ok, "freed": freed, "models": b.model_info()}
```

> **Hardware reality.** The two composite hybrids genuinely need *several* heavy bases at once
> (Stacked ≈ 13.8 GB, Dual ≈ 13.9 GB), so on a ~15 GB WSL2 VM they will hit the 507 guard rather
> than load. Single heavy models (SVD/Weighted ~7 GB, the kNNs ~3 GB) load one at a time; the
> small content models always fit. Raise the WSL2 cap (`.wslconfig` → `memory=22GB`) to run the
> big hybrids.

### `RECSYS_MODELS` — optional startup preload

| Value | Effect |
|---|---|
| *(empty / unset)* | **default** — preload nothing; load on demand |
| `lite` | preload the content models + LightGCN (~2 GB) |
| `all` | preload everything (~20 GB free required) |
| `svd,content_genome` | preload an explicit comma-separated set |

Set in `docker-compose.yml` (default empty) or `run_local.ps1`.

---

## Readiness gates & status codes

| Condition | Code | Body | Meaning |
|---|---|---|---|
| catalogue still loading | **503** | `{status: loading}` | `require_data` dependency; retry shortly |
| model not loaded yet | **409** | `{status: not_loaded, model}` | click Load first |
| load won't fit in RAM | **507** | `{status: insufficient_memory, needed_gb, free_gb, freeable_gb}` | free memory / load lighter |
| loader crashed | **500** | `{status: error, message}` | check logs |
| bad request | **400** | string | e.g. neither `user_id` nor `ratings` given |
| unknown id | **404** | string | user/movie/model not found |

`/recommend/compare` is lenient: instead of 409-ing, it marks each unloaded model inline
(`{error: "not loaded", items: []}`) so a partial comparison still renders.

---

## The `RecommenderBundle` (serving.py)

Single object that owns data + models and implements every operation:

- **`MODEL_REGISTRY`** — the single source of truth: `key → (label, family, ranking_only,
  cold_start)`. `ranking_only` models (LightGCN) emit a relevance score, not a rating.
  `cold_start` models (the 3 content models + 3 hybrids) can recommend for a brand-new user
  from an in-session ratings dict; pure CF (SVD/kNN/LightGCN) needs a user seen at train time.
- **`_build_dispatch(user_id, ratings)`** — builds the `{key: predict_fn}` map, **guarding each
  entry by whether its object(s) actually loaded**, so the API only ever advertises models it
  can score. `loaded_models()` is the key-set of this map.
- **`get_recommendations`** — scores unseen candidates, drops NaN, sorts desc, returns top-N +
  metadata. Genre/year **filters** prune candidates first. **Heavy models** (`dual`, `lightgcn`)
  use **two-stage retrieval**: take the top-3000 by popularity, then re-rank (they make many base
  calls per candidate, or have a restricted vocabulary).
- **`predict_all`** — every *loaded* model's prediction for one (user, movie) + the held-out true
  rating (for the Prediction Inspector).
- **`explain`** — the user's rated movies most content-similar to the target (the "because you
  liked…") + each model's score on it.
- **`search_movies` / `similar_movies` / `user_profile` / `sample_users` / `popular_movies`** —
  catalogue/user helpers (need data only, no models).
- **`model_info()`** — per model `{available (artifact on disk), loaded (in RAM), ram_gb}`; drives
  the app's loader UI and selectbox badges.

`static_model_info()` reports availability by **artifact-file existence** without loading the
bundle, so `/api/models` answers instantly at boot.

---

## Endpoints

Base path `/api`. Full request/response schemas live at `/docs`.

### Meta / lightweight (work at any time)

| Method · Path | Purpose |
|---|---|
| `GET /api/health` | `{status, data_ready, loaded[], n_models, error}`; also kicks the loader |
| `GET /api/models` | per-model `available` / `loaded` / `ram_gb` (the loader UI source) |
| `POST /api/models/{key}/load` | load a model + deps on demand → `{key, loaded, freed[], models}` |
| `GET /api/metrics` | the full `all_metrics.json` (Comparison tab) |
| `GET /api/figures` | list of deep-eval PNG stems served under `/figures` |

### Catalogue / users (need data only → 503 while loading)

| Method · Path | Purpose |
|---|---|
| `GET /api/genres` | sorted genre list (filter UI) |
| `GET /api/users/sample?n=` | sample of dataset userIds (20–500 ratings) |
| `GET /api/users/{id}/profile?top=` | a user's rating history + mean |
| `GET /api/movies/search?q=&limit=` | title search, popularity-ranked |
| `GET /api/movies/popular?n=` | most-rated movies |
| `GET /api/movies/{id}` | one movie's metadata |

### Model-dependent (need the specific model loaded → 409 otherwise)

| Method · Path | Body / params | Purpose |
|---|---|---|
| `GET /api/movies/{id}/similar?space=&k=` | `space ∈ {genome,tfidf,embed}` | item-item neighbours under a content space |
| `POST /api/recommend` | `RecommendRequest` | top-K for an existing user **or** a cold-start ratings dict |
| `POST /api/recommend/compare` | `CompareRequest` | same user through several models at once (lenient on unloaded) |
| `GET /api/predict?user_id=&movie_id=` | — | every loaded model's predicted rating vs the true rating |
| `POST /api/explain` | `ExplainRequest` | "why this?" — content-similar history + per-model agreement |

### Request bodies ([schemas.py](../backend/schemas.py))

```python
RecommendRequest:  user_id?: int | ratings?: {movieId: rating}    # one of the two
                   model="dual", k=10, genres?: [str], year_min?, year_max?
CompareRequest:    user_id? | ratings?, models=[…], k=10, genres?, year_min?, year_max?
ExplainRequest:    user_id? | ratings?, movie_id: int, model="dual"
```

Passing `ratings` (a `{movieId: rating}` map) instead of `user_id` is the **cold-start path** —
the synthetic user (id −1) the app's *New User (live)* tab uses. Only `cold_start=True` models
accept it.

### Static mount

`/figures` serves `artifacts/figures/*.png` directly (mounted via `StaticFiles`) — the
Comparison tab renders these.

---

## Anatomy of a request — `POST /api/recommend`, end to end

What actually happens between the HTTP call and the JSON answer (the other model endpoints
follow the same skeleton):

```
POST /api/recommend  {"user_id": 142403, "model": "dual", "k": 10, "genres": ["Sci-Fi"]}
 │
 1. require_data        → 503 if the catalogue is still loading (boot window)
 2. _require_loaded     → 409 {not_loaded} if "dual" isn't in RAM yet (click Load first)
 3. _resolve            → user 142403's train-ratings dict from `train_ratings`
 │                        (or, for a cold-start body, the in-session {movieId: rating} map)
 4. candidate retrieval → "dual" is a heavy model (_POOLED_MODELS) → take the top-3000
 │                        most-popular movies the user hasn't seen, instead of all 62K
 5. _filter_candidates  → drop candidates not matching the genre/year filters
 6. _build_dispatch     → the {key: predict_fn} map, containing ONLY loaded models;
 │                        for "dual" the fn builds the 8-feature vector (_dual_feats:
 │                        genome-CB, user-kNN, item-kNN, SVD, LightGCN + side features)
 │                        and calls dual.predict_rating_one(...)
 7. score loop          → predict every candidate; drop NaN; sort descending
 8. top-N + metadata    → join titles/genres/year; add predicted_rating unless the
                          model is ranking_only → JSON out
```

Notes that matter in practice: step 4 is why the Dual-Head answers in ~seconds instead of
minutes (each of its scores costs five base-model calls); step 6 is why an unloaded model can
never be reached by accident; and `/api/recommend/compare` simply runs steps 3–8 once per
requested model, marking unloaded ones inline instead of failing the whole call.

## Try it from the terminal (curl)

With the backend on `localhost:8000` (compose maps it to host **8806** — substitute accordingly):

```bash
curl localhost:8000/api/health                       # {status, data_ready, loaded[], …}
curl localhost:8000/api/models                       # availability + RAM cost per model
curl -X POST localhost:8000/api/models/content_genome/load     # load a model (small, ~10 s)

# existing dataset user → top-5
curl -X POST localhost:8000/api/recommend -H "Content-Type: application/json" \
  -d '{"user_id": 142403, "model": "content_genome", "k": 5}'

# cold-start user → ratings dict instead of user_id (content models + hybrids only)
curl -X POST localhost:8000/api/recommend -H "Content-Type: application/json" \
  -d '{"ratings": {"296": 5.0, "318": 4.5, "858": 5.0}, "model": "content_genome", "k": 5}'

# every loaded model's prediction vs the held-out truth for one (user, movie)
curl "localhost:8000/api/predict?user_id=142403&movie_id=296"

# why was this recommended?
curl -X POST localhost:8000/api/explain -H "Content-Type: application/json" \
  -d '{"user_id": 142403, "movie_id": 296, "model": "content_genome"}'

curl "localhost:8000/api/movies/search?q=matrix&limit=3"
curl "localhost:8000/api/movies/2571/similar?space=genome&k=5"
```

A failed-load probe is equally informative: requesting a model that doesn't fit returns the
**507** body with `needed_gb` / `free_gb` / `freeable_gb`, and a recommend against an unloaded
model returns the **409** `not_loaded` body — the same objects the Streamlit UI renders as
warnings and Load buttons.

---

## Running it

```bash
# local (Windows): use `python -m` so it runs under the interpreter that has scikit-surprise
python -m uvicorn backend.main:app --port 8000          # → http://localhost:8000/docs
# or both processes at once:
./run_local.ps1

# docker (the compose stack maps host 8806 → container 8000)
docker compose up --build
```

CORS is open (`allow_origins=["*"]`) so the Streamlit app on another port can call it directly.
See [app.md](app.md) for the front-end and the end-to-end picture; see the project root
[README.md](../README.md) for install prerequisites.
