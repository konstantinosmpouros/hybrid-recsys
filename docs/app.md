# App — Streamlit Front-End (in depth)

The app ([`app/app.py`](../app/app.py)) is the **demo deliverable (D)**: it takes user
ratings and recommends new items, demonstrated on sample dataset users. It is a **thin HTTP
client** — it contains *no* model logic, no data loading, no scoring. Every action is a call
to the [backend API](backend.md); the app only renders responses and manages a little UI
session state.

- **Run:** `python -m streamlit run app/app.py` → `http://localhost:8501`
  (docker compose maps it to host **8807**).
- **Talks to:** `BACKEND_URL` (default `http://localhost:8000`; `http://backend:8000` inside compose).
- **Stack:** Streamlit + `requests` + Plotly, with `@st.cache_data` over the read endpoints.

---

## Architecture — a thin client

```
┌────────────────────┐     HTTP (requests)      ┌──────────────────────┐
│  Streamlit app/     │ ───────────────────────► │  FastAPI backend/     │
│  - 6 tabs (UI only) │                          │  - RecommenderBundle  │
│  - session_state    │ ◄─────────────────────── │  - models + data      │
│  - caches responses │      JSON / PNG          └──────────────────────┘
└────────────────────┘
```

Why this split (vs the old single-process Streamlit app that loaded models itself): the
Surprise models are several GB each, so loading them inside the Streamlit process froze the UI
and made every rerun risk an OOM. Moving models behind a REST service lets the **UI render
instantly** and load models **on demand**, and lets the same API back a CLI, tests, or `/docs`.

### Request plumbing (`_request`)

A single helper wraps every call and translates backend status codes into typed Python
exceptions the UI can handle gracefully:

| Backend response | Exception | UI behaviour |
|---|---|---|
| `503 loading` | `BackendLoading` | show "starting up", auto-poll |
| `507 insufficient_memory` | `InsufficientMemory` | warn (can't fit), don't crash |
| connection refused / timeout | `BackendDown` | "backend restarting", auto-retry after a short sleep |
| `409 not_loaded` | (handled per call) | render an inline **Load** button |

`@st.cache_data` memoises the read endpoints (`/api/models`, `/api/genres`, `/api/users/sample`,
search, profiles, metrics, figures). The cache key includes a **`load_nonce`** counter that is
bumped every time a model is loaded — so loading a model invalidates the cached `/api/models`
and the selectboxes immediately re-badge.

### On-demand loading UX

- A **sidebar loader panel** lists every model grouped by family with its **RAM cost**
  (`ram_gb`, ⚠️ on ≥ 3 GB), a ✅/⬜ loaded badge, and a **Load** button.
- Inline `ensure_model_ui(key)` guards model-dependent actions: if the chosen model isn't
  loaded it renders a Load button (with the RAM cost) and returns `False` instead of running.
- `load_model(key)` POSTs to `/api/models/{key}/load`, then — if the backend **evicted** other
  models to make room — shows a `♻️ Unloaded …` toast (from the response's `freed` list), bumps
  `load_nonce`, and reruns. It catches `InsufficientMemory` (warn) and `BackendDown` (the load
  OOM'd the backend → tell the user it's restarting and stop).
- The sidebar caption states the policy plainly: **heavy (> 1 GB) models don't coexist —
  loading one frees the previous; the small content models + LightGCN stay resident.** (See the
  eviction model in [backend.md](backend.md).)

A connectivity gate at the top of the script handles the brief startup window: while the
catalogue is loading it shows a spinner and auto-refreshes; if the backend is unreachable it
shows a retry screen rather than a raw traceback.

---

## The six tabs

### 👤 Existing User

The core demo. Pick a sample `userId` (or type one), choose any of the 12 models, set Top-K,
optionally apply **genre/year filters**, and get recommendations **side-by-side with the user's
own top-rated history** (so a grader can eyeball that the recs match the taste). Below that, a
**"why was this recommended?"** explainer: pick a recommended title → the app calls `/api/explain`
and shows (a) the user's rated movies most **content-similar** to it ("because you liked…") and
(b) a bar chart of **what each loaded model predicts** for that pair. Cold-start-incapable models
(pure CF) require the user to exist in the training data.

### 🆕 New User (live — "watch it learn")

The standout demo. A **synthetic user (id −1, not in the dataset)** rates movies one at a time —
from a popular-movies picker or a search box — and the recommendations **re-compute after every
rating** via the cold-start path (`POST /api/recommend` with a `ratings` map, not a `user_id`).
Successive top-K lists are diffed to badge each item **🆕 / ▲ / ▼ / •** (new / moved up / down /
unchanged), and a genre-mix bar chart shows how the recommendation profile shifts. Only
`cold_start=True` models appear here (the 3 content models + 3 hybrids). This directly
demonstrates assignment requirement (D): *takes user ratings, recommends new items.*

### 🆚 Side-by-side

Run the **same user through several models at once** and compare their top-K columns; movies
recommended by **≥ 2 models are starred ⭐** with an overlap count. Uses `/api/recommend/compare`,
which marks any unloaded model inline rather than failing. Because the backend keeps only **one
heavy model resident at a time**, the tab warns when you select more than one > 1 GB model (they
can't be loaded together on a typical machine) — the realistic comparison here is the small
content models together, plus at most one heavy model.

### 🔍 Movie Explorer

Search a movie, then see its **nearest neighbours under each content space** in parallel
columns — **Tag Genome**, **TF-IDF**, **Embeddings** (`/api/movies/{id}/similar?space=`). Each
column has its own Load button (the three content models are small and coexist), so you can load
all three and directly compare how the representations disagree about "similar movies" — a visual
counterpart to the content-model results in [models.md](models.md).

### 🎯 Prediction Inspector

Pick a (user, movie) pair → every **loaded** model's predicted rating as a bar chart, with the
**held-out true rating** drawn as a dashed reference line (`/api/predict`). Ranking-only models
(LightGCN) are listed separately as relevance scores, not stars. A concrete, per-instance view of
where each model lands relative to ground truth — load more models in the sidebar to add bars.

### 📊 Comparison

The offline leaderboard for **all 12 models**, read from `all_metrics.json` (`/api/metrics`) —
**independent of what's loaded**, so it always works. Shows the RMSE/MAE table (min-highlighted),
the full Precision/Recall/F1@K table with an **F1@10 bar chart**, the **rating-vs-ranking
scatter** (top-left = best on both), and a selector for the **deep-evaluation figures** from
notebook 14 (NDCG/AUC, segmented RMSE, beyond-accuracy, bootstrap CIs) served as static PNGs from
`/figures`.

---

## Suggested demo script (for the 10–15 min presentation)

A tight path through the app that hits every assignment requirement — in particular **Ε:
present results for indicative users from the dataset**. Total ≈ 6–8 minutes of live demo,
leaving time for slides. Pre-load `content_genome` (~0.25 GB, seconds) before you start; load
one heavy model only if the machine has RAM to spare.

1. **📊 Comparison (1 min).** Open with the leaderboard — "12 models, one protocol; the
   learned hybrids win RMSE/MAE, the ranking-trained model wins F1." Show the
   rating-vs-ranking scatter. *Zero models need to be loaded for this tab.*
2. **👤 Existing User (2–3 min) — the requirement-Ε moment.** Pick a known archetype user —
   **142403** (mainstream) or **62122** (niche, Comedy/Romance) — so you can narrate their
   taste. Get top-10 with `content_genome`; show the recommendations sitting next to their
   real history; apply a **genre filter** live. Then click a recommendation → **Explain** —
   "because you rated these similar films highly" + the per-model agreement chart.
3. **🆕 New User (2 min) — "watch it learn".** Reset, rate 2–3 sci-fi titles 5★ → top-10
   turns sci-fi; now rate a comedy 5★ and point at the **🆕/▲/▼ badges** and the genre-mix
   chart shifting. This is the cold-start path (a ratings dict, no userId) — only content
   models + hybrids appear in this tab, and you can say *why* (predict-signature argument).
4. **🔍 Movie Explorer (1 min).** Search *The Matrix* → three columns of neighbours
   (Genome / TF-IDF / Embeddings) → "same algorithm, three representations — the genome's
   neighbours are visibly better, which is the content finding of the project."
5. **🎯 Prediction Inspector (1 min, optional).** A (user, movie) with a held-out truth —
   bars vs the dashed true-rating line.
6. **Close on 📊** with the F1@10-by-archetype case-study figure: "and per user type, the
   hybrid is never the worst — NDCG/AUC winner in all four."

Fallback if RAM is tight on the demo machine: the whole script above runs on **content
models alone** (~0.3 GB each); the Comparison tab carries all 12 models' numbers regardless.

---

## Running the whole thing

The app needs the backend up first. Three ways:

```bash
# 1. One script (Windows) — starts backend + app
./run_local.ps1

# 2. Two terminals (use `python -m` so they share the interpreter with scikit-surprise)
python -m uvicorn backend.main:app --port 8000
python -m streamlit run app/app.py

# 3. Docker compose — backend on host :8806, app on host :8807
docker compose up --build
```

Then open the app, click **Load** on a model or two (start with the small content models or
LightGCN — they're ~0.2–0.4 GB and load in seconds), and explore. The **Comparison** tab works
with zero models loaded. See [backend.md](backend.md) for the API the app calls and the memory
model behind the Load buttons.
