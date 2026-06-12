"""Streamlit front-end for the Hybrid Movie Recommender.

A thin HTTP client over the FastAPI backend (`backend.main`). All model logic
lives behind the API; this file is pure UI. Set BACKEND_URL to point elsewhere.

Models load **on demand**: nothing heavy is loaded at startup. You click "Load"
for a model (in the sidebar or inline), it loads into the backend, then you use
it. This keeps memory low — the Surprise models (SVD, k-NN) are several GB each.

Run:  streamlit run app/app.py     (backend must be up: uvicorn backend.main:app)
"""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")


# ── HTTP layer ────────────────────────────────────────────────────────────────
class BackendLoading(Exception):
    """Backend returned 503 — the catalogue is still loading."""


class BackendDown(Exception):
    """Couldn't reach the backend (connection refused / dropped / timed out)."""


class InsufficientMemory(Exception):
    """Backend returned 507 — not enough RAM to load the requested model."""


def _detail_msg(r):
    try:
        d = r.json().get("detail", {})
        return d.get("message") if isinstance(d, dict) else str(d)
    except Exception:
        return None


def _request(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=600, **kw)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise BackendDown(str(e))
    if r.status_code == 503:
        raise BackendLoading(_detail_msg(r) or "Backend is still loading…")
    if r.status_code == 507:
        raise InsufficientMemory(_detail_msg(r) or "Not enough memory to load this model.")
    r.raise_for_status()
    return r.json()


def api_get(path: str, **params):
    return _request("GET", path, params=params)


def api_post(path: str, payload: dict):
    return _request("POST", path, json=payload)


def get_health():
    return api_get("/api/health")


if "load_nonce" not in st.session_state:
    st.session_state.load_nonce = 0


@st.cache_data(ttl=120, show_spinner=False)
def get_models(nonce):                       # nonce busts cache after a model loads
    return api_get("/api/models")


@st.cache_data(ttl=600, show_spinner=False)
def get_genres():
    return api_get("/api/genres")


@st.cache_data(ttl=600, show_spinner=False)
def get_sample_users(n: int = 250):
    return api_get("/api/users/sample", n=n)


@st.cache_data(ttl=300, show_spinner=False)
def get_profile(user_id: int, top: int = 20):
    return api_get(f"/api/users/{user_id}/profile", top=top)


@st.cache_data(ttl=300, show_spinner=False)
def search_movies(q: str, limit: int = 25):
    return api_get("/api/movies/search", q=q, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def get_popular(n: int = 60):
    return api_get("/api/movies/popular", n=n)


@st.cache_data(ttl=120, show_spinner=False)
def get_similar(movie_id: int, space: str, k: int, nonce: int):
    return api_get(f"/api/movies/{movie_id}/similar", space=space, k=k)


@st.cache_data(ttl=120, show_spinner=False)
def get_predictions(user_id: int, movie_id: int, nonce: int):
    return api_get("/api/predict", user_id=user_id, movie_id=movie_id)


@st.cache_data(ttl=600, show_spinner=False)
def get_metrics():
    return api_get("/api/metrics")


@st.cache_data(ttl=600, show_spinner=False)
def get_figures():
    return api_get("/api/figures")


@st.cache_data(ttl=120, show_spinner=False)
def recommend_cached(user_id, ratings_items, model, k, genres_items, year_min, year_max, nonce):
    payload = {"model": model, "k": k}
    if ratings_items:
        payload["ratings"] = {str(m): r for m, r in ratings_items}
    if user_id is not None:
        payload["user_id"] = user_id
    if genres_items:
        payload["genres"] = list(genres_items)
    if year_min is not None:
        payload["year_min"] = year_min
    if year_max is not None:
        payload["year_max"] = year_max
    return api_post("/api/recommend", payload)


def do_recommend(user_id=None, ratings=None, model="content_genome", k=10,
                 genres=None, year_min=None, year_max=None):
    ratings_items = tuple(sorted(ratings.items())) if ratings else None
    genres_items = tuple(sorted(genres)) if genres else None
    return recommend_cached(user_id, ratings_items, model, k, genres_items,
                            year_min, year_max, st.session_state.load_nonce)


def load_model(key: str):
    """Trigger the backend to load a model (and its deps), then refresh."""
    label = LABELS.get(key, key)
    with st.spinner(f"Loading {label} into the backend… (large models can take a minute)"):
        try:
            resp = api_post(f"/api/models/{key}/load", {})
        except InsufficientMemory as e:
            st.warning(f"⚠️ {e}")
            return
        except BackendDown:
            st.error("The backend went down during loading (most likely out of memory). "
                     "It's restarting — wait a few seconds and click 🔄 Refresh.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load {label}: {e}")
            st.stop()
    if resp.get("freed"):
        st.session_state["_freed_note"] = resp["freed"]
    st.session_state.load_nonce += 1
    get_models.clear()
    st.rerun()


def ensure_model_ui(key: str, btn_key: str) -> bool:
    """If `key` isn't loaded, render a Load button and return False; else True."""
    if key in LOADED:
        return True
    c1, c2 = st.columns([3, 2])
    c1.info(f"**{LABELS.get(key, key)}** isn't loaded yet ({ram_tag(key)}).")
    if c2.button(f"⬇️ Load {LABELS.get(key, key)} ({ram_tag(key)})", key=btn_key, type="primary"):
        load_model(key)
    return False


# ── small helpers ───────────────────────────────────────────────────────────
def recs_to_df(items, ranking_only):
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    col = "score" if ranking_only or "predicted_rating" not in df else "predicted_rating"
    cols = [c for c in ["title", "year", "genres", col] if c in df]
    return df[cols].rename(columns={"title": "Title", "year": "Year", "genres": "Genres",
                                    "predicted_rating": "Pred. rating ★", "score": "Score"})


def models_by_family(only_cold_start=False):
    fam_order = {"Hybrid": 0, "Collaborative": 1, "Content-Based": 2}
    ms = [m for m in MODELS if m["available"] and (m["cold_start"] or not only_cold_start)]
    ms.sort(key=lambda m: (fam_order.get(m["family"], 9), m["label"]))
    return ms


def model_selectbox(label="Model", key=None, only_cold_start=False, default="content_genome"):
    ms = models_by_family(only_cold_start=only_cold_start)
    keys = [m["key"] for m in ms]
    if not keys:
        st.warning("No models available on disk.")
        return None
    idx = keys.index(default) if default in keys else 0

    def fmt(k):
        m = next(mm for mm in ms if mm["key"] == k)
        return f"{'✅' if m['loaded'] else '⬜'} {m['label']} · {m['family']} · {ram_tag(k)}"

    return st.selectbox(label, keys, index=idx, format_func=fmt, key=key)


# ── connectivity + data readiness ─────────────────────────────────────────────
try:
    health = get_health()
except BackendDown as e:
    st.title("🎬 Hybrid Movie Recommender")
    st.error("⚠️ The backend is unreachable — it may be **restarting** (e.g. after a heavy "
             "model load ran out of memory). It usually comes back within a few seconds.")
    st.caption(f"`{e}`")
    if st.button("🔄 Retry now"):
        st.rerun()
    import time
    time.sleep(3)
    st.rerun()
except Exception as e:
    st.error(
        f"❌ Cannot reach the backend at `{BACKEND}`.\n\n"
        "Start it first:\n\n```\npython -m uvicorn backend.main:app --port 8000\n```\n\n"
        f"`{type(e).__name__}: {e}`"
    )
    st.stop()

if health.get("status") == "error":
    st.error(f"❌ The backend failed to start:\n\n`{health.get('error')}`")
    st.stop()

if not health.get("data_ready"):
    st.title("🎬 Hybrid Movie Recommender")
    st.info("⏳ The backend is starting up (loading the movie catalogue)… a few seconds.")
    if st.button("🔄 Refresh now"):
        st.rerun()
    import time
    time.sleep(2.5)
    st.rerun()

MODELS = get_models(st.session_state.load_nonce)
LOADED = {m["key"] for m in MODELS if m.get("loaded")}
AVAIL = {m["key"] for m in MODELS if m.get("available")}
LABELS = {m["key"]: m["label"] for m in MODELS}
RAM = {m["key"]: m.get("ram_gb", 0) for m in MODELS}


def ram_tag(key):
    g = RAM.get(key, 0)
    return f"~{g:.1f} GB" + (" ⚠️" if g >= 3 else "")


_freed = st.session_state.pop("_freed_note", None)
if _freed:
    st.toast("♻️ Unloaded " + ", ".join(_freed) + " to free memory", icon="♻️")


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Hybrid Recommender")
    st.caption("MSc AI · Εφαρμογές Τεχνητής Νοημοσύνης · MovieLens 25M")
    st.caption(f"{len(LOADED)}/{len(AVAIL)} models loaded into memory")
    st.divider()
    st.markdown("**⚙️ Load models on demand**")
    st.caption("Models load only when you ask. Heavy models (>1 GB — SVD, Weighted, k-NN) "
               "don't fit together: loading one frees the previous. The small content models "
               "and LightGCN stay resident and coexist freely.")
    for fam in ["Hybrid", "Collaborative", "Content-Based"]:
        fam_ms = [m for m in MODELS if m["family"] == fam and m["available"]]
        if not fam_ms:
            continue
        st.markdown(f"*{fam}*")
        for m in fam_ms:
            c1, c2 = st.columns([3, 1])
            c1.markdown(("✅ " if m["loaded"] else "⬜ ") + m["label"]
                        + f"  \n<small>{ram_tag(m['key'])}"
                        + (" · ranking-only" if m["ranking_only"] else "") + "</small>",
                        unsafe_allow_html=True)
            if not m["loaded"]:
                if c2.button("Load", key=f"side_load_{m['key']}"):
                    load_model(m["key"])
    st.divider()
    st.caption("⭐ = predicted rating (0.5–5) · Score = relevance for ranking-only models")


# ── tabs ────────────────────────────────────────────────────────────────────
tab_user, tab_new, tab_side, tab_explore, tab_inspect, tab_compare = st.tabs([
    "👤 Existing User", "🆕 New User (live)", "🆚 Side-by-side",
    "🔍 Movie Explorer", "🎯 Prediction Inspector", "📊 Comparison",
])


# ══ Tab 1 · Existing User ═════════════════════════════════════════════════════
with tab_user:
    st.header("Recommendations for an existing user")
    sample_users = get_sample_users()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        pick = st.selectbox("Sample user", sample_users, key="eu_user")
        manual = st.number_input("…or type a userId", min_value=1, value=int(pick), step=1, key="eu_manual")
        user_id = int(manual)
    with c2:
        model = model_selectbox(key="eu_model", default="content_genome")
    with c3:
        k = st.slider("Top-K", 5, 30, 10, key="eu_k")

    with st.expander("🎛️ Filters (genre / year)"):
        fc1, fc2 = st.columns(2)
        with fc1:
            sel_genres = st.multiselect("Genres", get_genres(), key="eu_genres")
        with fc2:
            yr = st.slider("Release year", 1902, 2019, (1902, 2019), key="eu_year")
        year_min = yr[0] if yr[0] > 1902 else None
        year_max = yr[1] if yr[1] < 2019 else None

    if model and ensure_model_ui(model, "eu_load"):
        if st.button("Get recommendations", type="primary", key="eu_go"):
            prof = get_profile(user_id) if any(u == user_id for u in sample_users) or True else None
            try:
                prof = get_profile(user_id)
            except Exception:
                prof = None
            if not prof or prof.get("n_ratings", 0) == 0:
                st.warning(f"User {user_id} not found in the training data — try a sample user.")
            else:
                with st.spinner("Scoring candidates…"):
                    resp = do_recommend(user_id=user_id, model=model, k=k,
                                        genres=sel_genres or None, year_min=year_min, year_max=year_max)
                st.session_state["eu_resp"] = resp
                st.session_state["eu_uid"] = user_id
                st.session_state["eu_modelkey"] = model

    if st.session_state.get("eu_resp"):
        resp = st.session_state["eu_resp"]
        uid = st.session_state["eu_uid"]
        cur_model = st.session_state.get("eu_modelkey", "content_genome")
        prof = get_profile(uid)
        left, right = st.columns(2)
        with left:
            tag = "relevance score" if resp["ranking_only"] else "predicted rating ★"
            st.subheader(f"Top-{len(resp['items'])} · {resp['label']}")
            st.caption(f"Ranked by {tag}")
            st.dataframe(recs_to_df(resp["items"], resp["ranking_only"]),
                         use_container_width=True, hide_index=True)
        with right:
            st.subheader(f"User {uid} — top-rated history")
            st.caption(f"{prof['n_ratings']} ratings · mean {prof['mean_rating']}")
            hist = pd.DataFrame(prof["history"])
            if not hist.empty:
                st.dataframe(hist[["title", "year", "genres", "rating"]].rename(
                    columns={"title": "Title", "year": "Year", "genres": "Genres", "rating": "Rating ★"}),
                    use_container_width=True, hide_index=True)

        st.markdown("##### 💡 Why was a movie recommended?")
        titles = {it["title"]: it["movieId"] for it in resp["items"]}
        if titles:
            chosen = st.selectbox("Explain a recommendation", list(titles), key="eu_explain")
            if st.button("Explain", key="eu_explain_go"):
                exp = api_post("/api/explain", {"user_id": uid, "movie_id": titles[chosen], "model": cur_model})
                because = exp["because_you_liked"]
                if because:
                    st.markdown(f"**{chosen}** was recommended because you rated these similar films highly:")
                    st.dataframe(pd.DataFrame(because)[["title", "your_rating", "similarity", "genres"]].rename(
                        columns={"title": "You rated", "your_rating": "Your ★",
                                 "similarity": "Content sim.", "genres": "Genres"}),
                        use_container_width=True, hide_index=True)
                else:
                    st.info("No strongly similar rated movies — driven by the CF / popularity signal.")
                agree = pd.DataFrame([p for p in exp["model_agreement"]
                                      if not p["ranking_only"] and p["value"] is not None])
                if not agree.empty:
                    fig = px.bar(agree, x="label", y="value", title="What each loaded model predicts",
                                 text_auto=".2f")
                    fig.update_layout(xaxis_title="", yaxis_title="pred. rating",
                                      xaxis_tickangle=-25, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)


# ══ Tab 2 · New User (live "watch it learn") ══════════════════════════════════
with tab_new:
    st.header("New user — watch the recommendations learn")
    st.caption("A synthetic user (not in the dataset). Rate movies and see how the "
               "recommendations shift as ratings accumulate.")

    if "live_ratings" not in st.session_state:
        st.session_state.live_ratings = {}
        st.session_state.live_titles = {}
        st.session_state.live_prev = []

    popular = get_popular(60)
    pop_lookup = {p["movieId"]: p for p in popular}

    lc1, lc2 = st.columns([3, 2])
    with lc1:
        st.markdown("**Rate a movie**")
        opts = {f"{p['title']}  ({p['year']})" if p["year"] else p["title"]: p["movieId"]
                for p in popular if p["movieId"] not in st.session_state.live_ratings}
        rc1, rc2, rc3 = st.columns([3, 2, 1])
        with rc1:
            chosen = st.selectbox("Popular movie", list(opts), key="live_pick", label_visibility="collapsed")
        with rc2:
            rating = st.select_slider("Rating", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                                      value=4.0, key="live_rate", label_visibility="collapsed")
        with rc3:
            if st.button("➕ Add", key="live_add") and opts:
                mid = opts[chosen]
                st.session_state.live_ratings[mid] = float(rating)
                st.session_state.live_titles[mid] = pop_lookup[mid]["title"]
                st.rerun()

        q = st.text_input("…or search any movie to rate", key="live_search")
        if q:
            for h in search_movies(q, limit=8):
                bc1, bc2 = st.columns([4, 1])
                bc1.write(f"{h['title']} ({h['year'] or '—'}) · {h['genres']}")
                if bc2.button("Rate 4.5★", key=f"live_addsrch_{h['movieId']}"):
                    st.session_state.live_ratings[h["movieId"]] = 4.5
                    st.session_state.live_titles[h["movieId"]] = h["title"]
                    st.rerun()

    with lc2:
        st.markdown(f"**Your ratings ({len(st.session_state.live_ratings)})**")
        if st.session_state.live_ratings:
            st.dataframe(pd.DataFrame(
                [{"Title": st.session_state.live_titles.get(m, m), "★": r}
                 for m, r in st.session_state.live_ratings.items()]),
                use_container_width=True, hide_index=True, height=240)
            if st.button("🗑️ Reset", key="live_reset"):
                st.session_state.live_ratings, st.session_state.live_titles, st.session_state.live_prev = {}, {}, []
                st.rerun()
        else:
            st.info("Rate ≥1 movie to get started.")

    st.divider()
    mc1, mc2 = st.columns([2, 1])
    with mc1:
        live_model = model_selectbox(key="live_model", only_cold_start=True, default="content_genome")
    with mc2:
        live_k = st.slider("Top-K", 5, 20, 10, key="live_k")

    if live_model and not st.session_state.live_ratings:
        pass
    elif live_model and not ensure_model_ui(live_model, "live_load"):
        pass
    elif live_model:
        with st.spinner("Recomputing recommendations…"):
            resp = do_recommend(ratings=st.session_state.live_ratings, model=live_model, k=live_k)
        items = resp["items"]
        cur_ids = [it["movieId"] for it in items]
        prev = st.session_state.live_prev
        st.subheader(f"Recommendations after {len(st.session_state.live_ratings)} rating(s) · {resp['label']}")
        rows = []
        for rank, it in enumerate(items, 1):
            if it["movieId"] not in prev:
                badge = "🆕"
            else:
                old = prev.index(it["movieId"]) + 1
                badge = "▲" if rank < old else ("▼" if rank > old else "•")
            rows.append({"#": rank, "": badge, "Title": it["title"], "Year": it.get("year"),
                         "Genres": it["genres"], "★": it.get("predicted_rating", it.get("score"))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("🆕 new since last rating · ▲ moved up · ▼ moved down")
        gc = {}
        for it in items:
            for g in str(it["genres"]).split("|"):
                gc[g] = gc.get(g, 0) + 1
        if gc:
            gdf = pd.DataFrame(sorted(gc.items(), key=lambda x: -x[1])[:10], columns=["Genre", "Count"])
            fig = px.bar(gdf, x="Genre", y="Count", title="Genre mix of current recommendations")
            fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.session_state.live_prev = cur_ids


# ══ Tab 3 · Side-by-side ══════════════════════════════════════════════════════
with tab_side:
    st.header("Compare models side-by-side")
    st.caption("Same user, several models. Load the ones you want, then compare. "
               "Movies recommended by ≥2 models are starred.")
    sample_users = get_sample_users()
    sc1, sc2, sc3 = st.columns([2, 3, 1])
    with sc1:
        s_user = st.selectbox("User", sample_users, key="side_user")
    with sc2:
        avail = [m for m in MODELS if m["available"]]
        defaults = [m["key"] for m in avail if m["key"] in ("content_genome", "svd", "dual")]
        s_models = st.multiselect("Models", [m["key"] for m in avail],
                                  default=defaults or [avail[0]["key"]],
                                  format_func=lambda k: ("✅ " if k in LOADED else "⬜ ") + LABELS[k],
                                  key="side_models")
    with sc3:
        s_k = st.slider("Top-K", 5, 20, 10, key="side_k")

    missing = [m for m in s_models if m not in LOADED]
    heavy_sel = [m for m in s_models if RAM.get(m, 0) > 1.0]
    if len(heavy_sel) > 1:
        st.warning("⚠️ Only **one model larger than 1 GB fits at a time** on this machine, so "
                   f"the {len(heavy_sel)} heavy models you picked ("
                   + ", ".join(LABELS[m] for m in heavy_sel) + ") can't be compared side-by-side — "
                   "loading one evicts the others. Keep at most one heavy model; the small content "
                   "models (and LightGCN) coexist freely, so compare those together plus one heavy.")
    elif missing:
        total = sum(RAM.get(m, 0) for m in missing)
        st.warning("Not loaded: " + ", ".join(f"{LABELS[m]} ({ram_tag(m)})" for m in missing))
        if st.button(f"⬇️ Load {len(missing)} model(s)  ·  ~{total:.1f} GB total",
                     key="side_loadmissing", type="primary"):
            prog = st.progress(0.0)
            for i, m in enumerate(missing, 1):
                with st.spinner(f"Loading {LABELS[m]}…"):
                    try:
                        api_post(f"/api/models/{m}/load", {})
                    except InsufficientMemory as e:
                        st.warning(f"{LABELS[m]}: {e}")
                    except BackendDown:
                        st.error("Backend went down (out of memory) — restarting. Refresh shortly.")
                        st.stop()
                    except Exception as e:
                        st.error(f"{LABELS[m]}: {e}")
                prog.progress(i / len(missing))
            st.session_state.load_nonce += 1
            get_models.clear()
            st.rerun()

    if st.button("Compare", type="primary", key="side_go", disabled=bool(missing)) and s_models:
        with st.spinner("Scoring across models…"):
            resp = api_post("/api/recommend/compare", {"user_id": int(s_user), "models": s_models, "k": s_k})
        results = resp["models"]
        freq = {}
        for m in s_models:
            for it in results[m]["items"]:
                freq[it["movieId"]] = freq.get(it["movieId"], 0) + 1
        cols = st.columns(len(s_models))
        for col, m in zip(cols, s_models):
            with col:
                r = results[m]
                st.markdown(f"**{r['label']}**")
                if r.get("error"):
                    st.error(r["error"])
                    continue
                rows = [{"#": rank, "Title": ("⭐" if freq.get(it["movieId"], 0) > 1 else "") + it["title"],
                         "★": it.get("predicted_rating", it.get("score"))}
                        for rank, it in enumerate(r["items"], 1)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)
        st.caption(f"⭐ recommended by ≥2 models · {sum(1 for v in freq.values() if v > 1)} overlap")


# ══ Tab 4 · Movie Explorer ════════════════════════════════════════════════════
with tab_explore:
    st.header("Movie explorer — items like this")
    st.caption("Search a movie, then see its nearest neighbours under each content space. "
               "Load a content model to enable its column.")
    q = st.text_input("Search a movie", key="exp_q", placeholder="e.g. Matrix, Toy Story, Inception")
    if q:
        hits = search_movies(q, limit=20)
        if hits:
            opts = {f"{h['title']} ({h['year'] or '—'})": h["movieId"] for h in hits}
            chosen = st.selectbox("Pick a movie", list(opts), key="exp_pick")
            mid = opts[chosen]
            meta = next(h for h in hits if h["movieId"] == mid)
            st.markdown(f"**{meta['title']}** · {meta['genres']} · {meta['n_ratings']:,} ratings")
            ek = st.slider("Neighbours", 5, 20, 10, key="exp_k")
            spaces = [("genome", "content_genome", "Tag Genome"),
                      ("tfidf", "content", "TF-IDF"), ("embed", "content_embed", "Embeddings")]
            cols = st.columns(len(spaces))
            for col, (space, key, name) in zip(cols, spaces):
                with col:
                    st.markdown(f"**{name}**")
                    if key not in AVAIL:
                        st.caption("artifact missing")
                    elif key not in LOADED:
                        if st.button(f"⬇️ Load ({ram_tag(key)})", key=f"exp_load_{space}"):
                            load_model(key)
                    else:
                        sims = get_similar(mid, space, ek, st.session_state.load_nonce)
                        st.dataframe(pd.DataFrame(sims)[["title", "similarity"]].rename(
                            columns={"title": "Title", "similarity": "Sim."}),
                            use_container_width=True, hide_index=True, height=380)
        else:
            st.info("No matches.")


# ══ Tab 5 · Prediction Inspector ══════════════════════════════════════════════
with tab_inspect:
    st.header("Prediction inspector")
    st.caption("Pick a user and a movie — see the predicted rating from every **loaded** "
               "model vs the held-out true rating. Load more models in the sidebar to compare them.")
    sample_users = get_sample_users()
    ic1, ic2 = st.columns([1, 2])
    with ic1:
        i_user = st.selectbox("User", sample_users, key="insp_user")
    with ic2:
        iq = st.text_input("Search a movie", key="insp_q", placeholder="movie title…")

    movie_id = None
    if iq:
        hits = search_movies(iq, limit=15)
        if hits:
            opts = {f"{h['title']} ({h['year'] or '—'})": h["movieId"] for h in hits}
            movie_id = opts[st.selectbox("Movie", list(opts), key="insp_pick")]

    if not LOADED:
        st.info("No models loaded yet — load one or more in the sidebar to see predictions.")
    if movie_id and st.button("Inspect", type="primary", key="insp_go"):
        data = get_predictions(int(i_user), int(movie_id), st.session_state.load_nonce)
        m = data["movie"]
        st.markdown(f"**{m['title']}** · {m['genres']}")
        tr = data["true_rating"]
        if tr:
            st.info(f"True rating (from {tr['source']} set): **{tr['rating']} ★**")
        else:
            st.caption("No recorded rating for this (user, movie) pair.")
        rating_preds = [p for p in data["predictions"] if not p["ranking_only"] and p["value"] is not None]
        if rating_preds:
            df = pd.DataFrame(rating_preds)
            fig = px.bar(df, x="label", y="value", text_auto=".2f", title="Predicted rating by loaded model",
                         color="value", color_continuous_scale="Blues", range_color=[0.5, 5])
            if tr:
                fig.add_hline(y=tr["rating"], line_dash="dash", line_color="crimson",
                              annotation_text="true rating")
            fig.update_layout(xaxis_title="", yaxis_title="pred. rating", xaxis_tickangle=-25,
                              coloraxis_showscale=False, yaxis_range=[0, 5.2])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No rating models loaded — load SVD / a content model / a hybrid in the sidebar.")
        for p in [p for p in data["predictions"] if p["ranking_only"] and p["value"] is not None]:
            st.caption(f"{p['label']}: relevance score = {p['value']} (not a star rating)")


# ══ Tab 6 · Comparison ════════════════════════════════════════════════════════
with tab_compare:
    st.header("Model comparison — offline evaluation")
    st.caption("Metrics for **all 12 models** (from `all_metrics.json`) — independent of "
               "what's loaded in memory.")
    metrics = get_metrics()
    if not metrics:
        st.info("No metrics yet. Run the model notebooks (03–13) then 14_advanced_eval.")
    else:
        st.subheader("Rating prediction")
        rating_rows = [{"Model": n, "RMSE": v["rmse"], "MAE": v["mae"]}
                       for n, v in metrics.items() if v.get("rmse") is not None]
        if rating_rows:
            st.dataframe(pd.DataFrame(rating_rows).sort_values("RMSE").set_index("Model")
                         .style.highlight_min(color="#1b5e20"), use_container_width=True)

        st.subheader("Ranking metrics")
        rank_rows = []
        for n, v in metrics.items():
            for kk in (5, 10, 20):
                if f"k{kk}" in v:
                    rank_rows.append({"Model": n, "K": kk,
                                      "Precision@K": round(v[f"k{kk}"]["precision"], 4),
                                      "Recall@K": round(v[f"k{kk}"]["recall"], 4),
                                      "F1@K": round(v[f"k{kk}"]["f1"], 4)})
        if rank_rows:
            dfk = pd.DataFrame(rank_rows)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.dataframe(dfk, use_container_width=True, hide_index=True, height=420)
            with cc2:
                d10 = dfk[dfk["K"] == 10].sort_values("F1@K", ascending=False)
                fig = px.bar(d10, x="Model", y="F1@K", color="Model", text_auto=".3f", title="F1@10 by model")
                fig.update_layout(showlegend=False, xaxis_tickangle=-35, height=420)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rating accuracy vs ranking quality")
        scat = [{"Model": n, "RMSE": v["rmse"], "F1@10": v["k10"]["f1"]}
                for n, v in metrics.items() if v.get("rmse") is not None and "k10" in v]
        if scat:
            fig = px.scatter(pd.DataFrame(scat), x="RMSE", y="F1@10", text="Model", color="Model",
                             title="Lower RMSE ↔ higher F1@10 (top-left is best)")
            fig.update_traces(textposition="top center", marker_size=12)
            fig.update_layout(showlegend=False, height=480)
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Deep-evaluation figures")
        names = get_figures().get("figures", [])
        deep = [n for n in names if any(t in n for t in
                ("ndcg", "auc", "segmented", "beyond", "bootstrap", "rating_vs_ranking", "f1_curves"))]
        if deep:
            for n in st.multiselect("Show figures", deep, default=deep[:2], key="cmp_figs"):
                st.image(f"{BACKEND}/figures/{n}.png", caption=n, use_container_width=True)
        else:
            st.caption("Run 14_advanced_eval to generate the deep-eval figures.")
