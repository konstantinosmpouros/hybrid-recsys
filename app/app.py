"""Streamlit front-end for the Hybrid Movie Recommender.

A thin HTTP client over the FastAPI backend (`backend.main`). All model logic
lives behind the API; this file is pure UI. Set BACKEND_URL to point elsewhere.

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
    """Backend returned 503 because models/data are still loading."""


def _maybe_loading(r):
    if r.status_code == 503:
        try:
            detail = r.json().get("detail", {})
            msg = detail.get("message") if isinstance(detail, dict) else str(detail)
        except Exception:
            msg = None
        raise BackendLoading(msg or "Backend is still loading…")


def api_get(path: str, **params):
    r = requests.get(f"{BACKEND}{path}", params=params, timeout=600)
    _maybe_loading(r)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{BACKEND}{path}", json=payload, timeout=600)
    _maybe_loading(r)
    r.raise_for_status()
    return r.json()


def get_health():
    """Uncached — pinging it also triggers the backend's background load."""
    return api_get("/api/health")


def guarded(fn, *args, **kwargs):
    """Run a model-dependent call; if the backend is still loading, show a notice
    and return None instead of crashing the tab."""
    try:
        return fn(*args, **kwargs)
    except BackendLoading as e:
        st.warning(f"⏳ {e}")
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_models():
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


@st.cache_data(ttl=300, show_spinner=False)
def get_similar(movie_id: int, space: str, k: int = 10):
    return api_get(f"/api/movies/{movie_id}/similar", space=space, k=k)


@st.cache_data(ttl=300, show_spinner=False)
def get_predictions(user_id: int, movie_id: int):
    return api_get("/api/predict", user_id=user_id, movie_id=movie_id)


@st.cache_data(ttl=600, show_spinner=False)
def get_metrics():
    return api_get("/api/metrics")


@st.cache_data(ttl=600, show_spinner=False)
def get_figures():
    return api_get("/api/figures")


@st.cache_data(ttl=120, show_spinner=False)
def recommend_cached(user_id, ratings_items, model, k, genres_items, year_min, year_max):
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


def do_recommend(user_id=None, ratings=None, model="dual", k=10,
                 genres=None, year_min=None, year_max=None):
    ratings_items = tuple(sorted(ratings.items())) if ratings else None
    genres_items = tuple(sorted(genres)) if genres else None
    return recommend_cached(user_id, ratings_items, model, k, genres_items, year_min, year_max)


# ── small helpers ───────────────────────────────────────────────────────────
def recs_to_df(items: list[dict], ranking_only: bool) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    score_col = "score" if ranking_only else "predicted_rating"
    if score_col not in df:
        score_col = "score"
    cols = [c for c in ["title", "year", "genres", score_col] if c in df]
    return df[cols].rename(columns={
        "title": "Title", "year": "Year", "genres": "Genres",
        "predicted_rating": "Pred. rating ★", "score": "Score",
    })


def models_by_family(model_list, only_cold_start=False, only_available=True):
    fam_order = {"Hybrid": 0, "Collaborative": 1, "Content-Based": 2}
    ms = [m for m in model_list
          if (m["available"] or not only_available)
          and (m["cold_start"] or not only_cold_start)]
    ms.sort(key=lambda m: (fam_order.get(m["family"], 9), m["label"]))
    return ms


def model_selectbox(model_list, label="Model", key=None, only_cold_start=False, default="dual"):
    ms = models_by_family(model_list, only_cold_start=only_cold_start)
    keys = [m["key"] for m in ms]
    labels = {m["key"]: f"{m['label']}  ·  {m['family']}" for m in ms}
    idx = keys.index(default) if default in keys else 0
    return st.selectbox(label, keys, index=idx, format_func=lambda k: labels[k], key=key)


# ── connectivity + readiness (non-blocking) ──────────────────────────────────
try:
    health = get_health()           # pinging health kicks off the background load
except Exception as e:
    st.error(
        f"❌ Cannot reach the backend at `{BACKEND}`.\n\n"
        "Start it first:\n\n```\npython -m uvicorn backend.main:app --port 8000\n```\n\n"
        f"`{type(e).__name__}: {e}`"
    )
    st.stop()

MODELS = get_models()               # static (file existence) — returns instantly
DATA_READY = health.get("data_ready")
MODELS_READY = health.get("models_ready")

if health.get("status") == "error":
    st.error(f"❌ The backend failed to load models:\n\n`{health.get('error')}`")
    st.stop()

# Catalogue still loading (first few seconds) — auto-poll until ready.
if not DATA_READY:
    st.title("🎬 Hybrid Movie Recommender")
    st.info("⏳ The backend is starting up (loading the movie catalogue)… a few seconds.")
    if st.button("🔄 Refresh now"):
        st.rerun()
    import time
    time.sleep(2.5)
    st.rerun()

# Data is ready → the whole UI renders. Models may still be loading in the
# background; model-dependent actions degrade gracefully (see `guarded`).
if not MODELS_READY:
    bc1, bc2 = st.columns([8, 1])
    with bc1:
        st.warning("⏳ **Models are still loading in the background.** Catalogue, search and "
                   "the comparison tab work now; recommendations, predictions and explanations "
                   "will be available in ~1 minute.")
    with bc2:
        if st.button("🔄 Refresh"):
            st.rerun()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Hybrid Recommender")
    st.caption("MSc AI · Εφαρμογές Τεχνητής Νοημοσύνης · MovieLens 25M")
    n_avail = sum(1 for m in MODELS if m["available"])
    if MODELS_READY:
        st.success(f"Backend ready · {n_avail}/{len(MODELS)} models")
    else:
        st.info(f"⏳ Loading models… ({n_avail} on disk)")
    st.divider()
    st.markdown("**Models**")
    for fam in ["Hybrid", "Collaborative", "Content-Based"]:
        fam_models = [m for m in MODELS if m["family"] == fam and m["available"]]
        if fam_models:
            st.markdown(f"*{fam}*")
            for m in fam_models:
                tag = " · ranking-only" if m["ranking_only"] else ""
                st.markdown(f"- {m['label']}{tag}")
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
        model = model_selectbox(MODELS, key="eu_model", default="dual")
    with c3:
        k = st.slider("Top-K", 5, 30, 10, key="eu_k")

    with st.expander("🎛️ Filters (genre / year)"):
        fc1, fc2 = st.columns([2, 2])
        with fc1:
            sel_genres = st.multiselect("Genres", get_genres(), key="eu_genres")
        with fc2:
            yr = st.slider("Release year", 1902, 2019, (1902, 2019), key="eu_year")
        year_min = yr[0] if yr[0] > 1902 else None
        year_max = yr[1] if yr[1] < 2019 else None

    if not MODELS_READY:
        st.caption("⏳ Models loading — recommendations available shortly. Use 🔄 Refresh above.")
    if st.button("Get recommendations", type="primary", key="eu_go", disabled=not MODELS_READY):
        try:
            prof = get_profile(user_id)
        except Exception:
            prof = None
        if prof is None or prof.get("n_ratings", 0) == 0:
            st.warning(f"User {user_id} not found in the training data — try a sample user.")
        else:
            with st.spinner("Scoring candidates…"):
                resp = do_recommend(user_id=user_id, model=model, k=k,
                                    genres=sel_genres or None, year_min=year_min, year_max=year_max)
            st.session_state["eu_resp"] = resp
            st.session_state["eu_uid"] = user_id
            st.session_state["eu_model"] = model

    if st.session_state.get("eu_resp"):
        resp = st.session_state["eu_resp"]
        uid = st.session_state["eu_uid"]
        cur_model = st.session_state.get("eu_model", "dual")
        prof = get_profile(uid)
        left, right = st.columns(2)
        with left:
            tag = "relevance score" if resp["ranking_only"] else "predicted rating ★"
            st.subheader(f"Top-{len(resp['items'])} · {resp['label']}")
            st.caption(f"Ranked by {tag}" + ("  ·  (popularity-retrieval + re-rank)" if cur_model == "dual" else ""))
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

        # "Why this?" explanation
        st.markdown("##### 💡 Why was a movie recommended?")
        titles = {it["title"]: it["movieId"] for it in resp["items"]}
        if titles:
            chosen = st.selectbox("Explain a recommendation", list(titles), key="eu_explain")
            if st.button("Explain", key="eu_explain_go", disabled=not MODELS_READY):
                exp = api_post("/api/explain", {"user_id": uid, "movie_id": titles[chosen], "model": cur_model})
                because = exp["because_you_liked"]
                if because:
                    st.markdown(f"**{chosen}** was recommended because you rated these similar films highly:")
                    st.dataframe(pd.DataFrame(because)[["title", "your_rating", "similarity", "genres"]].rename(
                        columns={"title": "You rated", "your_rating": "Your ★",
                                 "similarity": "Content sim.", "genres": "Genres"}),
                        use_container_width=True, hide_index=True)
                else:
                    st.info("No strongly similar rated movies — the score is driven by the CF / popularity signal.")
                agree = pd.DataFrame([p for p in exp["model_agreement"]
                                      if not p["ranking_only"] and p["value"] is not None])
                if not agree.empty:
                    fig = px.bar(agree, x="label", y="value", title="What each model predicts for this movie",
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
        st.session_state.live_ratings = {}      # movieId -> rating
        st.session_state.live_titles = {}        # movieId -> title
        st.session_state.live_prev = []          # previous top movieIds

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

        # search-to-rate (beyond the popular list)
        q = st.text_input("…or search any movie to rate", key="live_search")
        if q:
            hits = search_movies(q, limit=8)
            for h in hits:
                bc1, bc2 = st.columns([4, 1])
                bc1.write(f"{h['title']} ({h['year'] or '—'}) · {h['genres']}")
                if bc2.button("Rate 4.5★", key=f"live_addsrch_{h['movieId']}"):
                    st.session_state.live_ratings[h["movieId"]] = 4.5
                    st.session_state.live_titles[h["movieId"]] = h["title"]
                    st.rerun()

    with lc2:
        st.markdown(f"**Your ratings ({len(st.session_state.live_ratings)})**")
        if st.session_state.live_ratings:
            rated = pd.DataFrame(
                [{"Title": st.session_state.live_titles.get(m, m), "★": r}
                 for m, r in st.session_state.live_ratings.items()])
            st.dataframe(rated, use_container_width=True, hide_index=True, height=240)
            if st.button("🗑️ Reset", key="live_reset"):
                st.session_state.live_ratings = {}
                st.session_state.live_titles = {}
                st.session_state.live_prev = []
                st.rerun()
        else:
            st.info("Rate ≥1 movie to get started.")

    st.divider()
    mc1, mc2 = st.columns([2, 1])
    with mc1:
        live_model = model_selectbox(MODELS, key="live_model", only_cold_start=True, default="content_genome")
    with mc2:
        live_k = st.slider("Top-K", 5, 20, 10, key="live_k")

    if st.session_state.live_ratings and not MODELS_READY:
        st.info("⏳ Models are loading — your live recommendations will appear once ready. Use 🔄 Refresh above.")
    elif st.session_state.live_ratings:
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

        # genre drift of the recommendation list
        genre_counts = {}
        for it in items:
            for g in str(it["genres"]).split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        if genre_counts:
            gdf = pd.DataFrame(sorted(genre_counts.items(), key=lambda x: -x[1])[:10],
                               columns=["Genre", "Count"])
            fig = px.bar(gdf, x="Genre", y="Count", title="Genre mix of current recommendations")
            fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.live_prev = cur_ids


# ══ Tab 3 · Side-by-side ══════════════════════════════════════════════════════
with tab_side:
    st.header("Compare models side-by-side")
    st.caption("Same user, several models — see how their top-K lists differ. "
               "Movies recommended by more than one model are starred.")
    sample_users = get_sample_users()
    sc1, sc2, sc3 = st.columns([2, 3, 1])
    with sc1:
        s_user = st.selectbox("User", sample_users, key="side_user")
    with sc2:
        avail = [m for m in MODELS if m["available"]]
        default_models = [m["key"] for m in avail if m["key"] in ("dual", "svd", "content_genome", "item_knn")]
        s_models = st.multiselect("Models", [m["key"] for m in avail],
                                  default=default_models or [avail[0]["key"]],
                                  format_func=lambda k: next(m["label"] for m in avail if m["key"] == k),
                                  key="side_models")
    with sc3:
        s_k = st.slider("Top-K", 5, 20, 10, key="side_k")

    if not MODELS_READY:
        st.caption("⏳ Models loading — comparison available shortly. Use 🔄 Refresh above.")
    if st.button("Compare", type="primary", key="side_go", disabled=not MODELS_READY) and s_models:
        with st.spinner("Scoring across models…"):
            resp = api_post("/api/recommend/compare",
                            {"user_id": int(s_user), "models": s_models, "k": s_k})
        results = resp["models"]
        # overlap counts
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
                rows = []
                for rank, it in enumerate(r["items"], 1):
                    star = "⭐" if freq.get(it["movieId"], 0) > 1 else ""
                    rows.append({"#": rank, "Title": f"{star}{it['title']}",
                                 "★": it.get("predicted_rating", it.get("score"))})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)
        shared = sum(1 for v in freq.values() if v > 1)
        st.caption(f"⭐ recommended by ≥2 models · {shared} movies overlap across the selected models")


# ══ Tab 4 · Movie Explorer ════════════════════════════════════════════════════
with tab_explore:
    st.header("Movie explorer — items like this")
    st.caption("Search a movie, then see its nearest neighbours under each content "
               "representation (tag genome / TF-IDF / sentence embeddings).")
    q = st.text_input("Search a movie", key="exp_q", placeholder="e.g. Matrix, Toy Story, Inception")
    if q:
        hits = search_movies(q, limit=20)
        if hits:
            opts = {f"{h['title']} ({h['year'] or '—'})": h["movieId"] for h in hits}
            chosen = st.selectbox("Pick a movie", list(opts), key="exp_pick")
            mid = opts[chosen]
            meta = next(h for h in hits if h["movieId"] == mid)
            st.markdown(f"**{meta['title']}** · {meta['genres']} · {meta['n_ratings']:,} ratings")

            spaces = [("genome", "Tag Genome"), ("tfidf", "TF-IDF"), ("embed", "Embeddings")]
            ek = st.slider("Neighbours", 5, 20, 10, key="exp_k")
            cols = st.columns(len(spaces))
            for col, (space, name) in zip(cols, spaces):
                with col:
                    st.markdown(f"**{name}**")
                    try:
                        sims = get_similar(mid, space, ek)
                    except BackendLoading:
                        sims = None
                    except Exception:
                        sims = []
                    if sims:
                        st.dataframe(pd.DataFrame(sims)[["title", "similarity"]].rename(
                            columns={"title": "Title", "similarity": "Sim."}),
                            use_container_width=True, hide_index=True, height=420)
                    elif sims is None:
                        st.caption("⏳ models loading…")
                    else:
                        st.caption("model not available")
        else:
            st.info("No matches.")


# ══ Tab 5 · Prediction Inspector ══════════════════════════════════════════════
with tab_inspect:
    st.header("Prediction inspector")
    st.caption("Pick a user and a movie — see every model's predicted rating vs the "
               "held-out true rating. Great for the oral demo.")
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
            pick = st.selectbox("Movie", list(opts), key="insp_pick")
            movie_id = opts[pick]

    if not MODELS_READY:
        st.caption("⏳ Models loading — prediction inspector available shortly. Use 🔄 Refresh above.")
    if movie_id and st.button("Inspect", type="primary", key="insp_go", disabled=not MODELS_READY):
        data = get_predictions(int(i_user), int(movie_id))
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
            fig = px.bar(df, x="label", y="value", text_auto=".2f",
                         title="Predicted rating by model", color="value",
                         color_continuous_scale="Blues", range_color=[0.5, 5])
            if tr:
                fig.add_hline(y=tr["rating"], line_dash="dash", line_color="crimson",
                              annotation_text="true rating")
            fig.update_layout(xaxis_title="", yaxis_title="pred. rating", xaxis_tickangle=-25,
                              coloraxis_showscale=False, yaxis_range=[0, 5.2])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[["label", "value"]].rename(columns={"label": "Model", "value": "Pred ★"}),
                         use_container_width=True, hide_index=True)
        rank_only = [p for p in data["predictions"] if p["ranking_only"] and p["value"] is not None]
        for p in rank_only:
            st.caption(f"{p['label']}: relevance score = {p['value']} (not a star rating)")


# ══ Tab 6 · Comparison ════════════════════════════════════════════════════════
with tab_compare:
    st.header("Model comparison — offline evaluation")
    metrics = get_metrics()
    if not metrics:
        st.info("No metrics yet. Run the model notebooks (03–13) then 14_advanced_eval.")
    else:
        # rating table
        st.subheader("Rating prediction")
        rating_rows = [{"Model": n, "RMSE": v["rmse"], "MAE": v["mae"]}
                       for n, v in metrics.items() if v.get("rmse") is not None]
        if rating_rows:
            dfr = pd.DataFrame(rating_rows).sort_values("RMSE").set_index("Model")
            st.dataframe(dfr.style.highlight_min(color="#1b5e20"), use_container_width=True)

        # ranking table
        st.subheader("Ranking metrics")
        rank_rows = []
        for n, v in metrics.items():
            for kk in (5, 10, 20):
                key = f"k{kk}"
                if key in v:
                    rank_rows.append({"Model": n, "K": kk,
                                      "Precision@K": round(v[key]["precision"], 4),
                                      "Recall@K": round(v[key]["recall"], 4),
                                      "F1@K": round(v[key]["f1"], 4)})
        if rank_rows:
            dfk = pd.DataFrame(rank_rows)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.dataframe(dfk, use_container_width=True, hide_index=True, height=420)
            with cc2:
                d10 = dfk[dfk["K"] == 10].sort_values("F1@K", ascending=False)
                fig = px.bar(d10, x="Model", y="F1@K", color="Model", text_auto=".3f",
                             title="F1@10 by model")
                fig.update_layout(showlegend=False, xaxis_tickangle=-35, height=420)
                st.plotly_chart(fig, use_container_width=True)

        # rating-vs-ranking trade-off scatter
        st.subheader("Rating accuracy vs ranking quality")
        scat = [{"Model": n, "RMSE": v["rmse"], "F1@10": v["k10"]["f1"]}
                for n, v in metrics.items() if v.get("rmse") is not None and "k10" in v]
        if scat:
            dfs = pd.DataFrame(scat)
            fig = px.scatter(dfs, x="RMSE", y="F1@10", text="Model", color="Model",
                             title="Lower RMSE ↔ higher F1@10 (top-left is best)")
            fig.update_traces(textposition="top center", marker_size=12)
            fig.update_layout(showlegend=False, height=480)
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # deep-eval figures (served by the backend as static PNGs)
        st.subheader("Deep-evaluation figures")
        figs = get_figures()
        names = figs.get("figures", [])
        deep = [n for n in names if any(t in n for t in
                ("ndcg", "auc", "segmented", "beyond", "bootstrap", "rating_vs_ranking", "f1_curves"))]
        if deep:
            pick = st.multiselect("Show figures", deep, default=deep[:2], key="cmp_figs")
            for n in pick:
                st.image(f"{BACKEND}/figures/{n}.png", caption=n, use_container_width=True)
        else:
            st.caption("Run 14_advanced_eval to generate NDCG/AUC/segmented/bootstrap figures.")
