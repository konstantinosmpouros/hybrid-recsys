import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_bundle():
    from hybrid_recsys.serving import RecommenderBundle
    return RecommenderBundle().load()


@st.cache_data(show_spinner=False)
def load_movies():
    from hybrid_recsys.config import DATA_PROCESSED
    return pd.read_parquet(DATA_PROCESSED / "movies.parquet")


@st.cache_data(show_spinner=False)
def load_train_ratings():
    from hybrid_recsys.config import DATA_PROCESSED
    return pd.read_parquet(DATA_PROCESSED / "split_train.parquet")


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Hybrid Recommender")
    st.caption("MSc AI · MovieLens 25M")
    st.divider()
    st.markdown(
        "**Models available**\n"
        "- Weighted Hybrid (SVD + Content)\n"
        "- Stacked Hybrid (meta-learner)\n"
        "- SVD · Item-kNN · User-kNN · Content"
    )


# ── tabs ──────────────────────────────────────────────────────────────────────
tab_existing, tab_new, tab_compare = st.tabs(
    ["👤 Existing User", "🆕 New User Onboarding", "📊 Model Comparison"]
)


# ── Tab 1 · Existing User ─────────────────────────────────────────────────────
with tab_existing:
    st.header("Recommendations for an Existing User")

    models_ready = True
    try:
        bundle = load_bundle()
        ratings = load_train_ratings()
        movies = load_movies()
    except Exception as e:
        st.warning(f"⚠️ Models not trained yet — run the training notebooks first.\n\n`{e}`")
        models_ready = False

    if models_ready:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            user_ids = sorted(ratings["userId"].unique())
            selected_user = st.selectbox("Select User ID", user_ids)
        with col2:
            model_choice = st.selectbox(
                "Model",
                ["weighted", "svd", "content", "item_knn", "user_knn"],
                format_func=lambda x: {
                    "weighted":  "Weighted Hybrid",
                    "svd":       "SVD (Matrix Factorization)",
                    "content":   "Content-Based",
                    "item_knn":  "Item-Based k-NN",
                    "user_knn":  "User-Based k-NN",
                }[x],
            )
        with col3:
            n_recs = st.slider("Top-K", 5, 20, 10)

        if st.button("Get Recommendations", type="primary"):
            user_df = ratings[ratings["userId"] == selected_user]
            user_ratings = dict(zip(user_df["movieId"], user_df["rating"]))

            with st.spinner("Generating recommendations…"):
                recs = bundle.get_recommendations(
                    selected_user, user_ratings, model=model_choice, n=n_recs
                )

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader(f"Top {n_recs} Recommendations")
                st.dataframe(
                    recs[["title", "genres", "predicted_rating"]],
                    use_container_width=True,
                    hide_index=True,
                )

            with col_right:
                st.subheader("User's Top-Rated Movies (training)")
                top_rated = (
                    user_df.sort_values("rating", ascending=False)
                    .head(10)
                    .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
                )
                st.dataframe(
                    top_rated[["title", "genres", "rating"]],
                    use_container_width=True,
                    hide_index=True,
                )


# ── Tab 2 · New User Onboarding ───────────────────────────────────────────────
with tab_new:
    st.header("New User — Rate Some Movies to Get Started")
    st.caption("Rate at least 3 movies; we'll use the content model to recommend similar ones.")

    try:
        movies = load_movies()
        bundle = load_bundle()

        seed_movies = movies.sample(12, random_state=42).reset_index(drop=True)
        seed_ratings: dict[int, float] = {}

        cols = st.columns(3)
        for idx, row in seed_movies.iterrows():
            with cols[idx % 3]:
                val = st.select_slider(
                    f"**{row['title']}**",
                    options=["Skip", 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                    value="Skip",
                    key=f"seed_{row['movieId']}",
                )
                if val != "Skip":
                    seed_ratings[int(row["movieId"])] = float(val)

        st.caption(f"Rated: {len(seed_ratings)} / 12")

        if st.button("Recommend for me!", type="primary", disabled=len(seed_ratings) < 3):
            with st.spinner("Generating recommendations…"):
                recs = bundle.get_recommendations(
                    user_id=-1,
                    user_ratings=seed_ratings,
                    model="content",
                    n=10,
                    exclude=set(seed_ratings),
                )
            st.subheader("Recommended for you")
            st.dataframe(
                recs[["title", "genres", "predicted_rating"]],
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.warning(f"⚠️ Models not trained yet.\n\n`{e}`")


# ── Tab 3 · Model Comparison ──────────────────────────────────────────────────
with tab_compare:
    st.header("Model Comparison (Offline Evaluation)")

    try:
        bundle = load_bundle()
        metrics = bundle.load_metrics()

        if not metrics:
            st.info("No evaluation metrics found yet. Run notebook `03_train_evaluate.ipynb` first.")
        else:
            # Rating prediction table
            st.subheader("Rating Prediction")
            rating_rows = [
                {"Model": name, "RMSE": m["rmse"], "MAE": m["mae"]}
                for name, m in metrics.items()
                if "rmse" in m
            ]
            if rating_rows:
                df_rating = pd.DataFrame(rating_rows).set_index("Model")
                st.dataframe(df_rating.style.highlight_min(color="#d4edda"), use_container_width=True)

            # Ranking metrics table
            st.subheader("Ranking Metrics")
            ranking_rows = []
            for name, m in metrics.items():
                for k in [5, 10, 20]:
                    key = f"k{k}"
                    if key in m:
                        ranking_rows.append({
                            "Model": name, "K": k,
                            "Precision@K": round(m[key]["precision"], 4),
                            "Recall@K":    round(m[key]["recall"], 4),
                            "F1@K":        round(m[key]["f1"], 4),
                        })

            if ranking_rows:
                df_ranking = pd.DataFrame(ranking_rows)
                st.dataframe(df_ranking, use_container_width=True, hide_index=True)

                import plotly.express as px

                fig = px.bar(
                    df_ranking[df_ranking["K"] == 10].sort_values("F1@K", ascending=False),
                    x="Model", y="F1@K",
                    title="F1@10 by Model",
                    color="Model",
                    text_auto=".4f",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not load metrics.\n\n`{e}`")
