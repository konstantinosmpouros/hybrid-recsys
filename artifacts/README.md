# artifacts/

Auto-generated outputs: `models/` from `notebooks/03_train.ipynb`; `metrics/` and the
evaluation `figures/` from `notebooks/04_evaluation.ipynb`.
No files in `models/` or `metrics/` are tracked by git.

## models/

Serialised model objects saved with `joblib`. Loaded at runtime by `RecommenderBundle`.

| File | Class | Description |
| --- | --- | --- |
| `content_model.joblib` | `ContentBasedRecommender` | Fitted item-item cosine similarity model |
| `user_knn_model.joblib` | `UserKNNModel` | Surprise KNNWithMeans, user-based |
| `item_knn_model.joblib` | `ItemKNNModel` | Surprise KNNWithMeans, item-based |
| `svd_model.joblib` | `SVDModel` | Surprise SVD with tuned hyperparameters |
| `weighted_hybrid.joblib` | `WeightedHybrid` | α·SVD + (1−α)·CB with tuned α |
| `stacked_hybrid.joblib` | `StackedHybrid` | Ridge meta-learner trained on OOF predictions |

## metrics/

| File | Description |
| --- | --- |
| `all_metrics.json` | RMSE, MAE, P@K, R@K, F1@K for every model — loaded by the Streamlit app |

## figures/

Plotly charts exported as HTML (always) and PNG (if kaleido is installed).
These are tracked by git as reference outputs.

| File | Notebook | Description |
| --- | --- | --- |
| `01_rating_distribution.html/png` | 01_eda | Rating value histogram |
| `02_ratings_per_user.html/png` | 01_eda | Log-scale ratings per user |
| `03_ratings_per_movie.html/png` | 01_eda | Long-tail movie popularity |
| `04_ratings_over_time.html/png` | 01_eda | Monthly rating volume |
| `05_top_genres.html/png` | 01_eda | Genre frequency bar chart |
| `06_genre_correlation.html/png` | 02_features | Genre co-occurrence heatmap |
| `07_feature_variance.html/png` | 02_features | Explained variance of LSA components |
| `08_rmse_mae.html/png` | 04_evaluation | RMSE & MAE grouped bar chart |
| `09_f1_at_10.html/png` | 04_evaluation | F1@10 by model |
| `10_ranking_metrics_k.html/png` | 04_evaluation | P/R/F1 vs K for best model |
| `11_rating_dist_by_split.html/png` | 01_eda | Rating distribution across train/val/test |
| `12_long_tail_pareto.html/png` | 01_eda | Cumulative share of ratings vs share of movies |
| `13_genre_frequency.html/png` | 02_features | Movies per genre (matrix column sums) |
| `14_tagtext_wordcount.html/png` | 02_features | Tag-text length distribution |
| `15_feature_space_pca.html/png` | 02_features | PCA-2D of item feature vectors, by genre |
| `16_feature_space_tsne.html/png` | 02_features | t-SNE-2D of item feature vectors, by genre |
| `17_feature_space_umap.html/png` | 02_features | UMAP-2D of item feature vectors, by genre (needs `umap-learn`) |
