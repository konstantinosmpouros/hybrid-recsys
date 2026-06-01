# artifacts/

Auto-generated outputs from `notebooks/03_train_evaluate.ipynb`.
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
| `08_rmse_mae.html/png` | 03_train_evaluate | RMSE & MAE grouped bar chart |
| `09_f1_at_10.html/png` | 03_train_evaluate | F1@10 by model |
| `10_ranking_metrics_k.html/png` | 03_train_evaluate | P/R/F1 vs K for best model |
