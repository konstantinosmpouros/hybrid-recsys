"""High-level evaluation helpers shared by the per-model notebooks.

Each model notebook trains its model, then calls `full_metrics(...)` to get the
RMSE/MAE + ranking metrics, `save_metric(...)` to record them in the shared
`all_metrics.json`, and `top_n(...)` for qualitative example recommendations.
"""
import json

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_METRICS, K_VALUES
from .metrics import evaluate_rating_prediction, evaluate_ranking_sampled


def full_metrics(
    predict_fn,
    test,
    test_sample,
    train_val,
    all_movie_ids,
    n_negatives: int = 100,
    k_values=None,
    random_state: int = 42,
):
    """Rating (RMSE/MAE over full test) + ranking (P/R/F1@K, sampled-negatives).

    Returns (metrics_dict, preds) — `preds` is the full-test prediction array, handy
    for plotting an error distribution.
    """
    k_values = list(k_values or K_VALUES)
    preds = np.array([predict_fn(r.userId, r.movieId) for r in test.itertuples()])
    rp = evaluate_rating_prediction(test["rating"].to_numpy(), preds)
    ranking = evaluate_ranking_sampled(
        test_sample, predict_fn, train_val,
        all_movie_ids=all_movie_ids, n_negatives=n_negatives,
        k_values=k_values, random_state=random_state,
    )
    metrics = {
        "rmse": round(rp["rmse"], 4),
        "mae":  round(rp["mae"], 4),
        **{f"k{k}": {m: round(v, 4) for m, v in kv.items()} for k, kv in ranking.items()},
    }
    return metrics, preds


def save_metric(label: str, metrics: dict, path=None) -> dict:
    """Insert/overwrite one model's entry in all_metrics.json, preserving the rest."""
    path = path or (ARTIFACTS_METRICS / "all_metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[label] = metrics
    path.write_text(json.dumps(data, indent=2))
    return data


def top_n(predict_fn, user_id, seen, candidate_ids, movies, n: int = 10) -> pd.DataFrame:
    """Top-n recommended movies for a user from a candidate pool, joined with titles."""
    scored = [(int(m), predict_fn(user_id, m)) for m in candidate_ids if m not in seen]
    scored = [(m, s) for m, s in scored if not np.isnan(s)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = pd.DataFrame(scored[:n], columns=["movieId", "pred"])
    top["pred"] = top["pred"].round(3)
    return top.merge(movies[["movieId", "clean_title", "genres"]], on="movieId", how="left")
