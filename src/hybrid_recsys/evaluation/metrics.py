import numpy as np
import pandas as pd
from ..config import RELEVANCE_THRESHOLD, K_VALUES


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    mask = ~np.isnan(pred)
    return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2)))


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    mask = ~np.isnan(pred)
    return float(np.mean(np.abs(true[mask] - pred[mask])))


def precision_at_k(recommended: list, relevant: set) -> float:
    if not recommended:
        return 0.0
    return len(set(recommended) & relevant) / len(recommended)


def recall_at_k(recommended: list, relevant: set) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended) & relevant) / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_ranking(
    test_df: pd.DataFrame,
    predict_fn,
    train_val_df: pd.DataFrame,
    k_values: list[int] | None = None,
    threshold: float | None = None,
    random_state: int = 42,
) -> dict[int, dict[str, float]]:
    """Macro-averaged Precision@K, Recall@K, F1@K over all users in test_df.

    predict_fn: callable(user_id: int, movie_id: int) -> float
    Candidates are all items in test_df not seen by the user in train_val_df.
    """
    k_values = k_values or K_VALUES
    threshold = threshold or RELEVANCE_THRESHOLD

    seen_items = train_val_df.groupby("userId")["movieId"].apply(set).to_dict()
    relevant_items = (
        test_df[test_df["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    all_movies = test_df["movieId"].unique()
    base_rng = np.random.default_rng(random_state)

    accum = {k: {"precision": [], "recall": []} for k in k_values}

    for user_id, relevant in relevant_items.items():
        user_seen = seen_items.get(user_id, set())
        candidates = np.array([m for m in all_movies if m not in user_seen])
        if len(candidates) == 0:
            continue
        # Random tie-break so constant-output models don't ride the original
        # candidate ordering through the stable sort (see evaluate_ranking_sampled).
        user_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1) ^ int(user_id))
        user_rng.shuffle(candidates)

        scored = [(m, predict_fn(user_id, m)) for m in candidates]
        scored = [(m, s) for m, s in scored if not np.isnan(s)]
        scored.sort(key=lambda x: x[1], reverse=True)

        for k in k_values:
            top_k = [m for m, _ in scored[:k]]
            accum[k]["precision"].append(precision_at_k(top_k, relevant))
            accum[k]["recall"].append(recall_at_k(top_k, relevant))

    # F1 from macro-averaged P and R (kept consistent — F1 lies between P and R).
    result = {}
    for k, v in accum.items():
        p = float(np.mean(v["precision"])) if v["precision"] else 0.0
        r = float(np.mean(v["recall"])) if v["recall"] else 0.0
        result[k] = {"precision": p, "recall": r, "f1": f1_at_k(p, r)}
    return result


def evaluate_rating_prediction(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(true, pred), "mae": mae(true, pred)}


# ── NDCG & AUC (ranking-quality metrics, less brittle than F1@K) ──────────────

def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """Binary-relevance NDCG@K for a ranked candidate list."""
    rel = np.array([1.0 if m in relevant else 0.0 for m in ranked[:k]])
    if rel.sum() == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel * disc).sum())
    ideal = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _auc(pos_scores, neg_scores) -> float:
    """Rank-based AUC = P(relevant scored above non-relevant), via Mann-Whitney."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    rsum = ranks[:len(pos)].sum()
    return float((rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate_ranking_extended(
    test_df, predict_fn, train_val_df, all_movie_ids,
    n_negatives: int = 100, k_values=None, threshold=None, random_state: int = 42,
):
    """Sampled-negatives ranking with Precision/Recall/F1/NDCG@K + overall AUC.

    Same protocol as `evaluate_ranking_sampled` (shuffled candidates, per-user seed);
    returns ({k: {precision, recall, f1, ndcg}}, macro_auc)."""
    k_values = k_values or K_VALUES
    threshold = threshold or RELEVANCE_THRESHOLD
    seen_items = train_val_df.groupby("userId")["movieId"].apply(set).to_dict()
    relevant_items = (
        test_df[test_df["rating"] >= threshold]
        .groupby("userId")["movieId"].apply(set).to_dict()
    )
    all_movie_ids = np.asarray(all_movie_ids)
    base_rng = np.random.default_rng(random_state)
    accum = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    aucs = []

    for user_id, relevant in relevant_items.items():
        if not relevant:
            continue
        user_seen = seen_items.get(user_id, set())
        pool = all_movie_ids[~np.isin(all_movie_ids, list(user_seen | relevant))]
        if len(pool) == 0:
            continue
        urng = np.random.default_rng(base_rng.integers(0, 2**32 - 1) ^ int(user_id))
        negatives = urng.choice(pool, size=min(n_negatives, len(pool)), replace=False)
        candidates = np.array(list(relevant) + list(negatives))
        urng.shuffle(candidates)
        scored = [(m, predict_fn(user_id, m)) for m in candidates]
        scored = [(m, s) for m, s in scored if not np.isnan(s)]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [m for m, _ in scored]
        for k in k_values:
            top = ranked[:k]
            accum[k]["precision"].append(precision_at_k(top, relevant))
            accum[k]["recall"].append(recall_at_k(top, relevant))
            accum[k]["ndcg"].append(ndcg_at_k(ranked, relevant, k))
        a = _auc([s for m, s in scored if m in relevant],
                 [s for m, s in scored if m not in relevant])
        if not np.isnan(a):
            aucs.append(a)

    result = {}
    for k, v in accum.items():
        p = float(np.mean(v["precision"])) if v["precision"] else 0.0
        r = float(np.mean(v["recall"])) if v["recall"] else 0.0
        result[k] = {"precision": p, "recall": r, "f1": f1_at_k(p, r),
                     "ndcg": float(np.mean(v["ndcg"])) if v["ndcg"] else 0.0}
    return result, (float(np.mean(aucs)) if aucs else float("nan"))


# ── Beyond-accuracy metrics (recommendation-list quality) ─────────────────────

def catalogue_coverage(rec_lists: list[list], n_items: int) -> float:
    """Fraction of the catalogue that appears in at least one user's top-K list."""
    recommended = set()
    for lst in rec_lists:
        recommended.update(lst)
    return len(recommended) / n_items if n_items else 0.0


def intra_list_diversity(rec_lists: list[list], feature_lookup) -> float:
    """Mean pairwise (1 − cosine similarity) of items within each list, averaged
    over users. `feature_lookup(movie_id) -> L2-normalised vector or None`."""
    per_user = []
    for lst in rec_lists:
        vecs = [feature_lookup(m) for m in lst]
        vecs = [v for v in vecs if v is not None]
        if len(vecs) < 2:
            continue
        V = np.vstack(vecs)
        sims = V @ V.T
        n = len(vecs)
        off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
        per_user.append(1.0 - off)
    return float(np.mean(per_user)) if per_user else 0.0


def novelty(rec_lists: list[list], item_popularity: dict, n_interactions: int) -> float:
    """Mean self-information −log2(p(item)) of recommended items (higher = more
    novel / less popular). p(item) = interactions(item) / total interactions."""
    per_item = []
    for lst in rec_lists:
        for m in lst:
            p = item_popularity.get(m, 0) / n_interactions
            if p > 0:
                per_item.append(-np.log2(p))
    return float(np.mean(per_item)) if per_item else 0.0


def evaluate_ranking_sampled(
    test_df: pd.DataFrame,
    predict_fn,
    train_val_df: pd.DataFrame,
    all_movie_ids: np.ndarray,
    n_negatives: int = 100,
    k_values: list[int] | None = None,
    threshold: float | None = None,
    random_state: int = 42,
) -> dict[int, dict[str, float]]:
    """Sampled-negatives ranking evaluation.

    For each test user, ranks their relevant test items against `n_negatives`
    randomly-sampled non-relevant items the user has not seen in train_val.

    Standard protocol used by NCF, BPR, and most modern recsys papers — produces
    metrics that meaningfully distinguish models even when the underlying CF
    model has a restricted item/user vocabulary.

    Reproducibility: a per-user-seeded RNG so the same negatives are sampled
    for every model in the experiment.
    """
    k_values = k_values or K_VALUES
    threshold = threshold or RELEVANCE_THRESHOLD

    seen_items = train_val_df.groupby("userId")["movieId"].apply(set).to_dict()
    relevant_items = (
        test_df[test_df["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    all_movie_ids = np.asarray(all_movie_ids)
    base_rng = np.random.default_rng(random_state)

    accum = {k: {"precision": [], "recall": []} for k in k_values}

    for user_id, relevant in relevant_items.items():
        if not relevant:
            continue
        user_seen = seen_items.get(user_id, set())
        excluded = user_seen | relevant
        pool = all_movie_ids[~np.isin(all_movie_ids, list(excluded))]
        if len(pool) == 0:
            continue
        size = min(n_negatives, len(pool))
        user_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1) ^ int(user_id))
        negatives = user_rng.choice(pool, size=size, replace=False)

        # Shuffle the candidate pool so that tied predictions break *randomly*.
        # Otherwise a constant-output model (e.g. Global Mean) keeps the original
        # relevant-first ordering through the stable sort and scores artificially
        # high precision/recall.
        candidates = np.array(list(relevant) + list(negatives))
        user_rng.shuffle(candidates)

        scored = [(m, predict_fn(user_id, m)) for m in candidates]
        scored = [(m, s) for m, s in scored if not np.isnan(s)]
        scored.sort(key=lambda x: x[1], reverse=True)

        for k in k_values:
            top_k = [m for m, _ in scored[:k]]
            accum[k]["precision"].append(precision_at_k(top_k, relevant))
            accum[k]["recall"].append(recall_at_k(top_k, relevant))

    # F1 is computed from the macro-averaged precision and recall so the three
    # numbers stay mutually consistent (F1 always lies between P and R). This
    # differs from averaging per-user F1, which can fall below both P and R.
    result = {}
    for k, v in accum.items():
        p = float(np.mean(v["precision"])) if v["precision"] else 0.0
        r = float(np.mean(v["recall"])) if v["recall"] else 0.0
        result[k] = {"precision": p, "recall": r, "f1": f1_at_k(p, r)}
    return result
