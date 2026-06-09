"""LightGCN — graph collaborative filtering (extension model).

A compact, self-contained implementation (no RecBole/torch-geometric). PyTorch is
required to ``fit``, but ``predict`` uses the cached numpy embeddings, so loading and
serving a trained model needs only numpy. Trained with a BPR ranking loss over the
user-item interaction graph.

Note: on a CPU laptop, train on a user subsample (``max_users``) — this is a
reduced-scale demonstration of the method, not a full-25M SOTA run.
"""
import numpy as np
import joblib
from scipy.sparse import coo_matrix, bmat

from ..config import ARTIFACTS_MODELS, RANDOM_STATE


class LightGCNRecommender:
    def __init__(self, dim: int = 64, n_layers: int = 3, epochs: int = 200,
                 lr: float = 5e-3, batch_size: int = 1_000_000, reg: float = 1e-4,
                 max_users: int = 10_000, random_state: int = RANDOM_STATE):
        self.dim = dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.reg = reg
        self.max_users = max_users
        self.random_state = random_state
        self.user_map: dict = {}
        self.item_map: dict = {}
        self.user_emb = None   # cached numpy embeddings (set after fit)
        self.item_emb = None

    def fit(self, train_df):
        import torch

        rng = np.random.default_rng(self.random_state)
        df = train_df[["userId", "movieId"]]
        if self.max_users and df["userId"].nunique() > self.max_users:
            keep = rng.choice(df["userId"].unique(), self.max_users, replace=False)
            df = df[df["userId"].isin(keep)]

        users = df["userId"].unique()
        items = df["movieId"].unique()
        self.user_map = {int(u): i for i, u in enumerate(users)}
        self.item_map = {int(m): i for i, m in enumerate(items)}
        nu, ni = len(users), len(items)
        u_idx = df["userId"].map(self.user_map).to_numpy()
        i_idx = df["movieId"].map(self.item_map).to_numpy()

        # Symmetric-normalised bipartite adjacency  Â = D^-1/2 A D^-1/2.
        R = coo_matrix((np.ones(len(u_idx), dtype=np.float32), (u_idx, i_idx)), shape=(nu, ni))
        A = bmat([[None, R], [R.T, None]]).tocoo()
        deg = np.asarray(A.sum(1)).ravel()
        dinv = np.zeros_like(deg)
        dinv[deg > 0] = np.power(deg[deg > 0], -0.5)
        vals = (dinv[A.row] * A.data * dinv[A.col]).astype(np.float32)

        torch.manual_seed(self.random_state)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
        norm_adj = torch.sparse_coo_tensor(
            idx, torch.tensor(vals), (nu + ni, nu + ni)
        ).coalesce().to(device)

        emb = torch.nn.Parameter(torch.empty(nu + ni, self.dim, device=device))
        torch.nn.init.normal_(emb, std=0.1)
        opt = torch.optim.Adam([emb], lr=self.lr)

        def propagate():
            x, out = emb, [emb]
            for _ in range(self.n_layers):
                x = torch.sparse.mm(norm_adj, x)
                out.append(x)
            return torch.mean(torch.stack(out, 0), 0)

        pos_u = torch.tensor(u_idx, dtype=torch.long, device=device)
        pos_i = torch.tensor(i_idx, dtype=torch.long, device=device)
        n_pos = len(u_idx)
        print(f"LightGCN on {device}: {nu:,} users, {ni:,} items, {n_pos:,} interactions")

        # Full-batch BPR: propagate the WHOLE graph ONCE per epoch (the expensive
        # spmm), then accumulate the BPR gradient over chunks of triples (bounds
        # memory) and take a single optimizer step. This does `epochs` propagations
        # total instead of one per minibatch — the key optimisation.
        for ep in range(self.epochs):
            opt.zero_grad()
            E = propagate()
            eu, ei = E[:nu], E[nu:]
            perm = torch.randperm(n_pos, device=device)
            neg = torch.randint(0, ni, (n_pos,), device=device)
            total, n_chunks = 0.0, 0
            for s in range(0, n_pos, self.batch_size):
                b = perm[s:s + self.batch_size]
                xu, xpos, xneg = eu[pos_u[b]], ei[pos_i[b]], ei[neg[b]]
                pos_s = (xu * xpos).sum(1)
                neg_s = (xu * xneg).sum(1)
                loss = -torch.nn.functional.logsigmoid(pos_s - neg_s).mean()
                loss = loss + self.reg * (xu.pow(2).sum() + xpos.pow(2).sum() + xneg.pow(2).sum()) / len(b)
                loss.backward(retain_graph=(s + self.batch_size < n_pos))
                total += float(loss.item())
                n_chunks += 1
            opt.step()
            if ep == 0 or (ep + 1) % 10 == 0 or ep == self.epochs - 1:
                print(f"  epoch {ep + 1:>3}/{self.epochs}  loss={total / n_chunks:.4f}")

        with torch.no_grad():
            E = propagate()
            self.user_emb = E[:nu].cpu().numpy()
            self.item_emb = E[nu:].cpu().numpy()
        return self

    def predict(self, user_id, movie_id) -> float:
        u = self.user_map.get(int(user_id))
        i = self.item_map.get(int(movie_id))
        if u is None or i is None:
            return np.nan
        return float(self.user_emb[u] @ self.item_emb[i])

    def save(self, path=None) -> None:
        path = path or ARTIFACTS_MODELS / "lightgcn_model.joblib"
        ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None) -> "LightGCNRecommender":
        return joblib.load(path or ARTIFACTS_MODELS / "lightgcn_model.joblib")
