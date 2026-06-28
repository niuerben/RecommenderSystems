# -*- coding: utf-8 -*-
"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

Reference:
    Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, Meng Wang.
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    SIGIR 2020. https://arxiv.org/abs/2002.02126

This implementation follows the RecSys-master model convention (same interface as
ItemCF / UserCF / XSimGCL):
    - MovieLens 1M data (:: separator)
    - Rating threshold >= 3 to define positive interactions
    - 8:1:1 train / valid / test split
    - Early stopping on validation MRR
    - Top-N recommendation evaluation (Recall / MRR / NDCG / Hit / Precision / MAP)

Architecture highlights (differences from full NGCF):
    - No feature transformation matrices (W1, W2)
    - No non-linear activation (no LeakyReLU / tanh)
    - No self-connections in the adjacency matrix
    - Mean pooling over layer embeddings (instead of concatenation)
"""

import argparse
import copy
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ==============================================================================
#  Core LightGCN Model (PyTorch nn.Module)
# ==============================================================================

class LightGCNModel(nn.Module):
    """
    LightGCN propagation module.

    Builds the symmetrically normalised adjacency matrix from the bipartite
    user-item graph (no self-loops) and propagates embeddings through K layers.
    The final representation is the *mean* of the layer-wise embeddings.
    """

    def __init__(self, n_users, n_items, edge_index, embedding_dim=64, n_layers=3):
        """
        Parameters
        ----------
        n_users : int
            Number of users.
        n_items : int
            Number of items.
        edge_index : torch.LongTensor [2, 2*|E|]
            COO-format edges of the bipartite graph.  Row 0 = source, row 1 = target.
            Both directions (user→item and item→user) must be present.
        embedding_dim : int
            Dimension of the embedding vectors (default 64).
        n_layers : int
            Number of LightGCN propagation layers (default 3).
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        # Shared embedding table for users + items
        self.embedding = nn.Embedding(self.n_nodes, embedding_dim)
        # Initialise following the original LightGCN: N(0, 0.1²)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # Build symmetrically normalised adjacency: D^{-1/2} A D^{-1/2}
        row, col = edge_index
        deg = torch.bincount(row, minlength=self.n_nodes).float().clamp(min=1)
        edge_weight = torch.rsqrt(deg[row]) * torch.rsqrt(deg[col])
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            torch.Size([self.n_nodes, self.n_nodes]),
            device=edge_index.device,
        ).coalesce()
        self.register_buffer("adj", adj)

    # ------------------------------------------------------------------
    def computer(self):
        """
        LightGCN propagation:  mean( E^(0), E^(1), ..., E^(K) ).

        Returns
        -------
        users : Tensor [n_users, dim]
        items : Tensor [n_items, dim]
        """
        all_emb = self.embedding.weight          # E^(0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)   # E^(k+1) = D^{-½}AD^{-½} E^(k)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)          # [N, K+1, d]
        light_out = torch.mean(embs, dim=1)      # mean pooling  [N, d]
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    # ------------------------------------------------------------------
    def forward(self, users, items):
        """Predict scores for (user, item) pairs.  Return shape [batch]."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = torch.sum(users_emb * items_emb, dim=1)
        return scores

    # ------------------------------------------------------------------
    def getUsersRating(self, users):
        """Return [len(users), n_items] score matrix (for evaluation)."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        rating = torch.matmul(users_emb, all_items.t())
        return rating


# ==============================================================================
#  LightGCN wrapper (RecSys-master interface)
# ==============================================================================

class LightGCN(object):
    """
    LightGCN recommender following the RecSys-master model convention.

    Usage
    -----
        lgn = LightGCN(topn=10, embedding_dim=64, n_layers=3)
        lgn.generate_dataset("data/ml-1m/ratings.dat", usersfile="data/ml-1m/users.dat")
        lgn.calc_movie_sim()          # alias for train()
        lgn.evaluate()                # test-set metrics
        lgn.generate_recommendation("outputs/lightgcn_rec.csv")
    """

    def __init__(
        self,
        topn=10,
        recommendation_topn=100,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        embedding_dim=64,
        n_layers=3,
        epochs=500,
        batch_size=2048,
        learning_rate=0.001,
        reg_weight=1e-4,
        seed=2020,
        valid_interval=1,
        early_stop_patience=10,
        min_delta=1e-6,
        save_epoch_recommendations=False,
        epoch_recommendation_dir="./outputs/lightgcn_epoch_recommendations",
    ):
        self.topn = topn
        self.recommendation_topn = recommendation_topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.seed = seed
        self.valid_interval = valid_interval
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.save_epoch_recommendations = save_epoch_recommendations
        self.epoch_recommendation_dir = epoch_recommendation_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Data containers (populated by generate_dataset)
        self.user_ids = []
        self.item_ids = []
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.train_items_by_user = defaultdict(set)
        self.valid_items_by_user = defaultdict(set)
        self.test_items_by_user = defaultdict(set)
        self.all_items = None
        self.train_pairs = None
        self.valid_pairs = None
        self.test_pairs = None
        self.edge_index = None

        # Model
        self.model: LightGCNModel = None
        self.best_epoch = 0
        self.best_valid_mrr = -1.0

    # ==================================================================
    #  Data loading  (same convention as XSimGCL / ItemCF)
    # ==================================================================

    def generate_dataset(self, ratingsfile, usersfile=None):
        """
        Load MovieLens-1M ratings, filter rating < threshold, split 8:1:1.

        Parameters
        ----------
        ratingsfile : str   Path to ratings.dat (user::item::rating::ts).
        usersfile   : str   Optional path to users.dat for fixed user list.
        """
        print("使用设备:%s" % self.device)
        print("加载 LightGCN 数据...")

        # --- read interactions -------------------------------------------------
        interactions = []
        with open(ratingsfile, "r", encoding="latin-1") as f:
            for line in f:
                user, item, rating, _ = line.rstrip("\n").split("::")
                if int(rating) < self.rating_threshold:
                    continue
                interactions.append((int(user), int(item)))

        self.rng.shuffle(interactions)

        # --- user list ---------------------------------------------------------
        if usersfile and os.path.exists(usersfile):
            self.user_ids = self._load_user_ids(usersfile)
        else:
            self.user_ids = sorted({user for user, _ in interactions})

        self.item_ids = sorted({item for _, item in interactions})
        self.user2idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item2idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.all_items = np.arange(len(self.item_ids), dtype=np.int64)

        # --- group by user ----------------------------------------------------
        interactions_by_user = defaultdict(list)
        for user, item in interactions:
            interactions_by_user[user].append(item)

        # --- split 8:1:1 -------------------------------------------------------
        ratios = [self.train_ratio, self.valid_ratio,
                  1 - self.train_ratio - self.valid_ratio]
        train_pairs, valid_pairs, test_pairs = [], [], []

        for user in self.user_ids:
            items = interactions_by_user.get(user, [])
            if not items:
                continue
            train_items, valid_items, test_items = self._split_user_items(items, ratios)
            user_idx = self.user2idx[user]
            for item in train_items:
                item_idx = self.item2idx[item]
                self.train_items_by_user[user_idx].add(item_idx)
                train_pairs.append((user_idx, item_idx))
            for item in valid_items:
                item_idx = self.item2idx[item]
                self.valid_items_by_user[user_idx].add(item_idx)
                valid_pairs.append((user_idx, item_idx))
            for item in test_items:
                item_idx = self.item2idx[item]
                self.test_items_by_user[user_idx].add(item_idx)
                test_pairs.append((user_idx, item_idx))

        self.train_pairs = np.asarray(train_pairs, dtype=np.int64)
        self.valid_pairs = np.asarray(valid_pairs, dtype=np.int64)
        self.test_pairs = np.asarray(test_pairs, dtype=np.int64)
        self.edge_index = self._build_edge_index(train_pairs)

        print(
            "用户数:%d，电影数:%d，交互数:%d，训练:%d，Top%d 推荐列: %d"
            % (
                len(self.user_ids),
                len(self.item_ids),
                len(interactions),
                len(self.train_pairs),
                self.recommendation_topn,
                len(self.user_ids) * self.recommendation_topn,
            )
        )

    def gernate_dataset(self, ratingsfile, usersfile=None):
        """Alias for generate_dataset (legacy spelling used by some callers)."""
        self.generate_dataset(ratingsfile, usersfile=usersfile)

    # ==================================================================
    #  Internal helpers
    # ==================================================================

    @staticmethod
    def _load_user_ids(usersfile):
        user_ids = []
        with open(usersfile, "r", encoding="latin-1") as f:
            for line in f:
                user_id = line.rstrip("\n").split("::", 1)[0]
                user_ids.append(int(user_id))
        return sorted(user_ids)

    @staticmethod
    def _split_user_items(items, ratios):
        """Split a user's items into train / valid / test according to ratios."""
        tot = len(items)
        norm_ratios = [r / sum(ratios) for r in ratios]
        cnt = [int(norm_ratios[i] * tot) for i in range(len(norm_ratios))]
        cnt[0] = tot - sum(cnt[1:])
        for i in range(1, len(norm_ratios)):
            if cnt[0] <= 1:
                break
            if 0 < norm_ratios[-i] * tot < 1:
                cnt[-i] += 1
                cnt[0] -= 1
        train_end = cnt[0]
        valid_end = cnt[0] + cnt[1]
        return items[:train_end], items[train_end:valid_end], items[valid_end:]

    def _build_edge_index(self, train_pairs):
        """Build COO edge index for the bipartite user-item graph (both directions)."""
        rows, cols = [], []
        item_offset = len(self.user_ids)
        for user_idx, item_idx in train_pairs:
            item_node = item_offset + item_idx
            rows.extend([user_idx, item_node])   # user→item, item→user
            cols.extend([item_node, user_idx])
        return torch.tensor([rows, cols], dtype=torch.long, device=self.device)

    def _sample_negative_items(self, users):
        """Sample one negative item per user, avoiding items seen in training."""
        neg_items = np.empty(len(users), dtype=np.int64)
        for idx, user_idx in enumerate(users):
            while True:
                item_idx = int(self.rng.integers(0, len(self.item_ids)))
                if item_idx not in self.train_items_by_user[int(user_idx)]:
                    neg_items[idx] = item_idx
                    break
        return neg_items

    # ==================================================================
    #  Training
    # ==================================================================

    def calc_movie_sim(self):
        """Alias for train() — follows the ItemCF / UserCF naming convention."""
        self.train()

    def train(self):
        """Train LightGCN with BPR loss and early stopping on validation MRR."""
        print("加载模型 LightGCN...")
        self.model = LightGCNModel(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            edge_index=self.edge_index,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state_dict = None
        stale_validations = 0

        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(len(self.train_pairs))
            total_loss = 0.0
            total_count = 0
            self.model.train()

            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                users = self.train_pairs[batch_idx, 0]
                pos_items = self.train_pairs[batch_idx, 1]
                neg_items = self._sample_negative_items(users)

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=self.device)

                # ---- LightGCN propagation -----------------------------------------
                all_users, all_items = self.model.computer()
                users_emb = all_users[users_t]
                pos_emb = all_items[pos_t]
                neg_emb = all_items[neg_t]

                # ---- BPR loss  (softplus form ≡ −log σ(pos − neg)) -----------------
                pos_scores = torch.sum(users_emb * pos_emb, dim=1)
                neg_scores = torch.sum(users_emb * neg_emb, dim=1)
                bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

                # ---- L2 regularisation on *initial* embeddings only -----------------
                ego_emb = self.model.embedding.weight
                ego_users = ego_emb[users_t]
                ego_pos = ego_emb[len(self.user_ids) + pos_t]
                ego_neg = ego_emb[len(self.user_ids) + neg_t]
                reg_loss = (ego_users.norm(2).pow(2) +
                            ego_pos.norm(2).pow(2) +
                            ego_neg.norm(2).pow(2)) / float(len(users_t))

                loss = bpr_loss + self.reg_weight * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(users_t)
                total_count += len(users_t)

            # ---- validation --------------------------------------------------
            message = "Epoch %d/%d - loss: %.4f" % (
                epoch, self.epochs, total_loss / max(total_count, 1))

            if self.valid_interval > 0 and epoch % self.valid_interval == 0:
                valid_metrics = self.evaluate_split(
                    self.valid_items_by_user,
                    self.train_items_by_user,
                    label="Valid",
                    verbose=False,
                )
                valid_mrr = valid_metrics["mrr"]
                message += " - valid_mrr@%d: %.4f" % (self.topn, valid_mrr)

                if valid_mrr > self.best_valid_mrr + self.min_delta:
                    self.best_valid_mrr = valid_mrr
                    self.best_epoch = epoch
                    stale_validations = 0
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    message += " *best*"
                else:
                    stale_validations += 1
                    message += " - stale:%d/%d" % (stale_validations, self.early_stop_patience)

            # ---- epoch recommendations (optional) ----------------------------
            if self.save_epoch_recommendations:
                filename = "lightgcn_recommendation_%03d.csv" % epoch
                filepath = os.path.join(self.epoch_recommendation_dir, filename)
                self.generate_recommendation(filepath=filepath, mask_valid=True, progress=False)
                message += " - saved %s" % filepath

            print(message)

            # ---- early stopping ----------------------------------------------
            if self.early_stop_patience > 0 and stale_validations >= self.early_stop_patience:
                print(
                    "早停触发: valid MRR@%d 连续 %d 次没有提升，停止于 epoch %d。"
                    % (self.topn, self.early_stop_patience, epoch)
                )
                break

        # Restore best checkpoint
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(
                "加载验证集最佳 checkpoint: epoch %d, Valid MRR@%d=%.4f"
                % (self.best_epoch, self.topn, self.best_valid_mrr)
            )

    # ==================================================================
    #  Evaluation
    # ==================================================================

    def evaluate(self):
        """Evaluate on test set (masking train + valid items)."""
        mask_items_by_user = defaultdict(set)
        for user_idx, items in self.train_items_by_user.items():
            mask_items_by_user[user_idx].update(items)
        for user_idx, items in self.valid_items_by_user.items():
            mask_items_by_user[user_idx].update(items)
        return self.evaluate_split(
            self.test_items_by_user,
            mask_items_by_user,
            label="Test",
            verbose=True,
        )

    def evaluate_split(self, eval_items_by_user, mask_items_by_user,
                       label="Test", verbose=True):
        """
        Compute Recall, MRR, NDCG, Hit, Precision, MAP @ topn.

        Parameters
        ----------
        eval_items_by_user : dict  {user_idx: set(item_idx)}  ground-truth items.
        mask_items_by_user : dict  {user_idx: set(item_idx)}  items to exclude.
        label : str               Label for logging.
        verbose : bool            Whether to print progress & final metrics.

        Returns
        -------
        metrics : dict  with keys recall, mrr, ndcg, hit, precision, map, users, hits.
        """
        N = self.topn
        hit = 0
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        map_sum = 0.0
        mrr_sum = 0.0
        user_hit_count = 0
        eval_user_count = 0

        self.model.eval()
        with torch.no_grad():
            all_users, all_items = self.model.computer()
            user_emb = all_users.detach()
            item_emb = all_items.detach()

        for user_idx in range(len(self.user_ids)):
            if verbose and user_idx % 500 == 0:
                print("%s topn evaluate for %d users" % (label.lower(), user_idx),
                      file=sys.stderr)

            eval_items = eval_items_by_user.get(user_idx, set())
            if not eval_items:
                continue

            scores = torch.matmul(item_emb, user_emb[user_idx]).detach().cpu().numpy()
            for item_idx in mask_items_by_user.get(user_idx, set()):
                scores[item_idx] = -np.inf

            top_k = min(N, len(scores))
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            dcg = 0.0
            ap = 0.0
            user_hit = 0
            for rank, item_idx in enumerate(top_idx, start=1):
                if int(item_idx) in eval_items:
                    hit += 1
                    user_hit += 1
                    dcg += 1 / math.log2(rank + 1)
                    ap += user_hit / rank
                    if user_hit == 1:
                        mrr_sum += 1 / rank

            ideal_hits = min(len(eval_items), N)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            precision_sum += user_hit / N
            recall_sum += user_hit / len(eval_items)
            ndcg_sum += dcg / idcg if idcg else 0
            map_sum += ap / ideal_hits if ideal_hits else 0
            if user_hit > 0:
                user_hit_count += 1
            eval_user_count += 1

        precision = precision_sum / eval_user_count if eval_user_count else 0
        recall = recall_sum / eval_user_count if eval_user_count else 0
        ndcg = ndcg_sum / eval_user_count if eval_user_count else 0
        mean_ap = map_sum / eval_user_count if eval_user_count else 0
        mrr = mrr_sum / eval_user_count if eval_user_count else 0
        hit_rate = user_hit_count / eval_user_count if eval_user_count else 0

        metrics = {
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "hit": hit_rate,
            "precision": precision,
            "map": mean_ap,
            "users": eval_user_count,
            "hits": hit,
        }

        if verbose:
            print(
                "测试集 %s RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    "
                "HIT@%d : %.4f    PRECISION@%d : %.4f"
                % (label, N, recall, N, mrr, N, ndcg, N, hit_rate, N, precision)
            )

        return metrics

    # ==================================================================
    #  Recommendation generation
    # ==================================================================

    def generate_recommendation(self, filepath="./outputs/lightgcn_recommendation.csv",
                                topn=None, mask_valid=False, progress=True):
        """
        Write per-user Top-N recommendations to a CSV file.

        CSV format:  user_id, rec1, rec2, ..., recN
        """
        topn = topn or self.recommendation_topn
        print("generating LightGCN recommendation result: %s" % filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            all_users, all_items = self.model.computer()
            user_emb = all_users.detach()
            item_emb = all_items.detach()

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id"] + ["rec%d" % idx for idx in range(1, topn + 1)])
            for user_idx, user_id in enumerate(self.user_ids):
                if progress and user_idx % 500 == 0:
                    print("generate LightGCN recommendation for %d users" % user_idx,
                          file=sys.stderr)

                scores = torch.matmul(item_emb, user_emb[user_idx]).detach().cpu().numpy()
                for item_idx in self.train_items_by_user.get(user_idx, set()):
                    scores[item_idx] = -np.inf
                if mask_valid:
                    for item_idx in self.valid_items_by_user.get(user_idx, set()):
                        scores[item_idx] = -np.inf

                top_k = min(topn, len(scores))
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                items = [str(self.idx2item[int(idx)]) for idx in top_idx]
                if len(items) < topn:
                    items.extend([""] * (topn - len(items)))
                writer.writerow([user_id] + items)

        print("LightGCN recommendation written to %s" % filepath)

    def gernate_recommendation(self):
        """Alias for generate_recommendation (legacy spelling)."""
        self.generate_recommendation()


# ==============================================================================
#  CLI
# ==============================================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="LightGCN: Simplifying and Powering Graph Convolution Network"
    )
    parser.add_argument("--ratings-file",
                        default="./data/ml-1m/ratings.dat",
                        help="Path to ratings.dat (ML-1M format)")
    parser.add_argument("--users-file",
                        default="./data/ml-1m/users.dat",
                        help="Path to users.dat (optional)")
    parser.add_argument("--topn", type=int, default=10,
                        help="N for Recall@N / MRR@N / NDCG@N evaluation")
    parser.add_argument("--recommendation-topn", type=int, default=100,
                        help="Number of items in output recommendation CSV")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of LightGCN propagation layers")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg-weight", type=float, default=1e-4,
                        help="L2 regularisation coefficient")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--valid-interval", type=int, default=1,
                        help="Run validation every N epochs")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Stop after N validations without improvement")
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--save-epoch-recommendations", action="store_true",
                        help="Write per-epoch recommendation CSV")
    parser.add_argument("--epoch-recommendation-dir",
                        default="./outputs/lightgcn_epoch_recommendations")
    parser.add_argument("--skip-recommendation", action="store_true",
                        help="Skip writing the final recommendation CSV")
    return parser


def main():
    args = build_arg_parser().parse_args()
    lightgcn = LightGCN(
        topn=args.topn,
        recommendation_topn=args.recommendation_topn,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        seed=args.seed,
        valid_interval=args.valid_interval,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        save_epoch_recommendations=args.save_epoch_recommendations,
        epoch_recommendation_dir=args.epoch_recommendation_dir,
    )
    lightgcn.generate_dataset(args.ratings_file, usersfile=args.users_file)
    lightgcn.calc_movie_sim()
    lightgcn.evaluate()
    if not args.skip_recommendation:
        lightgcn.generate_recommendation(topn=args.recommendation_topn, mask_valid=True)


if __name__ == "__main__":
    main()
