# -*- coding: utf-8 -*-
"""
Lightweight XSimGCL-style recommender for MovieLens 1M.

This module is intentionally self-contained for the course project. It follows
the same data convention as the RecBole-GNN ML-1M setting used in the report:
filter rating < 3, split interactions by 8:1:1, and evaluate Top-10 ranking.
"""

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class XSimGCLModel(nn.Module):
    def __init__(self, n_users, n_items, edge_index, embedding_dim=64, n_layers=2, eps=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.n_layers = n_layers
        self.eps = eps
        self.embedding = nn.Embedding(self.n_nodes, embedding_dim)
        self.register_buffer("edge_index", edge_index)
        nn.init.xavier_uniform_(self.embedding.weight)

    def _propagate(self, perturbed=False):
        row, col = self.edge_index
        deg = torch.bincount(row, minlength=self.n_nodes).float().clamp(min=1)
        norm = torch.rsqrt(deg[row]) * torch.rsqrt(deg[col])

        all_emb = self.embedding.weight
        embeddings = [all_emb]
        for _ in range(self.n_layers):
            next_emb = torch.zeros_like(all_emb)
            next_emb.index_add_(0, row, all_emb[col] * norm.unsqueeze(1))
            all_emb = next_emb
            if perturbed:
                noise = F.normalize(torch.rand_like(all_emb), dim=1)
                all_emb = all_emb + torch.sign(all_emb) * noise * self.eps
            embeddings.append(all_emb)
        output = torch.stack(embeddings, dim=0).mean(dim=0)
        return output[: self.n_users], output[self.n_users :]

    def forward(self, perturbed=False):
        return self._propagate(perturbed=perturbed)


class XSimGCL(object):
    def __init__(
        self,
        topn=10,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        embedding_dim=64,
        n_layers=2,
        epochs=5,
        batch_size=4096,
        learning_rate=1e-3,
        reg_weight=1e-4,
        ssl_weight=0.1,
        eps=0.1,
        seed=0,
    ):
        self.topn = topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.ssl_weight = ssl_weight
        self.eps = eps
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.user_ids = []
        self.item_ids = []
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.train_items_by_user = defaultdict(set)
        self.test_items_by_user = defaultdict(set)
        self.all_items = None
        self.train_pairs = None
        self.model = None

    def generate_dataset(self, ratingsfile):
        print("使用设备:%s" % self.device)
        print("加载XSimGCL数据...")
        interactions = []
        with open(ratingsfile, "r", encoding="latin-1") as f:
            for line in f:
                user, item, rating, _ = line.rstrip("\n").split("::")
                if int(rating) < self.rating_threshold:
                    continue
                interactions.append((int(user), int(item)))

        self.rng.shuffle(interactions)
        self.user_ids = sorted({user for user, _ in interactions})
        self.item_ids = sorted({item for _, item in interactions})
        self.user2idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item2idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.all_items = np.arange(len(self.item_ids), dtype=np.int64)

        train_split = int(len(interactions) * self.train_ratio)
        valid_split = int(len(interactions) * (self.train_ratio + self.valid_ratio))
        train_interactions = interactions[:train_split]
        test_interactions = interactions[valid_split:]

        train_pairs = []
        for user, item in train_interactions:
            user_idx = self.user2idx[user]
            item_idx = self.item2idx[item]
            self.train_items_by_user[user_idx].add(item_idx)
            train_pairs.append((user_idx, item_idx))

        for user, item in test_interactions:
            user_idx = self.user2idx[user]
            item_idx = self.item2idx[item]
            self.test_items_by_user[user_idx].add(item_idx)

        self.train_pairs = np.asarray(train_pairs, dtype=np.int64)
        self.edge_index = self._build_edge_index(train_pairs)
        print(
            "用户数:%d，电影数:%d，Top%d评估推荐位:%d"
            % (len(self.user_ids), len(self.item_ids), self.topn, len(self.user_ids) * self.topn)
        )

    def gernate_dataset(self, ratingsfile):
        self.generate_dataset(ratingsfile)

    def _build_edge_index(self, train_pairs):
        rows = []
        cols = []
        item_offset = len(self.user_ids)
        for user_idx, item_idx in train_pairs:
            item_node = item_offset + item_idx
            rows.extend([user_idx, item_node])
            cols.extend([item_node, user_idx])
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index.to(self.device)

    def calc_movie_sim(self):
        self.train()

    def train(self):
        print("加载模型 XSimGCL...")
        self.model = XSimGCLModel(
            n_users=len(self.user_ids),
            n_items=len(self.item_ids),
            edge_index=self.edge_index,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            eps=self.eps,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(1, self.epochs + 1):
            order = self.rng.permutation(len(self.train_pairs))
            total_loss = 0.0
            total_count = 0
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                users = self.train_pairs[batch_idx, 0]
                pos_items = self.train_pairs[batch_idx, 1]
                neg_items = self._sample_negative_items(users)

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=self.device)

                user_emb, item_emb = self.model(perturbed=False)
                aug_user_1, aug_item_1 = self.model(perturbed=True)
                aug_user_2, aug_item_2 = self.model(perturbed=True)

                u = user_emb[users_t]
                pos = item_emb[pos_t]
                neg = item_emb[neg_t]
                pos_scores = torch.sum(u * pos, dim=1)
                neg_scores = torch.sum(u * neg, dim=1)
                bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                reg_loss = (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / len(users_t)
                ssl_loss = self._ssl_loss(aug_user_1[users_t], aug_user_2[users_t])
                ssl_loss = ssl_loss + self._ssl_loss(aug_item_1[pos_t], aug_item_2[pos_t])
                loss = bpr_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(users_t)
                total_count += len(users_t)
            print("Epoch %d/%d - loss: %.4f" % (epoch, self.epochs, total_loss / total_count))

    def _sample_negative_items(self, users):
        neg_items = np.empty(len(users), dtype=np.int64)
        for idx, user_idx in enumerate(users):
            while True:
                item_idx = int(self.rng.integers(0, len(self.item_ids)))
                if item_idx not in self.train_items_by_user[int(user_idx)]:
                    neg_items[idx] = item_idx
                    break
        return neg_items

    def _ssl_loss(self, emb1, emb2, temperature=0.2):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / temperature)
        all_score = torch.exp(torch.matmul(emb1, emb2.t()) / temperature).sum(dim=1)
        return -torch.log(pos_score / all_score.clamp(min=1e-12)).mean()

    def evaluate(self):
        N = self.topn
        hit = 0
        test_count = 0
        ndcg_sum = 0.0
        map_sum = 0.0
        eval_user_count = 0
        user_emb, item_emb = self.model(perturbed=False)
        user_emb = user_emb.detach()
        item_emb = item_emb.detach()

        print("======================")
        print("【TopN推荐评估】(rating>=%d, N=%d)" % (self.rating_threshold, N))
        print("======================")

        for user_idx in range(len(self.user_ids)):
            if user_idx % 500 == 0:
                print("topn evaluate for %d users" % user_idx, file=sys.stderr)
            test_items = self.test_items_by_user.get(user_idx, set())
            if not test_items:
                continue
            scores = torch.matmul(item_emb, user_emb[user_idx]).detach().cpu().numpy()
            for item_idx in self.train_items_by_user.get(user_idx, set()):
                scores[item_idx] = -np.inf
            top_idx = np.argpartition(scores, -N)[-N:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

            dcg = 0.0
            ap = 0.0
            user_hit = 0
            for rank, item_idx in enumerate(top_idx, start=1):
                if int(item_idx) in test_items:
                    hit += 1
                    user_hit += 1
                    dcg += 1 / math.log2(rank + 1)
                    ap += user_hit / rank

            ideal_hits = min(len(test_items), N)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_sum += dcg / idcg if idcg else 0
            map_sum += ap / ideal_hits if ideal_hits else 0
            test_count += len(test_items)
            eval_user_count += 1

        precision = hit / (1.0 * eval_user_count * N) if eval_user_count else 0
        recall = hit / (1.0 * test_count) if test_count else 0
        ndcg = ndcg_sum / eval_user_count if eval_user_count else 0
        mean_ap = map_sum / eval_user_count if eval_user_count else 0

        print("Precision@%d: %.4f" % (N, precision))
        print("Recall@%d: %.4f" % (N, recall))
        print("NDCG@%d: %.4f" % (N, ndcg))
        print("MAP@%d: %.4f" % (N, mean_ap))

        os.makedirs("./outputs", exist_ok=True)
        with open("./outputs/metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["XSimGCL", "%.4f" % precision, "%.4f" % recall, "%.4f" % ndcg, "%.4f" % mean_ap])
        topn_metrics_file = "./outputs/topn_metrics.csv"
        write_header = not os.path.exists(topn_metrics_file)
        with open(topn_metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["model", "N", "precision", "recall", "ndcg", "map"])
            writer.writerow(["XSimGCL", N, "%.4f" % precision, "%.4f" % recall, "%.4f" % ndcg, "%.4f" % mean_ap])

    def generate_recommendation(self):
        print("generating XSimGCL recommendation result...", file=sys.stderr)
        os.makedirs("./outputs", exist_ok=True)
        user_emb, item_emb = self.model(perturbed=False)
        user_emb = user_emb.detach()
        item_emb = item_emb.detach()
        with open("./outputs/xsimgcl_recommendation.csv", "w", newline="") as f:
            for user_idx, user_id in enumerate(self.user_ids):
                if user_idx % 500 == 0:
                    print("generate XSimGCL recommendation for %d users" % user_idx, file=sys.stderr)
                scores = torch.matmul(item_emb, user_emb[user_idx]).detach().cpu().numpy()
                for item_idx in self.train_items_by_user.get(user_idx, set()):
                    scores[item_idx] = -np.inf
                top_idx = np.argpartition(scores, -self.topn)[-self.topn:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                items = [str(self.idx2item[int(idx)]) for idx in top_idx]
                f.write("%s,%s\n" % (user_id, ",".join(items)))

    def gernate_recommendation(self):
        self.generate_recommendation()
