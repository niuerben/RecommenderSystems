# -*- coding: utf-8 -*-
"""
RecBole-tuned DCN runner for MovieLens 1M ratings.

This implementation uses only the rating table and follows RecBole's
recommended DCN hyper-parameters for ML-1M.
"""

import argparse
import copy
import csv
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn.init import constant_, xavier_normal_


def log(message):
    print(message, flush=True)


def parse_hidden_size(value):
    if isinstance(value, str):
        value = value.strip().strip("[]")
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(value)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )
        for weight in self.weights:
            xavier_normal_(weight.view(1, -1))

    def forward(self, x0):
        xl = x0
        for weight, bias in zip(self.weights, self.biases):
            xl_w = torch.tensordot(xl, weight, dims=([1], [0]))
            xl = x0 * xl_w.unsqueeze(1) + bias + xl
        return xl

    def reg_loss(self):
        return sum(weight.norm(2) for weight in self.weights)


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_prob):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_dim, hidden_size)
            xavier_normal_(linear.weight)
            constant_(linear.bias, 0)
            layers.extend(
                [
                    linear,
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )
            prev_dim = hidden_size
        self.layers = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.layers(x)


class DCNModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_movies,
        embedding_dim=64,
        mlp_hidden_size=(512, 512, 512),
        cross_layer_num=6,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.movie_emb = nn.Embedding(n_movies, embedding_dim)
        xavier_normal_(self.user_emb.weight)
        xavier_normal_(self.movie_emb.weight)

        input_dim = embedding_dim * 2
        self.cross = CrossNetwork(input_dim, cross_layer_num)
        self.deep = MLPBlock(input_dim, mlp_hidden_size, dropout_prob)
        self.output = nn.Linear(input_dim + self.deep.output_dim, 1)
        xavier_normal_(self.output.weight)
        constant_(self.output.bias, 0)

    def forward(self, user_idx, movie_idx):
        x0 = torch.cat([self.user_emb(user_idx), self.movie_emb(movie_idx)], dim=1)
        x = torch.cat([self.cross(x0), self.deep(x0)], dim=1)
        return self.output(x).squeeze(1)

    def reg_loss(self):
        return self.cross.reg_loss()


class DCN(object):
    def __init__(
        self,
        topn=10,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        train_neg_per_positive=2,
        embedding_dim=64,
        mlp_hidden_size=(512, 512, 512),
        cross_layer_num=6,
        dropout_prob=0.2,
        reg_weight=1.0,
        epochs=500,
        batch_size=4096,
        learning_rate=1e-3,
        seed=2020,
        recommendation_topn=100,
        valid_interval=5,
        early_stop_patience=10,
        min_delta=1e-6,
        eval_user_batch_size=512,
        prediction_batch_size=1048576,
        use_amp=True,
    ):
        self.topn = topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.train_neg_per_positive = train_neg_per_positive
        self.embedding_dim = embedding_dim
        self.mlp_hidden_size = parse_hidden_size(mlp_hidden_size)
        self.cross_layer_num = cross_layer_num
        self.dropout_prob = dropout_prob
        self.reg_weight = reg_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.recommendation_topn = recommendation_topn
        self.valid_interval = valid_interval
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.eval_user_batch_size = eval_user_batch_size
        self.prediction_batch_size = prediction_batch_size
        self.use_amp = use_amp

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.use_amp and self.device.type == "cuda"
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass

        self.model = None
        self.user_ids = []
        self.movie_ids = []
        self.user2idx = {}
        self.movie2idx = {}
        self.idx2movie = {}
        self.rated_by_user_idx = defaultdict(set)
        self.train_rated_by_user_idx = defaultdict(set)
        self.valid_positive_by_user_idx = defaultdict(set)
        self.test_positive_by_user_idx = defaultdict(set)
        self.best_epoch = 0
        self.best_valid_mrr = -1.0

    def generate_dataset(self, ratingsfile):
        log("使用设备:%s" % self.device)
        log("加载 DCN 数据...")
        rows = self._load_rating_rows(ratingsfile)
        positives = rows[rows[:, 2] >= self.rating_threshold]
        rng = np.random.default_rng(self.seed)

        self.user_ids = sorted(np.unique(rows[:, 0]).tolist())
        self.movie_ids = sorted(np.unique(positives[:, 1]).tolist())
        self.user2idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie2idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.idx2movie = {idx: movie_id for movie_id, idx in self.movie2idx.items()}

        rng.shuffle(positives)
        train_split = int(len(positives) * self.train_ratio)
        valid_split = int(len(positives) * (self.train_ratio + self.valid_ratio))
        train_rows = positives[:train_split]
        valid_rows = positives[train_split:valid_split]
        test_rows = positives[valid_split:]

        for user_id, movie_id, _ in positives:
            self.rated_by_user_idx[self.user2idx[int(user_id)]].add(self.movie2idx[int(movie_id)])
        for user_id, movie_id, _ in train_rows:
            self.train_rated_by_user_idx[self.user2idx[int(user_id)]].add(
                self.movie2idx[int(movie_id)]
            )
        for user_id, movie_id, _ in valid_rows:
            self.valid_positive_by_user_idx[self.user2idx[int(user_id)]].add(
                self.movie2idx[int(movie_id)]
            )
        for user_id, movie_id, _ in test_rows:
            self.test_positive_by_user_idx[self.user2idx[int(user_id)]].add(
                self.movie2idx[int(movie_id)]
            )

        self.train_arrays = self._build_train_arrays(train_rows, rng)
        log(
            "用户数:%d，电影数:%d，交互数:%d，训练:%d，Top%d 推荐列: %d"
            % (
                len(self.user_ids),
                len(self.movie_ids),
                len(positives),
                len(train_rows),
                self.recommendation_topn,
                len(self.user_ids) * self.recommendation_topn,
            )
        )

    def _load_rating_rows(self, ratingsfile):
        rows = []
        with open(ratingsfile, "r", encoding="latin-1") as f:
            first_line = f.readline()
            f.seek(0)
            if "::" in first_line:
                for line in f:
                    user_id, movie_id, rating, _ = line.rstrip("\n").split("::")
                    rows.append((int(user_id), int(movie_id), int(rating)))
            else:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append((int(row["user_id"]), int(row["movie_id"]), int(row["rating"])))
        return np.asarray(rows, dtype=np.int32)

    def _build_train_arrays(self, rows, rng):
        user_arr = []
        movie_arr = []
        label_arr = []
        positives_by_user = defaultdict(int)
        for user_id, movie_id, _ in rows:
            user_arr.append(self.user2idx[int(user_id)])
            movie_arr.append(self.movie2idx[int(movie_id)])
            label_arr.append(1)
            positives_by_user[int(user_id)] += 1

        all_movie_idx = np.arange(len(self.movie_ids), dtype=np.int64)
        for user_id in self.user_ids:
            user_idx = self.user2idx[int(user_id)]
            rated = np.fromiter(self.rated_by_user_idx[user_idx], dtype=np.int64)
            candidates = np.setdiff1d(all_movie_idx, rated, assume_unique=True)
            sample_size = positives_by_user[int(user_id)] * self.train_neg_per_positive
            if len(candidates) == 0 or sample_size <= 0:
                continue
            sampled = rng.choice(candidates, size=sample_size, replace=sample_size > len(candidates))
            user_arr.extend([user_idx] * len(sampled))
            movie_arr.extend(sampled.tolist())
            label_arr.extend([0] * len(sampled))

        user_arr = np.asarray(user_arr, dtype=np.int64)
        movie_arr = np.asarray(movie_arr, dtype=np.int64)
        label_arr = np.asarray(label_arr, dtype=np.float32)
        order = rng.permutation(len(label_arr))
        return user_arr[order], movie_arr[order], label_arr[order]

    def calc_movie_sim(self):
        log("加载模型 DCN...")
        self.model = DCNModel(
            n_users=len(self.user_ids),
            n_movies=len(self.movie_ids),
            embedding_dim=self.embedding_dim,
            mlp_hidden_size=self.mlp_hidden_size,
            cross_layer_num=self.cross_layer_num,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

        train_user_idx, train_movie_idx, train_labels = self.train_arrays
        train_user_idx = torch.as_tensor(train_user_idx, dtype=torch.long, device=self.device)
        train_movie_idx = torch.as_tensor(train_movie_idx, dtype=torch.long, device=self.device)
        train_labels = torch.as_tensor(train_labels, dtype=torch.float32, device=self.device)
        train_count = train_labels.numel()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        scaler = self._make_grad_scaler()
        best_state_dict = None
        stale_validations = 0

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_count = 0
            order = torch.randperm(train_count, device=self.device)
            self.model.train()
            for start in range(0, train_count, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                user_idx = train_user_idx[batch_idx]
                movie_idx = train_movie_idx[batch_idx]
                labels = train_labels[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(user_idx, movie_idx)
                    loss = criterion(logits, labels) + self.reg_weight * self.model.reg_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * labels.size(0)
                total_count += labels.size(0)

            message = "Epoch %d/%d - loss: %.4f" % (epoch, self.epochs, total_loss / total_count)
            if self.valid_interval > 0 and epoch % self.valid_interval == 0:
                valid_metrics = self._evaluate_topn_split(
                    self.valid_positive_by_user_idx,
                    self.train_rated_by_user_idx,
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

            log(message)
            if self.early_stop_patience > 0 and stale_validations >= self.early_stop_patience:
                log(
                    "早停触发: valid MRR@%d 连续 %d 次没有提升，停止于 epoch %d。"
                    % (self.topn, self.early_stop_patience, epoch)
                )
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            log(
                "加载验证集最佳 checkpoint: epoch %d, Valid MRR@%d=%.4f"
                % (self.best_epoch, self.topn, self.best_valid_mrr)
            )

    def evaluate(self):
        return self.evaluate_topn()

    def evaluate_topn(self):
        mask_items_by_user = defaultdict(set)
        for user_idx, items in self.train_rated_by_user_idx.items():
            mask_items_by_user[user_idx].update(items)
        for user_idx, items in self.valid_positive_by_user_idx.items():
            mask_items_by_user[user_idx].update(items)
        return self._evaluate_topn_split(self.test_positive_by_user_idx, mask_items_by_user, label="Test")

    def _evaluate_topn_split(self, eval_items_by_user, mask_items_by_user, label="Test", verbose=True):
        N = self.topn
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        hit_user_count = 0
        eval_user_count = 0
        eval_user_indices = sorted(eval_items_by_user.keys())

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(eval_user_indices), self.eval_user_batch_size):
                batch_user_indices = eval_user_indices[start : start + self.eval_user_batch_size]
                users_t = torch.tensor(batch_user_indices, dtype=torch.long, device=self.device)
                scores = self._predict_user_item_scores(users_t)
                for row_idx, user_idx in enumerate(batch_user_indices):
                    mask_items = mask_items_by_user.get(user_idx, set())
                    if mask_items:
                        scores[row_idx, torch.tensor(list(mask_items), dtype=torch.long, device=self.device)] = -float("inf")

                top_k = min(N, scores.size(1))
                top_idx = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()

                for row_idx, user_idx in enumerate(batch_user_indices):
                    eval_movies = eval_items_by_user.get(user_idx, set())
                    if not eval_movies:
                        continue
                    dcg = 0.0
                    user_hit = 0
                    reciprocal_rank = 0.0
                    for rank, movie_idx in enumerate(top_idx[row_idx], start=1):
                        if int(movie_idx) in eval_movies:
                            user_hit += 1
                            dcg += 1 / np.log2(rank + 1)
                            if reciprocal_rank == 0.0:
                                reciprocal_rank = 1 / rank

                    ideal_hits = min(len(eval_movies), N)
                    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
                    precision_sum += user_hit / N
                    recall_sum += user_hit / len(eval_movies)
                    ndcg_sum += dcg / idcg if idcg else 0
                    mrr_sum += reciprocal_rank
                    if user_hit > 0:
                        hit_user_count += 1
                    eval_user_count += 1

        metrics = {
            "recall": recall_sum / eval_user_count if eval_user_count else 0,
            "mrr": mrr_sum / eval_user_count if eval_user_count else 0,
            "ndcg": ndcg_sum / eval_user_count if eval_user_count else 0,
            "hit": hit_user_count / eval_user_count if eval_user_count else 0,
            "precision": precision_sum / eval_user_count if eval_user_count else 0,
        }
        if verbose:
            log(
                "测试集 %s RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    HIT@%d : %.4f    PRECISION@%d : %.4f"
                % (label, N, metrics["recall"], N, metrics["mrr"], N, metrics["ndcg"], N, metrics["hit"], N, metrics["precision"])
            )
        return metrics

    def generate_recommendation(self, filepath="./outputs/dcn_recommendation.csv", topn=None):
        topn = topn or self.recommendation_topn
        log("generating DCN recommendation result: %s" % filepath)
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id"] + ["rec%d" % idx for idx in range(1, topn + 1)])
            user_indices = list(range(len(self.user_ids)))
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(user_indices), self.eval_user_batch_size):
                    batch_user_indices = user_indices[start : start + self.eval_user_batch_size]
                    users_t = torch.tensor(batch_user_indices, dtype=torch.long, device=self.device)
                    scores = self._predict_user_item_scores(users_t)
                    for row_idx, user_idx in enumerate(batch_user_indices):
                        rated = self.train_rated_by_user_idx.get(user_idx, set())
                        if rated:
                            scores[row_idx, torch.tensor(list(rated), dtype=torch.long, device=self.device)] = -float("inf")
                    top_k = min(topn, scores.size(1))
                    top_idx = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()
                    for row_idx, user_idx in enumerate(batch_user_indices):
                        movies = [str(self.idx2movie[int(idx)]) for idx in top_idx[row_idx]]
                        if len(movies) < topn:
                            movies.extend([""] * (topn - len(movies)))
                        writer.writerow([self.user_ids[user_idx]] + movies)

    def _predict_user_item_scores(self, user_idx, movie_idx=None):
        if movie_idx is None:
            movie_idx = torch.arange(len(self.movie_ids), dtype=torch.long, device=self.device)
        elif movie_idx.device != self.device:
            movie_idx = movie_idx.to(self.device)

        user_idx = user_idx.to(self.device)
        batch_users = user_idx.numel()
        scores = torch.empty((batch_users, movie_idx.numel()), dtype=torch.float32, device=self.device)
        max_rows = max(1, int(self.prediction_batch_size))
        movie_chunk = max(1, max_rows // max(1, batch_users))

        for start in range(0, movie_idx.numel(), movie_chunk):
            movies = movie_idx[start : start + movie_chunk]
            flat_users = user_idx.repeat_interleave(movies.numel())
            flat_movies = movies.repeat(batch_users)
            with self._autocast():
                logits = self.model(flat_users, flat_movies)
            scores[:, start : start + movies.numel()] = torch.sigmoid(logits).view(
                batch_users,
                movies.numel(),
            )
        return scores

    def _make_grad_scaler(self):
        if hasattr(torch, "amp"):
            try:
                return torch.amp.GradScaler("cuda", enabled=self.use_amp)
            except TypeError:
                return torch.amp.GradScaler(enabled=self.use_amp)
        return torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        if hasattr(torch, "amp"):
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run RecBole-tuned DCN on MovieLens-1M ratings.")
    parser.add_argument("--ratings-file", default="./data/ml-1m/ml-1m/ratings.dat")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--mlp-hidden-size", default="512,512,512")
    parser.add_argument("--cross-layer-num", type=int, default=6)
    parser.add_argument("--dropout-prob", type=float, default=0.2)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-neg-per-positive", type=int, default=2)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--recommendation-topn", type=int, default=100)
    parser.add_argument("--rating-threshold", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--valid-interval", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--eval-user-batch-size", type=int, default=512)
    parser.add_argument("--prediction-batch-size", type=int, default=1048576)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-final-recommendation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    dcn = DCN(
        topn=args.topn,
        rating_threshold=args.rating_threshold,
        embedding_dim=args.embedding_dim,
        mlp_hidden_size=parse_hidden_size(args.mlp_hidden_size),
        cross_layer_num=args.cross_layer_num,
        dropout_prob=args.dropout_prob,
        reg_weight=args.reg_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_neg_per_positive=args.train_neg_per_positive,
        seed=args.seed,
        recommendation_topn=args.recommendation_topn,
        valid_interval=args.valid_interval,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
        eval_user_batch_size=args.eval_user_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        use_amp=not args.no_amp,
    )
    dcn.generate_dataset(args.ratings_file)
    dcn.calc_movie_sim()
    if not args.skip_evaluation:
        dcn.evaluate()
    if not args.skip_final_recommendation:
        dcn.generate_recommendation(topn=args.recommendation_topn)


if __name__ == "__main__":
    main()
