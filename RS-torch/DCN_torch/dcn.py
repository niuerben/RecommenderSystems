# -*- coding: utf-8 -*-
"""
Project-level DCN runner for MovieLens 1M.

This wrapper keeps the same call style as the collaborative filtering modules:
generate dataset -> train/calc -> evaluate -> generate recommendations.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0):
        xl = x0
        for weight, bias in zip(self.weights, self.biases):
            xw = xl @ weight
            xl = x0 * xw + bias + xl
        return xl


class DCNModel(nn.Module):
    def __init__(self, n_users, n_movies, n_genders, n_ages, n_occupations, genre_features):
        super().__init__()
        self.register_buffer("genre_features", genre_features)

        self.user_emb = nn.Embedding(n_users, 16)
        self.movie_emb = nn.Embedding(n_movies, 24)
        self.gender_emb = nn.Embedding(n_genders, 4)
        self.age_emb = nn.Embedding(n_ages, 4)
        self.occupation_emb = nn.Embedding(n_occupations, 8)
        self.genre_layer = nn.Linear(genre_features.shape[1], 16)

        input_dim = 16 + 24 + 4 + 4 + 8 + 16
        self.cross = CrossNetwork(input_dim, num_layers=3)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 64, 1)

    def forward(self, user_idx, movie_idx, gender_idx, age_idx, occupation_idx):
        genre_vec = self.genre_features[movie_idx]
        x = torch.cat(
            [
                self.user_emb(user_idx),
                self.movie_emb(movie_idx),
                self.gender_emb(gender_idx),
                self.age_emb(age_idx),
                self.occupation_emb(occupation_idx),
                torch.relu(self.genre_layer(genre_vec)),
            ],
            dim=1,
        )
        x = torch.cat([self.cross(x), self.deep(x)], dim=1)
        return self.output(x).squeeze(1)


class DCN(object):
    def __init__(
        self,
        topn=10,
        rating_threshold=3,
        train_ratio=0.8,
        valid_ratio=0.1,
        train_neg_per_positive=2,
        epochs=5,
        batch_size=8192,
        seed=0,
    ):
        self.topn = topn
        self.rating_threshold = rating_threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.train_neg_per_positive = train_neg_per_positive
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.metrics = None

        self.user_ids = []
        self.movie_ids = []
        self.user2idx = {}
        self.movie2idx = {}
        self.idx2movie = {}
        self.user_features = {}
        self.user_feature_matrix = None
        self.movie_index_lookup = None
        self.rated_by_user = defaultdict(set)
        self.train_rated_by_user = defaultdict(set)
        self.test_positive_by_user = defaultdict(set)

    def generate_dataset(self, mergedfile):
        print("使用设备:%s" % self.device)
        print("加载数据...")
        data = pd.read_csv(mergedfile)
        data = data[data["rating"] >= self.rating_threshold].copy()
        rng = np.random.default_rng(self.seed)

        self.user_ids = sorted(data["user_id"].unique())
        self.movie_ids = sorted(data["movie_id"].unique())
        self.user2idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie2idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.idx2movie = {idx: movie_id for movie_id, idx in self.movie2idx.items()}
        self.movie_index_lookup = np.full(max(self.movie_ids) + 1, -1, dtype=np.int64)
        for movie_id, movie_idx in self.movie2idx.items():
            self.movie_index_lookup[int(movie_id)] = movie_idx

        gender2idx = {value: idx for idx, value in enumerate(sorted(data["gender"].unique()))}
        age2idx = {value: idx for idx, value in enumerate(sorted(data["age"].unique()))}
        occupation2idx = {value: idx for idx, value in enumerate(sorted(data["occupation"].unique()))}

        users = data[["user_id", "gender", "age", "occupation"]].drop_duplicates("user_id")
        for row in users.itertuples(index=False):
            self.user_features[int(row.user_id)] = (
                gender2idx[row.gender],
                age2idx[row.age],
                occupation2idx[row.occupation],
            )
        self.user_feature_matrix = np.zeros((len(self.user_ids), 3), dtype=np.int64)
        for user_id, user_idx in self.user2idx.items():
            self.user_feature_matrix[user_idx] = self.user_features[int(user_id)]

        movie_genres = data[["movie_id", "genres"]].drop_duplicates("movie_id")
        genres = sorted({genre for text in movie_genres["genres"] for genre in str(text).split("|")})
        genre2idx = {genre: idx for idx, genre in enumerate(genres)}
        genre_features = np.zeros((len(self.movie_ids), len(genres)), dtype=np.float32)
        for row in movie_genres.itertuples(index=False):
            movie_idx = self.movie2idx[int(row.movie_id)]
            for genre in str(row.genres).split("|"):
                genre_features[movie_idx, genre2idx[genre]] = 1.0
        self.genre_features = torch.tensor(genre_features, dtype=torch.float32)

        rows = data[["user_id", "movie_id", "rating"]].to_numpy(dtype=np.int32)
        rng.shuffle(rows)
        train_split = int(len(rows) * self.train_ratio)
        valid_split = int(len(rows) * (self.train_ratio + self.valid_ratio))
        train_rows = rows[:train_split]
        test_rows = rows[valid_split:]

        for user_id, movie_id, _ in rows:
            self.rated_by_user[int(user_id)].add(int(movie_id))
        for user_id, movie_id, _ in train_rows:
            self.train_rated_by_user[int(user_id)].add(int(movie_id))
        for user_id, movie_id, rating in test_rows:
            self.test_positive_by_user[int(user_id)].add(int(movie_id))

        self.train_arrays = self._build_train_arrays(train_rows, rng)

        print(
            "用户数:%d，电影数:%d，Top%d评估推荐位:%d"
            % (len(self.user_ids), len(self.movie_ids), self.topn, len(self.user_ids) * self.topn)
        )

    def gernate_dataset(self, mergedfile):
        self.generate_dataset(mergedfile)

    def _build_train_arrays(self, rows, rng):
        user_arr = []
        movie_arr = []
        label_arr = []

        positives_by_user = defaultdict(int)
        for user_id, movie_id, rating in rows:
            user_arr.append(self.user2idx[int(user_id)])
            movie_arr.append(self.movie2idx[int(movie_id)])
            label_arr.append(1)
            positives_by_user[int(user_id)] += 1

        all_movies = np.array(self.movie_ids, dtype=np.int32)
        for user_id in self.user_ids:
            rated = self.rated_by_user[int(user_id)]
            candidates = np.array([movie_id for movie_id in all_movies if int(movie_id) not in rated], dtype=np.int32)
            if len(candidates) == 0:
                continue
            sample_size = positives_by_user[int(user_id)] * self.train_neg_per_positive
            if sample_size <= 0:
                continue
            sampled = rng.choice(candidates, size=sample_size, replace=sample_size > len(candidates))
            user_arr.extend([self.user2idx[int(user_id)]] * len(sampled))
            movie_arr.extend(self.movie_index_lookup[sampled].tolist())
            label_arr.extend([0] * len(sampled))

        user_arr = np.asarray(user_arr, dtype=np.int64)
        movie_arr = np.asarray(movie_arr, dtype=np.int64)
        label_arr = np.asarray(label_arr, dtype=np.float32)
        order = rng.permutation(len(label_arr))
        return user_arr[order], movie_arr[order], label_arr[order]

    def _make_loader(self, arrays, shuffle=False):
        user_arr, movie_arr, label_arr = arrays
        user_feature_arr = self.user_feature_matrix[user_arr]
        gender_arr = user_feature_arr[:, 0]
        age_arr = user_feature_arr[:, 1]
        occupation_arr = user_feature_arr[:, 2]

        dataset = TensorDataset(
            torch.tensor(user_arr, dtype=torch.long),
            torch.tensor(movie_arr, dtype=torch.long),
            torch.tensor(gender_arr, dtype=torch.long),
            torch.tensor(age_arr, dtype=torch.long),
            torch.tensor(occupation_arr, dtype=torch.long),
            torch.tensor(label_arr, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def calc_movie_sim(self):
        print("加载模型 DCN...")
        self.model = DCNModel(
            n_users=len(self.user_ids),
            n_movies=len(self.movie_ids),
            n_genders=2,
            n_ages=len({features[1] for features in self.user_features.values()}),
            n_occupations=len({features[2] for features in self.user_features.values()}),
            genre_features=self.genre_features,
        ).to(self.device)

        loader = self._make_loader(self.train_arrays, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_count = 0
            for user_idx, movie_idx, gender_idx, age_idx, occupation_idx, labels in loader:
                user_idx = user_idx.to(self.device)
                movie_idx = movie_idx.to(self.device)
                gender_idx = gender_idx.to(self.device)
                age_idx = age_idx.to(self.device)
                occupation_idx = occupation_idx.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(user_idx, movie_idx, gender_idx, age_idx, occupation_idx)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_count += labels.size(0)
            print("Epoch %d/%d - loss: %.4f" % (epoch, self.epochs, total_loss / total_count))

    def evaluate(self):
        self.evaluate_topn()

    def evaluate_classification(self, threshold=0.5, eval_neg_per_user=1900):
        rng = np.random.default_rng(self.seed + 1)
        eval_rows = []
        for user_id, movies in self.test_positive_by_user.items():
            for movie_id in movies:
                eval_rows.append([user_id, movie_id, self.rating_threshold])
        if not eval_rows:
            print("No positive test samples for classification evaluation.")
            return

        user_arr = []
        movie_arr = []
        label_arr = []
        all_movies = np.array(self.movie_ids, dtype=np.int32)
        for user_id, movie_id, _ in eval_rows:
            user_arr.append(self.user2idx[int(user_id)])
            movie_arr.append(self.movie2idx[int(movie_id)])
            label_arr.append(1)
            rated = self.rated_by_user[int(user_id)]
            candidates = np.array([mid for mid in all_movies if int(mid) not in rated], dtype=np.int32)
            if len(candidates) == 0:
                continue
            sample_size = min(eval_neg_per_user, len(candidates))
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            user_arr.extend([self.user2idx[int(user_id)]] * len(sampled))
            movie_arr.extend(self.movie_index_lookup[sampled].tolist())
            label_arr.extend([0] * len(sampled))

        eval_arrays = (
            np.asarray(user_arr, dtype=np.int64),
            np.asarray(movie_arr, dtype=np.int64),
            np.asarray(label_arr, dtype=np.float32),
        )
        loader = self._make_loader(eval_arrays, shuffle=False)
        y_true, y_score = self._predict_loader(loader)
        y_pred = (y_score >= threshold).astype(np.int32)

        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0,
        }

        print("======================")
        print("【分类指标评估】(threshold=%.1f)" % threshold)
        print("======================")
        print("Accuracy: %.4f" % self.metrics["accuracy"])
        print("Precision: %.4f" % self.metrics["precision"])
        print("Recall:  %.4f" % self.metrics["recall"])
        print("F1-Score: %.4f" % self.metrics["f1"])
        print("AUC: %.4f" % self.metrics["auc"])

        os.makedirs("./outputs", exist_ok=True)
        with open("./outputs/dcn_metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "accuracy", "precision", "recall", "f1", "auc"])
            writer.writerow(
                [
                    "DCN",
                    "%.4f" % self.metrics["accuracy"],
                    "%.4f" % self.metrics["precision"],
                    "%.4f" % self.metrics["recall"],
                    "%.4f" % self.metrics["f1"],
                    "%.4f" % self.metrics["auc"],
                ]
            )

    def evaluate_topn(self):
        N = self.topn
        hit = 0
        test_count = 0
        ndcg_sum = 0
        map_sum = 0
        eval_user_count = 0
        all_movie_idx = torch.arange(len(self.movie_ids), dtype=torch.long, device=self.device)

        print("======================")
        print("【TopN推荐评估】(rating>=%d, N=%d)" % (self.rating_threshold, N))
        print("======================")

        for i, user_id in enumerate(self.user_ids):
            if i % 500 == 0:
                print("topn evaluate for %d users" % i, file=sys.stderr)
            test_movies = self.test_positive_by_user.get(int(user_id), set())
            if not test_movies:
                continue
            scores = self._predict_user_movies(user_id, all_movie_idx)
            for movie_id in self.train_rated_by_user.get(int(user_id), set()):
                scores[self.movie2idx[int(movie_id)]] = -1.0
            top_idx = np.argpartition(scores, -N)[-N:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            rec_movies = [self.idx2movie[int(idx)] for idx in top_idx]

            dcg = 0
            ap = 0
            user_hit = 0
            for rank, movie_id in enumerate(rec_movies, start=1):
                if movie_id in test_movies:
                    hit += 1
                    user_hit += 1
                    dcg += 1 / np.log2(rank + 1)
                    ap += user_hit / rank

            ideal_hits = min(len(test_movies), N)
            idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_sum += dcg / idcg if idcg else 0
            map_sum += ap / ideal_hits if ideal_hits else 0
            test_count += len(test_movies)
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
            writer.writerow(["DCN", "%.4f" % precision, "%.4f" % recall, "%.4f" % ndcg, "%.4f" % mean_ap])
        topn_metrics_file = "./outputs/topn_metrics.csv"
        write_header = not os.path.exists(topn_metrics_file)
        with open(topn_metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["model", "N", "precision", "recall", "ndcg", "map"])
            writer.writerow(["DCN", N, "%.4f" % precision, "%.4f" % recall, "%.4f" % ndcg, "%.4f" % mean_ap])

    def _predict_loader(self, loader):
        self.model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for user_idx, movie_idx, gender_idx, age_idx, occupation_idx, labels in loader:
                logits = self.model(
                    user_idx.to(self.device),
                    movie_idx.to(self.device),
                    gender_idx.to(self.device),
                    age_idx.to(self.device),
                    occupation_idx.to(self.device),
                )
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                y_score.append(scores)
                y_true.append(labels.numpy())
        return np.concatenate(y_true).astype(np.int32), np.concatenate(y_score)

    def generate_recommendation(self):
        print("generating DCN recommendation result...", file=sys.stderr)
        os.makedirs("./outputs", exist_ok=True)
        all_movie_idx = torch.arange(len(self.movie_ids), dtype=torch.long, device=self.device)
        with open("./outputs/dcn_recommendation.csv", "w", newline="") as f:
            for i, user_id in enumerate(self.user_ids):
                if i % 500 == 0:
                    print("generate DCN recommendation for %d users" % i, file=sys.stderr)
                scores = self._predict_user_movies(user_id, all_movie_idx)
                rated = self.train_rated_by_user[int(user_id)]
                for movie_id in rated:
                    scores[self.movie2idx[int(movie_id)]] = -1.0
                top_idx = np.argpartition(scores, -self.topn)[-self.topn:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                movies = [str(self.idx2movie[int(idx)]) for idx in top_idx]
                f.write("%s,%s\n" % (user_id, ",".join(movies)))
        print("generate DCN recommendation result succ", file=sys.stderr)

    def gernate_recommendation(self):
        self.generate_recommendation()

    def _predict_user_movies(self, user_id, all_movie_idx):
        self.model.eval()
        gender_idx, age_idx, occupation_idx = self.user_features[int(user_id)]
        user_idx = torch.full_like(all_movie_idx, self.user2idx[int(user_id)])
        gender_tensor = torch.full_like(all_movie_idx, gender_idx)
        age_tensor = torch.full_like(all_movie_idx, age_idx)
        occupation_tensor = torch.full_like(all_movie_idx, occupation_idx)
        with torch.no_grad():
            logits = self.model(user_idx, all_movie_idx, gender_tensor, age_tensor, occupation_tensor)
            return torch.sigmoid(logits).detach().cpu().numpy()
