# -*- coding: utf-8 -*-
"""
基于物品的协同过滤
"""


import random
import sys
import math
import os
import csv
from operator import itemgetter
random.seed(0)


class ItemCF(object):
    def __init__(self, rating_threshold=3):
        self.trainset = {}
        self.testset = {}
        self.train_ratedset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10
        self.rating_threshold = rating_threshold

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommendend movie number = %d' % self.n_rec_movie, file=sys.stderr)
    # 先过滤rating<3，再按8:1:1切分，验证集当前不参与训练和评估
    def generate_dataset(self, filename, pivot=0.8, valid_ratio=0.1):
        trainset_len = 0
        testset_len = 0
        train_rated_len = 0
        fp = open(filename, 'r')
        for line in fp:
            user, movie, rating, _ = line.split('::')
            rating = int(rating)
            if rating < self.rating_threshold:
                continue

            rand_value = random.random()
            if rand_value < pivot:
                self.train_ratedset.setdefault(user, set())
                self.train_ratedset[user].add(movie)
                train_rated_len += 1
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = rating
                trainset_len += 1
            elif rand_value >= pivot + valid_ratio:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = rating
                testset_len += 1

        print('split succ , positive trainset is %d , positive testset is %d' % (trainset_len, testset_len), file=sys.stderr)
        print('train rated set is %d' % train_rated_len, file=sys.stderr)

    def calc_movie_sim(self):
        for user, movies in self.trainset.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print('count movies number and pipularity succ', file=sys.stderr)

        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix', file=sys.stderr)
        for user, movies in self.trainset.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat.setdefault(m1, {})
                    itemsim_mat[m1].setdefault(m2, 0)
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)
        print('calculating movie similarity matrix', file=sys.stderr)

        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calcu movie similarity factor(%d)' % simfactor_count, file=sys.stderr)
        print('calcu similiarity succ', file=sys.stderr)

    def recommend(self, user):
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        # watched_movies表示用户user看过的电影和评分
        watched_movies = self.train_ratedset.get(user, set())

        for movie, rating in self.trainset.get(user, {}).items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat.get(movie, {}).items(), key=itemgetter(1),
                                                           reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        print('evaluation start', file=sys.stderr)

        N = self.n_rec_movie

        hit = 0
        test_count = 0
        ndcg_sum = 0
        map_sum = 0
        eval_user_count = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommend for %d users ' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            if not test_movies:
                continue
            rec_movies = self.recommend(user)

            dcg = 0
            ap = 0
            user_hit = 0
            for rank, (movie, _) in enumerate(rec_movies, start=1):
                if movie in test_movies:
                    hit += 1
                    user_hit += 1
                    dcg += 1 / math.log2(rank + 1)
                    ap += user_hit / rank

            ideal_hits = min(len(test_movies), N)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_sum += dcg / idcg if idcg else 0
            map_sum += ap / ideal_hits if ideal_hits else 0
            test_count += len(test_movies)
            eval_user_count += 1

        precision = hit / (1.0 * eval_user_count * N) if eval_user_count else 0
        recall = hit / (1.0 * test_count) if test_count else 0
        ndcg = ndcg_sum / eval_user_count if eval_user_count else 0
        mean_ap = map_sum / eval_user_count if eval_user_count else 0

        print('Precision@%d: %.4f' % (N, precision))
        print('Recall@%d: %.4f' % (N, recall))
        print('NDCG@%d: %.4f' % (N, ndcg))
        print('MAP@%d: %.4f' % (N, mean_ap))

        with open('./outputs/metrics.csv', 'a') as f:
            f.write('"ItemCF",%.4f,%.4f,%.4f,%.4f\n' % (precision, recall, ndcg, mean_ap))
        topn_metrics_file = './outputs/topn_metrics.csv'
        write_header = not os.path.exists(topn_metrics_file)
        with open(topn_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['model', 'N', 'precision', 'recall', 'ndcg', 'map'])
            writer.writerow(['ItemCF', N, '%.4f' % precision, '%.4f' % recall, '%.4f' % ndcg, '%.4f' % mean_ap])

    def generate_recommendation(self):
        ''' 输出推荐结果 '''
        print('generating recommendation result...', file=sys.stderr)
        with open('./outputs/itemcf_recommendation.csv', 'w') as f:
            for i, user in enumerate(self.trainset):
                if i % 500 == 0:
                    print('generate recommendation for %d users' % i, file=sys.stderr)
                rec_movies = self.recommend(user)
                f.write('%s,%s\n' % (user, ','.join([movie for movie, _ in rec_movies])))
        print('generate recommendation result succ', file=sys.stderr)

if __name__ == '__main__':
    ratingfile = "./data/ml-1m/ml-1m/ratings.dat"
    item_cf = ItemCF()
    item_cf.generate_dataset(ratingfile)
    item_cf.calc_movie_sim()
    item_cf.evaluate()
    item_cf.generate_recommendation()

