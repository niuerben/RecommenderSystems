# -*- coding: utf-8 -*-
"""
基于用户的协同过滤
"""

import sys
import random
import math
import os
import time
import csv
from operator import itemgetter


random.seed(0)


class UserCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self, rating_threshold=3):
        self.trainset = {}
        self.testset = {}
        self.train_ratedset = {}

        self.n_sim_user = 20
        self.n_rec_movie = 10
        self.rating_threshold = rating_threshold
        
        
        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print ('recommended movie number = %d' %
               self.n_rec_movie, file=sys.stderr)


    # 先过滤rating<3，再按8:1:1切分，验证集当前不参与训练和评估
    def generate_dataset(self, filename, pivot=0.8, valid_ratio=0.1):
        ''' load rating data and split it to training set and test set '''
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

        print ('split training set and test set succ', file=sys.stderr)
        print ('positive train set = %s' % trainset_len, file=sys.stderr)
        print ('positive test set = %s' % testset_len, file=sys.stderr)
        print ('train rated set = %s' % train_rated_len, file=sys.stderr)

    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # 建立电影到用户的倒排表
        # key=movieID, value=list of userIDs who have seen this movie
        print ('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print ('total movie number = %d' % self.movie_count, file=sys.stderr)

        # 计算用户相似度矩阵
        usersim_mat = self.user_sim_mat
        print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1
        print ('build user co-rated movies matrix succ', file=sys.stderr)

        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' 根据K个相似用户推荐N个该用户没看过的电影. '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.train_ratedset.get(user, set())

        for similar_user, similarity_factor in sorted(self.user_sim_mat.get(user, {}).items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie, rating in self.trainset[similar_user].items():
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor*rating
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' print evaluation result: precision@K, ndcg@K and map@K '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        hit = 0
        test_count = 0
        ndcg_sum = 0
        map_sum = 0
        eval_user_count = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
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

        print ('Precision@%d: %.4f' % (N, precision))
        print ('Recall@%d: %.4f' % (N, recall))
        print ('NDCG@%d: %.4f' % (N, ndcg))
        print ('MAP@%d: %.4f' % (N, mean_ap))

        with open('./outputs/metrics.csv', 'a') as f:
            f.write('"UserCF",%.4f,%.4f,%.4f,%.4f\n' % (precision, recall, ndcg, mean_ap))
        topn_metrics_file = './outputs/topn_metrics.csv'
        write_header = not os.path.exists(topn_metrics_file)
        with open(topn_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['model', 'N', 'precision', 'recall', 'ndcg', 'map'])
            writer.writerow(['UserCF', N, '%.4f' % precision, '%.4f' % recall, '%.4f' % ndcg, '%.4f' % mean_ap])

    def generate_recommendation(self):
        ''' 输出推荐结果 '''
        print ('generating recommendation result...', file=sys.stderr)
        with open('./outputs/usercf_recommendation.csv', 'w') as f:
            for i, user in enumerate(self.trainset):
                if i % 500 == 0:
                    print ('generate recommendation for %d users' % i, file=sys.stderr)
                rec_movies = self.recommend(user)
                f.write('%s,%s\n' % (user, ','.join([movie for movie, _ in rec_movies])))
        print ('generate recommendation result succ', file=sys.stderr)

if __name__ == '__main__':
    ratingfile = "./data/ml-1m/ml-1m/ratings.dat"
    usercf = UserCF()
    usercf.generate_dataset(ratingfile)
    usercf.calc_user_sim()
    usercf.evaluate()
    usercf.generate_recommendation()
