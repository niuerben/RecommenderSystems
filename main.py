import time
import csv
import os
import importlib.util
from pathlib import Path

from py3x import user_cf
from py3x import item_cf


def load_dcn():
    module_path = Path(__file__).resolve().parent / "RS-torch" / "DCN_torch" / "dcn.py"
    spec = importlib.util.spec_from_file_location("dcn", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_xsimgcl():
    module_path = Path(__file__).resolve().parent / "RS-torch" / "XSimGCL" / "xsimgcl.py"
    spec = importlib.util.spec_from_file_location("xsimgcl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge(moviesfile, ratingsfile, usersfile, outputfile="./data/ml-1m/ml-1m/merged.dat"):
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)

    movies = {}
    with open(moviesfile, "r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.rstrip("\n").split("::")
            movies[movie_id] = (title, genres)

    users = {}
    with open(usersfile, "r", encoding="latin-1") as f:
        for line in f:
            user_id, gender, age, occupation, zipcode = line.rstrip("\n").split("::")
            users[user_id] = (gender, age, occupation, zipcode)

    with open(ratingsfile, "r", encoding="latin-1") as rf, open(
        outputfile, "w", encoding="utf-8", newline=""
    ) as wf:
        writer = csv.writer(wf)
        writer.writerow(
            [
                "user_id",
                "movie_id",
                "rating",
                "timestamp",
                "gender",
                "age",
                "occupation",
                "zipcode",
                "title",
                "genres",
            ]
        )
        for line in rf:
            user_id, movie_id, rating, timestamp = line.rstrip("\n").split("::")
            gender, age, occupation, zipcode = users[user_id]
            title, genres = movies[movie_id]
            writer.writerow(
                [user_id, movie_id, rating, timestamp, gender, age, occupation, zipcode, title, genres]
            )
    return outputfile


if __name__ == '__main__':
    ratingfile = "./data/ml-1m/ml-1m/ratings.dat"
    moviesfile = "./data/ml-1m/ml-1m/movies.dat"
    usersfile = "./data/ml-1m/ml-1m/users.dat"

    # user_cf_obj = user_cf.UserCF()
    # t1=time.time()
    # user_cf_obj.generate_dataset(ratingfile)
    # user_cf_obj.calc_user_sim()
    # user_cf_obj.evaluate()
    # user_cf_obj.generate_recommendation()
    # t2=time.time()
    # print('user_cf算法耗时: %.2f' % (t2-t1))

    # item_cf_obj = item_cf.ItemCF()
    # t1=time.time()
    # item_cf_obj.generate_dataset(ratingfile)
    # item_cf_obj.calc_movie_sim()
    # item_cf_obj.evaluate()
    # item_cf_obj.generate_recommendation()
    # t2=time.time()
    # print('item_cf算法耗时: %.2f' % (t2-t1))

    # t1=time.time()
    # dcn=load_dcn()
    # dcn_obj=dcn.DCN()
    # mergedfile=merge(moviesfile,ratingfile,usersfile)
    # dcn_obj.gernate_dataset(mergedfile)
    # dcn_obj.calc_movie_sim()
    # dcn_obj.evaluate()
    # dcn_obj.gernate_recommendation()
    # t2=time.time()
    # print('DCN耗时：%.2f'%(t2-t1))

    t1=time.time()
    xsimgcl=load_xsimgcl()
    xsimgcl_obj=xsimgcl.XSimGCL()
    xsimgcl_obj.gernate_dataset(ratingfile)
    xsimgcl_obj.calc_movie_sim()
    xsimgcl_obj.evaluate()
    xsimgcl_obj.gernate_recommendation()
    t2=time.time()
    print('XSimGCL耗时：%.2f'%(t2-t1))
