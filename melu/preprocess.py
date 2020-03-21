import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
import pandas as pd
import datetime

class movielens_1m(object):
    def __init__(self):
        self.user_data, self.item_data, self.score_data = self.load()

    def load(self):
        path = "movielens/ml-1m"
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = np.array([[rate_list.index(str(row['rate']))]])
    genre_idx = np.zeros((1, 25))
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = np.zeros((1, 2186))
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = np.zeros((1, 8030))
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    # return np.concatenate([rate_idx, genre_idx, director_idx, actor_idx], 1)
    return {
        'rate': rate_idx,
        'genre': genre_idx,
        'director': director_idx,
        'actor': actor_idx
    }


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx = np.array([[gender_list.index(str(row['gender']))]])
    age_idx = np.array([[age_list.index(str(row['age']))]])
    occupation_idx = np.array([[occupation_list.index(str(row['occupation_code']))]])
    zip_idx = np.array([[zipcode_list.index(str(row['zip'])[:5])]])
    return {
        'gender': gender_idx,
        'age': age_idx,
        'occupation': occupation_idx,
        'zipcode': zip_idx
    }


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]


def generate(master_path):
    dataset_path = "movielens/ml-1m"
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = movielens_1m()

    # hashmap for item information
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            support_x_app = []
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = {}
                tmp_x_converted.update(movie_dict[m_id])
                tmp_x_converted.update(user_dict[u_id])
                # tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                support_x_app.append(tmp_x_converted)
                # try:
                #     support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                # except:
                #     support_x_app = tmp_x_converted
            support_x_app = np.array(support_x_app)

            query_x_app = []
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = {}
                tmp_x_converted.update(movie_dict[m_id])
                tmp_x_converted.update(user_dict[u_id])
                # tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                query_x_app.append(tmp_x_converted)
                # try:
                #     query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                # except:
                #     query_x_app = tmp_x_converted
            query_x_app = np.array(query_x_app)
            support_y_app = tmp_y[indices[:-10]]
            query_y_app = tmp_y[indices[-10:]]

            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1

if __name__ == '__main__':
    path = './datasets'
    if not os.path.isdir(path):
        os.makedirs(path)
    generate(path)
