import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from basek.utils.tf_compat import keras


def read_csv(data_path):
    user_names = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv(
        data_path + '/users.dat', sep='::', header=None, names=user_names, encoding='latin-1'
    )
    rating_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        data_path + '/ratings.dat', sep='::', header=None, names=rating_names, encoding='latin-1'
    )
    movie_names = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(
        data_path + '/movies.dat', sep='::', header=None, names=movie_names, encoding='latin-1'
    )
    data = pd.merge(pd.merge(ratings, movies), users)
    return data


# def read_csv(data_path):
#     return pd.read_csv('./datasets/movielens_sample.txt')


def index_sparse_features(data, sparse_features):
    max_idx = {}
    for sp in sparse_features:
        lbe = LabelEncoder()
        data[sp] = lbe.fit_transform(data[sp]) + 1
        max_idx[sp] = data[sp].max() + 1
    return max_idx


def read_raw_data(data_path, sparse_features):
    data = read_csv(data_path)
    max_idx = index_sparse_features(data, sparse_features)
    return data, max_idx


def gen_dataset(data, neg_samples=0):

    data.sort_values('timestamp', inplace=True)
    item_ids = data['movie_id'].unique()

    # user_item_seq = data.groupby("user_id")['movie_id'].apply(list)
    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates('user_id')
    user_profile.set_index('user_id', inplace=True)
    item_profile = data[['movie_id']].drop_duplicates('movie_id')

    train_dataset = []
    test_dataset = []
    for uid, hist in tqdm(data.groupby('user_id')):
        pos_label = 1.0
        neg_label = 0.0
        hist_item_seq = hist['movie_id'].tolist()
        hist_rating_seq = hist['rating'].tolist()
        # TODO: sample negative samples once or per positive sample
        if neg_samples > 0:
            candidate_set = list(set(item_ids) - set(hist_item_seq))
            neg_list = np.random.choice(
                candidate_set, size=(len(hist_item_seq) - 1) * neg_samples
            )
        for i in range(1, len(hist_item_seq)):
            hist_item_sub_seq = hist_item_seq[:i]
            iid = hist_item_seq[i]
            if i != len(hist_item_seq) - 1:
                train_dataset.append(
                    (uid, iid, pos_label, hist_item_sub_seq[:], len(hist_item_sub_seq), hist_rating_seq[i])
                )
                for neg_i in range(neg_samples):
                    train_dataset.append(
                        (uid, neg_list[i * neg_samples + neg_i], neg_label, hist[::-1], 0,len(hist[::-1]))
                    )
            else:
                test_dataset.append(
                    (uid, iid, pos_label, hist_item_sub_seq[:], len(hist_item_sub_seq), hist_rating_seq[i])
                )

    np.random.shuffle(train_dataset)
    # np.random.shuffle(test_dataset)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    print(f'-------------------- {train_size} training samples, {test_size} testing samples. --------------------')

    return (train_dataset, test_dataset), (user_profile, item_profile)


def gen_model_input(dataset, user_profile, item_profile, seq_max_len, validaton_split=None):

    uid = np.array([[line[0]] for line in dataset])
    iid = np.array([[line[1]] for line in dataset])
    label = np.array([[line[2]] for line in dataset])

    hist_item_seq = [line[3] for line in dataset]
    hist_item_len = np.array([[line[4]] for line in dataset])

    padded_hist_item_seq = keras.preprocessing.sequence.pad_sequences(
        hist_item_seq, maxlen=seq_max_len, padding='post', truncating='pre', value=0, dtype='int64'
    )

    gender = user_profile.loc[uid.reshape(-1)]['gender'].values.reshape(-1, 1)
    age = user_profile.loc[uid.reshape(-1)]['age'].values.reshape(-1, 1)
    occupation = user_profile.loc[uid.reshape(-1)]['gender'].values.reshape(-1, 1)
    zip = user_profile.loc[uid.reshape(-1)]['gender'].values.reshape(-1, 1)

    total_size = uid.shape[0]

    if validaton_split is not None:
        if not isinstance(validaton_split, float) or validaton_split <= 0.0:
            raise ValueError(
                'validaton_split should be a float in the (0, 1) range!'
            )
        data_idx = np.arange(total_size)
        train_idx, val_idx = train_test_split(data_idx, test_size=validaton_split, shuffle=False)
        train_dataset = {
            'uid': uid[train_idx], 'iid': iid[train_idx], 'label': label[train_idx],
            'hist_item_seq': padded_hist_item_seq[train_idx], 'hist_item_len': hist_item_len[train_idx],
            'gender': gender[train_idx], 'age': age[train_idx],
            'occupation': occupation[train_idx], 'zip': zip[train_idx],
            'size': len(train_idx)
        }
        val_dataset = {
            'uid': uid[val_idx], 'iid': iid[val_idx], 'label': label[val_idx],
            'hist_item_seq': padded_hist_item_seq[val_idx], 'hist_item_len': hist_item_len[val_idx],
            'gender': gender[val_idx], 'age': age[val_idx],
            'occupation': occupation[val_idx], 'zip': zip[val_idx],
            'size': len(val_idx)
        }

    else:
        train_dataset = {
            'uid': uid, 'iid': iid, 'label': label,
            'hist_item_seq': padded_hist_item_seq, 'hist_item_len': hist_item_len,
            'gender': gender, 'age': age,
            'occupation': occupation, 'zip': zip,
            'size': total_size
        }
        val_dataset = {'size': 0}
    return train_dataset, val_dataset

