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


def split_dataset(dataset, validaton_split):
    if not isinstance(validaton_split, float) or validaton_split <= 0.0:
        raise ValueError('validaton_split should be a float in the (0, 1) range!')
    total_size = len(dataset)
    train_size = int(np.ceil(total_size * (1.0 - validaton_split)))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    return train_dataset, val_dataset


def gen_dataset(data, validaton_split=None, neg_samples=0, neg_weight=1.0):

    data.sort_values('timestamp', inplace=True)
    item_ids = data['movie_id'].unique()

    # user_item_seq = data.groupby("user_id")['movie_id'].apply(list)
    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates('user_id')
    user_profile.set_index('user_id', inplace=True)
    item_profile = data[['movie_id']].drop_duplicates('movie_id')

    train_dataset = []
    neg_sample_dataset = []
    test_dataset = []
    pos_label = 1.0
    neg_label = 0.0
    pos_weight = 1.0
    neg_weight = neg_weight
    for uid, hist in tqdm(data.groupby('user_id')):
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
                    (
                        uid, iid, pos_label,
                        hist_item_sub_seq[:], len(hist_item_sub_seq), pos_weight, hist_rating_seq[i]
                    )
                )
                for neg_i in range(neg_samples):
                    neg_sample_dataset.append(
                        (
                            uid, neg_list[i * neg_samples + neg_i], neg_label,
                            hist_item_sub_seq[:], len(hist_item_sub_seq), neg_weight
                        )
                    )
            else:
                test_dataset.append(
                    (
                        uid, iid, pos_label,
                        hist_item_sub_seq[:], len(hist_item_sub_seq), pos_weight, hist_rating_seq[i]
                    )
                )

    np.random.shuffle(train_dataset)
    np.random.shuffle(neg_sample_dataset)
    # np.random.shuffle(test_dataset)

    if validaton_split is not None:
        train_dataset, val_dataset = split_dataset(train_dataset, validaton_split)
    else:
        train_dataset, val_dataset = train_dataset, []

    train_dataset = train_dataset + neg_sample_dataset
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print('-' * 120)
    print('-' * 16 + f'  {train_size} training samples, {val_size} validation samples, {test_size} testing samples.    ' + '-' * 16)

    return (train_dataset, val_dataset, test_dataset), (user_profile, item_profile)


def gen_model_input(dataset, user_profile, item_profile, seq_max_len, ):

    total_size = len(dataset)
    if total_size == 0:
        return {'size': 0}

    uid = np.array([[line[0]] for line in dataset])
    iid = np.array([[line[1]] for line in dataset])
    label = np.array([[line[2]] for line in dataset])

    hist_item_seq = [line[3] for line in dataset]
    hist_item_len = np.array([[line[4]] for line in dataset])
    sample_weight = np.array([[line[5]] for line in dataset])

    padded_hist_item_seq = keras.preprocessing.sequence.pad_sequences(
        hist_item_seq, maxlen=seq_max_len, padding='post', truncating='pre', value=0, dtype='int64'
    )

    gender = user_profile.loc[uid.reshape(-1)]['gender'].values.reshape(-1, 1)
    age = user_profile.loc[uid.reshape(-1)]['age'].values.reshape(-1, 1)
    occupation = user_profile.loc[uid.reshape(-1)]['gender'].values.reshape(-1, 1)
    zip = user_profile.loc[uid.reshape(-1)]['gender'].values.reshape(-1, 1)

    model_input = {
        'uid': uid, 'iid': iid, 'label': label,
        'hist_item_seq': padded_hist_item_seq, 'hist_item_len': hist_item_len,
        'gender': gender, 'age': age,
        'occupation': occupation, 'zip': zip,
        'sample_weight': sample_weight, 'size': total_size
    }

    return model_input

