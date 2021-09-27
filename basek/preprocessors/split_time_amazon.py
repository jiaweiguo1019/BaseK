import os
import networkx as nx
import pickle as pkl
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Queue, Process

import pandas as pd
from tqdm import tqdm

from basek.utils.tf_compat import keras
from basek.utils.imports import numpy as np


def read_raw_dataset(
    meta_path, review_path,
    item_to_cate_path='./item_to_cate_path.pkl',
    raw_dataset_path='./raw_dataset_path.pkl'
):

    item_to_cate = {'null': 'null', 'default': 'default'}
    with open(meta_path, 'r') as f:
        for line in tqdm(f):
            line = eval(line)
            item = str(line['asin'])
            cate = str(line['categories'][-1][-1])
            item_to_cate[item] = cate
    item_to_cate = pd.Series(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)

    with open(review_path, 'r') as f:
        data = {}
        for i, line in tqdm(enumerate(f)):
            line = eval(line)
            user = str(line['reviewerID'])
            item = str(line['asin'])
            rating = float(line['overall'])
            timestamp = int(line['unixReviewTime'])
            cate = item_to_cate[item]
            data[i] = {
                'user': user,
                'item': item,
                'cate': cate,
                'rating': rating,
                'timestamp': timestamp
            }
    raw_dataset_df = pd.DataFrame.from_dict(data, orient='index')
    raw_dataset_df.drop_duplicates(['user', 'item', 'timestamp'], inplace=True)
    raw_dataset_df.to_pickle(raw_dataset_path)

    return item_to_cate, raw_dataset_df


def reduce_to_k_core(dataset_df, k_core):

    edges = list(dataset_df[['user', 'item']].to_records(index=False))
    edges = list(map(lambda edge: (f'user^{edge[0]}', f'item^{edge[1]}'), edges))
    g = nx.Graph()
    g.add_edges_from(edges)
    user_and_item = list(nx.k_core(g, k_core).nodes)
    if not user_and_item:
        raise ValueError(f'k_core numbser: {k_core} is too much, no more interactons left')

    user_set = set(filter(lambda x: x.startswith('user'), user_and_item))
    item_set = set(filter(lambda x: x.startswith('item'), user_and_item))
    user_set = set(map(lambda x: x.split('^')[-1], user_set))
    item_set = set(map(lambda x: x.split('^')[-1], item_set))
    filter_idx = dataset_df['user'].isin(user_set) & dataset_df['item'].isin(item_set)

    return dataset_df[filter_idx].copy()


def process_raw_dataset(
    raw_dataset_df,
    processed_raw_dataset_path='./processed_raw_dataset.pkl',
    drop_dups=True, k_core=None
):

    if drop_dups is True:
        processed_raw_dataset_df = raw_dataset_df.drop_duplicates(['user', 'item']).copy()
    else:
        processed_raw_dataset_df = raw_dataset_df

    if k_core is not None:
        if not isinstance(k_core, int) or k_core <= 1:
            raise ValueError(f'k_core should be an integer greater than 1, got {k_core}')
        processed_raw_dataset_df = reduce_to_k_core(processed_raw_dataset_df, k_core)

    processed_raw_dataset_df.sort_values('timestamp', inplace=True)
    processed_raw_dataset_df.index = list(range(len(raw_dataset_df)))
    processed_raw_dataset_df.to_pickle(processed_raw_dataset_path)

    return processed_raw_dataset_df


def build_entity_id_map(entity_count, entity_prefix='default', id_prefix='default', dirpath='./'):

    entity_to_id_path = os.path.join(dirpath, f'{entity_prefix}_to_{id_prefix}.pkl')
    id_to_entity_path = os.path.join(dirpath, f'{id_prefix}_to_{entity_prefix}.pkl')

    entity_to_id = {'null': 0, 'default': 1}
    id_to_entity = {0: 'null', 1: 'default'}
    entity_count = dict(sorted(entity_count.items(), key=lambda x: x[-1], reverse=True))
    for idx, entity in enumerate(entity_count):
        entity_to_id[entity] = idx + 2
        id_to_entity[idx + 2] = entity

    entity_to_id = pd.Series(entity_to_id)
    id_to_entity = pd.Series(id_to_entity)
    entity_to_id.to_pickle(entity_to_id_path)
    id_to_entity.to_pickle(id_to_entity_path)

    return entity_to_id, id_to_entity


def read_reviews(
    meta_path, review_path, split_ratio=0.1,
    neg_sampes=0, unique_neg_sample=True,
    true_sample_weight=1.0, neg_sample_weight=1.0,
    pos_label=1.0, neg_label=0.0, neg_sample_rating=0.0,
    drop_dups=True, k_core=None
):
    abspath = os.path.abspath(review_path)
    dirpath = os.path.split(abspath)[0]
    raw_dataset_path = os.path.join(dirpath, 'raw_dataset.pkl')
    item_to_cate_path = os.path.join(dirpath, 'item_to_cate.pkl')
    processed_raw_dataset_path = os.path.join(dirpath, 'processed_raw_dataset.pkl')
    all_indices_path = os.path.join(dirpath, 'all_indices.pkl')
    sparse_features_max_idx_path = os.path.join(dirpath, 'sparse_features_max_idx.pkl')
    train_path, test_path = \
        os.path.join(dirpath, 'train.pkl'), os.path.join(dirpath, 'test.pkl')

    item_to_cate, raw_dataset_df,  = \
        read_raw_dataset(meta_path, review_path, item_to_cate_path, raw_dataset_path)
    # item_to_cate, raw_dataset_df = \
    #    pd.read_pickle(item_to_cate_path), pd.read_pickle(raw_dataset_path)
    processed_raw_dataset_df = \
        process_raw_dataset(raw_dataset_df, processed_raw_dataset_path, drop_dups, k_core)
    # processed_raw_dataset_df = \
    #    pd.read_pickle(processed_raw_dataset_path)

    user_count = processed_raw_dataset_df.groupby('user')['user'].count()
    item_count = processed_raw_dataset_df.groupby('item')['item'].count()
    cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count()
    user_to_uid, uid_to_user = build_entity_id_map(user_count, 'user', 'uid', dirpath)
    item_to_iid, iid_to_item = build_entity_id_map(item_count, 'item', 'iid', dirpath)
    cate_to_cid, cid_to_cate = build_entity_id_map(cate_count, 'cate', 'cid', dirpath)

    uid_count = {}
    for user, count in user_count.items():
        uid = user_to_uid[user]
        uid_count[uid] = count
        iid_to_cid = {}
    uid_count = pd.Series(uid_count)
    for item, iid in item_to_iid.items():
        cate = item_to_cate[item]
        cid = cate_to_cid[cate]
        iid_to_cid[iid] = cid
    iid_to_cid = pd.Series(iid_to_cid)

    uid_size, iid_size, cid_size = len(uid_to_user), len(iid_to_item), len(cid_to_cate)
    sparse_features_max_idx = {'uid': uid_size, 'iid': iid_size, 'cid': cid_size}
    with open(sparse_features_max_idx_path, 'wb') as f:
        pkl.dump(sparse_features_max_idx, f)

    all_iid_index = list(iid_to_item.index)
    all_cid_index = list(iid_to_cid[all_iid_index])
    all_indices = all_iid_index, all_cid_index
    with open(all_indices_path, 'wb') as f:
        pkl.dump(all_indices, f)

    converted_dataset_df = pd.DataFrame()
    converted_dataset_df['uid'] = processed_raw_dataset_df['user'].map(user_to_uid)
    converted_dataset_df['iid'] = processed_raw_dataset_df['item'].map(item_to_iid)
    converted_dataset_df['cid'] = processed_raw_dataset_df['cate'].map(cate_to_cid)
    converted_dataset_df['rating'] = processed_raw_dataset_df['rating']

    split_idx = int(len(converted_dataset_df) * (1.0 - split_ratio))
    train_df = converted_dataset_df.iloc[:split_idx]
    test_df = converted_dataset_df.iloc[split_idx:]

    curr_uid_count = defaultdict(int)
    hist_iid_seq = defaultdict(list)
    hist_cid_seq = defaultdict(list)
    hist_rating_seq = defaultdict(list)
    num_true_samples, num_neg_samples, num_test_samples = 0, 0, 0

    with open(train_path, 'wb') as f:
        print('-' * 32 + '    writing traing samples    ' + '-' * 32)
        for row in tqdm(train_df.itertuples()):
            _, uid, iid, cid, rating = row

            uid_hist_seq_len = curr_uid_count[uid]
            curr_uid_count[uid] += 1
            uid_hist_iid_seq = hist_iid_seq[uid]
            uid_hist_cid_seq = hist_cid_seq[uid]
            uid_hist_rating_seq = hist_rating_seq[uid]
            hist_iid_seq[uid] = uid_hist_iid_seq + [iid]
            hist_cid_seq[uid] = uid_hist_cid_seq + [cid]
            hist_rating_seq[uid] = uid_hist_rating_seq + [rating]
            train_sample = (
                uid, iid, cid, rating, pos_label,
                uid_hist_iid_seq, uid_hist_cid_seq, uid_hist_rating_seq, uid_hist_seq_len,
                true_sample_weight
            )
            pkl.dump(train_sample, f)
            num_true_samples += 1

            if neg_sampes == 0:
                continue
            neg_samples_iid = set()
            neg_samples_cid = set()
            for _ in range(neg_sampes):
                while True:
                    neg_iid = np.random.choice(iid_size)
                    if neg_iid == iid:
                        continue
                    else:
                        if unique_neg_sample and neg_iid in neg_samples_iid:
                            continue
                        else:
                            neg_samples_iid.add(neg_iid)
                            neg_cid = iid_to_cid[neg_iid]
                            neg_samples_cid.add(cid)
                            break
                neg_train_sample = (
                    uid, neg_iid, neg_cid, neg_sample_rating, neg_label,
                    uid_hist_iid_seq, uid_hist_cid_seq, uid_hist_rating_seq, uid_hist_seq_len,
                    neg_sample_weight
                )
                pkl.dump(neg_train_sample, f)
                num_neg_samples += 1

        total_train_samples = num_true_samples + num_neg_samples

    with open(test_path, 'wb') as f:
        for uid, u_hist in test_df.groupby('uid'):
            uid_hist_seq_len = curr_uid_count[uid]
            uid_hist_iid_seq = hist_iid_seq[uid]
            uid_hist_cid_seq = hist_cid_seq[uid]
            uid_hist_rating_seq = hist_rating_seq[uid]
            groud_truth_iid_seq = u_hist['iid'].tolist()
            groud_truth_cid_seq = u_hist['cid'].tolist()
            groud_truth_rating_seq = u_hist['raing'].tolist()
            num_test_samples += len(groud_truth_iid_seq)
            test_sample = (
                uid, uid_hist_iid_seq, uid_hist_cid_seq, uid_hist_rating_seq, uid_hist_seq_len
            )
            hist_iid_rating_map = dict(zip(uid_hist_iid_seq, uid_hist_rating_seq))
            hist_cid_rating_map = dict(zip(uid_hist_cid_seq, uid_hist_rating_seq))
            groud_truth_iid_rating_map = dict(zip(groud_truth_iid_seq, groud_truth_rating_seq))
            groud_truth_cid_rating_map = dict(zip(groud_truth_cid_seq, groud_truth_rating_seq))
            groud_truth = (
                groud_truth_iid_rating_map,
                groud_truth_cid_rating_map,
                hist_iid_rating_map,
                hist_cid_rating_map
            )
            pkl.dump((test_sample, groud_truth), f)

    print('=' * 120)
    print(
        '-' * 8 + f'  {total_train_samples} training samples, ' +
        f'{num_true_samples} of them are true_samples, ' +
        f'{num_neg_samples} of them are neg_samples  ' + '-' * 8
    )
    print('-' * 32 + f'    {num_test_samples} testing samples    ' + '-' * 32)

    dataset_files = train_path, test_path
    return dataset_files, sparse_features_max_idx_path, all_indices_path


def process_data(batch_data, max_seq_len, shuffle, sort_by_length, kind):
    if shuffle is True:
        np.random.shuffle(batch_data)
    if sort_by_length is True:
        batch_data.sort(key=lambda x: float(x[6]))
    uid, iid, cid, label, hist_iid_seq, hist_cid_seq, hist_seq_len, rating, sample_weight, ground_truth = \
        [], [], [], [], [], [], [], [], []
    if kind == 'train':
        for sample in batch_data:
            uid.append(sample[0])
            iid.append(sample[1])
            cid.append(sample[2])
            label.append(sample[3])
            hist_iid_seq.append(sample[4])
            hist_cid_seq.append(sample[5])
            hist_seq_len.append(sample[6])
            rating.append(sample[7])
            sample_weight.append(sample[8])
    else:
        for sample, per_ground_truth in batch_data:
            uid.append(sample[0])
            hist_iid_seq.append(sample[1])
            hist_cid_seq.append(sample[2])
            hist_seq_len.append(sample[3])
            ground_truth.append(per_ground_truth)
    uid = np.array(uid).reshape(-1, 1).astype(np.int64)
    iid = np.array(iid).reshape(-1, 1).astype(np.int64)
    cid = np.array(cid).reshape(-1, 1).astype(np.int64)
    label = np.array(label).reshape(-1, 1).astype(np.float32)
    hist_iid_seq = keras.preprocessing.sequence.pad_sequences(
        hist_iid_seq, maxlen=max_seq_len, dtype='int64',
        padding='post', truncating='pre', value=0
    ).astype(np.int64)
    hist_cid_seq = keras.preprocessing.sequence.pad_sequences(
        hist_cid_seq, maxlen=max_seq_len, dtype='int64',
        padding='post', truncating='pre', value=0
    ).astype(np.int64)
    hist_seq_len = np.array(hist_seq_len).reshape(-1, 1).astype(np.int64)
    rating = np.array(rating).reshape(-1, 1).astype(np.float32)
    sample_weight = np.array(sample_weight).reshape(-1, 1).astype(np.float32)
    if kind == 'train':
        return uid, iid, cid, label, hist_iid_seq, hist_cid_seq, hist_seq_len, rating, sample_weight
    else:
        return (uid, hist_iid_seq, hist_cid_seq, hist_seq_len), ground_truth


class DataLoader():

    def __init__(
        self, source, batch_size,
        max_seq_len=100, prefetch_batches=100,
        shuffle=True, sort_by_length=True,
        max_workers=8, kind='train'
    ):
        self.source = source
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.prefetch_batches = prefetch_batches
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.max_workers = max_workers
        kind = kind.lower()
        if kind not in ('train', 'test'):
            raise ValueError(f'unknown kind for dataloader: {kind}!')
        self.kind = kind

        self.__init()

    def __init(self):
        self.buffer_size = self.batch_size * self.prefetch_batches
        self.data_buffer = Queue()
        self.reset()

    def reset(self):
        self.data_loader = Process(
            target=self.load_data,
            args=(
                self.source, self.data_buffer, self.batch_size, self.prefetch_batches,
                self.max_seq_len, self.shuffle, self.sort_by_length, self.max_workers, self.kind
            )
        )
        self.data_loader.daemon = True
        self.data_loader.start()

    @staticmethod
    def load_data(
        source, data_buffer, batch_size, prefetch_batches,
        max_seq_len, shuffle, sort_by_length, max_workers, kind
    ):
        exectuor = ThreadPoolExecutor(max_workers=max_workers)
        source = open(source, 'rb')
        buffer_size = batch_size * prefetch_batches
        processor = partial(
            process_data,
            max_seq_len=max_seq_len, shuffle=shuffle, sort_by_length=sort_by_length, kind=kind
        )
        while True:
            if data_buffer.qsize() >= prefetch_batches * max_workers:
                continue
            tasks_submitted = []
            for _ in range(10):
                raw_data = []
                for _ in range(buffer_size):
                    try:
                        sample = pkl.load(source)
                        raw_data.append(sample)
                    except EOFError:
                        break
                if raw_data:
                    tasks_submitted.append(exectuor.submit(processor, raw_data))
                else:
                    break
            if not tasks_submitted:
                break
            all_batch_data = []
            for task in tasks_submitted:
                processed_data = task.result()
                data_size = len(processed_data[0])
                for i in range(0, data_size, batch_size):
                    batch_data = []
                    for data in processed_data:
                        batch_data.append(data[i: i + batch_size])
                    batch_data = tuple(batch_data)
                    all_batch_data.append(batch_data)
            for batch_data in all_batch_data:
                data_buffer.put(batch_data)

        data_buffer.put(None)
        source.close()

    def __next__(self):
        while True:
            data = self.data_buffer.get()
            if not data:
                self.reset()
                raise StopIteration
            return data

    def __iter__(self):
        return self
