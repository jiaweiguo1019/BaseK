import os
import pickle
import time
from collections import defaultdict
from functools import partial
from queue import Queue
from threading import Thread

import pandas as pd
from tqdm import tqdm

from basek.utils.tf_compat import keras
from basek.utils.imports import numpy as np


def read_reviews(file_path, true_sample_weight=1.0, neg_sampes=0, neg_sample_weight=1.0):

    abspath = os.path.abspath(file_path)
    dir_path = os.path.split(abspath)[0]
    train_file = os.path.join(dir_path, 'train')
    test_file = os.path.join(dir_path, 'test')

    all_lines = []
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    cate_count = defaultdict(int)
    item_to_cate = {'null': 'null', 'default_item': 'default_cate'}
    with open(file_path, 'r') as f:
        # for line in tqdm(f):
        for _ in tqdm(range(100000)):
            line = f.readline()
            line = line.strip().split(',')
            user, item, cate, behavior, timestamp = line
            user_count[user] += 1
            item_count[item] += 1
            cate_count[cate] += 1
            item_to_cate[item] = cate
            all_lines.append(((user, item, cate, behavior), float(timestamp)))
    start_ts = time.mktime(time.strptime('2017-11-25 0:0:0', "%Y-%m-%d %H:%M:%S"))
    end_ts = time.mktime(time.strptime('2017-12-4 0:0:0', "%Y-%m-%d %H:%M:%S"))
    all_lines = list(filter(lambda x: start_ts <= x[1] < end_ts, all_lines))
    all_lines.sort(key=lambda x: x[-1])
    user_size = len(user_count) + 2
    item_size = len(item_count) + 2
    cate_size = len(cate_count) + 2
    sparse_features_max_idx = {'uid': user_size, 'iid': item_size, 'cid': cate_size}
    serialized_sparse_features_max_idx = pickle.dumps(sparse_features_max_idx)
    with open(os.path.join(dir_path, 'sparse_features_max_idx'), 'wb') as f:
        f.write(serialized_sparse_features_max_idx)

    item_to_iid = {'null': 0, 'default_item': 1}
    iid_to_item = {0: 'null', 1: 'default_item'}
    item_count = dict(sorted(item_count.items(), key=lambda x: x[-1], reverse=True))
    for idx, asin in enumerate(item_count):
        item_to_iid[asin] = idx + 2
        iid_to_item[idx + 2] = asin
    item_to_iid = pd.Series(item_to_iid)
    iid_to_item = pd.Series(iid_to_item)
    item_to_iid.to_csv(os.path.join(dir_path, 'item_to_iid.csv'), sep='^', header=False)
    iid_to_item.to_csv(os.path.join(dir_path, 'iid_to_item.csv'), sep='^', header=False)

    cate_to_cid = {'null': 0, 'default_cate': 1}
    cid_to_cate = {0: 'null', 1: 'default_cate'}
    for idx, cate in enumerate(cate_count):
        cate_to_cid[cate] = idx + 2
        cid_to_cate[idx + 2] = cate
    cate_to_cid = pd.Series(cate_to_cid)
    cid_to_cate = pd.Series(cid_to_cate)
    cate_to_cid.to_csv(os.path.join(dir_path, 'cate_to_cid.csv'), sep='^', header=False)
    cid_to_cate.to_csv(os.path.join(dir_path, 'cid_to_cate.csv'), sep='^', header=False)

    iid_to_cid = {}
    for item, iid in item_to_iid.items():
        cate = item_to_cate[item]
        cid = cate_to_cid[cate]
        iid_to_cid[iid] = cid
    iid_to_cid = pd.Series(iid_to_cid)
    iid_to_cid.to_csv(os.path.join(dir_path, 'iid_to_cid.csv'), sep='^', header=False)
    all_iid_index = list(range(item_size))
    all_cid_index = list(iid_to_cid.loc[all_iid_index].values)
    all_indices = all_iid_index, all_cid_index
    serialized_all_indices = pickle.dumps(all_indices)
    with open(os.path.join(dir_path, 'all_indices'), 'wb') as f:
        f.write(serialized_all_indices)

    user_to_uid = {'null': 0, 'default_user': 1}
    uid_to_user = {0: 'null', 1: 'default_user'}
    user_count = dict(sorted(user_count.items(), key=lambda x: x[-1], reverse=True))
    for idx, user in enumerate(user_count):
        user_to_uid[user] = idx + 2
        uid_to_user[idx + 2] = user
    user_to_uid = pd.Series(user_to_uid)
    uid_to_user = pd.Series(uid_to_user)
    user_to_uid.to_csv(os.path.join(dir_path, 'user_to_uid.csv'), sep='^', header=False)
    uid_to_user.to_csv(os.path.join(dir_path, 'uid_to_user.csv'), sep='^', header=False)

    pos_label = str(1.0)
    true_sample_weight = str(true_sample_weight)
    neg_label = str(0.0)
    neg_sample_weight = str(neg_sample_weight)
    neg_rating = str(0.0)
    user_hist_iid_seq = defaultdict(str)
    user_hist_cid_seq = defaultdict(str)
    write_user_count = defaultdict(int)
    true_train_samples, neg_train_samples, test_samples = 0, 0, 0
    with open(train_file, 'w') as f_train, open(test_file, 'w') as f_test:
        print('-' * 32 + '    writing samples    ' + '-' * 32)
        # print(user_count)
        for line in tqdm(all_lines):
            line = line[0]
            user, asin, cate, rating = line
            hist_seq_len = str(write_user_count[user])
            write_user_count[user] += 1
            write_test = False
            if write_user_count[user] == user_count[user]:
                f_out = f_test
                test_samples += 1
                write_test = True
            else:
                f_out = f_train
                true_train_samples += 1
            hist_iid_seq = user_hist_iid_seq[user] + ' '
            hist_cid_seq = user_hist_cid_seq[user] + ' '
            uid = str(user_to_uid[user])
            iid = str(item_to_iid[asin])
            cid = str(cate_to_cid[cate])
            user_hist_iid_seq[user] = hist_iid_seq + iid
            user_hist_cid_seq[user] = hist_cid_seq + cid
            f_out.write(
                '\t'.join(
                    (
                        uid, iid, cid, pos_label, hist_iid_seq, hist_cid_seq,
                        hist_seq_len, true_sample_weight, rating
                    )
                ) + '\n'
            )
            if neg_sampes > 0 and not write_test:
                all_neg_samples_iid = set()
                for _ in range(neg_sampes):
                    neg_train_samples += 1
                    while True:
                        neg_sampe_iid = np.random.choice(item_size) + 2
                        if iid_to_item[neg_sampe_iid] != asin and neg_sampe_iid not in all_neg_samples_iid:
                            all_neg_samples_iid.add(neg_sampe_iid)
                            break
                    neg_item = iid_to_item[neg_sampe_iid]
                    cate = item_to_cate.get(neg_item, 'default_cate')
                    neg_sample_cid = cate_to_cid[cate]
                    neg_sampe_iid, neg_sample_cid = str(neg_sampe_iid), str(neg_sample_cid)

                    f_out.write(
                        '\t'.join(
                            (
                                uid, neg_sampe_iid, neg_sample_cid, neg_label,
                                hist_iid_seq, hist_cid_seq, hist_seq_len,
                                true_sample_weight, rating, neg_rating
                            )
                        ) + '\n'
                    )
    total_train_samples = true_train_samples + neg_train_samples
    print('=' * 120)
    print(
        '-' * 4 + f'  {total_train_samples} training samples, ' +
        f'{true_train_samples} of them are true_train_samples, ' +
        f'{neg_train_samples} of them are neg_train_samples  ' + '-' * 4
    )
    print('-' * 32 + f'    {test_samples} testing samples    ' + '-' * 32)

    return (train_file, test_file), sparse_features_max_idx, all_indices


class DataLoader():

    def __init__(
        self, file_path, batch_size,
        max_seq_len=100, prefetch_batch=10,
        shuffle=False, sort_by_length=False
    ):
        self.source = open(file_path, 'r')
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.prefetch_batch = prefetch_batch
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.buffer_size = self.batch_size * self.prefetch_batch
        self.__init()

    def __init(self):
        self.processor = partial(
            self.process_data,
            max_seq_len=self.max_seq_len, shuffle=self.shuffle, sort_by_length=self.sort_by_length
        )
        self.data_buffer = Queue(maxsize=self.prefetch_batch)
        self.reset()

    def reset(self):
        self.source.seek(0)
        self.data_loader = Thread(target=self.load_data)
        self.data_loader.daemon = True
        self.data_loader.start()

    def load_data(self):
        end_of_file = False
        while True:
            if self.data_buffer.qsize() >= self.prefetch_batch:
                continue
            batch_raw_data = []
            for _ in range(self.batch_size):
                line = self.source.readline()
                if not line:
                    end_of_file = True
                    break
                batch_raw_data.append(line)
            if batch_raw_data:
                batch_data = self.processor(batch_raw_data)
                self.data_buffer.put(batch_data)
            if end_of_file:
                break
        self.data_buffer.put(None)

    @staticmethod
    def process_data(batch_raw_data, max_seq_len, shuffle, sort_by_length):
        batch = [sample.strip().split('\t') for sample in batch_raw_data]
        if shuffle is True:
            np.random.shuffle(batch)
        if sort_by_length is True:
            batch.sort(key=lambda x: float(x[6]))
        uid = []
        iid = []
        cid = []
        label = []
        hist_iid_seq = []
        hist_cid_seq = []
        hist_seq_len = []
        sample_weight = []
        rating = []
        for sample in batch:
            uid.append(int(sample[0]))
            iid.append(int(sample[1]))
            cid.append(int(sample[2]))
            label.append(float(sample[3]))
            hist_iid_seq.append(
                list(
                    map(
                        lambda x: int(x) if x else 0,
                        sample[4].strip().split(' ')
                    )
                )
            )
            hist_cid_seq.append(
                list(
                    map(
                        lambda x: int(x) if x else 0,
                        sample[5].strip().split(' ')
                    )
                )
            )
            hist_seq_len.append(int(sample[6]))
            sample_weight.append(float(sample[7]))
            rating.append(sample[8])
        uid = np.array(uid).reshape(-1, 1)
        iid = np.array(iid).reshape(-1, 1)
        cid = np.array(cid).reshape(-1, 1)
        label = np.array(label).reshape(-1, 1)
        hist_iid_seq = keras.preprocessing.sequence.pad_sequences(
            hist_iid_seq, maxlen=max_seq_len, dtype='int64',
            padding='post', truncating='pre', value=0
        )
        hist_cid_seq = keras.preprocessing.sequence.pad_sequences(
            hist_cid_seq, maxlen=max_seq_len, dtype='int64',
            padding='post', truncating='pre', value=0
        )
        hist_seq_len = np.array(hist_seq_len).reshape(-1, 1)
        sample_weight = np.array(sample_weight).reshape(-1, 1)
        return uid, iid, cid, label, hist_iid_seq, hist_cid_seq, hist_seq_len, sample_weight, rating

    def __next__(self):
        while True:
            data = self.data_buffer.get()
            if not data:
                self.reset()
                raise StopIteration
            return data

    def __iter__(self):
        return self
