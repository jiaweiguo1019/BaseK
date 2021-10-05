import os
import networkx as nx
import pickle as pkl
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Process

import pandas as pd
from tqdm import tqdm

from basek.utils.tf_compat import tf
from basek.utils.imports import numpy as np


def read_raw_dataset(review_path, item_to_cate_path, raw_dataset_path):
    if os.path.exists(item_to_cate_path) and os.path.exists(raw_dataset_path):
        print('read_review finished!')
        return

    raw_dataset_df = pd.read_csv(review_path, names=['user', 'item', 'cate', 'behavior', 'timestamp'])
    start_ts = int(time.mktime(time.strptime('2017-11-25 0:0:0', "%Y-%m-%d %H:%M:%S")))
    end_ts = int(time.mktime(time.strptime('2017-12-4 0:0:0', "%Y-%m-%d %H:%M:%S")))
    raw_dataset_df = raw_dataset_df[
        (raw_dataset_df['timestamp'] >= start_ts) & (raw_dataset_df['timestamp'] < end_ts)
    ]
    raw_dataset_df.fillna('default', inplace=True)
    raw_dataset_df.drop_duplicates(['user', 'item', 'behavior', 'timestamp'], inplace=True)
    raw_dataset_df.sort_values('timestamp', inplace=True)
    raw_dataset_df.reset_index(drop=True, inplace=True)

    item_to_cate = dict(zip(raw_dataset_df['item'], raw_dataset_df['cate']))
    item_to_cate.update({'null': 'null', 'default': 'default'})
    item_to_cate = pd.Series(item_to_cate)

    raw_dataset_df['cate'] = raw_dataset_df['item'].map(item_to_cate)
    item_to_cate.to_pickle(item_to_cate_path)
    raw_dataset_df.to_pickle(raw_dataset_path)

    print('read_review finished !')
    return


def reduce_to_k_core(dataset_df, dirpath, file_prefix, k_core):
    reduced_dataset_path = os.path.join(dirpath, f'{file_prefix}-reduced_dataset.pkl')
    if os.path.exists(reduced_dataset_path):
        reduced_dataset_df = pd.read_pickle(reduced_dataset_path)
        print(f'reduce_to_{k_core}_core finished!')
        return reduced_dataset_df

    user_list = dataset_df['user'].tolist()
    item_list = dataset_df['item'].tolist()
    user_offset = np.max(user_list) + 1
    item_offset = np.max(item_list) + 1
    edges = list(zip(np.array(user_list) - user_offset, np.array(item_list) + item_offset))

    g = nx.Graph()
    g.add_edges_from(edges)
    user_and_item = list(nx.k_core(g, k_core).nodes)
    if not user_and_item:
        raise ValueError(f'k_core numbser: {k_core} is too much, no more interactons left')

    reduced_user_list = np.array(list(filter(lambda x: x < 0, user_and_item))) + user_offset
    reduced_item_list = np.array(list(filter(lambda x: x > 0, user_and_item))) - item_offset
    reduced_dataset_df = dataset_df[
        dataset_df['user'].isin(set(reduced_user_list)) & dataset_df['item'].isin(set(reduced_item_list))
    ]

    print(f'reduce_to_{k_core}_core finished!')
    return reduced_dataset_df


def process_raw_dataset(
    raw_dataset_path,
    first_lines=None,
    dirpath='./',
    file_prefix='',
    drop_dups=False,
    only_click=False,
    k_core=None,
    min_hist_seq_len=None,
):
    if not file_prefix:
        processed_raw_dataset_path = os.path.join(dirpath, 'processed_raw_dataset.pkl')
    else:
        processed_raw_dataset_path = os.path.join(dirpath, f'{file_prefix}-processed_raw_dataset.pkl')

    if os.path.exists(processed_raw_dataset_path):
        processed_raw_dataset_df = pd.read_pickle(processed_raw_dataset_path)
        user_count = processed_raw_dataset_df.groupby('user')['user'].count()
        item_count = processed_raw_dataset_df.groupby('item')['item'].count().to_dict()
        cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count().to_dict()
        behavior_count = processed_raw_dataset_df.groupby('behavior')['behavior'].count().to_dict()
        print('process_raw_dataset finished!')
        return processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count

    processed_raw_dataset_df = pd.read_pickle(raw_dataset_path)
    if first_lines:
        processed_raw_dataset_df = processed_raw_dataset_df.iloc[:first_lines]
    if only_click:
        processed_raw_dataset_df = processed_raw_dataset_df[processed_raw_dataset_df['behavior'] == 'pv']
    if drop_dups is True:
        processed_raw_dataset_df.drop_duplicates(['user', 'item'], inplace=True)

    if k_core is not None:
        if not isinstance(k_core, int) or k_core <= 1:
            raise ValueError(f'k_core should be an integer greater than 1, got {k_core}')
        processed_raw_dataset_df = reduce_to_k_core(processed_raw_dataset_df, dirpath, file_prefix, k_core)

    user_count = processed_raw_dataset_df.groupby('user')['user'].count()
    if min_hist_seq_len is not None:
        user_count = user_count[user_count >= min_hist_seq_len]
        user_set = set(user_count.keys())
        processed_raw_dataset_df = processed_raw_dataset_df[processed_raw_dataset_df['user'].isin(user_set)]
    processed_raw_dataset_df.sort_values('timestamp', inplace=True)
    processed_raw_dataset_df.reset_index(drop=True, inplace=True)
    item_count = processed_raw_dataset_df.groupby('item')['item'].count().to_dict()
    cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count().to_dict()
    behavior_count = processed_raw_dataset_df.groupby('behavior')['behavior'].count().to_dict()

    print('process_raw_dataset finished!')
    return processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count


def build_entity_id_map(
    entity_count, entity_prefix='default',
    id_prefix='default', dirpath='./', file_prefix='', id_ordered_by_count=True
):
    if id_ordered_by_count:
        entity_count = dict(sorted(entity_count.items(), key=lambda x: x[-1], reverse=True))
    else:
        entity_count = dict(entity_count.items())
    entity_to_id_path = os.path.join(dirpath, f'{file_prefix}-{entity_prefix}_to_{id_prefix}.pkl')
    id_to_entity_path = os.path.join(dirpath, f'{file_prefix}-{id_prefix}_to_{entity_prefix}.pkl')

    default_count = entity_count.get('default', 0)
    if 'default' in entity_count:
        del entity_count['default']
    entity_to_id = {'null': 0, 'default': 1}
    id_to_entity = {0: 'null', 1: 'default'}
    for idx, entity in enumerate(entity_count):
        entity_to_id[entity] = idx + 2
        id_to_entity[idx + 2] = entity
    entity_count['default'] = default_count
    entity_to_id = dict(sorted(entity_to_id.items(), key=lambda x: x[1]))
    id_to_entity = dict(sorted(id_to_entity.items(), key=lambda x: x[0]))

    entity_to_id = pd.Series(entity_to_id)
    id_to_entity = pd.Series(id_to_entity)
    entity_to_id.to_pickle(entity_to_id_path)
    id_to_entity.to_pickle(id_to_entity_path)

    return entity_to_id, id_to_entity


def read_reviews(
    review_path, seq_len, from_raw=False,
    first_lines=None,
    drop_dups=False, only_click=False, k_core=None,  min_seq_len=None,
    id_ordered_by_count=False,
    neg_samples=None,
    test_drop_hist=None
):
    file_prefix = ''
    if first_lines:
        file_prefix = f'{file_prefix}-first_lines_{first_lines}'
    if drop_dups:
        file_prefix = f'{file_prefix}-drop_dups'
    if only_click:
        file_prefix = f'{file_prefix}-only_click'
    if k_core:
        file_prefix = f'{file_prefix}-k_core_{k_core}'
    if min_seq_len:
        file_prefix = f'{file_prefix}-min_seq_len_{min_seq_len}'
    if id_ordered_by_count:
        file_prefix = f'{file_prefix}-id_ordered_by_count'
    if file_prefix.startswith('-'):
        file_prefix = file_prefix[1:]

    abspath = os.path.abspath(review_path)
    dirpath = os.path.split(abspath)[0]
    item_to_cate_path = os.path.join(dirpath, f'item_to_cate.pkl')
    raw_dataset_path = os.path.join(dirpath, f'raw_dataset.pkl')

    uid_count_path = os.path.join(dirpath, f'{file_prefix}-uid_count.pkl')
    iid_to_cid_path = os.path.join(dirpath, f'{file_prefix}-iid_to_cid.pkl')
    bid_sample_tempature_path = os.path.join(dirpath, f'{file_prefix}-bid_sample_tempature.pkl')

    sparse_features_max_idx_path = os.path.join(dirpath, f'{file_prefix}-sparse_features_max_idx.pkl')
    all_indices_path = os.path.join(dirpath, f'{file_prefix}-all_indices.pkl')

    converted_dataset_df_path = os.path.join(dirpath, f'{file_prefix}-converted_dataset_df.pkl')
    train_dataset_df_path = os.path.join(dirpath, f'{file_prefix}-train_dataset_df.pkl')
    test_dataset_df_path = os.path.join(dirpath, f'{file_prefix}-test_dataset_df.pkl')

    tf_records_prefix = f'seq_len_{seq_len}'
    if neg_samples:
        tf_records_prefix = f'{tf_records_prefix}-neg_samples_{neg_samples}'
    if test_drop_hist:
        tf_records_prefix = f'{tf_records_prefix}-test_drop_hist'
    if file_prefix:
        tf_records_prefix = f'{tf_records_prefix}-{file_prefix}'
    train_tfrecords_path = os.path.join(dirpath, f'{tf_records_prefix}-train.tfrecords')
    test_tfrecords_path = os.path.join(dirpath, f'{tf_records_prefix}-test.tfrecords')

    return_str = '#' * 132 + '\n' + \
        'data_files:\n' + \
        '\ttrain_records_path:\n' + \
        f'\t\t{train_tfrecords_path}\n' + \
        '\test_records_path:\n' + \
        f'\t\t{test_tfrecords_path}\n' + \
        'sparse_features_max_idx_path:\n' + \
        f'\t{sparse_features_max_idx_path}\n' + \
        'all_indices_path:\n' + \
        f'\t{all_indices_path}\n' + \
        '#' * 132

    if not from_raw:
        print(return_str)
        return

    read_raw_dataset(review_path, item_to_cate_path, raw_dataset_path)
    item_to_cate = pd.read_pickle(item_to_cate_path)
    processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count = process_raw_dataset(
        raw_dataset_path, first_lines, dirpath, file_prefix, drop_dups=drop_dups, only_click=only_click, k_core=k_core
    )

    user_to_uid, uid_to_user = \
        build_entity_id_map(user_count, 'user', 'uid', dirpath, file_prefix, id_ordered_by_count)
    item_to_iid, iid_to_item = \
        build_entity_id_map(item_count, 'item', 'iid', dirpath, file_prefix, id_ordered_by_count)
    cate_to_cid, cid_to_cate = \
        build_entity_id_map(cate_count, 'cate', 'cid', dirpath, file_prefix, id_ordered_by_count)
    behavior_to_bid, bid_to_behavior = \
        build_entity_id_map(behavior_count, 'behavior', 'bid', dirpath, file_prefix, id_ordered_by_count)

    uid_count = {}
    for user, count in user_count.items():
        uid = user_to_uid[user]
        uid_count[uid] = count
    uid_count = pd.Series(uid_count)
    uid_count.to_pickle(uid_count_path)

    iid_to_cid = {}
    for item, iid in item_to_iid.items():
        cate = item_to_cate[item]
        cid = cate_to_cid[cate]
        iid_to_cid[iid] = cid
    iid_to_cid = pd.Series(iid_to_cid)
    iid_to_cid.to_pickle(iid_to_cid_path)

    bid_sample_tempature = dict()
    high_priority_behavior_set = set(['buy', 'cart', 'fav'])
    for beavior, bid in behavior_to_bid.items():
        if beavior in high_priority_behavior_set:
            bid_sample_tempature[bid] = 2
        else:
            bid_sample_tempature[bid] = 1
    bid_sample_tempature = pd.Series(bid_sample_tempature)
    bid_sample_tempature.to_pickle(bid_sample_tempature_path)

    uid_size, iid_size, cid_size, bid_size = \
        len(uid_to_user), len(iid_to_item), len(cid_to_cate), len(bid_to_behavior)
    sparse_features_max_idx = {'uid': uid_size, 'iid': iid_size, 'cid': cid_size, 'bid': bid_size}
    print('#' * 132)
    print('-' * 16 + f'    uid_size: {uid_size}, iid_size: {iid_size}, ' + f'cid_size: {cid_size}    ' + '-' * 16)
    print('#' * 132)
    with open(sparse_features_max_idx_path, 'wb') as f:
        pkl.dump(sparse_features_max_idx, f)

    all_iid_index = list(iid_to_item.keys())
    all_cid_index = list(iid_to_cid[all_iid_index])
    all_indices = all_iid_index, all_cid_index
    with open(all_indices_path, 'wb') as f:
        pkl.dump(all_indices, f)

    converted_dataset_df = pd.DataFrame()
    converted_dataset_df['uid'] = processed_raw_dataset_df['user'].map(user_to_uid)
    converted_dataset_df['iid'] = processed_raw_dataset_df['item'].map(item_to_iid)
    converted_dataset_df['cid'] = processed_raw_dataset_df['cate'].map(cate_to_cid)
    converted_dataset_df['bid'] = processed_raw_dataset_df['behavior'].map(behavior_to_bid)
    converted_dataset_df['timestamp'] = processed_raw_dataset_df['timestamp']
    converted_dataset_df.sort_values('timestamp', inplace=True)
    converted_dataset_df.reset_index(drop=True, inplace=True)
    if not first_lines:
        split_ts = int(time.mktime(time.strptime('2017-12-3 0:0:0', '%Y-%m-%d %H:%M:%S')))
    else:
        converted_dataset_df = converted_dataset_df.iloc[:first_lines]
        split_idx = int(len(converted_dataset_df) * 0.9)
        split_ts = converted_dataset_df.iloc[split_idx]['timestamp']

    train_dataset_df = converted_dataset_df[converted_dataset_df['timestamp'] < split_ts]
    test_dataset_df = converted_dataset_df[converted_dataset_df['timestamp'] >= split_ts]
    converted_dataset_df.to_pickle(converted_dataset_df_path)
    train_dataset_df.to_pickle(train_dataset_df_path)
    test_dataset_df.to_pickle(test_dataset_df_path)

    return


def train_writer(
    train_records_path, train_dataset_df_path,
    sparse_features_max_idx_path, iid_to_cid_path, bid_sample_tempature_path,
    seq_len, neg_samples
):

    train_dataset_df = pd.read_pickle(train_dataset_df_path)
    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max = pkl.load(f)
    iid_size = sparse_features_max['iid']
    bid_size = sparse_features_max['bid']
    iid_to_cid = pd.read_pickle(iid_to_cid_path)
    iid_to_cid = iid_to_cid[range(iid_size)].values
    bid_sample_tempature = pd.read_pickle(bid_sample_tempature_path)
    bid_sample_tempature = bid_sample_tempature[range(bid_size)].values

    print('=' * 32 + '    writing training samples    ' + '=' * 32)
    uid_hist_iid_seq, uid_hist_cid_seq, uid_hist_bid_seq, uid_hist_ts_seq, uid_hist_sample_tempature_seq = \
        {}, {}, {}, {}, {}

    for uid, uid_hist in train_dataset_df.groupby('uid'):
        uid_hist_iid_seq[uid] = np.array(uid_hist['iid'].to_list())
        uid_hist_cid_seq[uid] = np.array(uid_hist['cid'].to_list())
        hist_bid_seq = np.array(uid_hist['bid'].to_list())
        uid_hist_bid_seq[uid] = hist_bid_seq
        uid_hist_sample_tempature_seq[uid] = np.array(bid_sample_tempature[hist_bid_seq])
        uid_hist_ts_seq[uid] = np.array(uid_hist['timestamp'].to_list())

    total_train_samples = 0
    with tf.io.TFRecordWriter(train_records_path) as writer:
        curr_uid_count = defaultdict(int)
        for idx, row in tqdm(enumerate(train_dataset_df.itertuples())):
            _, uid, iid, cid, bid, timestamp = row

            hist_seq_len = curr_uid_count[uid]
            hist_iid_seq = uid_hist_iid_seq[uid][:hist_seq_len]
            hist_cid_seq = uid_hist_cid_seq[uid][:hist_seq_len]
            hist_bid_seq = uid_hist_bid_seq[uid][:hist_seq_len]
            hist_ts_seq = uid_hist_ts_seq[uid][:hist_seq_len]
            hist_ts_diff_seq = (timestamp - hist_ts_seq) / 3600
            hist_ts_diff_seq = hist_ts_diff_seq.astype(np.int64)

            hist_sample_tempature_seq = uid_hist_sample_tempature_seq[uid][:hist_seq_len]
            position_tempature_seq = np.arange(hist_seq_len) * 0.005
            total_sample_tempature_seq = hist_sample_tempature_seq + position_tempature_seq
            sample_prob = total_sample_tempature_seq / np.sum(total_sample_tempature_seq)

            sample_len = int(min(seq_len, hist_seq_len * 0.9))
            if not sample_len:
                sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq = (
                    np.array([], dtype=np.int64) for _ in range(4)
                )
            else:
                sample_idx = np.random.choice(hist_seq_len, sample_len, replace=False, p=sample_prob)
                sample_idx.sort()
                sample_iid_seq = np.array(hist_iid_seq)[sample_idx]
                sample_cid_seq = np.array(hist_cid_seq)[sample_idx]
                sample_bid_seq = np.array(hist_bid_seq)[sample_idx]
                sample_ts_diff_seq = hist_ts_diff_seq[sample_idx]

            train_hist_iid_seq = _pad_func(hist_iid_seq, seq_len)
            train_hist_cid_seq = _pad_func(hist_cid_seq, seq_len)
            train_hist_bid_seq = _pad_func(hist_bid_seq, seq_len)
            train_hist_ts_diff_seq = _pad_func(hist_ts_diff_seq, seq_len)
            train_hist_seq_len = min(hist_seq_len, seq_len)
            sample_iid_seq = _pad_func(sample_iid_seq, seq_len)
            sample_cid_seq = _pad_func(sample_cid_seq, seq_len)
            sample_bid_seq = _pad_func(sample_bid_seq, seq_len)
            sample_ts_diff_seq = _pad_func(sample_ts_diff_seq, seq_len)
            train_example = _build_train_example(
                uid, iid, cid, bid, timestamp,
                train_hist_iid_seq, train_hist_cid_seq, train_hist_bid_seq, train_hist_ts_diff_seq, train_hist_seq_len,
                sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq, sample_len
            )

            writer.write(train_example.SerializeToString())
            total_train_samples += 1
            curr_uid_count[uid] += 1
            if idx % 1000 == 0:
                writer.flush()
    print(
        '#' * 132 + '\n'
        + '=' * 32 + f'    writing training samples finished, {total_train_samples} total train samples   ' + '=' * 32 + '\n'
        + '-' * 4 + f'     test file saved in {train_records_path}    ' + '-' * 4 + '\n'
        + '#' * 132
    )


def _build_train_example(
    uid, iid, cid, bid, timestamp,
    hist_iid_seq, hist_cid_seq, hist_bid_seq, hist_ts_diff_seq, hist_seq_len,
    sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq, sample_len
):
    feature = {
        'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
        'iid': tf.train.Feature(int64_list=tf.train.Int64List(value=[iid])),
        'cid': tf.train.Feature(int64_list=tf.train.Int64List(value=[cid])),
        'bid': tf.train.Feature(int64_list=tf.train.Int64List(value=[bid])),
        'timestamp': tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),
        'hist_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_iid_seq)),
        'hist_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_cid_seq)),
        'hist_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_bid_seq)),
        'hist_ts_diff_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_ts_diff_seq)),
        'hist_seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[hist_seq_len])),
        'sample_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_iid_seq)),
        'sample_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_cid_seq)),
        'sample_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_bid_seq)),
        'sample_ts_diff_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_ts_diff_seq)),
        'sample_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample_len])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def test_writer(
    test_records_path, train_dataset_df_path, test_dataset_df_path, sparse_features_max_idx_path, iid_to_cid_path,
    split_ts, seq_len, test_drop_hist, neg_samples
):

    train_dataset_df = pd.read_pickle(train_dataset_df_path)
    test_dataset_df = pd.read_pickle(test_dataset_df_path)
    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max = pkl.load(f)
    iid_size = sparse_features_max['iid']
    iid_to_cid = pd.read_pickle(iid_to_cid_path)
    iid_to_cid = iid_to_cid[range(iid_size)].values

    print('=' * 32 + '    writing testing samples    ' + '=' * 32)
    uid_hist_iid_seq, uid_hist_cid_seq, uid_hist_bid_seq, uid_hist_ts_diff_seq, uid_hist_len = \
        {}, {}, {}, {}, defaultdict(int)

    for uid, uid_hist in tqdm(train_dataset_df.groupby('uid')):
        hist_iid_seq = np.array(uid_hist['iid'].to_list())
        uid_hist_iid_seq[uid] = hist_iid_seq
        uid_hist_cid_seq[uid] = np.array(uid_hist['cid'].to_list())
        uid_hist_bid_seq[uid] = np.array(uid_hist['bid'].to_list())
        hist_ts_seq = np.array(uid_hist['timestamp'].to_list())
        hist_ts_diff_seq = (split_ts - hist_ts_seq) / 3600
        uid_hist_ts_diff_seq[uid] = hist_ts_diff_seq.astype(np.int64)
        uid_hist_len[uid] = len(uid_hist)

    total_test_samples = 0
    with tf.io.TFRecordWriter(test_records_path) as writer:
        for idx, (uid, uid_ground_truth) in tqdm(enumerate(test_dataset_df.groupby('uid'))):
            all_hist_seq_len = uid_hist_len[uid]
            if all_hist_seq_len == 0:
                continue
            all_hist_iid_seq = uid_hist_iid_seq[uid]
            all_hist_cid_seq = uid_hist_cid_seq[uid]
            all_hist_bid_seq = uid_hist_bid_seq[uid]
            all_hist_ts_diff_seq = uid_hist_ts_diff_seq[uid]

            ground_truth_iid_seq = np.array(uid_ground_truth['iid'].to_list())
            ground_truth_cid_seq = np.array(uid_ground_truth['cid'].to_list())
            ground_truth_bid_seq = np.array(uid_ground_truth['bid'].to_list())
            ground_truth_ts_seq = np.array(uid_ground_truth['timestamp'].to_list())
            ground_truth_seq_len = len(ground_truth_iid_seq)
            if test_drop_hist:
                hist_iid_set = set(all_hist_iid_seq)
                ground_truth_iid_idx = list(
                    map(
                        lambda x: x[0],
                        filter(lambda x: x[1] not in hist_iid_set, enumerate(ground_truth_iid_seq))
                    )
                )
                ground_truth_seq_len = len(ground_truth_iid_idx)
                if ground_truth_seq_len == 0:
                    continue
                ground_truth_iid_seq = ground_truth_iid_seq[ground_truth_iid_idx]
                ground_truth_cid_seq = ground_truth_cid_seq[ground_truth_iid_idx]
                ground_truth_bid_seq = ground_truth_bid_seq[ground_truth_iid_idx]
                ground_truth_ts_seq = ground_truth_ts_seq[ground_truth_iid_idx]

            hist_iid_seq = _pad_func(all_hist_iid_seq, seq_len)
            hist_cid_seq = _pad_func(all_hist_cid_seq, seq_len)
            hist_bid_seq = _pad_func(all_hist_bid_seq, seq_len)
            hist_ts_diff_seq = _pad_func(all_hist_ts_diff_seq, seq_len)
            hist_seq_len = min(all_hist_seq_len, seq_len)
            test_example = _build_test_example(
                uid, hist_iid_seq, hist_cid_seq, hist_bid_seq, hist_ts_diff_seq, hist_seq_len,
                ground_truth_iid_seq, ground_truth_cid_seq, ground_truth_bid_seq, ground_truth_ts_seq, ground_truth_seq_len,
                all_hist_iid_seq, all_hist_cid_seq, all_hist_bid_seq, all_hist_ts_diff_seq, all_hist_seq_len
            )
            writer.write(test_example.SerializeToString())
            total_test_samples += ground_truth_seq_len
            if idx % 100 == 0:
                writer.flush()
    print(
        '#' * 132 + '\n'
        + '=' * 32 + f'     writing testing samples finished, {total_test_samples} total test samples    ' + '=' * 32 + '\n'
        + '-' * 4 + f'     test file saved in {test_records_path}    ' + '-' * 4 + '\n'
        + '#' * 132
    )


def _build_test_example(
    uid, hist_iid_seq, hist_cid_seq, hist_bid_seq, hist_ts_diff_seq, hist_seq_len,
    ground_truth_iid_seq, ground_truth_cid_seq, ground_truth_bid_seq, ground_truth_ts_seq, ground_truth_seq_len,
    all_hist_iid_seq, all_hist_cid_seq, all_hist_bid_seq, all_hist_ts_diff_seq, all_hist_seq_len
):
    feature = {
        'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
        'hist_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_iid_seq)),
        'hist_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_cid_seq)),
        'hist_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_bid_seq)),
        'hist_ts_diff_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_ts_diff_seq)),
        'hist_seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[hist_seq_len])),
        'ground_truth_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=ground_truth_iid_seq)),
        'ground_truth_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=ground_truth_cid_seq)),
        'ground_truth_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=ground_truth_bid_seq)),
        'ground_truth_ts_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=ground_truth_ts_seq)),
        'ground_truth_seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[ground_truth_seq_len])),
        'all_hist_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=all_hist_iid_seq)),
        'all_hist_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=all_hist_cid_seq)),
        'all_hist_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=all_hist_bid_seq)),
        'all_hist_ts_diff_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=all_hist_ts_diff_seq)),
        'all_hist_seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[all_hist_seq_len])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def _pad_func(inp, seq_len):
    inp = np.array(inp)
    inp_len = inp.shape[0]
    if inp_len >= seq_len:
        return inp[-seq_len:]
    res = np.zeros(shape=(seq_len,), dtype=np.int64)
    res[0: inp_len] = inp
    return res
