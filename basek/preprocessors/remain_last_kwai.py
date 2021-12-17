import os
from threading import Thread
import networkx as nx
import pickle as pkl
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from basek.utils.tf_compat import tf
from basek.utils.imports import numpy as np


def read_raw_dataset(review_path, savepath, pp):
    item_to_cate_path = os.path.join(savepath, 'item_to_cate.pkl')
    raw_dataset_path = os.path.join(savepath, 'raw_dataset.pkl')
    if os.path.exists(item_to_cate_path) and os.path.exists(raw_dataset_path):
        item_to_cate = pd.read_pickle(item_to_cate_path)
        raw_dataset_df = pd.read_pickle(raw_dataset_path)
        print('read_review finished!')
        return item_to_cate, raw_dataset_df

    raw_dataset_df = pd.read_pickle(review_path)
    start_ts = int(time.mktime(time.strptime('2021-08-20 0:0:0', "%Y-%m-%d %H:%M:%S")))
    end_ts = int(time.mktime(time.strptime('2021-08-29 0:0:0', "%Y-%m-%d %H:%M:%S")))
    raw_dataset_df = raw_dataset_df[
        (raw_dataset_df['timestamp'] >= start_ts) & (raw_dataset_df['timestamp'] < end_ts)
    ]
    raw_dataset_df.fillna('default', inplace=True)
    if pp:
        user_list = raw_dataset_df['user'].unique()
        sampled_user_list = []
        for user in user_list:
            if np.random.random() < pp / 100:
                sampled_user_list.append(user)
        sampled_user_set = set(sampled_user_list)
        raw_dataset_df = raw_dataset_df[raw_dataset_df['user'].isin(sampled_user_set)]
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
    return item_to_cate, raw_dataset_df


def reduce_to_k_core(dataset_df, savepath, k_core):

    reduced_dataset_path = os.path.join(savepath, f'reduced_k_core_{k_core}_dataset.pkl')
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
    reduced_dataset_df = reduced_dataset_df.copy()
    reduced_dataset_df.to_pickle(reduced_dataset_path)
    print(f'reduce_to_{k_core}_core finished!')
    return reduced_dataset_df


def process_raw_dataset(
    raw_dataset_df,
    savepath,
    drop_dups=False,
    k_core=None,
):

    processed_raw_dataset_path = os.path.join(savepath, 'processed_raw_dataset.pkl')

    if os.path.exists(processed_raw_dataset_path):
        processed_raw_dataset_df = pd.read_pickle(processed_raw_dataset_path)
        user_count = processed_raw_dataset_df.groupby('user')['user'].count()
        item_count = processed_raw_dataset_df.groupby('item')['item'].count().to_dict()
        cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count().to_dict()
        behavior_count = processed_raw_dataset_df.groupby('behavior')['behavior'].count().to_dict()
        print('process_raw_dataset finished!')
        return processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count

    processed_raw_dataset_df = raw_dataset_df
    if drop_dups is True:
        processed_raw_dataset_df.drop_duplicates(['user', 'item'], inplace=True)

    if k_core is not None:
        if not isinstance(k_core, int) or k_core <= 1:
            raise ValueError(f'k_core should be an integer greater than 1, got {k_core}')
        processed_raw_dataset_df = reduce_to_k_core(processed_raw_dataset_df, savepath, k_core)

    processed_raw_dataset_df.sort_values('timestamp', inplace=True)
    processed_raw_dataset_df.reset_index(drop=True, inplace=True)
    user_count = processed_raw_dataset_df.groupby('user')['user'].count().to_dict()
    item_count = processed_raw_dataset_df.groupby('item')['item'].count().to_dict()
    cate_count = processed_raw_dataset_df.groupby('cate')['cate'].count().to_dict()
    behavior_count = processed_raw_dataset_df.groupby('behavior')['behavior'].count().to_dict()

    print('process_raw_dataset finished!')
    return processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count


def build_entity_id_map(
    entity_count, entity_prefix='default',
    id_prefix='default', savepath='./', id_ordered_by_count=True
):
    if id_ordered_by_count:
        entity_count = dict(sorted(entity_count.items(), key=lambda x: x[-1], reverse=True))
        entity_to_id_path = os.path.join(savepath, f'id_ordered_by_count-{entity_prefix}_to_{id_prefix}.pkl')
        id_to_entity_path = os.path.join(savepath, f'id_ordered_by_count-{id_prefix}_to_{entity_prefix}.pkl')
    else:
        entity_count = dict(entity_count.items())
        entity_to_id_path = os.path.join(savepath, f'no_id_ordered_by_count-{entity_prefix}_to_{id_prefix}.pkl')
        id_to_entity_path = os.path.join(savepath, f'no_id_ordered_by_count-{id_prefix}_to_{entity_prefix}.pkl')

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


def build_ordinal_entity_id_map(
    entity_count, entity_prefix='default',
    id_prefix='default', savepath='./'
):
    entity_to_id_path = os.path.join(savepath, f'{entity_prefix}_to_{id_prefix}.pkl')
    id_to_entity_path = os.path.join(savepath, f'{id_prefix}_to_{entity_prefix}.pkl')

    entities = sorted(list(entity_count.keys()))

    entity_to_id = {'null': 0, 'default': 1}
    id_to_entity = {0: 'null', 1: 'default'}
    for idx, entity in enumerate(entities):
        entity_to_id[entity] = idx + 2
        id_to_entity[idx + 2] = entity

    entity_to_id = pd.Series(entity_to_id)
    id_to_entity = pd.Series(id_to_entity)
    entity_to_id.to_pickle(entity_to_id_path)
    id_to_entity.to_pickle(id_to_entity_path)

    return entity_to_id, id_to_entity


def read_reviews(
    review_file, dirpath, savepath,
    pp=None, drop_dups=False, k_core=None, id_ordered_by_count=True
):

    os.makedirs(savepath, exist_ok=True)
    review_path = os.path.join(dirpath, review_file)

    item_to_cate, raw_dataset_df = \
        read_raw_dataset(review_path, savepath, pp)
    processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count = process_raw_dataset(
        raw_dataset_df, savepath, drop_dups=drop_dups, k_core=k_core
    )

    user_to_uid, uid_to_user = \
        build_entity_id_map(user_count, 'user', 'uid', savepath, id_ordered_by_count)
    item_to_iid, iid_to_item = \
        build_entity_id_map(item_count, 'item', 'iid', savepath, id_ordered_by_count)
    cate_to_cid, cid_to_cate = \
        build_entity_id_map(cate_count, 'cate', 'cid', savepath, id_ordered_by_count)

    if id_ordered_by_count:
        uid_count_path = os.path.join(savepath, 'id_ordered_by_count-uid_count.pkl')
        iid_to_cid_path = os.path.join(savepath, 'id_ordered_by_count-iid_to_cid.pkl')
        bid_to_sample_tempature_path = os.path.join(savepath, 'id_ordered_by_count-bid_to_sample_tempature.pkl')
    else:
        uid_count_path = os.path.join(savepath, 'no_id_ordered_by_count-uid_count.pkl')
        iid_to_cid_path = os.path.join(savepath, 'no_id_ordered_by_count-iid_to_cid.pkl')
        bid_to_sample_tempature_path = os.path.join(savepath, 'no_id_ordered_by_count-bid_to_sample_tempature.pkl')
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

    behavior_to_bid, bid_to_behavior = \
        build_ordinal_entity_id_map(behavior_count, 'behavior', 'bid', savepath)
    bid_to_sample_tempature_path = os.path.join(savepath, 'bid_to_sample_tempature.pkl')
    bid_to_sample_tempature = {}
    for bid in bid_to_behavior.keys():
        bid_to_sample_tempature[bid] = bid - 1
    bid_to_sample_tempature = pd.Series(bid_to_sample_tempature)
    bid_to_sample_tempature.to_pickle(bid_to_sample_tempature_path)

    sparse_features_max_idx_path = os.path.join(savepath, 'sparse_features_max_idx.pkl')
    uid_size, iid_size, cid_size, bid_size = \
        len(uid_to_user), len(iid_to_item), len(cid_to_cate), len(bid_to_behavior)
    sparse_features_max_idx = {'uid': uid_size, 'iid': iid_size, 'cid': cid_size, 'bid': bid_size}
    print('#' * 132)
    print('-' * 16 + f'    uid_size: {uid_size}, iid_size: {iid_size}, ' + f'cid_size: {cid_size}    ' + '-' * 16)
    print('#' * 132)
    with open(sparse_features_max_idx_path, 'wb') as f:
        pkl.dump(sparse_features_max_idx, f)

    if id_ordered_by_count:
        dataset_df_path = os.path.join(savepath, 'id_ordered_by_count-dataset_df.pkl')
        all_indices_path = os.path.join(savepath, 'id_ordered_by_count-all_indices.pkl')
    else:
        dataset_df_path = os.path.join(savepath, 'no_id_ordered_by_count-dataset_df.pkl')
        all_indices_path = os.path.join(savepath, 'no_id_ordered_by_count-all_indices.pkl')
    all_iid_index = list(iid_to_item.keys())
    all_cid_index = list(iid_to_cid[all_iid_index])
    all_indices = all_iid_index, all_cid_index
    with open(all_indices_path, 'wb') as f:
        pkl.dump(all_indices, f)

    dataset_df = pd.DataFrame()
    dataset_df['uid'] = processed_raw_dataset_df['user'].map(user_to_uid)
    dataset_df['iid'] = processed_raw_dataset_df['item'].map(item_to_iid)
    dataset_df['cid'] = processed_raw_dataset_df['cate'].map(cate_to_cid)
    dataset_df['bid'] = processed_raw_dataset_df['behavior'].map(behavior_to_bid)
    dataset_df['timestamp'] = processed_raw_dataset_df['timestamp']
    dataset_df.sort_values('timestamp', inplace=True)
    dataset_df.reset_index(drop=True, inplace=True)
    dataset_df.to_pickle(dataset_df_path)


def records_writer(
    savepath, max_seq_len, neg_samples=0, id_ordered_by_count=True,
    write_train=True, write_test=True
):

    print('=' * 36 + '    writing samples    ' + '=' * 36)

    sparse_features_max_idx_path = os.path.join(savepath, 'sparse_features_max_idx.pkl')
    tfrecords_prefix = f'max_seq_len_{max_seq_len}'
    if neg_samples:
        tfrecords_prefix = f'{tfrecords_prefix}-neg_samples_{neg_samples}'
    if id_ordered_by_count:
        tfrecords_prefix = f'{tfrecords_prefix}-id_ordered_by_count'
        dataset_df_path = os.path.join(savepath, 'id_ordered_by_count-dataset_df.pkl')
        iid_to_cid_path = os.path.join(savepath, 'id_ordered_by_count-iid_to_cid.pkl')
    else:
        tfrecords_prefix = f'{tfrecords_prefix}-no_id_ordered_by_count'
        dataset_df_path = os.path.join(savepath, 'no_id_ordered_by_count-dataset_df.pkl')
        iid_to_cid_path = os.path.join(savepath, 'no_id_ordered_by_count-iid_to_cid.pkl')
    bid_to_sample_tempature_path = os.path.join(savepath, 'bid_to_sample_tempature.pkl')

    dataset_df = pd.read_pickle(dataset_df_path)
    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max = pkl.load(f)
    iid_size = sparse_features_max['iid']
    bid_size = sparse_features_max['bid']
    iid_to_cid = pd.read_pickle(iid_to_cid_path)
    iid_to_cid = iid_to_cid[range(iid_size)].values
    bid_to_sample_tempature = pd.read_pickle(bid_to_sample_tempature_path)
    bid_to_sample_tempature = bid_to_sample_tempature[range(bid_size)].values

    if id_ordered_by_count:
        uid_neg_iid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-id_ordered_by_count-uid_neg_iid_list.pkl')
        uid_neg_cid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-id_ordered_by_count-uid_neg_cid_list.pkl')
    else:
        uid_neg_iid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-no_id_ordered_by_count-uid_neg_iid_list.pkl')
        uid_neg_cid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-no_id_ordered_by_count-uid_neg_cid_list.pkl')
    uid_hist_iid_seq, uid_hist_cid_seq, uid_hist_bid_seq, uid_hist_ts_seq, uid_hist_sample_tempature_seq, uid_hist_seq_len = \
        {}, {}, {}, {}, {}, {}
    if os.path.exists(uid_neg_iid_list_path) and os.path.exists(uid_neg_cid_list_path):
        neg_sample_finished = True
        with open(uid_neg_iid_list_path, 'rb') as f:
            uid_neg_iid_list = pkl.load(f)
        with open(uid_neg_cid_list_path, 'rb') as f:
            uid_neg_cid_list = pkl.load(f)
    else:
        neg_sample_finished = False
        uid_neg_iid_list, uid_neg_cid_list = {}, {}

    all_sample_list = np.array(list(range(1, iid_size)))
    for uid, uid_hist in tqdm(dataset_df.groupby('uid')):
        hist_iid_seq = np.array(uid_hist['iid'].to_list())
        hist_cid_seq = np.array(uid_hist['cid'].to_list())
        hist_bid_seq = np.array(uid_hist['bid'].to_list())
        hist_ts_seq = np.array(uid_hist['timestamp'].to_list())
        hist_sample_tempature_seq = bid_to_sample_tempature[hist_bid_seq]
        hist_seq_len = hist_iid_seq.shape[0]
        uid_hist_iid_seq[uid] = hist_iid_seq
        uid_hist_cid_seq[uid] = hist_cid_seq
        uid_hist_bid_seq[uid] = hist_bid_seq
        uid_hist_ts_seq[uid] = hist_ts_seq
        uid_hist_sample_tempature_seq[uid] = hist_sample_tempature_seq
        uid_hist_seq_len[uid] = hist_seq_len
        if not neg_sample_finished:
            hist_iid_set = set(hist_iid_seq)
            neg_iid_list = []
            neg_iid_samples = 0
            while True:
                if neg_iid_samples == neg_samples * hist_seq_len:
                    break
                neg_iid = np.random.choice(all_sample_list)
                if neg_iid not in hist_iid_set:
                    neg_iid_list.append(neg_iid)
                    neg_iid_samples += 1
            neg_iid_list = np.array(neg_iid_list)
            neg_cid_list = iid_to_cid[neg_iid_list]
            uid_neg_iid_list[uid] = neg_iid_list
            uid_neg_cid_list[uid] = neg_cid_list
    if not neg_sample_finished:
        with open(uid_neg_iid_list_path, 'wb') as f:
            pkl.dump(uid_neg_iid_list, f)
        with open(uid_neg_cid_list_path, 'wb') as f:
            pkl.dump(uid_neg_cid_list, f)
        neg_sample_finished = True

    def _build_example(
        uid, iid, neg_iid_list, cid, neg_cid_list, bid, timestamp,
        hist_iid_seq, hist_cid_seq, hist_bid_seq, hist_ts_diff_seq, hist_seq_len,
        sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq, sample_len
    ):
        feature = {
            'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
            'iid': tf.train.Feature(int64_list=tf.train.Int64List(value=[iid])),
            'neg_iid_list': tf.train.Feature(int64_list=tf.train.Int64List(value=neg_iid_list)),
            'cid': tf.train.Feature(int64_list=tf.train.Int64List(value=[cid])),
            'neg_cid_list': tf.train.Feature(int64_list=tf.train.Int64List(value=neg_cid_list)),
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

    def _pad_func(inp, max_seq_len):
        inp = np.array(inp)
        inp_len = inp.shape[0]
        if inp_len >= max_seq_len:
            return inp[-max_seq_len:]
        res = np.zeros(shape=(max_seq_len,), dtype=np.int64)
        res[0: inp_len] = inp
        return res

    def _writer(mode):
        tfrecords_path = os.path.join(savepath, f'{tfrecords_prefix}-{mode}.tfrecords')
        total_samples = 0
        uid_curr_hist_seq_len = defaultdict(int)
        with tf.io.TFRecordWriter(tfrecords_path) as writer:
            for idx, row in tqdm(enumerate(dataset_df.itertuples())):
                _, uid, iid, cid, bid, timestamp = row
                curr_hist_seq_len = uid_curr_hist_seq_len[uid]
                if mode == 'test':
                    if curr_hist_seq_len != uid_hist_seq_len[uid] - 1:
                        uid_curr_hist_seq_len[uid] += 1
                        continue
                    if curr_hist_seq_len == 0:
                        continue
                if mode == 'train':
                    if curr_hist_seq_len == uid_hist_seq_len[uid] - 1:
                        continue
                neg_iid_list = uid_neg_iid_list[uid][curr_hist_seq_len * neg_samples: (curr_hist_seq_len + 1) * neg_samples]
                neg_cid_list = uid_neg_cid_list[uid][curr_hist_seq_len * neg_samples: (curr_hist_seq_len + 1) * neg_samples]
                hist_iid_seq = uid_hist_iid_seq[uid][:curr_hist_seq_len]
                hist_cid_seq = uid_hist_cid_seq[uid][:curr_hist_seq_len]
                hist_bid_seq = uid_hist_bid_seq[uid][:curr_hist_seq_len]
                hist_ts_seq = uid_hist_ts_seq[uid][:curr_hist_seq_len]
                hist_ts_diff_seq = (timestamp - hist_ts_seq) / 3600
                hist_ts_diff_seq = hist_ts_diff_seq.astype(np.int64)

                sample_len = int(min(max_seq_len, curr_hist_seq_len * 0.9))
                if not sample_len:
                    sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq = (
                        np.array([], dtype=np.int64) for _ in range(4)
                    )
                else:
                    hist_sample_tempature_seq = uid_hist_sample_tempature_seq[uid][:curr_hist_seq_len]
                    position_tempature_seq = np.arange(curr_hist_seq_len) * 0.005
                    aggregated_sample_tempature_seq = hist_sample_tempature_seq + position_tempature_seq
                    sample_prob = aggregated_sample_tempature_seq / np.sum(aggregated_sample_tempature_seq)
                    sample_idx = np.random.choice(curr_hist_seq_len, sample_len, replace=False, p=sample_prob)
                    sample_idx.sort()
                    sample_iid_seq = np.array(hist_iid_seq)[sample_idx]
                    sample_cid_seq = np.array(hist_cid_seq)[sample_idx]
                    sample_bid_seq = np.array(hist_bid_seq)[sample_idx]
                    sample_ts_diff_seq = hist_ts_diff_seq[sample_idx]

                hist_iid_seq = _pad_func(hist_iid_seq, max_seq_len)
                hist_cid_seq = _pad_func(hist_cid_seq, max_seq_len)
                hist_bid_seq = _pad_func(hist_bid_seq, max_seq_len)
                hist_ts_diff_seq = _pad_func(hist_ts_diff_seq, max_seq_len)
                hist_seq_len = min(curr_hist_seq_len, max_seq_len)
                sample_iid_seq = _pad_func(sample_iid_seq, max_seq_len)
                sample_cid_seq = _pad_func(sample_cid_seq, max_seq_len)
                sample_bid_seq = _pad_func(sample_bid_seq, max_seq_len)
                sample_ts_diff_seq = _pad_func(sample_ts_diff_seq, max_seq_len)

                example = _build_example(
                    uid, iid, neg_iid_list, cid, neg_cid_list, bid, timestamp,
                    hist_iid_seq, hist_cid_seq, hist_bid_seq, hist_ts_diff_seq, hist_seq_len,
                    sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq, sample_len
                )
                writer.write(example.SerializeToString())
                total_samples += 1
                uid_curr_hist_seq_len[uid] += 1
                if idx % 10000 == 0:
                    writer.flush()

        print(
            '#' * 132 + '\n'
            + '=' * 32 + f'    writing {mode} samples finished, {total_samples} total samples   '
            + '=' * 32 + '\n'
            + '-' * 4 + f'     file saved in {tfrecords_path}    ' + '-' * 4 + '\n'
            + '#' * 132
        )

    if write_train:
        train_writer_t = Thread(target=_writer, kwargs={'mode': 'train'})
        train_writer_t.setDaemon(True)
        train_writer_t.start()
    if write_test:
        test_writer_t = Thread(target=_writer, kwargs={'mode': 'test'})
        test_writer_t.setDaemon(True)
        test_writer_t.start()
    if write_train:
        train_writer_t.join()
    if write_test:
        test_writer_t.join()
