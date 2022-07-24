import os
import pickle as pkl
from collections import defaultdict
from threading import Thread

import pandas as pd
from tqdm import tqdm

from basek.utils.tf_compat import tf
from basek.utils.imports import numpy as np


def records_writer(
    savepath, max_seq_len, neg_samples=0,
    id_ordered_by_count=True, write_train=True, write_test=True
):
    print('=' * 36 + '    writing samples    ' + '=' * 36)

    sparse_features_max_idx_path = os.path.join(savepath, 'sparse_features_max_idx.pkl')
    tfrecords_prefix = f'max_seq_len_{max_seq_len}-neg_samples_{neg_samples}'
    if id_ordered_by_count:
        tfrecords_prefix = f'{tfrecords_prefix}-id_ordered_by_count'
        dataset_df_path = os.path.join(savepath, 'id_ordered_by_count-dataset_df.pkl')
        iid_to_cid_path = os.path.join(savepath, 'id_ordered_by_count-iid_to_cid.pkl')
        freq_path_prefix = os.path.join(savepath, 'id_ordered_by_count-freq_path')
    else:
        tfrecords_prefix = f'{tfrecords_prefix}-no_id_ordered_by_count'
        dataset_df_path = os.path.join(savepath, 'no_id_ordered_by_count-dataset_df.pkl')
        iid_to_cid_path = os.path.join(savepath, 'no_id_ordered_by_count-iid_to_cid.pkl')
        freq_path_prefix = os.path.join(savepath, 'no_id_ordered_by_count-freq_path')
    bid_to_sample_tempature_path = os.path.join(savepath, 'bid_to_sample_tempature.pkl')

    try:
        dataset_df = pd.read_pickle(dataset_df_path)
    except:
        raise ValueError(f'no data in {savepath}')

    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max = pkl.load(f)
    iid_size = sparse_features_max['iid']
    cid_size = sparse_features_max['cid']
    bid_size = sparse_features_max['bid']
    iid_to_cid = pd.read_pickle(iid_to_cid_path)
    iid_to_cid = iid_to_cid[range(iid_size)].values
    bid_to_sample_tempature = pd.read_pickle(bid_to_sample_tempature_path)
    bid_to_sample_tempature = bid_to_sample_tempature[range(bid_size)].values
    iid_freq = pd.read_pickle(f'{freq_path_prefix}_iid.pkl')
    iid_freq = iid_freq[range(iid_size)].values
    cid_freq = pd.read_pickle(f'{freq_path_prefix}_cid.pkl')
    cid_freq = cid_freq[range(cid_size)].values
    bid_freq = pd.read_pickle(f'{freq_path_prefix}_bid.pkl')
    bid_freq = bid_freq[range(bid_size)].values

    if id_ordered_by_count:
        uid_neg_iid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-id_ordered_by_count-uid_neg_iid_list.pkl')
        uid_neg_iid_freq_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-id_ordered_by_count-uid_neg_iid_freq_list.pkl')
        uid_neg_cid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-id_ordered_by_count-uid_neg_cid_list.pkl')
        uid_neg_cid_freq_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-id_ordered_by_count-uid_neg_cid_freq_list.pkl')
    else:
        uid_neg_iid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-no_id_ordered_by_count-uid_neg_iid_list.pkl')
        uid_neg_iid_freq_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-no_id_ordered_by_count-uid_neg_iid_freq_list.pkl')
        uid_neg_cid_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-no_id_ordered_by_count-uid_neg_cid_list.pkl')
        uid_neg_cid_freq_list_path = \
            os.path.join(savepath, f'neg_samples_{neg_samples}-no_id_ordered_by_count-uid_neg_cid_freq_list.pkl')
    uid_hist_iid_seq, uid_hist_iid_freq_seq, \
        uid_hist_cid_seq, uid_hist_cid_freq_seq, \
        uid_hist_bid_seq, uid_hist_bid_freq_seq, \
        uid_hist_ts_seq, uid_hist_sample_tempature_seq, uid_hist_seq_len = {}, {}, {}, {}, {}, {}, {}, {}, {}

    neg_sample_finished = False
    if os.path.exists(uid_neg_iid_list_path) and os.path.exists(uid_neg_iid_freq_list_path) \
        and os.path.exists(uid_neg_cid_list_path) and os.path.exists(uid_neg_cid_freq_list_path):
        neg_sample_finished = True
        with open(uid_neg_iid_list_path, 'rb') as f:
            uid_neg_iid_list = pkl.load(f)
        with open(uid_neg_iid_freq_list_path, 'rb') as f:
            uid_neg_iid_freq_list = pkl.load(f)
        with open(uid_neg_cid_list_path, 'rb') as f:
            uid_neg_cid_list = pkl.load(f)
        with open(uid_neg_cid_freq_list_path, 'rb') as f:
            uid_neg_cid_freq_list = pkl.load(f)
    else:
        uid_neg_iid_list, uid_neg_iid_freq_list, uid_neg_cid_list, uid_neg_cid_freq_list = \
            defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    all_sample_set = set(range(1, iid_size))
    for uid, uid_hist in tqdm(dataset_df.groupby('uid')):
        hist_iid_seq = np.array(uid_hist['iid'].to_list())
        hist_iid_freq_seq = np.array(uid_hist['iid_freq'].to_list())
        hist_cid_seq = np.array(uid_hist['cid'].to_list())
        hist_cid_freq_seq = np.array(uid_hist['cid_freq'].to_list())
        hist_bid_seq = np.array(uid_hist['bid'].to_list())
        hist_bid_freq_seq = np.array(uid_hist['bid_freq'].to_list())

        hist_ts_seq = np.array(uid_hist['timestamp'].to_list())
        hist_sample_tempature_seq = bid_to_sample_tempature[hist_bid_seq]
        hist_seq_len = hist_iid_seq.shape[0]

        uid_hist_iid_seq[uid] = hist_iid_seq
        uid_hist_iid_freq_seq[uid] = hist_iid_freq_seq
        uid_hist_cid_seq[uid] = hist_cid_seq
        uid_hist_cid_freq_seq[uid] = hist_cid_freq_seq
        uid_hist_bid_seq[uid] = hist_bid_seq
        uid_hist_bid_freq_seq[uid] = hist_bid_freq_seq

        uid_hist_ts_seq[uid] = hist_ts_seq
        uid_hist_sample_tempature_seq[uid] = hist_sample_tempature_seq
        uid_hist_seq_len[uid] = hist_seq_len

        if not neg_sample_finished:
            hist_iid_set = set(hist_iid_seq)
            all_neg_list = list(all_sample_set - hist_iid_set)
            all_neg_list = np.array(all_neg_list)
            neg_iid_list = []
            if neg_samples > len(all_neg_list):
                raise ValueError('neg_samples is too large, please check')
            if neg_samples < 0:
                raise ValueError('neg_samples is less than 0, please check')
            elif neg_samples == 0:
                neg_iid_list = [0] * hist_seq_len
            else:
                for _ in range(hist_seq_len):
                    neg_iid_list += list(np.random.choice(all_neg_list, neg_samples, replace=False))
            neg_iid_list = np.array(list(neg_iid_list))
            neg_iid_freq_list = iid_freq[neg_iid_list]
            neg_cid_list = iid_to_cid[neg_iid_list]
            neg_cid_freq_list = cid_freq[neg_cid_list]
            uid_neg_iid_list[uid] = neg_iid_list
            uid_neg_iid_freq_list[uid] = neg_iid_freq_list
            uid_neg_cid_list[uid] = neg_cid_list
            uid_neg_cid_freq_list[uid] = neg_cid_freq_list
    if not neg_sample_finished:
        with open(uid_neg_iid_list_path, 'wb') as f:
            pkl.dump(uid_neg_iid_list, f)
        with open(uid_neg_iid_freq_list_path, 'wb') as f:
            pkl.dump(uid_neg_iid_freq_list, f)
        with open(uid_neg_cid_list_path, 'wb') as f:
            pkl.dump(uid_neg_cid_list, f)
        with open(uid_neg_cid_freq_list_path, 'wb') as f:
            pkl.dump(uid_neg_cid_freq_list, f)
        neg_sample_finished = True

    def _writer(mode):
        tfrecords_path = os.path.join(savepath, f'{tfrecords_prefix}-{mode}.tfrecords')
        total_samples = 0
        uid_curr_hist_seq_len = defaultdict(int)
        with tf.io.TFRecordWriter(tfrecords_path) as writer:
            for idx, row in tqdm(enumerate(dataset_df.itertuples())):
                _, uid, iid, iid_freq, cid, cid_freq, bid, bid_freq, timestamp = row
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
                neg_iid_freq_list = uid_neg_iid_freq_list[uid][curr_hist_seq_len * neg_samples: (curr_hist_seq_len + 1) * neg_samples]
                neg_cid_list = uid_neg_cid_list[uid][curr_hist_seq_len * neg_samples: (curr_hist_seq_len + 1) * neg_samples]
                neg_cid_freq_list = uid_neg_cid_freq_list[uid][curr_hist_seq_len * neg_samples: (curr_hist_seq_len + 1) * neg_samples]

                hist_iid_seq = uid_hist_iid_seq[uid][:curr_hist_seq_len]
                hist_iid_freq_seq = uid_hist_iid_freq_seq[uid][:curr_hist_seq_len]
                hist_cid_seq = uid_hist_cid_seq[uid][:curr_hist_seq_len]
                hist_cid_freq_seq = uid_hist_cid_freq_seq[uid][:curr_hist_seq_len]
                hist_bid_seq = uid_hist_bid_seq[uid][:curr_hist_seq_len]
                hist_bid_freq_seq = uid_hist_bid_freq_seq[uid][:curr_hist_seq_len]

                hist_ts_seq = uid_hist_ts_seq[uid][:curr_hist_seq_len]
                hist_ts_diff_seq = (timestamp - hist_ts_seq) / 3600
                hist_ts_diff_seq = hist_ts_diff_seq.astype(np.int64)

                sample_len = int(min(max_seq_len, curr_hist_seq_len * 0.9))
                if not sample_len:
                    sample_iid_seq, sample_cid_seq, sample_bid_seq, sample_ts_diff_seq = (
                        np.array([], dtype=np.int64) for _ in range(4)
                    )
                    sample_iid_freq_seq, sample_cid_freq_seq, sample_bid_freq_seq = (
                        np.array([], dtype=np.float32) for _ in range(3)
                    )
                else:
                    hist_sample_tempature_seq = uid_hist_sample_tempature_seq[uid][:curr_hist_seq_len]
                    position_tempature_seq = np.arange(curr_hist_seq_len) * 0.005
                    aggregated_sample_tempature_seq = hist_sample_tempature_seq + position_tempature_seq
                    sample_prob = aggregated_sample_tempature_seq / np.sum(aggregated_sample_tempature_seq)
                    sample_idx = np.random.choice(curr_hist_seq_len, sample_len, replace=False, p=sample_prob)
                    sample_idx.sort()
                    sample_iid_seq = hist_iid_seq[sample_idx]
                    sample_iid_freq_seq = hist_iid_freq_seq[sample_idx]
                    sample_cid_seq = hist_cid_seq[sample_idx]
                    sample_cid_freq_seq = hist_cid_freq_seq[sample_idx]
                    sample_bid_seq = hist_bid_seq[sample_idx]
                    sample_bid_freq_seq = hist_bid_freq_seq[sample_idx]
                    sample_ts_diff_seq = hist_ts_diff_seq[sample_idx]

                hist_iid_seq = _pad_func(hist_iid_seq, max_seq_len)
                hist_iid_freq_seq = _pad_func(hist_iid_freq_seq, max_seq_len)
                hist_cid_seq = _pad_func(hist_cid_seq, max_seq_len)
                hist_cid_freq_seq = _pad_func(hist_cid_freq_seq, max_seq_len)
                hist_bid_seq = _pad_func(hist_bid_seq, max_seq_len)
                hist_bid_freq_seq = _pad_func(hist_bid_freq_seq, max_seq_len)
                hist_ts_diff_seq = _pad_func(hist_ts_diff_seq, max_seq_len)
                hist_seq_len = min(curr_hist_seq_len, max_seq_len)

                sample_iid_seq = _pad_func(sample_iid_seq, max_seq_len)
                sample_iid_freq_seq = _pad_func(sample_iid_freq_seq, max_seq_len)
                sample_cid_seq = _pad_func(sample_cid_seq, max_seq_len)
                sample_cid_freq_seq = _pad_func(sample_cid_freq_seq, max_seq_len)
                sample_bid_seq = _pad_func(sample_bid_seq, max_seq_len)
                sample_bid_freq_seq = _pad_func(sample_bid_freq_seq, max_seq_len)
                sample_ts_diff_seq = _pad_func(sample_ts_diff_seq, max_seq_len)

                example = _build_example(
                    uid, timestamp,
                    iid, iid_freq, cid, cid_freq, bid, bid_freq,
                    neg_iid_list, neg_iid_freq_list, neg_cid_list, neg_cid_freq_list,
                    hist_iid_seq, hist_iid_freq_seq, hist_cid_seq, hist_cid_freq_seq, hist_bid_seq, hist_bid_freq_seq,
                    hist_ts_diff_seq, hist_seq_len,
                    sample_iid_seq, sample_iid_freq_seq, sample_cid_seq, sample_cid_freq_seq, sample_bid_seq, sample_bid_freq_seq,
                    sample_ts_diff_seq, sample_len
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


def _build_example(
    uid, timestamp,
    iid, iid_freq, cid, cid_freq, bid, bid_freq,
    neg_iid_list, neg_iid_freq_list, neg_cid_list, neg_cid_freq_list,
    hist_iid_seq, hist_iid_freq_seq, hist_cid_seq, hist_cid_freq_seq, hist_bid_seq, hist_bid_freq_seq,
    hist_ts_diff_seq, hist_seq_len,
    sample_iid_seq, sample_iid_freq_seq, sample_cid_seq, sample_cid_freq_seq, sample_bid_seq, sample_bid_freq_seq,
    sample_ts_diff_seq, sample_len
):
    feature = {
        'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
        'timestamp': tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),

        'iid': tf.train.Feature(int64_list=tf.train.Int64List(value=[iid])),
        'iid_freq': tf.train.Feature(float_list=tf.train.FloatList(value=[iid_freq])),
        'cid': tf.train.Feature(int64_list=tf.train.Int64List(value=[cid])),
        'cid_freq': tf.train.Feature(float_list=tf.train.FloatList(value=[cid_freq])),
        'bid': tf.train.Feature(int64_list=tf.train.Int64List(value=[bid])),
        'bid_freq': tf.train.Feature(float_list=tf.train.FloatList(value=[bid_freq])),

        'neg_iid_list': tf.train.Feature(int64_list=tf.train.Int64List(value=neg_iid_list)),
        'neg_iid_freq_list': tf.train.Feature(float_list=tf.train.FloatList(value=neg_iid_freq_list)),
        'neg_cid_list': tf.train.Feature(int64_list=tf.train.Int64List(value=neg_cid_list)),
        'neg_cid_freq_list': tf.train.Feature(float_list=tf.train.FloatList(value=neg_cid_freq_list)),

        'hist_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_iid_seq)),
        'hist_iid_freq_seq': tf.train.Feature(float_list=tf.train.FloatList(value=hist_iid_freq_seq)),
        'hist_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_cid_seq)),
        'hist_cid_freq_seq': tf.train.Feature(float_list=tf.train.FloatList(value=hist_cid_freq_seq)),
        'hist_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_bid_seq)),
        'hist_bid_freq_seq': tf.train.Feature(float_list=tf.train.FloatList(value=hist_bid_freq_seq)),
        'hist_ts_diff_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=hist_ts_diff_seq)),
        'hist_seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[hist_seq_len])),

        'sample_iid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_iid_seq)),
        'sample_iid_freq_seq': tf.train.Feature(float_list=tf.train.FloatList(value=sample_iid_freq_seq)),
        'sample_cid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_cid_seq)),
        'sample_cid_freq_seq': tf.train.Feature(float_list=tf.train.FloatList(value=sample_cid_freq_seq)),
        'sample_bid_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_bid_seq)),
        'sample_bid_freq_seq': tf.train.Feature(float_list=tf.train.FloatList(value=sample_bid_freq_seq)),
        'sample_ts_diff_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_ts_diff_seq)),
        'sample_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample_len]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def _pad_func(inp, max_seq_len):
    inp = np.array(inp)
    inp_len = inp.shape[0]
    if inp_len >= max_seq_len:
        return inp[-max_seq_len:]
    res = np.zeros(shape=(max_seq_len,), dtype=inp.dtype)
    res[0: inp_len] = inp
    return res
