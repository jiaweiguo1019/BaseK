from collections import defaultdict
import os
import pickle as pkl
from time import localtime, strftime, time
from tqdm import tqdm

import faiss
from basek.utils.imports import numpy as np
from basek.utils.imports import random

from basek.params import args
from basek.preprocessors.split_time_taobao_copy import read_reviews
from basek.utils.tf_compat import tf, keras
import tensorflow_addons as tfa

from basek.layers.base import BiasAdd, Concatenate, Dense, Embedding, Flatten, Index, Input, Lambda

from basek.utils.compute_metrics import compute_metrics


SEQ_LEN = 50
NEG_SAMPLES = 1024
USER_EMB_DIM = 64
ITEM_EMB_DIM = 64
CATE_EMB_DIM = 16
EPOCHS = args.epochs
BATCH_SIZE = 1024
MATCH_POINTS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400]


def train_parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'uid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'iid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'cid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'bid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'timestamp': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'hist_iid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_cid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_bid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_ts_diff_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_seq_len': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'sample_iid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_cid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_bid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_ts_diff_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_len': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
        }
    )
    return features


def test_parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'uid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'hist_iid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_cid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_bid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_ts_diff_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_seq_len': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'ground_truth_iid_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'ground_truth_cid_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'ground_truth_bid_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'ground_truth_ts_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'ground_truth_seq_len': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'all_hist_iid_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'all_hist_cid_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'all_hist_bid_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'all_hist_ts_diff_seq': tf.io.VarLenFeature(dtype=tf.int64),
            'all_hist_seq_len': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
        }
    )

    features['ground_truth_iid_seq'] = tf.sparse_tensor_to_dense(features['ground_truth_iid_seq'])
    features['ground_truth_cid_seq'] = tf.sparse_tensor_to_dense(features['ground_truth_cid_seq'])
    features['ground_truth_bid_seq'] = tf.sparse_tensor_to_dense(features['ground_truth_bid_seq'])
    features['ground_truth_ts_seq'] = tf.sparse_tensor_to_dense(features['ground_truth_ts_seq'])
    features['all_hist_iid_seq'] = tf.sparse_tensor_to_dense(features['all_hist_iid_seq'])
    features['all_hist_cid_seq'] = tf.sparse_tensor_to_dense(features['all_hist_cid_seq'])
    features['all_hist_bid_seq'] = tf.sparse_tensor_to_dense(features['all_hist_bid_seq'])
    features['all_hist_ts_diff_seq'] = tf.sparse_tensor_to_dense(features['all_hist_ts_diff_seq'])
    return features


if __name__ == '__main__':

    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    MATCH_POINTS = sorted(list(set(MATCH_POINTS)))
    MAX_MATCH_POINT = MATCH_POINTS[-1]

    sparse_features = ['uid', 'iid', 'cid']

    sparse_features_max_idx_path = \
        '/home/web_server/guojiawei/basek/datasets/Taobao/k_core_20-id_ordered_by_count-sparse_features_max_idx.pkl'
    all_indices_path = \
        '/home/web_server/guojiawei/basek/datasets/Taobao/k_core_20-id_ordered_by_count-all_indices.pkl'

    train_file = '/data/project/tfrecords/seq_len_50-k_core_20-id_ordered_by_count-train.tfrecords'
    test_file = '/data/project/tfrecords/seq_len_50-test_drop_hist-k_core_20-id_ordered_by_count-test.tfrecords'

    train_dataset = tf.data.TFRecordDataset(train_file)
    train_dataset = train_dataset.map(train_parser, num_parallel_calls=-1)
    train_dataset = train_dataset.shuffle(100 * BATCH_SIZE, reshuffle_each_iteration=True)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE).prefetch(10)
    train_iterator = tf.data.make_initializable_iterator(train_dataset)
    train_batch = train_iterator.get_next()

    test_dataset = tf.data.TFRecordDataset(test_file)
    test_dataset = test_dataset.map(test_parser, num_parallel_calls=-1)
    test_dataset = test_dataset.padded_batch(16 * BATCH_SIZE).prefetch(10)
    test_iterator = tf.data.make_initializable_iterator(test_dataset)
    test_batch = test_iterator.get_next()

    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max_idx = pkl.load(f)
    with open(all_indices_path, 'rb') as f:
        all_indices = pkl.load(f)

    user_size = sparse_features_max_idx['uid']
    item_size = sparse_features_max_idx['iid']
    cate_size = sparse_features_max_idx['cid']
    all_iid_index, all_cid_index = all_indices
    print(
        '-' * 16 + f'    user_size: {user_size}, item_size: {item_size}, '
        + f'cate_size: {cate_size}    ' + '-' * 16 + '\n' + '#' * 132
    )

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    ckpt_path = './ckpts/' + timestamp
    log_path = './logs/' + timestamp
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    print('-' * 4 + f'  model weights are saved in {os.path.realpath(ckpt_path)}  ' + '-' * 4 + '\n' + '#' * 132)

    # bulid model
    train_uid = train_batch['uid']
    train_iid = train_batch['iid']
    train_cid = train_batch['cid']
    train_hist_iid_seq = train_batch['hist_iid_seq']
    train_hist_cid_seq = train_batch['hist_cid_seq']
    train_hist_seq_len = train_batch['hist_seq_len']

    test_uid = test_batch['uid']
    test_hist_iid_seq = test_batch['hist_iid_seq']
    test_hist_cid_seq = test_batch['hist_cid_seq']
    test_hist_seq_len = test_batch['hist_seq_len']
    test_ground_truth_iid_seq = test_batch['ground_truth_iid_seq']
    test_ground_truth_seq_len = test_batch['ground_truth_seq_len']
    all_hist_iid_seq = test_batch['all_hist_iid_seq']

    uid_emb_layer = Embedding(
        user_size, USER_EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='uid_emb_layer'
    )
    iid_emb_layer = Embedding(
        item_size, ITEM_EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='iid_emb_layer'
    )
    cid_emb_layer = Embedding(
        cate_size, CATE_EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='cid_emb_layer'
    )

    create_mask_layer = Lambda(
        lambda x: tf.transpose(
            tf.sequence_mask(x, SEQ_LEN, dtype=tf.float32),
            [0, 2, 1]
        ),
        name='create_mask_layer'
    )
    mask_layer = Lambda(
        lambda x: x[0] * x[1],
        name='mask_layer'
    )
    masked_hist_seq_len_layer = Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True),
        name='masked_hist_len_layer'
    )
    average_pooling_layer = Lambda(
        lambda x: tf.reduce_sum(x[0] / (x[1] + 1e-16), axis=1, keepdims=True),
        name='average_pooling_layer'
    )
    sum_pooling_layer = Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='sum_pooling_layer'
    )
    concatenate_layer = Concatenate(axis=-1, name='concatenate_layer')
    flatten_layer = Flatten(name='flatten_layer')

    def _user_all_embs(uid, hist_iid_seq, hist_cid_seq, hist_seq_len):
        uid_emb = uid_emb_layer(uid)
        hist_iid_seq_emb = iid_emb_layer(hist_iid_seq)
        hist_cid_seq_emb = cid_emb_layer(hist_cid_seq)

        mask = create_mask_layer(hist_seq_len)
        masked_hist_seq_len = masked_hist_seq_len_layer(mask)
        masked_hist_iid_seq_emb = mask_layer([hist_iid_seq_emb, mask])
        masked_hist_cid_seq_emb = mask_layer([hist_cid_seq_emb, mask])
        averaged_masked_hist_iid_seq_emb = \
            average_pooling_layer([masked_hist_iid_seq_emb, masked_hist_seq_len])
        averaged_masked_hist_cid_seq_emb = \
            average_pooling_layer([masked_hist_cid_seq_emb, masked_hist_seq_len])
        concatenated_user_all_embs = concatenate_layer(
            [
                uid_emb,
                averaged_masked_hist_iid_seq_emb,
                averaged_masked_hist_cid_seq_emb
            ]
        )
        all_embs = flatten_layer(concatenated_user_all_embs)
        return all_embs

    def _item_all_embs(iid, cid):
        iid_emb = iid_emb_layer(iid)
        cid_emb = cid_emb_layer(cid)
        concatenated_item_all_embs = concatenate_layer([iid_emb, cid_emb])
        item_all_embs = flatten_layer(concatenated_item_all_embs)
        return item_all_embs

    def _user_hidder(all_embs):
        user_hidden_1 = Dense(1024, 'relu', name='user_hidden_1')(all_embs)
        user_hidden_2 = Dense(512, 'relu', name='user_hidden_2')(user_hidden_1)
        user_hidden_3 = Dense(256, 'relu', name='user_hidden_3')(user_hidden_2)
        user_out = Dense(ITEM_EMB_DIM + CATE_EMB_DIM, name='user_out')(user_hidden_3)
        return user_out

    train_user_all_embs = _user_all_embs(train_uid, train_hist_iid_seq, train_hist_cid_seq, train_hist_seq_len)
    train_user_out = _user_hidder(train_user_all_embs)
    train_item_embs = _item_all_embs(train_iid, train_cid)

    # model = keras.Model(
    #     inputs=[train_uid, train_iid, train_cid, train_hist_iid_seq, train_hist_cid_seq, train_hist_seq_len],
    #     outputs=[train_user_out, train_item_embs]
    # )
    # print('=' * 120)
    # model.summary()

    # for evaluate
    test_user_all_embs = _user_all_embs(test_uid, test_hist_iid_seq, test_hist_cid_seq, test_hist_seq_len)
    test_user_out = _user_hidder(test_user_all_embs)

    all_iid_index = Index(all_iid_index, name='all_iid_index')()
    all_cid_index = Index(all_cid_index, name='all_cid_index')()
    all_item_embs = _item_all_embs(all_iid_index, all_cid_index)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, ITEM_EMB_DIM + CATE_EMB_DIM, flat_config)
    # index = faiss.IndexFlatIP(ITEM_EMB_DIM + CATE_EMB_DIM)

    # definde loss and train_op
    bias = tf.get_variable(
        name='bias', shape=(ITEM_EMB_DIM + CATE_EMB_DIM,),
        initializer=tf.initializers.zeros(), trainable=False
    )
    # bias = tf.zeros(shape=(ITEM_EMB_DIM + CATE_EMB_DIM,), dtype=tf.float32, name='unused_bias')
    loss = tf.nn.sampled_softmax_loss(
        weights=all_item_embs,
        biases=bias,
        labels=train_iid,
        inputs=train_user_out,
        num_sampled=NEG_SAMPLES,
        num_classes=item_size
    )
    loss = tf.reduce_mean(loss)
    optimizer = tfa.optimizers.AdamW(1e-5, 1e-2)
    model_vars = tf.trainable_variables()
    grads = tf.gradients(loss, model_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
    train_op = optimizer.apply_gradients(zip(clipped_grads, model_vars))

    # training loop
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.initializers.global_variables())
        prev_time = time()
        test_num = 0
        for epoch in range(1, EPOCHS + 1):
            sess.run(train_iterator.initializer)
            sess.run(test_iterator.initializer)
            total_loss = 0.0
            # model.save_weights(ckpt_path + f'/{epoch}')
            batch_num = 0
            while True:
                try:
                    batch_loss, _ = sess.run([loss, train_op])
                    total_loss += batch_loss
                    print('\r' + '-' * 32 + f'  batch: {batch_num}, loss: {batch_loss:.8f}  ' + '-' * 32, end='')
                    batch_num += 1
                    if batch_num % 10000 == 0:
                        curr_time = time()
                        time_elapsed = curr_time - prev_time
                        prev_time = curr_time
                        print('\n' + '-' * 72 + f'    time elapsed: {time_elapsed:.2f}s' + '\n' + '#' * 132)
                        test_num += 1
                        print('=' * 40 + f'  test times: {test_num:6d}  ' + '=' * 40)
                        sess.run(test_iterator.initializer)
                        all_item_out = sess.run(all_item_embs)
                        index.reset()
                        # faiss.normalize_L2(val_all_item_out)
                        index.add(all_item_out)
                        metrics = defaultdict(lambda: defaultdict(list))
                        test_batch = 1
                        while True:
                            try:
                                user_out, ground_truth_iid_seq, hist_iid_seq = sess.run(
                                    [test_user_out, test_ground_truth_iid_seq, all_hist_iid_seq]
                                )
                                # faiss.normalize_L2(val_user_out)
                                _, I = index.search(np.ascontiguousarray(user_out), MAX_MATCH_POINT)
                                batch_metrics = compute_metrics(I, MATCH_POINTS, ground_truth_iid_seq, hist_iid_seq)
                                for per_metric, per_batch_metric_values in batch_metrics.items():
                                    for match_point, per_batch_metric_value in per_batch_metric_values.items():
                                        metrics[per_metric][match_point].append(per_batch_metric_value)
                                print('\r' + '-' * 32 + f'   test batch {test_batch} finished   ' + '-' * 32, end='')
                                test_batch += 1
                            except tf.errors.OutOfRangeError:
                                aggregated_metrics = defaultdict(dict)
                                for per_metric, per_metric_values in metrics.items():
                                    for math_point, per_metric_value in per_metric_values.items():
                                        aggregated_metrics[per_metric][math_point] = \
                                            np.mean(np.concatenate(per_metric_value, axis=0))
                                # print(aggregated_metrics)
                                print()
                                for per_metric, per_aggregated_metric_values in aggregated_metrics.items():
                                    print('=' * 52 + f'    {per_metric}    ' + '=' * 52)
                                    per_metric_str = ''
                                    for math_point, per_aggregated_metric_value in per_aggregated_metric_values.items():
                                        per_metric_str = per_metric_str + '-' + \
                                            f' @{math_point:3d}: {per_aggregated_metric_value:.10f} ' + '-'
                                        if len(per_metric_str) > 120:
                                            print(per_metric_str)
                                            per_metric_str = ''
                                    if per_metric_str:
                                        print(per_metric_str)
                                curr_time = time()
                                time_elapsed = curr_time - prev_time
                                prev_time = curr_time
                                print('-' * 72 + f'    time elapsed: {time_elapsed:.2f}s' + '\n' + '#' * 132)
                                break

                except tf.errors.OutOfRangeError:
                    curr_time = time()
                    time_elapsed = curr_time - prev_time
                    prev_time = curr_time
                    print(
                        '#' * 132 + '\n'
                        + '=' * 32 + f'    train epoch: {epoch} finished    ' + '=' * 48
                        + '\n' + f'train_loss of epoch-{epoch}: {(total_loss / batch_num):.8f}    '
                        + '-' * 36 + f'    time elapsed: {time_elapsed:.2f}s'
                        + '\n' + '#' * 132
                    )
                    test_num += 1
                    print('=' * 40 + f'test times: {test_num:6d}' + '=' * 40)
                    sess.run(test_iterator.initializer)
                    all_item_out = sess.run(all_item_embs)
                    index.reset()
                    # faiss.normalize_L2(val_all_item_out)
                    index.add(all_item_out)
                    metrics = defaultdict(lambda: defaultdict(list))
                    test_batch = 1
                    while True:
                        try:
                            user_out, ground_truth_iid_seq, hist_iid_seq = sess.run(
                                [test_user_out, test_ground_truth_iid_seq, all_hist_iid_seq]
                            )
                            # faiss.normalize_L2(val_user_out)
                            _, I = index.search(np.ascontiguousarray(user_out), MAX_MATCH_POINT)
                            batch_metrics = compute_metrics(I, MATCH_POINTS, ground_truth_iid_seq, hist_iid_seq)
                            for per_metric, per_batch_metric_values in batch_metrics.items():
                                for match_point, per_batch_metric_value in per_batch_metric_values.items():
                                    metrics[per_metric][match_point].append(per_batch_metric_value)
                            print('\r' + '-' * 32 + f'   test batch {test_batch} finished   ' + '-' * 32, end='')
                            test_batch += 1
                        except tf.errors.OutOfRangeError:
                            aggregated_metrics = defaultdict(dict)
                            for per_metric, per_metric_values in metrics.items():
                                for math_point, per_metric_value in per_metric_values.items():
                                    aggregated_metrics[per_metric][math_point] = \
                                        np.mean(np.concatenate(per_metric_value, axis=0))
                            # print(aggregated_metrics)
                            for per_metric, per_aggregated_metric_values in aggregated_metrics.items():
                                print('=' * 52 + f'    {per_metric}    ' + '=' * 52)
                                per_metric_str = ''
                                for math_point, per_aggregated_metric_value in per_aggregated_metric_values.items():
                                    per_metric_str = per_metric_str + '-' + \
                                        f' @{math_point:3d}: {per_aggregated_metric_value:.10f} ' + '-'
                                    if len(per_metric_str) > 120:
                                        print(per_metric_str)
                                        per_metric_str = ''
                                if per_metric_str:
                                    print(per_metric_str)
                            curr_time = time()
                            time_elapsed = curr_time - prev_time
                            prev_time = curr_time
                            print('-' * 72 + f'    time elapsed: {time_elapsed:.2f}s' + '\n' + '#' * 132)
                            break
                    break
