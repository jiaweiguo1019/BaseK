from basek.params import args
import os

import pickle as pkl
from time import localtime, strftime, time
from multiprocessing import Process, Queue

import faiss
from basek.utils.imports import numpy as np
from basek.utils.imports import random

from basek.utils.tf_compat import tf, keras
import tensorflow_addons as tfa

from basek.layers.base import (
    BiasAdd, BatchNormalization,
    Concatenate,
    Dense, Dropout,
    Embedding,
    Flatten,
    Index,
    Lambda, LayerNormalization
)
from basek.layers.sequence import (
    SDMShortEncoderLayer, SDMLongEncoderLayer, SDMGateLayer
)

from basek.utils.metrics__ import ComputeMetrics


NEG_SAMPLES = 10
SEQ_LEN = 50
SHORT_SEQ_LEN = args.short_seq_len
LONG_SEQ_LEN = SEQ_LEN - SHORT_SEQ_LEN
EMB_DIM = args.emb_dim
HEADS = 2
BATCH_SIZE = 512
MATCH_POINTS = [10, 20, 25, 30, 50, 100, 120, 150, 200, 250, 300]
EPOCHS = args.epochs
SEED = args.seed
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'uid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'iid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'neg_iid_list': tf.io.FixedLenFeature(shape=(NEG_SAMPLES,), dtype=tf.int64),
            'cid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'neg_cid_list': tf.io.FixedLenFeature(shape=(NEG_SAMPLES,), dtype=tf.int64),
            'bid': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'timestamp': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'hist_iid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_cid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_bid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_ts_diff_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'hist_seq_len': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'sample_iid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_cid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_bid_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_ts_diff_seq': tf.io.FixedLenFeature(shape=(SEQ_LEN,), dtype=tf.int64),
            'sample_len': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        }
    )
    return features


if __name__ == '__main__':

    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    random.seed(SEED)
    MATCH_POINTS = sorted(list(set(MATCH_POINTS)))
    MAX_MATCH_POINT = MATCH_POINTS[-1]

    sparse_features = ['uid', 'iid', 'cid']

    dirpath = args.dirpath
    # dirpath = '/data/project/datasets/Taobao/pp_30-k_core_20-id_ordered_by_count'
    # dirpath = '/data/project/datasets/MovieLens/ml-20m/pp_50-k_core_10'
    sparse_features_max_idx_path = os.path.join(dirpath, 'sparse_features_max_idx.pkl')
    if args.id_ordered_by_count:
        all_indices_path = os.path.join(dirpath, 'id_ordered_by_count-all_indices.pkl')
        train_file = \
            os.path.join(dirpath, f'max_seq_len_{SEQ_LEN}-neg_samples_{NEG_SAMPLES}-id_ordered_by_count-train.tfrecords')
        test_file = \
            os.path.join(dirpath, f'max_seq_len_{SEQ_LEN}-neg_samples_{NEG_SAMPLES}-id_ordered_by_count-test.tfrecords')
        result_file = \
            f'by_count-{SEQ_LEN}-{args.short_seq_len}-{args.filter}-{args.lr}'
    else:
        all_indices_path = os.path.join(dirpath, 'no_id_ordered_by_count-all_indices.pkl')
        train_file = \
            os.path.join(dirpath, f'max_seq_len_{SEQ_LEN}-neg_samples_{NEG_SAMPLES}-no_id_ordered_by_count-train.tfrecords')
        test_file = \
            os.path.join(dirpath, f'max_seq_len_{SEQ_LEN}-neg_samples_{NEG_SAMPLES}-no_id_ordered_by_count-test.tfrecords')
        result_file = \
            f'no_by_count-{SEQ_LEN}-{args.short_seq_len}-{args.filter}-{args.lr}'

    ffn_hidden_unit = args.ffn_hidden_unit
    dropout = args.dropout
    result_file = f'{result_file}-{ffn_hidden_unit}-{EMB_DIM}-{dropout}'
    if args.emb_dropout:
        result_file = result_file + '-emb_dropout'
    shuffle = args.shuffle
    if shuffle:
        result_file = result_file + '-shuffle'

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    result_file = timestamp + '-' + result_file
    result_path = './results/' + result_file
    ckpt_path = './ckpts/' + timestamp + '-' + result_file
    log_path = './logs/' + timestamp + '-' + result_file
    os.makedirs(ckpt_path + '-' + result_file, exist_ok=True)
    os.makedirs(log_path + '-' + result_file, exist_ok=True)
    print('-' * 4 + f'  model weights are saved in {os.path.realpath(ckpt_path)}  ' + '-' * 4 + '\n' + '#' * 132)


    train_dataset = tf.data.TFRecordDataset(train_file)
    train_dataset = train_dataset.map(parser, num_parallel_calls=-1)
    train_dataset = train_dataset.filter(lambda x: x['hist_seq_len'] > args.filter)
    if shuffle:
        train_dataset = train_dataset.shuffle(64 * BATCH_SIZE, seed=args.seed, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(-1)
    train_iterator = tf.data.make_initializable_iterator(train_dataset)

    test_dataset = tf.data.TFRecordDataset(test_file)
    test_dataset = test_dataset.map(parser)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(-1)
    test_iterator = tf.data.make_initializable_iterator(test_dataset)

    handle = tf.placeholder(tf.string, shape=())
    training = tf.placeholder(tf.bool, shape=())
    iterator = tf.data.Iterator.from_string_handle(
        handle, tf.data.get_output_types(train_iterator), tf.data.get_output_shapes(train_iterator)
    )
    one_batch = iterator.get_next()

    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max_idx = pkl.load(f)
    with open(all_indices_path, 'rb') as f:
        all_indices = pkl.load(f)

    user_size = sparse_features_max_idx['uid']
    item_size = sparse_features_max_idx['iid']
    cate_size = sparse_features_max_idx['cid']
    behavior_size = sparse_features_max_idx['bid']
    all_iid_index, all_cid_index = all_indices
    print(
        '-' * 16 + ' ' * 4
        + f'user_size: {user_size}, item_size: {item_size}, cate_size: {cate_size}, behavior_size: {behavior_size}'
        + ' ' * 4 + '-' * 16 + '\n' + '#' * 132
    )

    # bulid model
    uid = one_batch['uid']
    iid = one_batch['iid']
    cid = one_batch['cid']
    hist_iid_seq = one_batch['hist_iid_seq']
    hist_cid_seq = one_batch['hist_cid_seq']
    hist_bid_seq = one_batch['hist_bid_seq']
    hist_ts_diff_seq = one_batch['hist_ts_diff_seq']
    hist_seq_len = one_batch['hist_seq_len']

    hist_iid_seq_short = hist_iid_seq[:, :SHORT_SEQ_LEN]
    hist_cid_seq_short = hist_cid_seq[:, :SHORT_SEQ_LEN]
    hist_bid_seq_short = hist_bid_seq[:, :SHORT_SEQ_LEN]
    hist_ts_diff_seq_short = hist_ts_diff_seq[:, :SHORT_SEQ_LEN]
    hist_iid_seq_long = hist_iid_seq[:, SHORT_SEQ_LEN:]
    hist_cid_seq_long = hist_cid_seq[:, SHORT_SEQ_LEN:]
    hist_bid_seq_long = hist_bid_seq[:, SHORT_SEQ_LEN:]
    hist_ts_diff_seq_long = hist_ts_diff_seq[:, SHORT_SEQ_LEN:]

    mask_short, mask_long = tf.split(
        tf.sequence_mask(hist_seq_len, SEQ_LEN, dtype=tf.float32),
        [SHORT_SEQ_LEN, LONG_SEQ_LEN],
        axis=-1
    )

    uid_emb_layer = Embedding(
        user_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='uid_emb_layer'
    )
    iid_emb_layer = Embedding(
        item_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='iid_emb_layer'
    )
    cid_emb_layer = Embedding(
        cate_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='cid_emb_layer'
    )
    bid_emb_layer = Embedding(
        behavior_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='cid_emb_layer'
    )
    ts_diff_emb_layer = Embedding(
        1000, EMB_DIM,
        embeddings_initializer=keras.initializers.he_uniform(),
        name='cid_emb_layer'
    )
    flatten_layer = Flatten(name='flatten_layer')


    uid_emb = uid_emb_layer(uid)
    hist_iid_emd_short = iid_emb_layer(hist_iid_seq_short)
    hist_cid_emb_short = cid_emb_layer(hist_cid_seq_short)
    hist_bid_emb_short = bid_emb_layer(hist_bid_seq_short)
    hist_ts_diff_emb_short = cid_emb_layer(hist_cid_seq_short)
    hist_iid_emd_long = iid_emb_layer(hist_iid_seq_long)
    hist_cid_emd_long = cid_emb_layer(hist_cid_seq_long)
    hist_bid_emd_long = iid_emb_layer(hist_bid_seq_long)
    hist_ts_diff_emd_long = iid_emb_layer(hist_ts_diff_seq_long)

    emb_dropout = args.emb_dropout
    if emb_dropout:
        drop_uid_layer = Dropout(0.1)
        drop_iid_layer = Dropout(0.1)
        drop_cid_layer = Dropout(0.1)
        drop_bid_layer = Dropout(0.1)
        drop_ts_diff_layer = Dropout(0.1)
        uid_emb = drop_uid_layer(uid_emb, training=training)
        hist_iid_emd_short = drop_iid_layer(hist_iid_emd_short, training=training)
        hist_cid_emb_short = drop_cid_layer(hist_cid_emb_short, training=training)
        hist_bid_emb_short = drop_bid_layer(hist_bid_emb_short, training=training)
        hist_ts_diff_emb_short = drop_ts_diff_layer(hist_ts_diff_emb_short, training=training)
        hist_iid_emd_long = drop_iid_layer(hist_iid_emd_long, training=training)
        hist_cid_emd_long = drop_cid_layer(hist_cid_emd_long, training=training)
        hist_bid_emd_long = drop_bid_layer(hist_bid_emd_long, training=training)
        hist_ts_diff_emd_long = drop_ts_diff_layer(hist_ts_diff_emd_long, training=training)


    short_emb_layer = SDMShortEncoderLayer(EMB_DIM, HEADS, ffn_hidden_unit=ffn_hidden_unit, dropout=dropout)
    long_emb_layer = SDMLongEncoderLayer(EMB_DIM, HEADS, ffn_hidden_unit=ffn_hidden_unit, dropout=dropout)
    gate_layer = SDMGateLayer(EMB_DIM, ffn_hidden_unit=ffn_hidden_unit)

    hist_emb_short = hist_iid_emd_short + hist_cid_emb_short + hist_bid_emb_short + hist_ts_diff_emb_short

    short_emb = short_emb_layer(
        [uid_emb, hist_emb_short],
        mask_short,
        training=training
    )
    long_emb = long_emb_layer(
        [uid_emb, hist_iid_emd_long, hist_cid_emd_long, hist_bid_emd_long, hist_ts_diff_emd_long],
        mask_long,
        training=training
    )

    output = gate_layer([uid_emb, short_emb, long_emb])
    output = flatten_layer(output)

    all_iid_index = Index(all_iid_index, name='all_iid_index')()
    all_iid_emb = iid_emb_layer(all_iid_index)
    all_iid_emb = flatten_layer(all_iid_emb)
    if emb_dropout:
        all_iid_emb = drop_iid_layer(all_iid_emb, training=training)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, EMB_DIM, flat_config)
    # index = faiss.IndexFlatIP(EMB_DIM)

    metrics_q = Queue()
    metrics_computer = ComputeMetrics(metrics_q, MATCH_POINTS, result_path)
    metrics_computer_p = Process(target=metrics_computer.compute_metrics)
    metrics_computer_p.daemon = True
    metrics_computer_p.start()

    # definde loss and train_op
    # bias = tf.get_variable(
    #     name='bias', shape=(ITEM_EMB_DIM + CATE_EMB_DIM,),
    #     initializer=tf.initializers.zeros(), trainable=True
    # )
    bias = tf.zeros(shape=(item_size,), dtype=tf.float32, name='unused_bias')
    loss = tf.nn.sampled_softmax_loss(
        weights=all_iid_emb,
        biases=bias,
        labels=iid,
        inputs=output,
        num_sampled=1024,
        num_classes=item_size
    )
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(args.lr)
    model_vars = tf.trainable_variables()
    grads = tf.gradients(loss, model_vars)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    train_op = optimizer.apply_gradients(zip(grads, model_vars))

    # training loop
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        sess.run(tf.initializers.global_variables())
        prev_time = time()
        test_num = 0
        batch_num = 0
        epoch = 0
        sess.run(train_iterator.initializer)
        exit_flag = False

        def _eval(prev_time, test_num):
            sess.run(test_iterator.initializer)
            all_item_out = sess.run(all_iid_emb, feed_dict={training: False})
            index.reset()
            # faiss.normalize_L2(val_all_item_out)
            index.add(all_item_out)
            test_batch = 1
            print('\n' + '=' * 40 + f'  test times: {test_num:6d}  ' + '=' * 40)
            while True:
                try:
                    user_out, test_iid = sess.run([output, iid], feed_dict={handle: test_handle, training: False})
                    # faiss.normalize_L2(val_user_out)
                    _, I = index.search(np.ascontiguousarray(user_out), MAX_MATCH_POINT)
                    metrics_q.put((I, test_iid, False, False))
                    print('\r' + '-' * 32 + f'   test batch {test_batch} finished   ' + '-' * 32, end='')
                    test_batch += 1
                except tf.errors.OutOfRangeError:
                    metrics_q.put((None, None, True, False))
                    curr_time = time()
                    time_elapsed = curr_time - prev_time
                    print('\n' + '-' * 72 + f'    time elapsed: {time_elapsed:.2f}s' + '\n' + '#' * 132)
                    break
            return curr_time

        while True:
            total_loss = 0.0
            for i in range(2048):
                try:
                    batch_loss, _ = sess.run([loss, train_op], feed_dict={handle: train_handle, training: True})
                    total_loss += batch_loss
                    batch_num += 1
                    print('\r' + '-' * 32 + f'  batch: {batch_num + 1}, loss: {batch_loss:.8f}  ' + '-' * 32, end='')
                except tf.errors.OutOfRangeError:
                    epoch += 1
                    test_num += 1
                    curr_time = _eval(prev_time, test_num)
                    time_elapsed = curr_time - prev_time
                    prev_time = curr_time
                    print(
                        '\n' + '#' * 132 + '\n'
                        + '=' * 32 + f'    train epoch: {epoch} finished    ' + '=' * 32
                        + '\n' + f'train_loss of epoch-{epoch}: {(total_loss / batch_num):.8f}    '
                        + '-' * 36 + f'    time elapsed: {time_elapsed:.2f}s'
                        + '\n' + '#' * 132
                    )
                    batch_num = 0
                    if (epoch == EPOCHS):
                        exit_flag = True
                    else:
                        sess.run(train_iterator.initializer)
            if exit_flag:
                break
            else:
                curr_time = time()
                time_elapsed = curr_time - prev_time
                print('\n' + '-' * 72 + f'    time elapsed: {time_elapsed:.2f}s' + '\n' + '#' * 132)
                test_num += 1
                prev_time = _eval(curr_time, test_num)

    metrics_q.put((None, None, False, True))
    metrics_computer_p.join()
