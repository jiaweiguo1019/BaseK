import os
import pickle as pkl
from time import localtime, strftime, time

import faiss
from basek.utils.imports import numpy as np
from basek.utils.imports import random

from basek.params import args
from basek.preprocessors.amazon_preprocessor import read_reviews, DataLoader
from basek.utils.metrics import compute_metrics
from basek.utils.tf_compat import tf, keras
import tensorflow_addons as tfa

from basek.layers.base import BiasAdd, Concatenate, Dense, Embedding, Flatten, Index, Input, Lambda


SEQ_LEN = 100
VALIDATION_SPLIT = True
NEG_SAMPLES = 0
EMB_DIM = 32
EPOCHS = args.epochs
BATCH_SIZE = 2560
MATCH_NUMS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]
MAX_MATCH_NUM = 300


if __name__ == '__main__':

    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    sparse_features = ['uid', 'iid', 'cid']
#    meta_path = './datasets/Amazon/meta_Books.json'
#    review_path = './datasets/Amazon/reviews_Books_5.json'
#    data_files, sparse_features_max_idx_path, all_indices_path = read_reviews(meta_path, review_path, NEG_SAMPLES)
    data_files = './datasets/Amazon/train.pkl', './datasets/Amazon/test.pkl'
    sparse_features_max_idx_path = './datasets/Amazon/sparse_features_max_idx.pkl'
    all_indices_path = './datasets/Amazon/all_indices.pkl'
    with open(sparse_features_max_idx_path, 'rb') as f:
        sparse_features_max_idx = pkl.load(f)
    with open(all_indices_path, 'rb') as f:
        all_indices = pkl.load(f)

    train_file, test_file = data_files
    train_input = DataLoader(train_file, SEQ_LEN, BATCH_SIZE, )
    val_input = DataLoader(test_file, SEQ_LEN, BATCH_SIZE * 10)
    user_size = sparse_features_max_idx['uid']
    item_size = sparse_features_max_idx['iid']
    cate_size = sparse_features_max_idx['cid']
    all_iid_index, all_cid_index = all_indices
    print('=' * 120)
    print(
        '-' * 16 + f'    user_size: {user_size}, item_size: {item_size}, ' +
        f'cate_size: {cate_size}    ' + '-' * 16
    )

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    ckpt_path = './ckpts/' + timestamp
    log_path = './logs/' + timestamp
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    print('=' * 120)
    print('-' * 4 + f'  model weights are saved in {os.path.realpath(ckpt_path)}  ' + '-' * 4)

    # bulid model
    uid = Input(shape=[1], dtype=tf.int64, name='uid')
    iid = Input(shape=[1], dtype=tf.int64, name='iid')
    cid = Input(shape=[1], dtype=tf.int64, name='cid')
    label = Input(shape=[1], dtype=tf.float32, name='label')
    hist_iid_seq = Input(shape=[SEQ_LEN], dtype=tf.int64, name='hist_iid_seq')
    hist_cid_seq = Input(shape=[SEQ_LEN], dtype=tf.int64, name='hist_cid_seq')
    hist_seq_len = Input(shape=[1], dtype=tf.int64, name='hist_seq_len')

    uid_emb_layer = Embedding(
        user_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='uid_emb_layer'
    )
    iid_emb_layer = Embedding(
        item_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='iid_emb_layer'
    )
    cid_emb_layer = Embedding(
        cate_size, EMB_DIM, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='cid_emb_layer'
    )

    uid_emb = uid_emb_layer(uid)
    iid_emb = iid_emb_layer(iid)
    cid_emb = cid_emb_layer(cid)

    mask = Lambda(
        lambda x: tf.transpose(tf.sequence_mask(x, SEQ_LEN, dtype=tf.float32), [0, 2, 1]),
        name='mask'
    )(hist_seq_len)
    hist_seq_len_for_avg_pooling = Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True) + 1e-16, name='hist_seq_len_for_avg_pooling'
    )(mask)

    hist_iid_seq_emb = iid_emb_layer(hist_iid_seq)
    hist_iid_seq_emb = Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True), name='masked_hist_item_seq_emb'
    )([hist_iid_seq_emb, mask])
    hist_iid_seq_emb = Lambda(
        lambda x: x[0] / x[1], name='pooled_hist_item_seq_emb'
    )([hist_iid_seq_emb, hist_seq_len_for_avg_pooling])

    hist_cid_seq_emb = cid_emb_layer(hist_cid_seq)
    hist_cid_seq_emb = Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True), name='masked_hist_item_seq_emb'
    )([hist_cid_seq_emb, mask])
    hist_iid_seq_emb = Lambda(
        lambda x: x[0] / x[1], name='pooled_hist_item_seq_emb'
    )([hist_cid_seq_emb, hist_seq_len_for_avg_pooling])

    concatenated_all_embs = Concatenate(
        axis=-1,
        name='concatenated_all_embs'
    )([uid_emb, hist_iid_seq_emb, hist_cid_seq_emb])
    all_embs = Flatten(name='flattened_all_embs')(concatenated_all_embs)

    user_hidden_1 = Dense(1024, 'relu', name='user_hidden_1')(all_embs)
    user_hidden_2 = Dense(512, 'relu', name='user_hidden_2')(user_hidden_1)
    user_hidden_3 = Dense(256, 'relu', name='user_hidden_3')(user_hidden_2)
    user_out = Dense(EMB_DIM, name='user_out')(user_hidden_3)

    item_out = Flatten(name='item_out')(iid_emb + cid_emb)

    model = keras.Model(
        inputs=[uid, iid, cid, hist_iid_seq, hist_cid_seq, hist_seq_len],
        outputs=[user_out, item_out]
    )
    print('=' * 120)
    model.summary()

    # for evaluate
    all_iid_index, all_cid_index = \
        Index(all_iid_index, name='all_iid_index'), Index(all_cid_index, name='all_cid_index')
    all_iid_emb = iid_emb_layer(all_iid_index())
    all_cid_emb = cid_emb_layer(all_cid_index())
    all_item_out = Flatten()(all_iid_emb + all_cid_emb)
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, EMB_DIM, flat_config)

    # definde loss and train_op
    bias = tf.get_variable(name='bias', shape=[item_size], initializer=tf.initializers.zeros(), trainable=False)
    loss = tf.nn.sampled_softmax_loss(
        weights=all_item_out,
        biases=bias,
        labels=iid,
        inputs=user_out,
        num_sampled=1024,
        num_classes=item_size
    )
    loss = tf.reduce_mean(loss)
    optimizer = tfa.optimizers.AdamW(1e-4)
    model_vars = tf.trainable_variables()
    grads = tf.gradients(loss, model_vars)
    train_op = optimizer.apply_gradients(zip(grads, model_vars))

    # training loop
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    print('=' * 120)
    with tf.Session(config=config) as sess:
        sess.run(tf.initializers.global_variables())

        best_val_loss = float('inf')
        prev_time = time()

        for epoch in range(EPOCHS):
            total_loss = 0.0
            model.save_weights(ckpt_path + f'/{epoch}')
            batch_num = 0
            for train_data in train_input:
                batch_num += 1
                uid_, iid_, cid_, label_, hist_iid_seq_, hist_cid_seq_, \
                    hist_seq_len_, rating_, sample_weight_ = train_data
                batch_loss, _ = sess.run(
                    [
                        loss, train_op
                    ],
                    feed_dict={
                        uid: uid_, iid: iid_, cid: cid_,
                        hist_iid_seq: hist_iid_seq_,
                        hist_cid_seq: hist_cid_seq_,
                        hist_seq_len: hist_seq_len_
                    }
                )
                total_loss += batch_loss
                print('\r' + '-' * 42 + f'  batch_loss: {batch_loss:.8f}  ' + '-' * 42, end='')
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'\ntrain_loss of epoch-{epoch + 1}: {(total_loss / batch_num):.8f}    ' +
                  '-' * 36 + f'    time elapsed: {time_elapsed:.2f}s')

            total_loss = 0.0
            total_samples = 0
            val_all_item_out = sess.run(all_item_out)
            index.reset()
            index.add(val_all_item_out)
            hits, ndcgs = [[] for _ in range(len(MATCH_NUMS))], [[] for _ in range(len(MATCH_NUMS))]
            for val_data in val_input:
                uid_, iid_, cid_, label_, hist_iid_seq_, hist_cid_seq_, \
                    hist_seq_len_, rating_, sample_weight_ = val_data
                batch_loss, val_user_out = sess.run(
                    [loss, user_out],
                    feed_dict={
                        uid: uid_, iid: iid_, cid: cid_,
                        hist_iid_seq: hist_iid_seq_,
                        hist_cid_seq: hist_cid_seq_,
                        hist_seq_len: hist_seq_len_
                    }
                )
                print('\r' + '-' * 42 + f'  batch_loss: {batch_loss:.8f}  ' + '-' * 42, end='')
                num_batch_sample = len(uid_)
                total_loss += batch_loss * num_batch_sample
                total_samples += num_batch_sample
                _, I = index.search(np.ascontiguousarray(val_user_out), MAX_MATCH_NUM)
                batch_hits, batch_ndcgs = compute_metrics(I, MATCH_NUMS, uid_, iid_)
                for idx, (hit_i, ndcg_i) in enumerate(zip(hits, ndcgs)):
                    hit_i.extend(batch_hits[idx])
                    ndcg_i.extend(batch_ndcgs[idx])
            val_loss = total_loss / total_samples
            print(f'\nval_loss  of  epoch-{epoch + 1}: {val_loss:.8f}    ' + '-' * 72)
            # faiss.normalize_L2(val_user_out)
            # faiss.normalize_L2(val_all_item_out)
            for idx, (hit_i, ndcg_i) in enumerate(zip(hits, ndcgs)):
                mean_hr = np.mean(hit_i)
                mean_ndcg = np.mean(ndcg_i)
                print('-' * 36 + f'  HR@{MATCH_NUMS[idx]}: {mean_hr:.6f}, ' +
                      f'NDCG@{MATCH_NUMS[idx]}: {mean_ndcg:.6f}  ' + '-' * 36)
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print('-' * 74 + f'    time elapsed: {time_elapsed:.2f}s\n' + '=' * 120)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(ckpt_path + '/best')
