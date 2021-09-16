import os
from time import localtime, strftime, time

import faiss
from basek.utils.imports import numpy as np
from tqdm import tqdm

from basek.params import args
from basek.preprocessors.ml_preprocessor import read_raw_data, gen_dataset, gen_model_input
from basek.utils.metrics import compute_metrics
from basek.utils.tf_compat import tf, keras

from basek.layers.base import BiasAdd, Concatenate, Dense, Embedding, Flatten, Index, Input, Lambda


SEQ_LEN = 50
VALIDATION_SPLIT = True
NEG_SAMPLES = 0
NEG_WEIGHT = 0.1
EMB_DIM = 32
EPOCHS = 200
BATCH_SIZE = 256
SAMPLE_WEIGHT = 0.01
MATCH_NUMS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]
MAX_MATCH_NUM = 200


if __name__ == '__main__':

    data_path = './datasets/movielens'
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    data, sparse_features_max_idx = read_raw_data(data_path, sparse_features)
    user_size = sparse_features_max_idx['user_id']
    item_size = sparse_features_max_idx['movie_id']

    datasets, profiles = gen_dataset(data, VALIDATION_SPLIT, NEG_SAMPLES)
    train_dataset, val_dataset, test_dataset = datasets
    user_profile, item_profile = profiles

    train_input = gen_model_input(train_dataset, user_profile, item_profile, SEQ_LEN)
    val_input = gen_model_input(val_dataset, user_profile, item_profile, SEQ_LEN)
    test_input = gen_model_input(test_dataset, user_profile, item_profile, SEQ_LEN)

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

    uid_emb_layer = Embedding(
        user_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='uid_emb_layer'
    )
    iid_emb_layer = Embedding(
        item_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='iid_emb_layer'
    )
    uid_bias_emb_layer = Embedding(
        user_size, 1, embeddings_initializer=keras.initializers.Zeros(),
        mask_zero=True, name='uid_bias_emb_layer'
    )
    iid_bias_emb_layer = Embedding(
        item_size, 1, embeddings_initializer=keras.initializers.Zeros(),
        mask_zero=True, name='iid_bias_emb_layer'
    )

    uid_emb = uid_emb_layer(uid)
    iid_emb = iid_emb_layer(iid)
    uid_bias = uid_bias_emb_layer(uid)
    iid_bias = iid_bias_emb_layer(iid)

    user_out = Lambda(
        lambda x: tf.squeeze(x[0] + x[1], axis=1),
        name='user_out'
    )([uid_emb, uid_bias])
    item_out = Lambda(
        lambda x: tf.squeeze(x[0] + x[1], axis=1),
        name='item_out'
    )([iid_emb, iid_bias])

    true_prod_logits = Lambda(
        lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=-1, keepdims=True),
        name='prod_logits'
    )([user_out, item_out])
    bias_add_layer = BiasAdd(dim=1)
    logits = bias_add_layer(true_prod_logits)

    pred_prob = Lambda(
        lambda x: tf.nn.sigmoid(x), axis=1, name='pred_prob'
    )(logits)

    model = keras.Model(inputs=[uid, iid], outputs=[pred_prob])
    print('=' * 120)
    model.summary()

    # for evaluate
    all_item_index = Index(item_size)
    all_item_emb = iid_emb_layer(all_item_index())
    all_item_bias = iid_bias_emb_layer(all_item_index())
    all_item_out = all_item_emb + all_item_bias
    all_item_out = Flatten()(all_item_out)
    index = faiss.IndexFlatIP(EMB_DIM)

    # definde loss and train_op
    sampled_values = tf.random.uniform_candidate_sampler(
        true_classes=iid,
        num_true=1,
        num_sampled=10,
        unique=True,
        range_max=item_size
    )
    sampled_iid, true_expected_count, sampled_expected_count = (tf.stop_gradient(s) for s in sampled_values)
    sampled_iid_emb = iid_emb_layer(sampled_iid)
    sampled_iid_bias = iid_bias_emb_layer(sampled_iid)
    sampled_item_out = sampled_iid_emb + sampled_iid_bias

    sampled_prod_logits = tf.matmul(user_out, sampled_iid_emb, transpose_b=True)
    true_prod_logits -= tf.log(true_expected_count)
    sampled_prod_logits -= tf.log(sampled_expected_count)
    all_logits = tf.concat([true_prod_logits, sampled_prod_logits], axis=-1)
    all_logits = bias_add_layer(all_logits)

    true_labels = tf.ones_like(true_prod_logits)
    samples_labels = tf.zeros_like(sampled_prod_logits)
    all_labels = tf.concat([true_labels, samples_labels], axis=-1)
    all_labels = tf.stop_gradient(all_labels)
    true_weigts = tf.ones_like(true_prod_logits)
    sample_weights = SAMPLE_WEIGHT * tf.ones_like(sampled_prod_logits)
    all_weights = tf.concat([true_weigts, sample_weights], axis=-1)
    all_weights = tf.stop_gradient(all_weights)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=all_labels, logits=all_logits
    )
    loss = loss * all_weights
    loss = tf.reduce_mean(loss)

    optimizer = keras.optimizers.SGD(0.01)
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
        train_size = train_input['size']
        val_size = val_input['size']
        batch_num = (train_size - 1) // BATCH_SIZE + 1
        best_val_loss = float('inf')
        prev_time = time()
        all_idx = np.arange(train_size)
        for epoch in range(EPOCHS):
            total_loss = 0.0
            model.save_weights(ckpt_path + f'/{epoch}')
            np.random.shuffle(all_idx)
            for i in range(batch_num):
                batch_idx = all_idx[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch_loss, _ = sess.run(
                    [
                        loss, train_op
                    ],
                    feed_dict={
                        uid: train_input['uid'][batch_idx],
                        iid: train_input['iid'][batch_idx]
                    }
                )
                total_loss += batch_loss
                print('\r' + '-' * 42 + f'  batch_loss: {batch_loss:.8f}  '  + '-' * 42, end='')
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'\ntrain_loss of epoch-{epoch + 1}: {(total_loss / batch_num):.8f}    ' +
                  '-' * 36 + f'    time elapsed: {time_elapsed:.2f}s')
            if val_size == 0:
                continue
            val_loss, val_user_out, val_all_item_out = sess.run(
                [loss, user_out, all_item_out],
                feed_dict={
                    uid: val_input['uid'],
                    iid: val_input['iid']
                }
            )
            print(f'val_loss  of  epoch-{epoch + 1}: {val_loss:.8f}    ' + '-' * 72)
            # faiss.normalize_L2(val_user_out)
            # faiss.normalize_L2(val_all_item_out)
            index.reset()
            index.add(val_all_item_out)
            _, I = index.search(np.ascontiguousarray(val_user_out), MAX_MATCH_NUM)
            compute_metrics(I, MATCH_NUMS, val_input['uid'], val_input['iid'])
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print('-' * 74 + f'    time elapsed: {time_elapsed:.2f}s\n' + '=' * 120)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(ckpt_path + '/best')

        eval_loss, eval_user_out, eval_all_item_out = sess.run(
            [loss, user_out, all_item_out],
            feed_dict={
                uid: test_input['uid'],
                iid: test_input['iid']
            }
        )
        print('=' * 120)
        print(f'test_loss  of: {eval_loss:.8f}        ' + '-' * 72)
        index.reset()
        index.add(eval_all_item_out)
        _, I = index.search(np.ascontiguousarray(eval_user_out), MAX_MATCH_NUM)
        compute_metrics(I, MATCH_NUMS, test_input['uid'], test_input['iid'])
        curr_time = time()
        time_elapsed = curr_time - prev_time
        print('-' * 74 + f'    time elapsed: {time_elapsed:.2f}s\n' + '=' * 120)
