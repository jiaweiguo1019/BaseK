import os
from time import localtime, strftime, time

import faiss
import numpy as np
from tqdm import tqdm

from basek.params import args
from basek.preprocessors.ml_preprocessor import read_raw_data, gen_dataset, gen_model_input
from basek.utils.metrics import compute_metrics
from basek.utils.tf_compat import tf, keras

from basek.layers.base import BiasAdd, Concatenate, Dense, Embedding, Flatten, Index, Input, Lambda


SEQ_LEN = 50
NEG_SAMPLES = 0
VALIDATION_SPLIT = 0.1
EMB_DIM = 32
EPOCHS = 1000
BATCH_SIZE = 256
MATCH_NUMS = 50


if __name__ == '__main__':

    data_path ='./datasets/movielens'
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    data, sparse_features_max_idx = read_raw_data(data_path, sparse_features)
    user_size = sparse_features_max_idx['user_id']
    item_size = sparse_features_max_idx['movie_id']

    datasets, profiles =  gen_dataset(data, VALIDATION_SPLIT, NEG_SAMPLES)
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
    uid = keras.Input(shape=[1, ], dtype=tf.int64, name='uid')
    label = keras.Input(shape=[1,], dtype=tf.float32, name='label')
    hist_item_seq = keras.Input(shape=[SEQ_LEN, ], dtype=tf.int64, name='hist_item_seq')
    hist_item_len = keras.Input(shape=[1, ], dtype=tf.int64, name='hist_item_len')
    gender = keras.Input(shape=[1, ], dtype=tf.int64, name='gender')
    age = keras.Input(shape=[1, ], dtype=tf.int64, name='age')
    occupation = keras.Input(shape=[1, ], dtype=tf.int64, name='occupation')
    zip_input = keras.Input(shape=[1, ], dtype=tf.int64, name='zip')

    linear_uid_emb_layer = keras.layers.Embedding(
        user_size, 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='linear_uid_emb_layer'
    )
    linear_iid_emb_layer = keras.layers.Embedding(
        item_size, 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='linear_iid_emb_layer'
    )
    linear_gender_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['gender'], 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='linear_gender_emb_layer'
    )
    linear_age_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['age'], 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='linear_age_emb_layer'
    )
    linear_occupation_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['occupation'], 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='linear_occupation_emb_layer'
    )
    linear_zip_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['zip'], 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='linear_zip_emb_layer'
    )
    linear_uid_emb = linear_uid_emb_layer(uid)
    linear_gender_emb = linear_gender_emb_layer(gender)
    linear_age_emb = linear_age_emb_layer(age)
    linear_occupation_emb = linear_occupation_emb_layer(occupation)
    linear_zip_emb = linear_zip_emb_layer(zip_input)

    cross_uid_emb_layer = keras.layers.Embedding(
        user_size, EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='cross_uid_emb_layer'
    )
    cross_iid_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='cross_iid_emb_layer'
    )
    cross_gender_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['gender'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='cross_gender_emb_layer'

    )
    cross_age_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['age'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='cross_age_emb_layer'
    )
    cross_occupation_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['occupation'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='cross_occupation_emb_layer'
    )
    cross_zip_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['zip'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='cross_zip_emb_layer'
    )

    cross_uid_emb = cross_uid_emb_layer(uid)
    cross_gender_emb = cross_gender_emb_layer(gender)
    cross_age_emb = cross_age_emb_layer(age)
    cross_occupation_emb = cross_occupation_emb_layer(occupation)
    cross_zip_emb = cross_zip_emb_layer(zip_input)

    mask = keras.layers.Lambda(
        lambda x: tf.transpose(tf.sequence_mask(x, SEQ_LEN, dtype=tf.float32), [0, 2, 1]),
        name='mask'
    )(hist_item_len)
    hist_item_len_for_avg_pooling = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='len_for_avg_pooling'
    )(mask)
    linear_hist_item_seq_emb = linear_iid_emb_layer(hist_item_seq)
    linear_hist_item_seq_emb = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True),
        name='masked_linear_hist_item_seq_emb'
    )([linear_hist_item_seq_emb, mask])

    linear_hist_item_seq_emb = keras.layers.Lambda(
        lambda x: x[0] / x[1], name='pooled_linear_hist_item_seq_emb'
    )([linear_hist_item_seq_emb, hist_item_len_for_avg_pooling])
    concatenated_linear_emb = keras.layers.Concatenate(axis=1, name='concatenated_linear_emb')(
        [
            linear_uid_emb, linear_hist_item_seq_emb,
            linear_gender_emb, linear_age_emb, linear_occupation_emb, linear_zip_emb
        ]
    )
    linear_logits = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1), name='linear_logits'
    )(concatenated_linear_emb)

    cross_hist_item_seq_emb = cross_iid_emb_layer(hist_item_seq)
    cross_hist_item_seq_emb = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True),
        name='masked_cross_hist_item_seq_emb'
    )([cross_hist_item_seq_emb, mask])
    cross_hist_item_seq_emb = keras.layers.Lambda(
        lambda x: x[0] / x[1], name='pooled_cross_hist_item_seq_emb'
    )([cross_hist_item_seq_emb, hist_item_len_for_avg_pooling])
    concatenated_cross_emb = keras.layers.Lambda(
        lambda x: tf.concat(x, axis=1),
        name='concatenated_cross_emb'
    )(
        [
            cross_uid_emb, cross_hist_item_seq_emb,
            cross_gender_emb, cross_age_emb, cross_occupation_emb, cross_zip_emb
        ]
    )
    cross_sum_of_square = keras.layers.Lambda(
        lambda x: tf.reduce_sum(tf.square(x), axis=1),
        name='cross_sum_of_square'
    )(concatenated_cross_emb)
    cross_square_of_sum = keras.layers.Lambda(
        lambda x: tf.reduce_sum(tf.square(x), axis=1),
        name='cross_square_of_sum'
    )(concatenated_cross_emb)
    cross_logits = keras.layers.Lambda(
        lambda x: 0.5 * tf.reduce_sum(x[0] - x[1], axis=-1, keepdims=True),
        name='cross_logits'
    )([cross_sum_of_square, cross_square_of_sum])

    added_logits = keras.layers.Lambda(
        lambda x: x[0] + x[1],
        name='added_logits'
    )([linear_logits, cross_logits])

    bias_add_layer = BiasAdd(dim=1, name='bias_add')
    final_logits = bias_add_layer(added_logits)

    model = keras.Model(
        inputs=[uid, hist_item_seq, hist_item_len, gender, age, occupation, zip_input],
        outputs=[final_logits]
    )
    print('=' * 120)
    model.summary()

    # for evaluate
    all_item_index = Index(item_size)
    all_item_emb = linear_iid_emb_layer(all_item_index())

    uid_emb_squeezed = tf.squeeze(uid_emb, axis=1)
    uid_bias_squeezed = tf.squeeze(uid_bias, axis=1)
    uid_emb_prob_all_item = tf.matmul(uid_emb_squeezed, all_item_emb, transpose_b=True)
    added_score_all_item = uid_emb_prob_all_item + uid_bias_squeezed + tf.transpose(all_item_bias)
    final_score = bias_add_layer(added_score_all_item)


    # definde loss and train_op
    iid = keras.Input(shape=[1,], dtype=tf.int64, name='iid')
    bias = tf.get_variable(name='bias', shape=[item_size,], initializer=tf.initializers.zeros(), trainable=False)
    loss = tf.nn.sampled_softmax_loss(
        weights=all_item_emb,
        biases=bias,
        labels=iid,
        inputs=user_emb,
        num_sampled=160,
        num_classes=item_size
    )
    loss = tf.reduce_mean(loss)
    optimizer = keras.optimizers.Adam()
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
        for epoch in range(EPOCHS):
            total_loss = 0.0
            model.save_weights(ckpt_path + f'/{epoch}')
            for i in range(batch_num):
                batch_loss, _ = sess.run(
                    [
                        loss, train_op
                    ],
                    feed_dict={
                        uid: train_input['uid'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        iid: train_input['iid'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        hist_item_seq: train_input['hist_item_seq'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        hist_item_len: train_input['hist_item_len'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        gender: train_input['gender'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        age: train_input['age'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        occupation: train_input['occupation'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        zip_input: train_input['zip'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    }
                )
                total_loss += batch_loss
                print('\r' + '-' * 42 + f'  batch_loss: {batch_loss:.8f}  '  + '-' * 42, end='')
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'\ntrain_loss of epoch-{epoch + 1}: {(total_loss / batch_num):.8f}    ' +
                  '-' * 36 + f'    tiem elapsed: {time_elapsed:.2f}s')
            if val_size == 0:
                continue
            val_loss, val_user_emb, val_all_item_emb = sess.run(
                [loss, user_emb, all_item_emb],
                feed_dict={
                    uid: val_input['uid'], iid: val_input['iid'],
                    hist_item_seq: val_input['hist_item_seq'], hist_item_len: val_input['hist_item_len'],
                    gender: val_input['gender'], age: val_input['age'],
                    occupation: val_input['occupation'], zip_input: val_input['zip']
                }
            )
            # faiss.normalize_L2(val_user_emb)
            # faiss.normalize_L2(val_all_item_emb)
            index.reset()
            index.add(val_all_item_emb)
            _, I = index.search(np.ascontiguousarray(val_user_emb), MATCH_NUMS)
            hr, ndcg = compute_metrics(I, val_input['uid'], val_input['iid'])
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'val_loss  of  epoch-{epoch + 1}: {val_loss:.8f}    ' +
                  '-' * 36 + f'    tiem elapsed: {time_elapsed:.2f}s')
            print('-' * 36 + f'  hr@{MATCH_NUMS}: {hr:.6f}, ndcg@{MATCH_NUMS}: {ndcg:.6f}  ' + '-' * 36)
            print('=' * 120)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(ckpt_path + '/best')

        eval_loss, eval_user_emb, eval_all_item_emb = sess.run(
            [loss, user_emb, all_item_emb],
            feed_dict={
                uid: test_input['uid'], iid: test_input['iid'],
                hist_item_seq: test_input['hist_item_seq'], hist_item_len: test_input['hist_item_len'],
                gender: test_input['gender'], age: test_input['age'],
                occupation: test_input['occupation'], zip_input: test_input['zip']
            }
        )
        index.reset()
        index.add(eval_all_item_emb)
        _, I = index.search(np.ascontiguousarray(eval_user_emb), MATCH_NUMS)
        hr, ndcg = compute_metrics(I, test_input['uid'], test_input['iid'])
        curr_time = time()
        time_elapsed = curr_time - prev_time
        prev_time = curr_time
        print('=' * 120)
        print(f'-------------test_loss of: {eval_loss:.8f}, \
            tiem elapsed: {time_elapsed:.2f}s -------------')
        print('-' * 32 + f'   hr: {hr:.6f}, ndcg: {ndcg:.6f}   ' + '-' * 32)

