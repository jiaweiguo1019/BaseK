import os
from time import localtime, strftime, time

import faiss
import numpy as np
from numpy.core.fromnumeric import transpose
from tqdm import tqdm

from basek.params import args
from basek.preprocessors.ml_preprocessor import read_raw_data, gen_dataset, gen_model_input
from basek.utils.metrics import compute_metrics
from basek.utils.tf_compat import tf, keras

from basek.layers.base import BiasAdd, Concatenate, Dense, Embedding, Flatten, Index, Input, Lambda

SEQ_LEN = 50
# NEG_SAMPLES = 5
NEG_SAMPLES = 0
NEG_WEIGHT = 0.2
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

    datasets, profiles =  gen_dataset(data, VALIDATION_SPLIT, NEG_SAMPLES, NEG_WEIGHT)
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
    uid = keras.Input(shape=[1,], dtype=tf.int64, name='uid')
    iid = keras.Input(shape=[1,], dtype=tf.int64, name='iid')

    uid_emb_layer = keras.layers.Embedding(
        user_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='uid_emb_layer'
    )
    iid_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='iid_emb_layer'
    )
    uid_bias_emb_layer = keras.layers.Embedding(
        user_size, 1, embeddings_initializer=keras.initializers.Zeros(),
        mask_zero=True, name='uid_bias_emb_layer'
    )
    iid_bias_emb_layer = keras.layers.Embedding(
        item_size, 1, embeddings_initializer=keras.initializers.Zeros(),
        mask_zero=True, name='iid_bias_emb_layer'
    )

    uid_emb = uid_emb_layer(uid)
    iid_emb = iid_emb_layer(iid)
    uid_bias = uid_bias_emb_layer(uid)
    iid_bias = iid_bias_emb_layer(iid)

    emb_prod = keras.layers.Lambda(
        lambda x: tf.matmul(x[0], tf.transpose(x[1], [0, 2, 1])),
        name='emb_prod'
    )([uid_emb, iid_emb])
    added_score = keras.layers.Lambda(
        lambda x: x[0] + x[1] + x[2], name='added_score'
    )([emb_prod, uid_bias, iid_bias])
    bias_add_layer = BiasAdd(dim=1, name='bias_add')
    logits = bias_add_layer(added_score)
    logits = keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=1), name='flattened_final_score'
    )(logits)

    pred_prob = keras.layers.Lambda(
        lambda x: tf.nn.sigmoid(x), name='sigmoid'
    )(logits)

    model = keras.Model(inputs=[uid, iid], outputs=[logits])
    print('=' * 120)
    model.summary()

    # for evaluate
    all_item_index = Index(item_size)
    all_item_emb = iid_emb_layer(all_item_index())
    all_item_bias = iid_bias_emb_layer(all_item_index())

    uid_emb_squeezed = tf.squeeze(uid_emb, axis=1)
    uid_bias_squeezed = tf.squeeze(uid_bias, axis=1)
    uid_emb_prob_all_item = tf.matmul(uid_emb_squeezed, all_item_emb, transpose_b=True)
    added_score_all_item = uid_emb_prob_all_item + uid_bias_squeezed + tf.transpose(all_item_bias)
    final_score = bias_add_layer(added_score_all_item)

    # definde loss and train_op
    label = keras.Input(shape=[1,], dtype=tf.float32, name='label')
    sample_weight = keras.Input(shape=[1,], dtype=tf.float32, name='sample_weight')

    # loss = -sample_weight * (label * tf.log(pred_prob) + (1.0 - label) * tf.log(1.0 - pred_prob))
    loss = (label - logits) ** 2
    loss = tf.reduce_mean(loss)
    optimizer = keras.optimizers.RMSprop(0.0001)
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
                        # label: train_input['label'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        label: train_input['feedback'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        sample_weight: train_input['sample_weight'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                    }
                )
                total_loss += batch_loss
                print('\r' + '-' * 32 + ' ' * 6 + f'batch_loss: {batch_loss:.8f}' + ' ' * 6  + '-' * 32, end='')
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'\ntrain_loss of epoch-{epoch + 1}: {(total_loss / batch_num):.8f}    ' +
                  '-' * 36 + f'    tiem elapsed: {time_elapsed:.2f}s')
            if val_size == 0:
                continue
            val_loss, val_score = sess.run(
                [loss, final_score],
                feed_dict={
                    uid: val_input['uid'],
                    iid: val_input['iid'],
                    # label: val_input['label'],
                    label: val_input['feedback'],
                    sample_weight: val_input['sample_weight']
                }
            )

            I = np.argsort(val_score)
            I = I[:, ::-1][:, :MATCH_NUMS]
            hr, ndcg = compute_metrics(I, val_input['uid'], val_input['iid'])
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'val_loss  of  epoch-{epoch + 1}: {val_loss:.8f}    ' +
                  '-' * 36 + f'    tiem elapsed: {time_elapsed:.2f}s')
            print('-' * 32 + f'   hr: {hr:.6f}, ndcg: {ndcg:.6f}   ' + '-' * 32)
            print('=' * 120)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(ckpt_path + '/best')

        eval_loss, val_score = sess.run(
            [loss, final_score],
            feed_dict={
                uid: test_input['uid'],
                iid: test_input['iid'],
                # label: test_input['label'],
                label: test_input['feedback'],
                sample_weight: test_input['sample_weight']
            }
        )
        I = np.argsort(val_score)
        I = I[:, ::-1][:, :MATCH_NUMS]
        hr, ndcg = compute_metrics(I, test_input['uid'], test_input['iid'])
        curr_time = time()
        time_elapsed = curr_time - prev_time
        prev_time = curr_time
        print('=' * 120)
        print(f'-------------test_loss of: {eval_loss:.8f}, \
            tiem elapsed: {time_elapsed:.2f}s -------------')
        print('-' * 32 + f'   hr: {hr:.6f}, ndcg: {ndcg:.6f}   ' + '-' * 32)
