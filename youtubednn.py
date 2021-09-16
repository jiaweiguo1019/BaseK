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
EMB_DIM = 32
EPOCHS = 200
BATCH_SIZE = 256
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
    label = Input(shape=[1], dtype=tf.float32, name='label')
    hist_item_seq = Input(shape=[SEQ_LEN], dtype=tf.int64, name='hist_item_seq')
    hist_item_len = Input(shape=[1], dtype=tf.int64, name='hist_item_len')
    gender = Input(shape=[1], dtype=tf.int64, name='gender')
    age = Input(shape=[1], dtype=tf.int64, name='age')
    occupation = Input(shape=[1], dtype=tf.int64, name='occupation')
    zip_input = Input(shape=[1], dtype=tf.int64, name='zip')

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
    gender_emb_layer = Embedding(
        sparse_features_max_idx['gender'], 2, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='gender_emb_layer'
    )
    age_emb_layer = Embedding(
        sparse_features_max_idx['age'], 4, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='age_emb_layer'
    )
    occupation_emb_layer = Embedding(
        sparse_features_max_idx['occupation'], 4, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='occupation_emb_layer'

    )
    zip_emb_layer = Embedding(
        sparse_features_max_idx['zip'], 4, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        name='zip_emb_layer'
    )

    uid_emb = uid_emb_layer(uid)
    gender_emb = gender_emb_layer(gender)
    age_emb = age_emb_layer(age)
    occupation_emb = occupation_emb_layer(occupation)
    zip_emb = zip_emb_layer(zip_input)

    mask = Lambda(
        lambda x: tf.transpose(tf.sequence_mask(x, SEQ_LEN, dtype=tf.float32), [0, 2, 1]),
        name='mask'
    )(hist_item_len)
    hist_item_len_for_avg_pooling = Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='len_for_avg_pooling'
    )(mask)
    hist_item_seq_emb = iid_emb_layer(hist_item_seq)
    hist_item_seq_emb = Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True), name='masked_hist_item_seq_emb'
    )([hist_item_seq_emb, mask])
    hist_item_seq_emb = Lambda(
        lambda x: x[0] / x[1], name='pooled_hist_item_seq_emb'
    )([hist_item_seq_emb, hist_item_len_for_avg_pooling])
    all_embs = Concatenate(
        axis=-1
    )([uid_emb, hist_item_seq_emb, gender_emb, age_emb, occupation_emb, zip_emb])
    all_embs = Flatten()(all_embs)

    user_hidden_1 = Dense(1024, 'relu', name='user_hidden_1')(all_embs)
    user_hidden_2 = Dense(512, 'relu', name='user_hidden_2')(user_hidden_1)
    user_hidden_3 = Dense(256, 'relu', name='user_hidden_3')(user_hidden_2)
    user_out = Dense(EMB_DIM, name='user_out')(user_hidden_3)

    model = keras.Model(
        inputs=[uid, hist_item_seq, hist_item_len, gender, age, occupation, zip_input],
        outputs=[user_out]
    )
    print('=' * 120)
    model.summary()

    # for evaluate
    all_item_index = Index(item_size)
    all_item_emb = iid_emb_layer(all_item_index())
    all_item_emb = Flatten()(all_item_emb)
    index = faiss.IndexFlatIP(EMB_DIM)

    # definde loss and train_op
    iid = Input(shape=[1], dtype=tf.int64, name='iid')
    bias = tf.get_variable(name='bias', shape=[item_size], initializer=tf.initializers.zeros(), trainable=False)
    loss = tf.nn.sampled_softmax_loss(
        weights=all_item_emb,
        biases=bias,
        labels=iid,
        inputs=user_out,
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
                        iid: train_input['iid'][batch_idx],
                        hist_item_seq: train_input['hist_item_seq'][batch_idx],
                        hist_item_len: train_input['hist_item_len'][batch_idx],
                        gender: train_input['gender'][batch_idx],
                        age: train_input['age'][batch_idx],
                        occupation: train_input['occupation'][batch_idx],
                        zip_input: train_input['zip'][batch_idx],
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
            val_loss, val_user_emb, val_all_item_emb = sess.run(
                [loss, user_out, all_item_emb],
                feed_dict={
                    uid: val_input['uid'], iid: val_input['iid'],
                    hist_item_seq: val_input['hist_item_seq'], hist_item_len: val_input['hist_item_len'],
                    gender: val_input['gender'], age: val_input['age'],
                    occupation: val_input['occupation'], zip_input: val_input['zip']
                }
            )
            print(f'val_loss  of  epoch-{epoch + 1}: {val_loss:.8f}    ' + '-' * 72)
            # faiss.normalize_L2(val_user_emb)
            # faiss.normalize_L2(val_all_item_emb)
            index.reset()
            index.add(val_all_item_emb)
            _, I = index.search(np.ascontiguousarray(val_user_emb), MAX_MATCH_NUM)
            compute_metrics(I, MATCH_NUMS, val_input['uid'], val_input['iid'])
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print('-' * 74 + f'    time elapsed: {time_elapsed:.2f}s\n' + '=' * 120)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(ckpt_path + '/best')

        eval_loss, eval_user_emb, eval_all_item_emb = sess.run(
            [loss, user_out, all_item_emb],
            feed_dict={
                uid: test_input['uid'], iid: test_input['iid'],
                hist_item_seq: test_input['hist_item_seq'], hist_item_len: test_input['hist_item_len'],
                gender: test_input['gender'], age: test_input['age'],
                occupation: test_input['occupation'], zip_input: test_input['zip']
            }
        )
        print('=' * 120)
        print(f'test_loss  of: {eval_loss:.8f}        ' + '-' * 72)
        index.reset()
        index.add(eval_all_item_emb)
        _, I = index.search(np.ascontiguousarray(eval_user_emb), MAX_MATCH_NUM)
        compute_metrics(I, MATCH_NUMS, test_input['uid'], test_input['iid'])
        curr_time = time()
        time_elapsed = curr_time - prev_time
        print('-' * 74 + f'    time elapsed: {time_elapsed:.2f}s\n' + '=' * 120)
