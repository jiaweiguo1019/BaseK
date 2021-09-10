import os
from time import localtime, strftime, time

import faiss
import numpy as np
from tqdm import tqdm

from basek.params import args
from basek.preprocessors.ml_preprocessor import read_raw_data, gen_dataset, gen_model_input
from basek.layers.embedding import EmbeddingIndex
from basek.utils.metrics import compute_metrics
from basek.utils.tf_compat import tf, keras


SEQ_LEN = 50
NEG_SAMPLES = 0
VALIDATION_SPLIT = 0.1
EMB_DIM = 32
EPOCHS = 2
BATCH_SIZE = 256
MATCH_NUMS = 50


if __name__ == '__main__':

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    ckpt_path = './ckpts/' + timestamp
    log_path = './logs/' + timestamp
    os.makedirs(ckpt_path)
    os.makedirs(log_path)

    data_path ='./datasets/movielens'
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    data, sparse_features_max_idx = read_raw_data(data_path, sparse_features)
    user_size = sparse_features_max_idx['user_id']
    item_size = sparse_features_max_idx['movie_id']

    datasets, profiles =  gen_dataset(data, NEG_SAMPLES)
    train_dataset, test_dataset = datasets
    user_profile, item_profile = profiles

    train_input, val_input = gen_model_input(train_dataset, user_profile, item_profile, SEQ_LEN, VALIDATION_SPLIT)
    test_input, _ = gen_model_input(test_dataset, user_profile, item_profile, SEQ_LEN)

    uid = keras.Input(shape=[1,], dtype=tf.int64, name='uid')
    lable = keras.Input(shape=[1,], dtype=tf.float32, name='lable')
    hist_item_seq = keras.Input(shape=[SEQ_LEN,], dtype=tf.int64, name='hist_item_seq')
    hist_item_len = keras.Input(shape=[1,], dtype=tf.int64, name='hist_item_len')
    gender = keras.Input(shape=[1,], dtype=tf.int64, name='gender')
    age = keras.Input(shape=[1,], dtype=tf.int64, name='age')
    occupation = keras.Input(shape=[1,], dtype=tf.int64, name='occupation')
    zip_input = keras.Input(shape=[1,], dtype=tf.int64, name='zip')

    uid_emb_layer = keras.layers.Embedding(
        user_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(), mask_zero=True
    )
    iid_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(), mask_zero=True
    )
    gender_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['gender'], 2,  mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal()
    )
    age_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['age'], 4, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal()
    )
    occupation_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['occupation'], 4, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal()
    )
    zip_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['zip'], 4, mask_zero=True,
        embeddings_initializer=keras.initializers.TruncatedNormal()
    )

    uid_emb = uid_emb_layer(uid)
    gender_emb = gender_emb_layer(gender)
    age_emb = age_emb_layer(age)
    occupation_emb = occupation_emb_layer(occupation)
    zip_emb = zip_emb_layer(zip_input)

    hist_item_seq_emb = iid_emb_layer(hist_item_seq)
    mask = keras.layers.Lambda(
        lambda x: tf.sequence_mask(x, SEQ_LEN, dtype=tf.float32)
    )(hist_item_len)
    mask = keras.layers.Lambda(
        lambda x: tf.transpose(x, [0, 2, 1])
    )(mask)
    hist_item_seq_emb = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True)
    )([hist_item_seq_emb, mask])
    hist_item_len_for_mean = keras.layers.Lambda(
        lambda x: tf.cast(tf.expand_dims(x, axis=1), tf.float32)
    )(hist_item_len)
    hist_item_seq_emb = keras.layers.Lambda(
        lambda x: x[0] / x[1]
    )([hist_item_seq_emb, hist_item_len_for_mean])

    all_embs = keras.layers.Concatenate(
        axis=-1
    )([uid_emb, hist_item_seq_emb, gender_emb, age_emb, occupation_emb, zip_emb])
    all_embs = keras.layers.Flatten()(all_embs)

    hidden_1 = keras.layers.Dense(1024, 'relu')(all_embs)
    hidden_2 = keras.layers.Dense(512, 'relu')(hidden_1)
    hidden_3 = keras.layers.Dense(256, 'relu')(hidden_2)
    final_out = keras.layers.Dense(EMB_DIM)(hidden_3)

    model = keras.Model(
        inputs=[uid, hist_item_seq, hist_item_len, gender, age, occupation, zip_input],
        outputs=[final_out]
    )
    user_emb = model(
        [uid, hist_item_seq, hist_item_len, gender, age, occupation, zip_input]
    )

    embedding_index = EmbeddingIndex()
    all_item_emb = iid_emb_layer(embedding_index(item_size))

    iid = keras.Input(shape=[1,], dtype=tf.int64, name='iid')
    bias = tf.get_variable(name='bias', shape=[item_size,], initializer=tf.initializers.zeros(), trainable=False)
    loss = tf.nn.sampled_softmax_loss(
        weights=all_item_emb,
        biases=bias,
        labels=iid,
        inputs=user_emb,
        num_sampled=16,
        num_classes=item_size
    )
    loss = tf.reduce_mean(loss)
    optimizer = keras.optimizers.Adam()
    model_vars = tf.trainable_variables()
    grads = tf.gradients(loss, model_vars)
    train_op = optimizer.apply_gradients(zip(grads, model_vars))

    index = faiss.IndexFlatIP(EMB_DIM)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    print('-' * 120)
    with tf.Session(config=config) as sess:
        sess.run(tf.initializers.global_variables())
        train_size = train_input['size']
        val_size = val_input['size']
        prev_time = time()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            model.save_weights(ckpt_path + f'/{epoch}')
            train_size = train_input['size']
            val_size = val_input['size']
            batch_num = (train_size - 1) // BATCH_SIZE + 1
            best_val_loss = float('inf')
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
                print('\r' + '-' * 32 + ' ' * 6 + f'batch_loss: {batch_loss:.8f}' + ' ' * 6  + '-' * 32, end='')
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
            print(f'val_loss  of  epoch-{epoch + 1}: {(val_loss / batch_num):.8f}    ' +
                '-' * 36 + f'    tiem elapsed: {time_elapsed:.2f}s')
            print(f'-' * 32 +  f'   hr: {hr:.6f}, ndcg: {ndcg:.6f}   ' +  '-' * 32)
            print('-' * 120)

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
        print('-' * 120)
        print(f'-------------test_loss of: {eval_loss:.8f}, \
            tiem elapsed: {time_elapsed:.2f}s -------------')
        print(f'-' * 32 +  f'   hr: {hr:.6f}, ndcg: {ndcg:.6f}   ' +  '-' * 32)
