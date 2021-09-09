import os
from time import localtime, strftime, time

import numpy as np

from basek.params import args
from basek.preprocessors.ml_preprocessor import read_raw_data, gen_data_set, gen_model_input
from basek.layers.embedding import EmbeddingIndex
from basek.utils.tf_compat import tf, keras


SEQ_LEN = 50
NEG_SAMPLES = 0
VALIDATION_SPLIT = 0.1
EMB_DIM = 16
EPOCHS = 10
BATCH_SIZE = 256


if __name__ == '__main__':

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    ckpt_path = './ckpts/' + timestamp
    log_path = './logs/' + timestamp
    os.makedirs(ckpt_path)
    os.makedirs(log_path)

    data_path ='./datasets/movielens'
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    data, sparse_features_max_idx = read_raw_data(data_path, sparse_features)

    datasets, profiles =  gen_data_set(data, NEG_SAMPLES)
    train_dataset, test_dataset = datasets
    user_profile, item_profile = profiles

    train_input, val_input = gen_model_input(train_dataset, VALIDATION_SPLIT)
    test_dataset, _ = gen_model_input(train_dataset)


    uid = keras.Input(shape=[1,], dtype=tf.int64, name='uid')
    iid = keras.Input(shape=[1,], dtype=tf.int64, name='iid')
    lable = keras.Input(shape=[1,], dtype=tf.float32, name='lable')
    hist_item_seq = keras.Input(shape=[SEQ_LEN,], dtype=tf.int64, name='hist_item_seq')
    hist_item_len = keras.Input(shape=[1,], dtype=tf.int64, name='hist_item_len')
    gender = keras.Input(shape=[1,], dtype=tf.int64, name='gender')
    age = keras.Input(shape=[1,], dtype=tf.int64, name='age')
    occupation = keras.Input(shape=[1,], dtype=tf.int64, name='occupation')
    zip_input = keras.Input(shape=[1,], dtype=tf.int64, name='zip')

    uid_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['user_id'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True
    )
    iid_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['movie_id'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True
    )
    gender_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['gender'], 1,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True
    )
    age_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['age'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True
    )
    occupation_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['occupation'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True
    )
    zip_emb_layer = keras.layers.Embedding(
        sparse_features_max_idx['zip'], EMB_DIM,
        embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True
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
        lambda x: tf.reduce_mean(x[0] * x[1], axis=1, keepdims=True)
    )([hist_item_seq_emb, mask])

    all_embs = keras.layers.Concatenate(
        axis=-1
    )([uid_emb, gender_emb, age_emb, occupation_emb, zip_emb, hist_item_seq_emb])
    all_embs = keras.layers.Flatten()(all_embs)

    hidden_1 = keras.layers.Dense(1024, 'relu')(all_embs)
    hidden_2 = keras.layers.Dense(512, 'relu')(hidden_1)
    hidden_3 = keras.layers.Dense(256, 'relu')(hidden_2)

    user_emb = keras.layers.Dense(EMB_DIM)(hidden_3)

    model = keras.Model(
        inputs=[
            uid, iid, hist_item_seq, hist_item_len,
            gender, age, occupation, zip_input
        ],
        outputs=[
            user_emb
        ]
    )

    embedding_index = EmbeddingIndex()
    all_item_emb = iid_emb_layer(embedding_index(sparse_features_max_idx['movie_id']))

    bias = tf.get_variable(
        name='bias', shape=[sparse_features_max_idx['movie_id'],],
        initializer='zeros', trainable=False
    )

    out = model(
        [
            uid, iid, hist_item_seq, hist_item_len,
            gender, age, occupation, zip_input
        ]
    )
    loss = tf.nn.sampled_softmax_loss(
        weights=all_item_emb,
        biases=bias,
        labels=lable,
        inputs=out,
        num_sampled=5,
        num_classes=sparse_features_max_idx['movie_id']
    )
    loss = tf.reduce_mean(loss)
    optimizer = keras.optimizers.Adam()
    model_vars = tf.trainable_variables()
    grads = tf.gradients(loss, model_vars)
    train_op = optimizer.apply_gradients(zip(grads, model_vars))

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for epoch in range(EPOCHS):
            total_loss = 0.0
            model.save_weights(ckpt_path + f'/{epoch}')
            batch_num = (train_size - 1) // BATCH_SIZE + 1
            for i in range(batch_num):
                batch_loss, _ = sess.run(
                    [
                        loss, train_op
                    ],
                    feed_dict={
                        uid: train_dataset['uid'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        iid: train_dataset['iid'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        hist_item_seq: train_dataset['hist_item_seq'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        hist_item_len: train_dataset['hist_item_len'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        gender: train_dataset['gender'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        age: train_dataset['age'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        occupation: train_dataset['occupation'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                        zip_input: train_dataset['zip'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    }
                )
                total_loss += batch_loss
            eval_loss = sess.run(
                loss,
                feed_dict={
                    uid: train_dataset['uid'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    iid: train_dataset['iid'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    hist_item_seq: train_dataset['hist_item_seq'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    hist_item_len: train_dataset['hist_item_len'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    gender: train_dataset['gender'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    age: train_dataset['age'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    occupation: train_dataset['occupation'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                    zip_input: train_dataset['zip'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                }
            )




