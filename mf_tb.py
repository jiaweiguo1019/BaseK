import os
from time import localtime, strftime, time

import faiss
import numpy as np
from tqdm import tqdm
import _pickle as pkl
from basek.layers.base import Index

from basek.params import args
#from basek.preprocessors.ml_preprocessor_my import read_raw_data, gen_dataset_my, gen_model_input_my, gen_dataset_user_augment, gen_model_input_user_augment, gen_all_user_hist_info
from basek.preprocessors.ml_preprocessor_my import *
from basek.layers.embedding import EmbeddingIndex
from basek.utils.metrics import compute_metrics, compute_metrics_u2u, compute_metrics_u2u_2, ComputeMetrics
from basek.utils.tf_compat import tf, keras


SEQ_LEN = 50
NEG_SAMPLES = 0
VALIDATION_SPLIT = None
EMB_DIM = 32
EPOCHS = 10
BATCH_SIZE = 128
alpha = 1.0
contrast_item = True
hit_and_ndcg = [1, 5, 10, 20, 30, 50, 100, 120, 150, 180, 200]
train_cycle = 1000
test_cycle = 10
train_data_path = "./taobao/train_s2.tfrecods"
test_data_path = "./taobao/test_s2.tfrecods"
idx_data_path = "./taobao/index.pkl"

def CSELoss(y_pred):
    n = tf.shape(y_pred)[0]
    #y_true = tf.reshape(y_true, [-1, ])
    #y_true = tf.linalg.tensor_diag(y_true)
    y_true = tf.eye(n, dtype = tf.float32)
    loss = tf.keras.metrics.categorical_crossentropy(y_true, y_pred, from_logits=True)
    #y_pred = tf.nn.softmax(y_pred, axis = -1)
    #y_pred = tf.math.exp(y_pred)
    #y_ner = tf.reduce_sum(tf.multiply(y_true, y_pred), axis = -1)
    #y_der = tf.reduce_sum(tf.multiply(1.0-y_true, y_pred), axis = -1)
    #loss = -tf.math.log(y_ner) + tf.math.log(y_der+1e-6)
    #loss = tf.keras.metrics.binary_crossentropy(y_true, y_pred, from_logits=False)
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits = y_pred)
    #loss = tf.reduce_sum(loss, axis = -1)
    return tf.reduce_mean(loss)

def eval_on_test_u2i(sess, index, test_handle):
    index.reset()
    all_items = sess.run([all_item_emb_norm])
    index.add(all_items[0])
    caculator = ComputeMetrics(hit_and_ndcg)
    MATCH_NUMS = max(hit_and_ndcg)
    for i in range(test_cycle):
        try:
            test_user_query, test_uids, target_items = sess.run(
                        [user_final_norm, uids, s2_items], feed_dict={handle: test_handle})
            S, I = index.search(test_user_query, MATCH_NUMS)
            caculator.add_one_batch(I, target_items)
        except tf.errors.OutOfRangeError:
            break
    caculator.print_metrics()
    return
        
## just skip TODO 
def eval_on_test_u2u(sess, index, test_handle, u2i_index):
    index.reset()
    #all_user_uids = list()
    #while True:
    #    try:
    #        user_emb_query, all_uids = sess.run([s1_user_norm, uids], feed_dict={handle: all_user_handle})
    #        all_user_uids.append(uids)
    #        index.add(np.ascontiguousarray(user_emb_index))
    #    except tf.errors.OutOfRangeError:
    #        break
    #all_user_uids = np.concat(all_user_uids, axis = 0)
    all_test_uids, all_test_iids, all_test_times = list(), list(), list()
    all_queries = list()
    for i in range(10):
        try:
            test_user_query, test_uids, test_iids, test_times, slens = sess.run(
                        [s1_user_norm, uids, iids, times, s1_lens], feed_dict={handle: test_handle})
            #print(test_uids, test_iids , test_times, slens)
            all_test_uids.append(test_uids)
            all_test_iids.append(test_iids)
            all_test_times.append(test_times)
            embed = np.ascontiguousarray(test_user_query)
            all_queries.append(embed)
            index.add(embed)
        except tf.errors.OutOfRangeError:
            break
    hr, ndcg, num = 0, 0, 0
    all_uids = np.concatenate(all_test_uids, axis = 0)
    #print(all_queries, all_test_uids)
    for i in range(len(all_test_uids)):
        S, I = index.search(all_queries[i], MATCH_NUMS)
        #print("eval: ", I)
        #print(uid_i, iid_i, {x:user_list[x] for x in sim_users})
        hr1, ndcg1 = compute_metrics_u2u_2(I, all_test_uids[i], all_test_iids[i],
                            all_test_times[i], all_uids, u2i_index)
        hr += hr1 * I.shape[0]
        ndcg += ndcg1 * I.shape[0]
        num += I.shape[0]
    hr = hr * 1.0 /num
    ndcg = ndcg * 1.0 /num
    print(f'-' * 32 +  f'   hr: {hr:.6f}, ndcg: {ndcg:.6f}   ' +  '-' * 32)

test_feature_desp = {
    "uid": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "iid": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "time": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "label": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "s1_hist_item_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_tag_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_act_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_time_seq":tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_item_len":tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
train_feature_desp = {
    "uid": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "iid": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "time": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "label": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "s1_hist_item_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_tag_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_act_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_time_seq":tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s1_hist_item_len":tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "s2_hist_item_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s2_hist_tag_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s2_hist_act_seq": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s2_hist_time_seq":tf.io.FixedLenFeature([], tf.string, default_value=""),
    "s2_hist_item_len":tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}


def train_parse_func(data_proto):
    feature_dict = tf.io.parse_single_example(data_proto, train_feature_desp)
    for name in ["s1_hist_item_seq", "s1_hist_tag_seq", "s1_hist_act_seq", "s1_hist_time_seq",
            "s2_hist_item_seq", "s2_hist_tag_seq", "s2_hist_act_seq", "s2_hist_time_seq"]:
        feature_dict[name] = tf.io.decode_raw(feature_dict[name], out_type = tf.int64)
        feature_dict[name] = tf.reshape(feature_dict[name], [SEQ_LEN,])
    return feature_dict

def test_parse_func(data_proto):
    feature_dict = tf.io.parse_single_example(data_proto, test_feature_desp)
    for name in ["s1_hist_item_seq", "s1_hist_tag_seq", "s1_hist_act_seq", "s1_hist_time_seq"]:
        feature_dict[name] = tf.io.decode_raw(feature_dict[name], out_type = tf.int64)
        feature_dict[name] = tf.reshape(feature_dict[name], [SEQ_LEN,])
    return feature_dict

if __name__ == '__main__':
    train_data = tf.data.TFRecordDataset(train_data_path)
    train_data = train_data.map(train_parse_func).shuffle(buffer_size=4096, reshuffle_each_iteration=True) \
                    .repeat(EPOCHS).batch(BATCH_SIZE).prefetch(10)
    
    test_data = tf.data.TFRecordDataset(test_data_path)
    test_data = test_data.map(train_parse_func).repeat(EPOCHS*10).batch(BATCH_SIZE).prefetch(10)
    
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
    #iterator = tf.data.Iterator.from_structure(train_data.output_types)
                                               #train_data.output_shapes)
    one_batch = iterator.get_next()
    #train_iter_op = iterator.make_initializer(train_data)
    #test_iter_op = iterator.make_initializer(test_data)

    train_iterator = train_data.make_initializable_iterator()
    test_iterator = test_data.make_initializable_iterator()
    
    feature_name = ['user_id', 'item_id', 'cate', 'rating']
    enc, u2i_index = pkl.load(open(idx_data_path, "rb"))

    feature_counts = {feature_name[i]: v.shape[0] for i, v in enumerate(enc.categories_)}    
    user_size = feature_counts['user_id']
    item_size = feature_counts['item_id']
    print(feature_counts)

    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    ckpt_path = './ckpts/' + timestamp
    log_path = './logs/' + timestamp
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    print('-' * 120)
    print('-' * 4 + f'  model weights are saved in {os.path.realpath(ckpt_path)}  ' + '-' * 4)

    uid_emb_layer = keras.layers.Embedding(
        user_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='uid_emb_layer'
    )
    iid_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='iid_emb_layer'
    )
    target_iid_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='target_iid_emb_layer'
    )
    act_emb_layer = keras.layers.Embedding(
        feature_counts['rating'], 8, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='action_emb_layer'
    )
    time_emb_layer = keras.layers.Embedding(
        2000, 8, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='time_emb_layer'
    )
    tag_emb_layer = keras.layers.Embedding(
        feature_counts['cate'], 8, embeddings_initializer=keras.initializers.TruncatedNormal(),
        mask_zero=True, name='tag_emb_layer'
    )
    uids = one_batch['uid']
    iids = one_batch['iid']
    times = one_batch['time']
    s2_items = one_batch['s2_hist_item_seq']

    uid_emb = uid_emb_layer(uids)
    iid_emb = iid_emb_layer(iids)

    user_final_norm = tf.math.l2_normalize(uid_emb, axis =-1)


    ui_score1 = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b = True))([uid_emb, iid_emb])
    ui_score1 = ui_score1 * 5 #tf.nn.sigmoid(ui_score * 8)
    
    iid_index = tf.constant(np.arange(item_size), dtype = tf.int64)
    all_item_emb = iid_emb_layer(iid_index)
    all_item_emb_norm = tf.math.l2_normalize(all_item_emb, axis = 1)
    if (contrast_item):
        #ui_loss = 0.5*CSELoss(ui_score1) +  0.5*CSELoss(ui_score2)
        ui_loss = CSELoss(ui_score1)
    else:
        bias = tf.get_variable(name='bias', shape=[item_size,], initializer=tf.initializers.zeros(), trainable=False)
        loss = tf.nn.sampled_softmax_loss(
            weights=all_item_emb,
            biases=bias,
            labels=tf.reshape(iids, [-1, 1]),
            inputs=uid_emb,
            num_sampled=BATCH_SIZE,
            num_classes=item_size
        )
        ui_loss = tf.reduce_mean(loss)

    # just u2i loss
    loss = ui_loss
    
    optimizer = keras.optimizers.Adam()
    model_vars = tf.trainable_variables()
    grads = tf.gradients(loss, model_vars)
    train_op = optimizer.apply_gradients(zip(grads, model_vars))

    index = faiss.IndexFlatIP(EMB_DIM)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    print('-' * 120)
    exit = False
    with tf.Session(config=config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        prev_time = time()
        epoch = 0
        sess.run(tf.initializers.global_variables())
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        while True:
            epoch += 1
            iterator_num = 0
            #eval_on_test_u2u(sess, index, test_handle, u2i_index)
            eval_on_test_u2i(sess, index, test_handle)
            total_loss = 0.0
            for i in range(train_cycle):
                iterator_num += 1
                try:
                    batch_loss, _ = sess.run([loss, train_op], feed_dict={handle: train_handle})
                except tf.errors.OutOfRangeError:
                    #print("reinitalization!!!")
                    #sess.run(train_iterator.initializer)
                    exit = True
                    break
                total_loss += batch_loss
                print('\r' + '-' * 32 + ' ' * 6 + f'batch_loss: {batch_loss:.8f}' + ' ' * 6  + '-' * 32, end='')
            curr_time = time()
            time_elapsed = curr_time - prev_time
            prev_time = curr_time
            print(f'\ntrain_loss of iteration-{epoch + 1}*1k: {(total_loss / iterator_num):.8f}    ' +
                '-' * 36 + f'    time elapsed: {time_elapsed:.2f}s')
            if (exit):
                break
