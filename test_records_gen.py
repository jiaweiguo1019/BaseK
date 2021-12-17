from basek.preprocessors.split_time_taobao_gen import test_writer

import argparse
import os
import time

from multiprocessing import Process


parser = argparse.ArgumentParser(description='parser')


args, _ = parser.parse_known_args()


if __name__ == '__main__':

    records_prefix = '/data/project/test_tfrecords'
    dataset_prefix = '/home/web_server/guojiawei/basek/datasets/Taobao'
    split_ts = int(time.mktime(time.strptime('2017-12-3 0:0:0', '%Y-%m-%d %H:%M:%S')))

    p_s = []
    for seq_len in (50, 100):
        for k_core in (20, 10, 5):
            for only_click in (True, False):
                for test_drop_hist in (True, False):
                    if test_drop_hist:
                        test_records_path = \
                            f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
                    else:
                        test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
                    train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
                    test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
                    sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
                    iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
                    if only_click:
                        test_records_path = f'only_click-{test_records_path}'
                        train_dataset_df_path = f'only_click-{train_dataset_df_path}'
                        test_dataset_df_path = f'only_click-{test_dataset_df_path}'
                        sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
                        iid_to_cid_path = f'only_click-{iid_to_cid_path}'
                    p = Process(
                        target=test_writer,
                        kwargs={
                            'test_records_path': os.path.join(records_prefix, test_records_path),
                            'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
                            'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
                            'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
                            'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
                            'split_ts': split_ts,
                            'seq_len': seq_len,
                            'test_drop_hist': test_drop_hist,
                            'neg_samples': None
                        }
                    )
                    p_s.append(p)
                    p.daemon = True
                    p.start()

    for p in p_s:
        p.join()

    # p_s = []
    # seq_len = 50
    # k_core = 20
    # for only_click in (True, False):
    #     for test_drop_hist in (True, False):
    #         if test_drop_hist:
    #             test_records_path = \
    #                 f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         else:
    #             test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
    #         test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
    #         sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
    #         iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
    #         if only_click:
    #             test_records_path = f'only_click-{test_records_path}'
    #             train_dataset_df_path = f'only_click-{train_dataset_df_path}'
    #             test_dataset_df_path = f'only_click-{test_dataset_df_path}'
    #             sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
    #             iid_to_cid_path = f'only_click-{iid_to_cid_path}'
    #         p = Process(
    #             target=test_writer,
    #             kwargs={
    #                 'test_records_path': os.path.join(records_prefix, test_records_path),
    #                 'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
    #                 'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
    #                 'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
    #                 'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
    #                 'split_ts': split_ts,
    #                 'seq_len': seq_len,
    #                 'test_drop_hist': test_drop_hist,
    #                 'neg_samples': None
    #             }
    #         )
    #         p_s.append(p)
    #         p.daemon = True
    #         p.start()

    # for p in p_s:
    #     p.join()

    # p_s = []
    # seq_len = 50
    # k_core = 10
    # for only_click in (True, False):
    #     for test_drop_hist in (True, False):
    #         if test_drop_hist:
    #             test_records_path = \
    #                 f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         else:
    #             test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
    #         test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
    #         sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
    #         iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
    #         if only_click:
    #             test_records_path = f'only_click-{test_records_path}'
    #             train_dataset_df_path = f'only_click-{train_dataset_df_path}'
    #             test_dataset_df_path = f'only_click-{test_dataset_df_path}'
    #             sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
    #             iid_to_cid_path = f'only_click-{iid_to_cid_path}'
    #         p = Process(
    #             target=test_writer,
    #             kwargs={
    #                 'test_records_path': os.path.join(records_prefix, test_records_path),
    #                 'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
    #                 'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
    #                 'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
    #                 'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
    #                 'split_ts': split_ts,
    #                 'seq_len': seq_len,
    #                 'test_drop_hist': test_drop_hist,
    #                 'neg_samples': None
    #             }
    #         )
    #         p_s.append(p)
    #         p.daemon = True
    #         p.start()

    # for p in p_s:
    #     p.join()

    # p_s = []
    # seq_len = 50
    # k_core = 5
    # for only_click in (True, False):
    #     for test_drop_hist in (True, False):
    #         if test_drop_hist:
    #             test_records_path = \
    #                 f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         else:
    #             test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
    #         test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
    #         sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
    #         iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
    #         if only_click:
    #             test_records_path = f'only_click-{test_records_path}'
    #             train_dataset_df_path = f'only_click-{train_dataset_df_path}'
    #             test_dataset_df_path = f'only_click-{test_dataset_df_path}'
    #             sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
    #             iid_to_cid_path = f'only_click-{iid_to_cid_path}'
    #         p = Process(
    #             target=test_writer,
    #             kwargs={
    #                 'test_records_path': os.path.join(records_prefix, test_records_path),
    #                 'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
    #                 'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
    #                 'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
    #                 'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
    #                 'split_ts': split_ts,
    #                 'seq_len': seq_len,
    #                 'test_drop_hist': test_drop_hist,
    #                 'neg_samples': None
    #             }
    #         )
    #         p_s.append(p)
    #         p.daemon = True
    #         p.start()

    # for p in p_s:
    #     p.join()

    # p_s = []
    # seq_len = 100
    # k_core = 20
    # for only_click in (True, False):
    #     for test_drop_hist in (True, False):
    #         if test_drop_hist:
    #             test_records_path = \
    #                 f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         else:
    #             test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
    #         test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
    #         sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
    #         iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
    #         if only_click:
    #             test_records_path = f'only_click-{test_records_path}'
    #             train_dataset_df_path = f'only_click-{train_dataset_df_path}'
    #             test_dataset_df_path = f'only_click-{test_dataset_df_path}'
    #             sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
    #             iid_to_cid_path = f'only_click-{iid_to_cid_path}'
    #         p = Process(
    #             target=test_writer,
    #             kwargs={
    #                 'test_records_path': os.path.join(records_prefix, test_records_path),
    #                 'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
    #                 'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
    #                 'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
    #                 'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
    #                 'split_ts': split_ts,
    #                 'seq_len': seq_len,
    #                 'test_drop_hist': test_drop_hist,
    #                 'neg_samples': None
    #             }
    #         )
    #         p_s.append(p)
    #         p.daemon = True
    #         p.start()

    # for p in p_s:
    #     p.join()

    # p_s = []
    # seq_len = 100
    # k_core = 10
    # for only_click in (True, False):
    #     for test_drop_hist in (True, False):
    #         if test_drop_hist:
    #             test_records_path = \
    #                 f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         else:
    #             test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
    #         test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
    #         sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
    #         iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
    #         if only_click:
    #             test_records_path = f'only_click-{test_records_path}'
    #             train_dataset_df_path = f'only_click-{train_dataset_df_path}'
    #             test_dataset_df_path = f'only_click-{test_dataset_df_path}'
    #             sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
    #             iid_to_cid_path = f'only_click-{iid_to_cid_path}'
    #         p = Process(
    #             target=test_writer,
    #             kwargs={
    #                 'test_records_path': os.path.join(records_prefix, test_records_path),
    #                 'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
    #                 'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
    #                 'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
    #                 'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
    #                 'split_ts': split_ts,
    #                 'seq_len': seq_len,
    #                 'test_drop_hist': test_drop_hist,
    #                 'neg_samples': None
    #             }
    #         )
    #         p_s.append(p)
    #         p.daemon = True
    #         p.start()

    # for p in p_s:
    #     p.join()

    # p_s = []
    # seq_len = 100
    # k_core = 5
    # for only_click in (True, False):
    #     for test_drop_hist in (True, False):
    #         if test_drop_hist:
    #             test_records_path = \
    #                 f'seq_len_{seq_len}-test_drop_hist-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         else:
    #             test_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-test.tfrecords'
    #         train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
    #         test_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-test_dataset_df.pkl'
    #         sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
    #         iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
    #         if only_click:
    #             test_records_path = f'only_click-{test_records_path}'
    #             train_dataset_df_path = f'only_click-{train_dataset_df_path}'
    #             test_dataset_df_path = f'only_click-{test_dataset_df_path}'
    #             sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
    #             iid_to_cid_path = f'only_click-{iid_to_cid_path}'
    #         p = Process(
    #             target=test_writer,
    #             kwargs={
    #                 'test_records_path': os.path.join(records_prefix, test_records_path),
    #                 'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
    #                 'test_dataset_df_path': os.path.join(dataset_prefix, test_dataset_df_path),
    #                 'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
    #                 'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
    #                 'split_ts': split_ts,
    #                 'seq_len': seq_len,
    #                 'test_drop_hist': test_drop_hist,
    #                 'neg_samples': None
    #             }
    #         )
    #         p_s.append(p)
    #         p.daemon = True
    #         p.start()

    # for p in p_s:
    #     p.join()
