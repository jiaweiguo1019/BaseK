from basek.preprocessors.split_time_taobao_gen import train_writer

import argparse
import os

from multiprocessing import Process

parser = argparse.ArgumentParser(description='parser')


args, _ = parser.parse_known_args()

if __name__ == '__main__':

    p_s = []

    records_prefix = '/data/project/tfrecords_dataset'
    dataset_prefix = '/home/web_server/guojiawei/basek/datasets/Taobao'

    for seq_len in (50, 100):
        for k_core in (20, 10, 5):
            for only_click in (True, False):
                train_records_path = f'seq_len_{seq_len}-k_core_{k_core}-id_ordered_by_count-train.tfrecords'
                train_dataset_df_path = f'k_core_{k_core}-id_ordered_by_count-train_dataset_df.pkl'
                sparse_features_max_idx_path = f'k_core_{k_core}-id_ordered_by_count-sparse_features_max_idx.pkl'
                iid_to_cid_path = f'k_core_{k_core}-id_ordered_by_count-iid_to_cid.pkl'
                bid_sample_tempature_path = f'k_core_{k_core}-id_ordered_by_count-bid_sample_tempature.pkl'
                if only_click:
                    train_records_path = f'only_click-{train_records_path}'
                    train_dataset_df_path = f'only_click-{train_dataset_df_path}'
                    sparse_features_max_idx_path = f'only_click-{sparse_features_max_idx_path}'
                    iid_to_cid_path = f'only_click-{iid_to_cid_path}'
                    bid_sample_tempature_path = f'only_click-{bid_sample_tempature_path}'
                p = Process(
                    target=train_writer,
                    kwargs={
                        'train_records_path': os.path.join(records_prefix, train_records_path),
                        'train_dataset_df_path': os.path.join(dataset_prefix, train_dataset_df_path),
                        'sparse_features_max_idx_path': os.path.join(dataset_prefix, sparse_features_max_idx_path),
                        'iid_to_cid_path': os.path.join(dataset_prefix, iid_to_cid_path),
                        'bid_sample_tempature_path': os.path.join(dataset_prefix, bid_sample_tempature_path),
                        'seq_len': seq_len,
                        'neg_samples': None
                    }
                )
                p_s.append(p)
                p.daemon = True
                p.start()

    for p in p_s:
        p.join()
