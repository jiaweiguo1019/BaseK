import argparse
import os

import pandas as pd
from tqdm import tqdm
import pickle as pkl

from multiprocessing import Process


parser = argparse.ArgumentParser(description='parser')


args, _ = parser.parse_known_args()


def uid_u2i_index(dataset_df_path):

    dataset_df = pd.read_pickle(dataset_df_path)
    abspath = os.path.abspath(dataset_df_path)
    dirpath, dataset_df_filename = os.path.split(abspath)
    index_prefix = '-'.join(dataset_df_filename.split('-')[:-1])
    index_filename = index_prefix + '-' + 'u2i_index.pkl'
    index_path = os.path.join(dirpath, index_filename)

    uid_all_iid_index = {}
    for uid, uid_hist in tqdm(dataset_df.groupby('uid')):
        all_iid = uid_hist['iid'].tolist()
        all_cid = uid_hist['cid'].tolist()
        all_bid = uid_hist['bid'].tolist()
        all_ts = uid_hist['timestamp'].tolist()
        all_iid_index = list(zip(all_iid, all_cid, all_bid, all_ts))
        all_iid_index.sort(key=lambda x: x[-1])
        uid_all_iid_index[uid] = all_iid_index

    with open(index_path, 'wb') as f:
        pkl.dump(uid_all_iid_index, f)


if __name__ == '__main__':
    dataset_path = '/home/web_server/guojiawei/basek/datasets/Taobao'
    all_files = os.listdir(dataset_path)
    all_files = list(map(lambda x: os.path.join(dataset_path, x), all_files))
    all_train_files = list(filter(lambda x: x.endswith('train_dataset_df.pkl'), all_files))

    len_all_train_files = len(all_train_files)
    for i in tqdm(range(0, len_all_train_files, 8)):
        part_train_files = all_train_files[i: i + 8]
        p_s = []
        for train_file in part_train_files:
            p = Process(
                target=uid_u2i_index,
                args=(train_file,)
            )
            p.daemon = True
            p.start()
            p_s.append(p)
        for p in p_s:
            p.join()
