import argparse
import os

import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np

from multiprocessing import Process


parser = argparse.ArgumentParser(description='parser')


args, _ = parser.parse_known_args()


if __name__ == '__main__':
    dataset_path = '/data/project/datasets/Taobao'
    all_dirs = list(map(lambda x: os.path.join(dataset_path, x), os.listdir(dataset_path)))
    all_dirs = list(filter(lambda x: os.path.isdir(x), all_dirs))

    for dataset_dir in all_dirs:
        dataset_file = os.path.join(dataset_dir, 'dataset_df.pkl')
        dataset_df = pd.read_pickle(dataset_file)
        uid_all_interaction_index = {}
        for uid, uid_hist in tqdm(dataset_df.groupby('uid')):
            hist_iid_seq = np.array(uid_hist['iid'].to_list())
            hist_cid_seq = np.array(uid_hist['cid'].to_list())
            hist_bid_seq = np.array(uid_hist['bid'].to_list())
            hist_ts_seq = np.array(uid_hist['timestamp'].to_list())
            all_interaction_index = list(zip(hist_iid_seq, hist_cid_seq, hist_bid_seq, hist_ts_seq))
            all_interaction_index.sort(key=lambda x: x[-1])
            uid_all_interaction_index[uid] = all_interaction_index
        u2i_index_path = os.path.join(dataset_dir, 'u2i_index.pkl')
        with open(u2i_index_path, 'wb') as f:
            pkl.dump(uid_all_interaction_index, f)
