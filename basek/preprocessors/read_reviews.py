import os
import pickle as pkl

import pandas as pd

from basek.preprocessors.read_raw_dataset import read_raw_dataset
from basek.preprocessors.utils import process_raw_dataset, build_entity_id_map, build_ordinal_entity_id_map


def read_reviews(
    dataset, dirpath, savepath,
    pp=None, drop_dups=False, k_core=None,
    id_ordered_by_count=True
):
    os.makedirs(savepath, exist_ok=True)
    item_to_cate, raw_dataset_df = \
        read_raw_dataset(dataset, dirpath, savepath, pp)
    processed_raw_dataset_df, user_count, item_count, cate_count, behavior_count = process_raw_dataset(
        dataset, raw_dataset_df, savepath, drop_dups=drop_dups, k_core=k_core
    )

    user_to_uid, uid_to_user = \
        build_entity_id_map(user_count, 'user', 'uid', savepath, id_ordered_by_count)
    item_to_iid, iid_to_item = \
        build_entity_id_map(item_count, 'item', 'iid', savepath, id_ordered_by_count)
    cate_to_cid, cid_to_cate = \
        build_entity_id_map(cate_count, 'cate', 'cid', savepath, id_ordered_by_count)

    if id_ordered_by_count:
        uid_count_path = os.path.join(savepath, 'id_ordered_by_count-uid_count.pkl')
        iid_to_cid_path = os.path.join(savepath, 'id_ordered_by_count-iid_to_cid.pkl')
        bid_to_sample_tempature_path = os.path.join(savepath, 'id_ordered_by_count-bid_to_sample_tempature.pkl')
    else:
        uid_count_path = os.path.join(savepath, 'no_id_ordered_by_count-uid_count.pkl')
        iid_to_cid_path = os.path.join(savepath, 'no_id_ordered_by_count-iid_to_cid.pkl')
        bid_to_sample_tempature_path = os.path.join(savepath, 'no_id_ordered_by_count-bid_to_sample_tempature.pkl')
    uid_count = {}
    for user, count in user_count.items():
        uid = user_to_uid[user]
        uid_count[uid] = count
    uid_count = pd.Series(uid_count)
    uid_count.to_pickle(uid_count_path)
    iid_to_cid = {}
    for item, iid in item_to_iid.items():
        cate = item_to_cate[item]
        cid = cate_to_cid[cate]
        iid_to_cid[iid] = cid
    iid_to_cid = pd.Series(iid_to_cid)
    iid_to_cid.to_pickle(iid_to_cid_path)

    behavior_to_bid, bid_to_behavior = \
        build_ordinal_entity_id_map(dataset, behavior_count, 'behavior', 'bid', savepath)
    bid_to_sample_tempature_path = os.path.join(savepath, 'bid_to_sample_tempature.pkl')
    bid_to_sample_tempature = {}
    for bid in bid_to_behavior.keys():
        bid_to_sample_tempature[bid] = bid - 1
    bid_to_sample_tempature = pd.Series(bid_to_sample_tempature)
    bid_to_sample_tempature.to_pickle(bid_to_sample_tempature_path)

    sparse_features_max_idx_path = os.path.join(savepath, 'sparse_features_max_idx.pkl')
    uid_size, iid_size, cid_size, bid_size = \
        len(uid_to_user), len(iid_to_item), len(cid_to_cate), len(bid_to_behavior)
    sparse_features_max_idx = {'uid': uid_size, 'iid': iid_size, 'cid': cid_size, 'bid': bid_size}
    print('#' * 132)
    print('-' * 16 + f'    uid_size: {uid_size}, iid_size: {iid_size}, ' + f'cid_size: {cid_size}    ' + '-' * 16)
    print('#' * 132)
    with open(sparse_features_max_idx_path, 'wb') as f:
        pkl.dump(sparse_features_max_idx, f)

    if id_ordered_by_count:
        dataset_df_path = os.path.join(savepath, 'id_ordered_by_count-dataset_df.pkl')
        all_indices_path = os.path.join(savepath, 'id_ordered_by_count-all_indices.pkl')
        freq_path_prefix = os.path.join(savepath, 'id_ordered_by_count-freq_path')
    else:
        dataset_df_path = os.path.join(savepath, 'no_id_ordered_by_count-dataset_df.pkl')
        all_indices_path = os.path.join(savepath, 'no_id_ordered_by_count-all_indices.pkl')
        freq_path_prefix = os.path.join(savepath, 'no_id_ordered_by_count-freq_path')
    all_iid_index = list(iid_to_item.keys())
    all_cid_index = list(iid_to_cid[all_iid_index])
    all_indices = all_iid_index, all_cid_index
    with open(all_indices_path, 'wb') as f:
        pkl.dump(all_indices, f)

    dataset_df = pd.DataFrame()
    dataset_df['uid'] = processed_raw_dataset_df['user'].map(user_to_uid)
    dataset_df['iid'] = processed_raw_dataset_df['item'].map(item_to_iid)
    iid_freq = dataset_df['iid'].map(dataset_df.groupby('iid')['iid'].count() / len(dataset_df))
    dataset_df['iid_freq'] = iid_freq
    dataset_df['cid'] = processed_raw_dataset_df['cate'].map(cate_to_cid)
    cid_freq = dataset_df['cid'].map(dataset_df.groupby('cid')['cid'].count() / len(dataset_df))
    dataset_df['cid_freq'] = cid_freq
    dataset_df['bid'] = processed_raw_dataset_df['behavior'].map(behavior_to_bid)
    bid_freq = dataset_df['bid'].map(dataset_df.groupby('bid')['bid'].count() / len(dataset_df))
    dataset_df['bid_freq'] = bid_freq
    dataset_df['timestamp'] = processed_raw_dataset_df['timestamp']
    iid_freq.to_pickle(f'{freq_path_prefix}_iid.pkl')
    cid_freq.to_pickle(f'{freq_path_prefix}_cid.pkl')
    bid_freq.to_pickle(f'{freq_path_prefix}_bid.pkl')
    dataset_df.sort_values('timestamp', inplace=True)
    dataset_df.reset_index(drop=True, inplace=True)
    dataset_df.to_pickle(dataset_df_path)
