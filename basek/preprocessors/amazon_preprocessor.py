import os
import pandas as pd
from collections import defaultdict
from basek.utils.imports import numpy as np
from tqdm import tqdm


def read_reviews(meta_path, review_path, true_sample_weight=1.0, neg_sampes=0, neg_sample_weight=1.0):

    abspath = os.path.abspath(meta_path)
    dir_path = os.path.split(abspath)[0]
    review_info_train = os.path.join(dir_path, 'reviw_info_train')
    review_info_test = os.path.join(dir_path, 'reviw_info_test')

    asin_to_cate = {}
    asin_set, cate_set = set(), set()
    with open(meta_path, 'r') as f_in:
        print('-' * 32 + '     processing items     ' + '-' * 32)
        for line in tqdm(f_in):
            line = eval(line)
            asin, cate = line['asin'], line['categories'][0][-1]
            asin_set.add(asin)
            cate_set.add(cate)
            asin_to_cate[asin] = cate

    asin_count = defaultdict(int)
    for asin in asin_set:
        asin_count[asin] = 0
    cate_count = defaultdict(int)
    for cate in cate_set:
        cate_count[cate] = 0

    user_count = defaultdict(int)
    all_lines = []
    with open(review_path, 'r') as f_in:
        print('-' * 32 + '    processing samples    ' + '-' * 32)
        for line in tqdm(f_in):
            line = eval(line)
            user = line['reviewerID']
            asin = line['asin']
            rating = str(line["overall"])
            timestamp = line["unixReviewTime"]
            user_count[user] += 1
            asin_count[asin] += 1
            cate = asin_to_cate.get(asin)
            if cate is not None:
                cate_count[cate] += 1
            else:
                cate = 'default_cate'
            all_lines.append(((user, asin, cate, rating), timestamp))
    all_lines.sort(key=lambda x: x[-1])
    user_size = len(user_count)
    item_size = len(asin_count)
    cate_size = len(cate_count)
    print('-' * 16 + f'    user_size: {user_size}, item_size: {item_size}, ' +
          f'cate_size: {cate_size}    ' + '-' * 16)

    item_to_iid = {'null': 0, 'default_item': 1}
    iid_to_item = {0: 'null', 1: 'default_item'}
    asin_count = dict(sorted(asin_count.items(), key=lambda x: x[-1]))
    for idx, asin in enumerate(asin_count):
        item_to_iid[asin] = idx + 2
        iid_to_item[idx + 2] = asin
    item_to_iid = pd.Series(item_to_iid)
    iid_to_item = pd.Series(iid_to_item)
    item_to_iid.to_csv(os.path.join(dir_path, 'item_to_iid.csv'), sep='^', header=False)
    iid_to_item.to_csv(os.path.join(dir_path, 'iid_to_item.csv'), sep='^', header=False)

    cate_to_cid = {'null': 0, 'default_cate': 1}
    cid_to_cate = {0: 'null', 1: 'default_cate'}
    cate_count = dict(sorted(cate_count.items(), key=lambda x: x[-1]))
    for idx, cate in enumerate(cate_count):
        cate_to_cid[cate] = idx + 2
        cid_to_cate[idx + 2] = cate
    cate_to_cid = pd.Series(cate_to_cid)
    cid_to_cate = pd.Series(cid_to_cate)
    cate_to_cid.to_csv(os.path.join(dir_path, 'cate_to_cid.csv'), sep='^', header=False)
    cid_to_cate.to_csv(os.path.join(dir_path, 'cid_to_cate.csv'), sep='^', header=False)

    user_to_uid = {'null': 0, 'default_user': 1}
    uid_to_user = {0: 'null', 1: 'default_user'}
    user_count = dict(sorted(user_count.items(), key=lambda x: x[-1]))
    for idx, user in enumerate(user_count):
        user_to_uid[user] = idx + 2
        uid_to_user[idx + 2] = user
    user_to_uid = pd.Series(user_to_uid)
    uid_to_user = pd.Series(uid_to_user)
    user_to_uid.to_csv(os.path.join(dir_path, 'user_to_uid.csv'), sep='^', header=False)
    uid_to_user.to_csv(os.path.join(dir_path, 'uid_to_user.csv'), sep='^', header=False)

    pos_label = str(1.0)
    true_sample_weight = str(true_sample_weight)
    neg_label = str(0.0)
    neg_sample_weight = str(neg_sample_weight)
    neg_rating = str(0.0)
    user_hist_iid_seq = defaultdict(str)
    user_hist_cid_seq = defaultdict(str)
    write_user_count = defaultdict(int)
    true_train_samples, neg_train_samples, test_samples = 0, 0, 0

    with open(review_info_train, 'w') as f_train, open(review_info_test, 'w') as f_test:
        print('-' * 32 + '    writing samples    ' + '-' * 32)
        # print(user_count)
        for line in tqdm(all_lines):
            line = line[0]
            user, asin, cate, rating = line
            hist_seq_len = str(write_user_count[user])
            write_user_count[user] += 1
            write_test = False
            if write_user_count[user] == user_count[user]:
                f_out = f_test
                test_samples += 1
                write_test = True
            else:
                f_out = f_train
                true_train_samples += 1
            hist_iid_seq = user_hist_iid_seq[user] + ' '
            hist_cid_seq = user_hist_cid_seq[user] + ' '
            uid = str(user_to_uid[user])
            iid = str(item_to_iid[asin])
            cid = str(cate_to_cid[cate])
            user_hist_iid_seq[user] = hist_iid_seq + iid
            user_hist_cid_seq[user] = hist_cid_seq + cid
            f_out.write(
                '\t'.join(
                    (
                        uid, iid, cid, pos_label, hist_iid_seq, hist_cid_seq,
                        hist_seq_len, true_sample_weight, rating
                    )
                )  + '\n'
            )
            if neg_sampes > 0 and not write_test:
                all_neg_samples_iid = set()
                for _ in range(neg_sampes):
                    neg_train_samples += 1
                    while True:
                        neg_sampe_iid = np.random.choice(item_size) + 2
                        if iid_to_item[neg_sampe_iid] != asin and neg_sampe_iid not in all_neg_samples_iid:
                            all_neg_samples_iid.add(neg_sampe_iid)
                            break
                    neg_asin = iid_to_item[neg_sampe_iid]
                    cate = asin_to_cate.get(neg_asin)
                    if cate is None:
                        cate = 'default_cate'
                    neg_sample_cid = cate_to_cid[cate]
                    neg_sampe_iid, neg_sample_cid = str(neg_sampe_iid), str(neg_sample_cid)

                    f_out.write(
                        '\t'.join(
                            (
                                uid, neg_sampe_iid, neg_sample_cid, neg_label,
                                hist_iid_seq, hist_cid_seq, hist_seq_len,
                                true_sample_weight, rating, neg_rating
                            )
                        ) + '\n'
                    )
    total_train_samples = true_train_samples + neg_train_samples
    print('-' * 4 + f'  {total_train_samples} training samples, ' +
          f'{true_train_samples} of them are true_train_samples, ' +
          f'{neg_train_samples} of them are neg_train_samples  ' + '-' * 4)
    print('-' * 32 + f'    {test_samples} testing samples    ' + '-' * 32)

    return (review_info_train, review_info_test), (user_size, item_size, cate_size)
