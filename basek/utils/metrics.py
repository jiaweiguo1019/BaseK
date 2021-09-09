from tqdm import tqdm

import numpy as np


def compute_metrics(I, uid, iid):
    hits, ndcgs = [], []
    #for i, (uid_i, iid_i) in tqdm(enumerate(zip(uid, iid))):
    for i, (uid_i, iid_i) in enumerate(zip(uid, iid)):
            uid_i, iid_i = uid_i[0], iid_i[0]
            pred = I[i]
            ndcg, hit = 0, 0
            for idx, pred_i in enumerate(pred):
                if pred_i == iid_i:
                    ndcg = np.log(2) / np.log(idx + 2)
                    hit = 1
                    break
            hits.append(hit)
            ndcgs.append(ndcg)
    return np.mean(hits), np.mean(ndcgs)
