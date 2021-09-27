

def ndcg(I, match_nums, ground_truth):





def compute_metrics(I, match_nums, uid, iid):
    match_nums = np.sort(match_nums)
    len_of_match_nums = len(match_nums)
    hits = [[] for _ in range(len_of_match_nums)]
    ndcgs = [[] for _ in range(len_of_match_nums)]

    for i, (uid_i, iid_i) in enumerate(zip(uid, iid)):
        uid_i, iid_i = uid_i[0], iid_i[0]
        pred = I[i]
        hit, ndcg = 0, 0.0
        for idx, pred_i in enumerate(pred):
            if pred_i == iid_i:
                ndcg = np.log(2) / np.log(idx + 2)
                hit = 1
                break
        hit_pos = idx + 1
        for idx, (hit_i, ndcg_i) in enumerate(zip(hits, ndcgs)):
            if match_nums[idx] >= hit_pos:
                hit_i.append(hit)
                ndcg_i.append(ndcg)
            else:
                hit_i.append(0)
                ndcg_i.append(0.0)

    return hits, ndcgs