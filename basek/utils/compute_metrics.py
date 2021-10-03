from collections import defaultdict

from basek.utils.imports import numpy as np


def compute_metrics(I, match_points, ground_truth_iid_seq, hist_iid_seq):
    batch_size = len(I)
    match_points = sorted(list(set(match_points)))
    max_match_point = match_points[-1]

    hits_matrix = np.zeros(shape=(batch_size, max_match_point), dtype=np.int64)
    novelty_hits_matrix = np.zeros(shape=(batch_size, max_match_point), dtype=np.int64)
    mrrs = np.zeros(shape=(batch_size, max_match_point))
    groud_truth_count = np.zeros(shape=(batch_size), dtype=np.int64)
    idcgs_cut_point = np.full(shape=(batch_size), fill_value=max_match_point)

    dcgs = np.zeros(shape=(batch_size, max_match_point))
    idcgs = np.cumsum([1.0 / np.log2(i + 2) for i in range(max_match_point)]).reshape(1, -1)
    idcgs = np.tile(idcgs, [batch_size, 1])

    for per in range(batch_size):
        per_pred = I[per]
        per_ground_truth_iid_seq = set(ground_truth_iid_seq[per]) - set([0])
        per_len_ground_truth = len(per_ground_truth_iid_seq)
        groud_truth_count[per] = per_len_ground_truth
        idcgs_cut_point[per] = min(per_len_ground_truth, max_match_point)
        per_hist_iid_seq = set(hist_iid_seq[per]) - set([0])

        per_hits = hits_matrix[per]
        per_dcgs = dcgs[per]
        per_mrrs = mrrs[per]
        per_novelty_hits = novelty_hits_matrix[per]

        first_match = False
        for pred_point, pred in enumerate(per_pred):
            if pred in per_ground_truth_iid_seq:
                if not first_match:
                    first_match = True
                    per_mrrs[pred_point:] = 1.0 / (pred_point + 1)
                per_hits[pred_point] = 1
                per_dcgs[pred_point] = per_dcgs[pred_point - 1] + 1.0 / np.log2(pred_point + 2)
            if pred not in per_hist_iid_seq:
                per_novelty_hits[pred_point] = 1

    groud_truth_count = groud_truth_count.reshape(-1, 1)
    cumsum_hits_matrix = np.cumsum(hits_matrix, axis=-1)
    recalls = cumsum_hits_matrix / groud_truth_count
    precisions = cumsum_hits_matrix / np.arange(1, max_match_point + 1).reshape(1, -1)
    f1_scores = (2 * recalls * precisions) / (recalls + precisions + 1e-16)

    ndcgs = np.zeros(shape=(batch_size, max_match_point))
    for per in range(batch_size):
        per_idcgs_cut_point = idcgs_cut_point[per]
        idcgs[per][per_idcgs_cut_point:] = idcgs[per][per_idcgs_cut_point - 1]
        ndcgs[per] = dcgs[per] / idcgs[per]
    maps = np.cumsum(np.where(hits_matrix, precisions, np.zeros_like(precisions)), axis=-1) / groud_truth_count
    novelties = np.cumsum(novelty_hits_matrix, axis=-1) / np.arange(1, max_match_point + 1).reshape(1, -1)

    metrics = defaultdict(dict)
    for match_point in match_points:
        metrics['Recall'][match_point] = recalls[:, match_point - 1]
        metrics['Precision'][match_point] = precisions[:, match_point - 1]
        metrics['F1_Score'][match_point] = f1_scores[:, match_point - 1]
        metrics['MAP'][match_point] = maps[:, match_point - 1]
        metrics['MRR'][match_point] = mrrs[:, match_point - 1]
        metrics['NDCG'][match_point] = ndcgs[:, match_point - 1]
        metrics['Novelty'][match_point] = novelties[:, match_point - 1]
    return metrics
