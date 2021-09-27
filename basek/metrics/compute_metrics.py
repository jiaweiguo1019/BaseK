
import enum
from numpy.core import shape_base
from tqdm import tqdm

from basek.utils.imports import numpy as np


def compute_metrics(I, match_points, ground_truth, threshold=3.0):
    batch_size = len(I)
    num_match_points = len(match_points)
    max_match_point = match_points[-1]
    recalls = np.zeros(shape=(batch_size, max_match_point))
    precisions = np.zeros(shape=(batch_size, max_match_point))
    ndcgs = np.zeros(shape=(batch_size, max_match_point))
    threshold_ndcgs = np.zeros(shape=(batch_size, max_match_point))
    for per in batch_size:
        per_pred = I[per]
        per_ground_truth = ground_truth[per]
        groud_truth_iid_rating_map, groud_truth_cid_rating_map, hist_iid_rating_map, hist_cid_rating_map = \
            per_ground_truth
        per_hits = np.zeros(shape=(max_match_point,))
        per_idcg_len = min(len(groud_truth_iid_rating_map), max_match_point)
        per_idcgs = np.cumsum([1.0 / np.log2(np.arange(2, max_match_point + 2))])
        per_idcgs[per_idcg_len:] = per_idcgs[per_idcg_len - 1]
        per_dcgs = np.zeros(shape=(max_match_point,))
        for match_point, pred in enumerate(per_pred):
            if pred in groud_truth_iid_rating_map:
                per_hits = per_hits[match_point - 1] + 1.0
                per_dcgs[match_point] = per_dcgs[match_point - 1] + np.log2(match_point + 2)
            else:
                per_hits = per_hits[match_point - 1]
                per_dcgs[match_point] = per_dcgs[match_point - 1]
        per_recalls = per_hits / len(groud_truth_iid_rating_map)
        recalls[per] = per_recalls
        per_precisons = per_hits / np.arange(1, 1 + max_match_point)
        precisions[per] = per_precisons
        ndcgs[per] = per_dcgs / (per_idcgs + 1e-16)
















